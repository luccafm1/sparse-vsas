from __future__ import annotations

"""Shared model, training loop and reporting code for the SparseJointCore image
classification tests (image_classification_test.py, fashion_classification_test.py).

Proves that the SparseJointCore VSA bottleneck (base.py) is domain-agnostic: no
video, no physics, just a small CNN image encoder feeding SparseJointCore, with
a linear head reading the resulting "program" vector.

The training loss includes a redundancy penalty: without it, every group's
discrete code independently converges to a near-complete guess of the label
(redundant, not complementary). The penalty pushes groups toward genuine
specialization -- individually weaker codes whose *combination* along
scaffold edges carries more class information than either alone.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, normalized_mutual_info_score
from torch.utils.data import DataLoader, Dataset

from base import SJConfig, SparseJointCore, make_rigid_scaffold, physical_edges

EPOCHS = 6
# Weight of the inter-group redundancy penalty (see pairwise_mi_penalty). 0
# disables it and lets groups converge to redundant copies of the label;
# ~4.0 is the smallest weight that reliably flips scaffold-edge pairs from
# redundant to complementary in this setup.
REDUNDANCY_WEIGHT = 4.0


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).flatten(1))


class ImageSJClassifier(nn.Module):
    def __init__(self, cfg: SJConfig, masks: torch.Tensor, seed: int, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = ImageEncoder(cfg.frame_embed)
        self.sj = SparseJointCore(cfg.frame_embed, cfg, masks, seed)
        self.head = nn.Linear(self.sj.program_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.sj(self.encoder(x), pair_writes=True)
        out["class_logits"] = self.head(out["program"])
        return out


def pairwise_mi_penalty(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batch estimate of mutual information between every pair of groups'
    (soft) discrete codes. probs: [B, G, C]. Returns the mean MI (nats)
    over all off-diagonal group pairs -- minimizing this pushes groups
    toward independent, non-redundant codes."""
    B, G, C = probs.shape
    joint = torch.einsum("bgi,bhj->ghij", probs, probs) / B          # [G,G,C,C]
    marginal = probs.mean(0)                                        # [G,C]
    outer = marginal[:, None, :, None] * marginal[None, :, None, :]  # [G,G,C,C]
    mi_terms = joint * (torch.log(joint + eps) - torch.log(outer + eps))
    mi = mi_terms.sum(dim=(-1, -2))                                  # [G,G]
    off_diag = ~torch.eye(G, dtype=torch.bool)
    return mi[off_diag].mean()


def specialization_report_text(codes: np.ndarray, labels: np.ndarray, edges, card: int) -> str:
    G = codes.shape[1]
    nmis = np.array([normalized_mutual_info_score(labels, codes[:, g]) for g in range(G)])
    pair_gains = []
    for i, j in edges:
        joint = codes[:, i].astype(np.int64) * card + codes[:, j].astype(np.int64)
        nmi_joint = normalized_mutual_info_score(labels, joint)
        pair_gains.append(nmi_joint - max(nmis[i], nmis[j]))
    code_code = [normalized_mutual_info_score(codes[:, i], codes[:, j])
                 for i in range(G) for j in range(i + 1, G)]

    lines = [
        "specialization report:",
        f"  NMI(code, label) per group: mean={nmis.mean():.4f} std={nmis.std():.4f} "
        f"values={[round(float(x), 3) for x in nmis]}",
        f"  NMI(code_i, code_j) between groups, averaged over all pairs: {np.mean(code_code):.4f}  "
        "(near 0 -> independent codes; high -> redundant)",
        f"  scaffold-edge pair gain vs best single group in the pair: {np.mean(pair_gains):+.4f}  "
        "(positive means the pair together beats either group alone -- real specialization)",
    ]
    return "\n".join(lines)


def plot_confusion_matrix(cm, class_names: list[str], title: str, path: str) -> None:
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6.5,
                     color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def plot_sample_predictions(images: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor,
                             class_names: list[str], path: str) -> None:
    n, cols = len(images), 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.4 * rows))
    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            continue
        ax.imshow(images[idx, 0].numpy(), cmap="gray")
        pred, target = preds[idx].item(), targets[idx].item()
        correct = pred == target
        ax.set_title(f"pred={class_names[pred]}\ntrue={class_names[target]}",
                      color="green" if correct else "red", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def run_classification_test(name: str, train_ds: Dataset, test_ds: Dataset, class_names: list[str],
                              out_dir: str, epochs: int = EPOCHS, redundancy_weight: float = REDUNDANCY_WEIGHT,
                              seed: int = 0) -> None:
    """Train an ImageSJClassifier on (train_ds, test_ds) and write
    confusion_matrix.png, sample_predictions.png and metrics.txt into out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed)
    # SJConfig novo: o alfabeto é por grupo (`cardinalities`) em vez de um `card`
    # único, e `n_factors`/`input_dim` passaram a ser obrigatórios. Aqui o encoder
    # é próprio (ImageEncoder), então `input_dim` só precisa ser coerente com o
    # que entra no core.
    cfg = SJConfig(cardinalities=(16,) * 8, n_factors=1, input_dim=64,
                   layers=2, units=32, topk=6, unit_width=8, hv_dim=64, frame_embed=64)
    masks = make_rigid_scaffold(cfg, seed=cfg.seed)
    model = ImageSJClassifier(cfg, masks, seed=cfg.seed, num_classes=len(class_names))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    log_lines = [f"=== {name} ==="]
    model.train()
    for epoch in range(epochs):
        last_loss = None
        for x, y in train_loader:
            out = model(x)
            vq, commit, kl, entropy = model.sj.vq_losses(out)
            redundancy = pairwise_mi_penalty(out["probs"])
            loss = (F.cross_entropy(out["class_logits"], y) + vq + 0.25 * commit + 0.05 * kl
                    + redundancy_weight * redundancy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        line = f"epoch {epoch + 1}/{epochs}  last batch loss={last_loss:.4f}"
        print(line)
        log_lines.append(line)

    model.eval()
    all_preds, all_targets, all_codes = [], [], []
    sample_images = sample_preds = sample_targets = None
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            out = model(x)
            preds = out["class_logits"].argmax(-1)
            all_preds.append(preds)
            all_targets.append(y)
            all_codes.append(out["codes"])
            if i == 0:
                sample_images, sample_preds, sample_targets = x[:16], preds[:16], y[:16]

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_codes = torch.cat(all_codes).numpy()
    accuracy = (all_preds == all_targets).mean()

    acc_line = f"\ntest accuracy: {accuracy:.4f} ({int((all_preds == all_targets).sum())}/{len(all_targets)})\n"
    report_text = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    spec_text = specialization_report_text(all_codes, all_targets, physical_edges(masks), cfg.max_card)

    print(acc_line)
    print("classification report:")
    print(report_text)
    print()
    print(spec_text)

    metrics_path = os.path.join(out_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
        f.write(acc_line + "\n")
        f.write("classification report:\n")
        f.write(report_text + "\n\n")
        f.write(spec_text + "\n")
    print(f"saved {metrics_path}")

    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names, f"Confusion matrix ({name} test set)",
                           os.path.join(out_dir, "confusion_matrix.png"))
    plot_sample_predictions(sample_images, sample_preds, sample_targets, class_names,
                             os.path.join(out_dir, "sample_predictions.png"))
