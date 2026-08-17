from __future__ import annotations

"""Proof that the SparseVQCore VSA bottleneck (base.py) is domain-agnostic:
no video, no physics, just a small CNN image encoder feeding SparseVQCore,
with a linear head reading the resulting "program" vector. Trains and
evaluates on MNIST digit classification, then reports a confusion matrix,
a per-class metrics table, and a grid of sample predictions.

Usage: python image_classification_test.py
Outputs: confusion_matrix.png, sample_predictions.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from base import SJConfig, SparseVQCore, make_rigid_scaffold

DATA_DIR = "data"
NUM_CLASSES = 10


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
        self.sj = SparseVQCore(cfg.frame_embed, cfg, masks, seed)
        self.head = nn.Linear(self.sj.program_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.sj(self.encoder(x), pair_writes=True)
        out["class_logits"] = self.head(out["program"])
        return out


def plot_confusion_matrix(cm, path: str = "confusion_matrix.png") -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (MNIST test set)")
    threshold = cm.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                     color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def plot_sample_predictions(images: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor,
                             path: str = "sample_predictions.png") -> None:
    n, cols = len(images), 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            continue
        ax.imshow(images[idx, 0].numpy(), cmap="gray")
        correct = preds[idx].item() == targets[idx].item()
        ax.set_title(f"pred={preds[idx].item()}  true={targets[idx].item()}",
                      color="green" if correct else "red", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    torch.manual_seed(0)
    cfg = SJConfig(groups=8, card=16, layers=2, units=32, topk=6, unit_width=8, hv_dim=64, frame_embed=64)
    masks = make_rigid_scaffold(cfg, seed=cfg.seed)
    model = ImageSJClassifier(cfg, masks, seed=cfg.seed, num_classes=10)

    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(3):
        last_loss = None
        for x, y in train_loader:
            out = model(x)
            vq, commit, kl, entropy = model.sj.vq_losses(out)
            loss = F.cross_entropy(out["class_logits"], y) + vq + 0.25 * commit + 0.05 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"epoch {epoch + 1}/3  last batch loss={last_loss:.4f}")

    model.eval()
    all_preds, all_targets = [], []
    sample_images = sample_preds = sample_targets = None
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            preds = model(x)["class_logits"].argmax(-1)
            all_preds.append(preds)
            all_targets.append(y)
            if i == 0:
                sample_images, sample_preds, sample_targets = x[:16], preds[:16], y[:16]

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    accuracy = (all_preds == all_targets).mean()

    print(f"\ntest accuracy: {accuracy:.4f} ({int((all_preds == all_targets).sum())}/{len(all_targets)})\n")
    print("classification report:")
    print(classification_report(all_targets, all_preds, digits=4))

    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm)
    plot_sample_predictions(sample_images, sample_preds, sample_targets)


if __name__ == "__main__":
    main()
