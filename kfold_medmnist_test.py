from __future__ import annotations

"""Bateria de testes k-fold da arquitetura sparse-joint em imagens médicas.

Usa MedMNIST v2 (Yang et al., Scientific Data 2023), o benchmark padronizado de
imagem médica em 28x28. As métricas principais (ACC e AUC macro) são as mesmas
que o MedMNIST reporta, então os números aqui são comparáveis aos baselines
publicados.

Além das métricas de classificação, extrai as métricas próprias da arquitetura:
especialização entre grupos, uso do alfabeto (codebook) e recuperação VSA.

Uso:
    python kfold_medmnist_test.py                          # bloodmnist, 5 folds
    python kfold_medmnist_test.py --dataset dermamnist
    python kfold_medmnist_test.py --dataset pathmnist --folds 5 --epochs 20
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import plots
from base import SJConfig, build_model, encode, vsa_cleanup_accuracy


# ------------------------------------------------------------------ dados

def load_medmnist(name: str) -> tuple[np.ndarray, np.ndarray, list[str], tuple[int, int, int]]:
    """Junta os splits oficiais; a validação aqui é por k-fold, não pelo split."""

    import medmnist
    from medmnist import INFO

    info = INFO[name]
    if info["task"] == "multi-label, binary-class":
        raise SystemExit(f"{name} é multi-label; esta bateria assume classe única")

    cls = getattr(medmnist, info["python_class"])
    parts = [cls(split=s, download=True) for s in ("train", "val", "test")]

    images = np.concatenate([p.imgs for p in parts], axis=0)
    labels = np.concatenate([p.labels for p in parts], axis=0).reshape(-1).astype(np.int64)

    if images.ndim == 3:                       # escala de cinza -> [N, H, W, 1]
        images = images[..., None]
    height, width, channels = images.shape[1:]

    x = images.reshape(len(images), -1).astype(np.float32) / 255.0
    class_names = [info["label"][str(i)] for i in range(len(info["label"]))]
    return x, labels, class_names, (channels, height, width)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estatísticas do fold de treino apenas, para não vazar o fold de teste."""

    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    return ((train_x - mean) / std).astype(np.float32), ((test_x - mean) / std).astype(np.float32)


# ------------------------------------------------------------------ predição

@torch.no_grad()
def predict_proba(model, x: np.ndarray, device: torch.device, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(x), batch):
        chunk = torch.from_numpy(x[start:start + batch]).to(device)
        logits = model(chunk)["factor_logits"][0]
        out.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


# ------------------------------------------------------------------ métricas

def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y_pred = probs.argmax(axis=1)
    n_classes = probs.shape[1]

    if n_classes == 2:
        auc = roc_auc_score(y_true, probs[:, 1])
    else:
        present = np.unique(y_true)
        if len(present) < n_classes:          # fold sem alguma classe
            auc = roc_auc_score(y_true, probs[:, present], multi_class="ovr",
                                average="macro", labels=present)
        else:
            auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_macro": float(auc),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def specialization_metrics(codes: np.ndarray, labels: np.ndarray,
                           edges, card: int) -> dict[str, float]:
    """Mesmas definições validadas na branch add-tests-and-docs."""

    groups = codes.shape[1]
    nmis = np.array([normalized_mutual_info_score(labels, codes[:, g]) for g in range(groups)])

    pair_gains = []
    for i, j in edges:
        joint = codes[:, i].astype(np.int64) * card + codes[:, j].astype(np.int64)
        nmi_joint = normalized_mutual_info_score(labels, joint)
        pair_gains.append(nmi_joint - max(nmis[i], nmis[j]))

    between = [normalized_mutual_info_score(codes[:, i], codes[:, j])
               for i in range(groups) for j in range(i + 1, groups)]

    return {
        "nmi_code_label_mean": float(nmis.mean()),
        "nmi_code_label_std": float(nmis.std()),
        "nmi_code_label_max": float(nmis.max()),
        "nmi_between_groups": float(np.mean(between)) if between else 0.0,
        "edge_pair_gain": float(np.mean(pair_gains)) if pair_gains else 0.0,
    }


def alphabet_metrics(codes: np.ndarray, card: int) -> dict[str, float]:
    """Quantos fillers cada grupo realmente usa (o loss de support mira nisso)."""

    groups = codes.shape[1]
    used, perplexities = [], []
    for g in range(groups):
        counts = np.bincount(codes[:, g], minlength=card).astype(np.float64)
        p = counts / counts.sum()
        nz = p[p > 0]
        used.append(float((counts > 0).sum()))
        perplexities.append(float(np.exp(-(nz * np.log(nz)).sum())))
    return {
        "fillers_used_mean": float(np.mean(used)),
        "code_perplexity_mean": float(np.mean(perplexities)),
        "alphabet_utilization": float(np.mean(used) / card),
    }


# ------------------------------------------------------------------ encoder

class ConvEncoder(torch.nn.Module):
    """CNN pequena no lugar da densa do `FactorSJ`.

    O `FactorSJ.forward` faz `self.sj(self.encoder(x))`, então qualquer módulo
    que leve `[B, input_dim]` em `[B, frame_embed]` serve. Isto recebe o vetor
    achatado, remonta a imagem e devolve o embedding — sem tocar no `base.py`.

    Mesma ideia do `sj_image_classifier.py` da branch add-tests-and-docs, onde o
    core era alimentado por uma CNN e não por pixel cru.
    """

    def __init__(self, image_shape: tuple[int, int, int], frame_embed: int):
        super().__init__()
        channels, height, width = image_shape
        self.image_shape = image_shape
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.GELU(),
            torch.nn.MaxPool2d(2),                                   # 28 -> 14
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.GELU(),
            torch.nn.MaxPool2d(2),                                   # 14 -> 7
        )
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * (height // 4) * (width // 4), frame_embed),
            torch.nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels, height, width = self.image_shape
        img = x.view(len(x), height, width, channels).permute(0, 3, 1, 2)
        return self.head(self.features(img))


# ------------------------------------------------- penalidade de redundância

def pairwise_mi_penalty(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Informação mútua média entre os códigos (suaves) de cada par de grupos.

    Portado da branch `add-tests-and-docs`, onde derrubou a NMI entre grupos de
    ~0,84 para ~0,08 no MNIST sem custo de acurácia. `probs` é [B, G, C];
    minimizar isto empurra os grupos para códigos independentes.
    """

    batch, groups, _ = probs.shape
    joint = torch.einsum("bgi,bhj->ghij", probs, probs) / batch      # [G,G,C,C]
    marginal = probs.mean(0)                                         # [G,C]
    outer = marginal[:, None, :, None] * marginal[None, :, None, :]  # [G,G,C,C]
    mi = (joint * (torch.log(joint + eps) - torch.log(outer + eps))).sum(dim=(-1, -2))
    off_diagonal = ~torch.eye(groups, dtype=torch.bool, device=probs.device)
    return mi[off_diagonal].mean()


def train_supervised(model, x: np.ndarray, z: np.ndarray, *, device: torch.device,
                     redundancy_weight: float):
    """Espelha `base.train(mode="acquisition", pairs=None)` e opcionalmente soma
    a penalidade de redundância.

    Replicado aqui em vez de alterar o `base.py`: esta branch é só de teste, e
    com `redundancy_weight=0` o loop reproduz exatamente o baseline (serve de
    controle da comparação).
    """

    from base import acquisition_losses

    cfg = model.cfg
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    xt = torch.from_numpy(x).to(device)
    zt = torch.from_numpy(z).long().to(device)

    n = len(xt)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 7717)
    for _ in range(cfg.epochs):
        order = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, cfg.batch):
            index = order[start:start + cfg.batch]
            out = model(xt[index])
            losses = acquisition_losses(model, out, zt[index], None)
            if redundancy_weight:
                losses["redundancy"] = redundancy_weight * pairwise_mi_penalty(out["probs"])
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model.eval()


# ------------------------------------------------------------------ execução

def run_fold(fold: int, x_tr, z_tr, x_te, z_te, *, cfg_kwargs, n_classes,
             device, card, locality: bool = False, redundancy_weight: float = 0.0,
             encoder: str = "linear",
             image_shape: tuple[int, int, int] | None = None
             ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    cfg = SJConfig(
        cardinalities=tuple([card] * cfg_kwargs["groups"]),
        n_factors=1,
        input_dim=x_tr.shape[1],
        units=cfg_kwargs["units"],
        layers=cfg_kwargs["layers"],
        topk=cfg_kwargs["topk"],
        hv_dim=cfg_kwargs["hv_dim"],
        frame_embed=cfg_kwargs["frame_embed"],
        epochs=cfg_kwargs["epochs"],
        batch=cfg_kwargs["batch"],
        lr=cfg_kwargs["lr"],
        seed=cfg_kwargs["seed"] + fold,
    )

    torch.manual_seed(cfg.seed)
    model = build_model(cfg, (n_classes,))
    if encoder == "cnn":
        model.encoder = ConvEncoder(image_shape, cfg.frame_embed)

    started = time.time()
    model = train_supervised(model, x_tr, z_tr.reshape(-1, 1), device=device,
                             redundancy_weight=redundancy_weight)
    train_seconds = time.time() - started

    probs = predict_proba(model, x_te, device)
    codes = encode(model, x_te, device)

    metrics = classification_metrics(z_te, probs)
    metrics.update(specialization_metrics(codes, z_te, model.sj.edges, card))
    metrics.update(alphabet_metrics(codes, card))
    metrics["vsa_cleanup_accuracy"] = vsa_cleanup_accuracy(model, codes, device)
    if locality:
        # ATENÇÃO: a definição desta métrica vem do `full_path_locality`, que hoje
        # é reconstruído. Confirmar contra o módulo original antes de reportar.
        from base import off_subgraph_influence
        metrics["scaffold_outside_fraction"] = float(
            off_subgraph_influence(model, x_te, device, rows=128))
    metrics["train_seconds"] = train_seconds

    return metrics, probs.argmax(axis=1), z_te, probs


def aggregate(per_fold: list[dict]) -> dict[str, dict[str, float]]:
    keys = per_fold[0].keys()
    return {
        k: {
            "mean": float(np.mean([f[k] for f in per_fold])),
            "std": float(np.std([f[k] for f in per_fold])),
            "min": float(np.min([f[k] for f in per_fold])),
            "max": float(np.max([f[k] for f in per_fold])),
        }
        for k in keys
    }


def write_report(path: Path, *, dataset, class_names, folds, cfg_kwargs, card,
                 summary, per_fold, y_true_all, y_pred_all) -> str:
    order = [
        ("accuracy", "acurácia"),
        ("auc_macro", "AUC macro (OvR)"),
        ("balanced_accuracy", "acurácia balanceada"),
        ("f1_macro", "F1 macro"),
        ("f1_weighted", "F1 ponderado"),
        ("precision_macro", "precisão macro"),
        ("recall_macro", "recall macro"),
        ("cohen_kappa", "kappa de Cohen"),
        ("mcc", "MCC"),
    ]
    arch = [
        ("nmi_code_label_mean", "NMI(código, rótulo) média por grupo"),
        ("nmi_code_label_max", "NMI(código, rótulo) do melhor grupo"),
        ("nmi_between_groups", "NMI entre grupos (redundância)"),
        ("edge_pair_gain", "ganho do par vs melhor grupo sozinho"),
        ("fillers_used_mean", f"fillers usados por grupo (de {card})"),
        ("code_perplexity_mean", "perplexidade do código"),
        ("alphabet_utilization", "utilização do alfabeto"),
        ("vsa_cleanup_accuracy", "recuperação VSA (cleanup)"),
        ("scaffold_outside_fraction", "gradiente fora do grupo (localidade)"),
        ("train_seconds", "segundos de treino por fold"),
    ]

    lines = [
        f"bateria k-fold | dataset: {dataset} | folds: {folds}",
        f"classes ({len(class_names)}): " + ", ".join(class_names),
        f"config: groups={cfg_kwargs['groups']} card={card} units={cfg_kwargs['units']} "
        f"layers={cfg_kwargs['layers']} topk={cfg_kwargs['topk']} hv_dim={cfg_kwargs['hv_dim']} "
        f"epochs={cfg_kwargs['epochs']} batch={cfg_kwargs['batch']} lr={cfg_kwargs['lr']} "
        f"redundancy_weight={cfg_kwargs.get('redundancy_weight', 0.0)}",
        "",
        "classificação (média ± desvio entre folds):",
    ]
    for key, label in order:
        s = summary[key]
        lines.append(f"  {label:<28} {s['mean']:.4f} ± {s['std']:.4f}   "
                     f"[{s['min']:.4f}, {s['max']:.4f}]")

    lines += ["", "arquitetura (média ± desvio entre folds):"]
    for key, label in arch:
        if key not in summary:
            continue
        s = summary[key]
        lines.append(f"  {label:<36} {s['mean']:.4f} ± {s['std']:.4f}")

    lines += ["", "por fold (acurácia / AUC macro):"]
    for i, f in enumerate(per_fold):
        lines.append(f"  fold {i}: acc={f['accuracy']:.4f}  auc={f['auc_macro']:.4f}  "
                     f"f1={f['f1_macro']:.4f}  ({f['train_seconds']:.1f}s)")

    lines += ["", "relatório por classe (todos os folds concatenados):",
              classification_report(y_true_all, y_pred_all, target_names=class_names,
                                    zero_division=0, digits=4),
              "matriz de confusão (linhas = verdadeiro, colunas = predito):",
              str(confusion_matrix(y_true_all, y_pred_all))]

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="bloodmnist")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--card", type=int, default=0, help="0 = max(16, n_classes)")
    parser.add_argument("--units", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--hv-dim", type=int, default=256)
    parser.add_argument("--frame-embed", type=int, default=128)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="subamostra para teste rápido")
    parser.add_argument("--encoder", choices=("linear", "cnn"), default="linear",
                        help="linear = densa do FactorSJ; cnn = CNN pequena no lugar dela")
    parser.add_argument("--redundancy-weight", type=float, default=0.0,
                        help="peso da penalidade de MI entre grupos (0 = baseline)")
    parser.add_argument("--locality", action="store_true",
                        help="mede localidade do scaffold (depende do sj_invariants reconstruído)")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dispositivo: {device}")

    x, y, class_names, image_shape = load_medmnist(args.dataset)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(x), size=min(args.limit, len(x)), replace=False)
        x, y = x[keep], y[keep]
    n_classes = len(class_names)
    card = args.card or max(16, n_classes)
    print(f"{args.dataset}: {len(x)} amostras, {x.shape[1]} features, {n_classes} classes")

    cfg_kwargs = {
        "groups": args.groups, "units": args.units, "layers": args.layers,
        "topk": args.topk, "hv_dim": args.hv_dim, "frame_embed": args.frame_embed,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": args.seed,
        "redundancy_weight": args.redundancy_weight, "encoder": args.encoder,
    }

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    per_fold, y_true_all, y_pred_all, test_index_all = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        x_tr, x_te = standardize(x[train_idx], x[test_idx])
        metrics, preds, truth, _ = run_fold(
            fold, x_tr, y[train_idx], x_te, y[test_idx],
            cfg_kwargs=cfg_kwargs, n_classes=n_classes, device=device, card=card,
            locality=args.locality, redundancy_weight=args.redundancy_weight,
            encoder=args.encoder, image_shape=image_shape)
        per_fold.append(metrics)
        y_pred_all.append(preds)
        y_true_all.append(truth)
        test_index_all.append(test_idx)
        print(f"  fold {fold}: acc={metrics['accuracy']:.4f} auc={metrics['auc_macro']:.4f} "
              f"f1={metrics['f1_macro']:.4f} ({metrics['train_seconds']:.1f}s)")

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    test_index_all = np.concatenate(test_index_all)
    summary = aggregate(per_fold)

    out_dir = Path(args.out or f"results/kfold/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps({
        "dataset": args.dataset, "folds": args.folds, "classes": class_names,
        "config": {**cfg_kwargs, "card": card, "n_samples": int(len(x)),
                   "redundancy_weight": args.redundancy_weight},
        "summary": summary, "per_fold": per_fold,
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all).tolist(),
    }, indent=2), encoding="utf-8")

    matrix = confusion_matrix(y_true_all, y_pred_all)
    plots.confusion_matrix_png(
        matrix, class_names, out_dir / "confusion_matrix.png",
        title=f"{args.dataset} — {args.folds} folds, encoder={args.encoder}")
    plots.sample_predictions_png(
        x[test_index_all], image_shape, y_true_all, y_pred_all, class_names,
        out_dir / "sample_predictions.png", seed=args.seed)
    plots.per_fold_png(per_fold, out_dir / "per_fold_metrics.png")

    text = write_report(out_dir / "report.txt", dataset=args.dataset,
                        class_names=class_names, folds=args.folds,
                        cfg_kwargs=cfg_kwargs, card=card, summary=summary,
                        per_fold=per_fold, y_true_all=y_true_all, y_pred_all=y_pred_all)
    print("\n" + text)
    print(f"\nsalvo em {out_dir}/")


if __name__ == "__main__":
    main()
