from __future__ import annotations

"""Saídas visuais da bateria k-fold.

Mesmas figuras que a branch `add-tests-and-docs` gerava (matriz de confusão e
grade de predições), mais as que fazem sentido com validação cruzada: métricas
por fold e a curva do trade-off da penalidade de redundância.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _short(name: str, limit: int = 18) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


def confusion_matrix_png(matrix: np.ndarray, class_names: list[str], path: Path,
                         title: str = "") -> None:
    """Matriz de confusão normalizada por linha, com contagens sobrepostas."""

    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    size = max(6.0, 0.72 * len(class_names) + 2.5)

    fig, ax = plt.subplots(figsize=(size, size * 0.86))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

    labels = [_short(c) for c in class_names]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predito")
    ax.set_ylabel("verdadeiro")
    ax.set_title(title or "matriz de confusão (todos os folds)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{normalized[i, j]:.2f}\n{matrix[i, j]}",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if normalized[i, j] > 0.55 else "black")

    fig.colorbar(image, ax=ax, fraction=0.046, label="fração da classe verdadeira")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sample_predictions_png(images: np.ndarray, image_shape, y_true: np.ndarray,
                           y_pred: np.ndarray, class_names: list[str], path: Path,
                           rows: int = 4, cols: int = 6, seed: int = 0) -> None:
    """Grade de exemplos com predição e rótulo (verde = acerto, vermelho = erro).

    Mostra acertos e erros de propósito, metade de cada quando houver erros.
    """

    rng = np.random.default_rng(seed)
    total = rows * cols
    wrong = np.flatnonzero(y_true != y_pred)
    right = np.flatnonzero(y_true == y_pred)

    n_wrong = min(len(wrong), total // 2)
    n_right = min(len(right), total - n_wrong)
    picks = np.concatenate([
        rng.choice(right, n_right, replace=False) if n_right else np.array([], int),
        rng.choice(wrong, n_wrong, replace=False) if n_wrong else np.array([], int),
    ])
    rng.shuffle(picks)

    channels, height, width = image_shape
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.95))
    for ax, index in zip(axes.ravel(), picks):
        img = images[index].reshape(height, width, channels)
        ax.imshow(img.squeeze(), cmap="gray" if channels == 1 else None)
        hit = y_true[index] == y_pred[index]
        ax.set_title(f"pred: {_short(class_names[y_pred[index]], 14)}\n"
                     f"real: {_short(class_names[y_true[index]], 14)}",
                     fontsize=6.5, color="green" if hit else "red")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")

    fig.suptitle("predições de exemplo", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def per_fold_png(per_fold: list[dict], path: Path) -> None:
    """Dispersão das métricas entre folds — mostra a estabilidade da validação."""

    groups = [
        ("classificação", ["accuracy", "auc_macro", "balanced_accuracy",
                           "f1_macro", "cohen_kappa", "mcc"]),
        ("arquitetura", ["nmi_code_label_mean", "nmi_between_groups",
                         "edge_pair_gain", "vsa_cleanup_accuracy",
                         "alphabet_utilization"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    for ax, (title, keys) in zip(axes, groups):
        keys = [k for k in keys if k in per_fold[0]]
        values = [[f[k] for f in per_fold] for k in keys]
        positions = np.arange(len(keys))

        ax.bar(positions, [np.mean(v) for v in values],
               yerr=[np.std(v) for v in values], capsize=4,
               color="#4C72B0", alpha=0.75, zorder=2)
        for pos, vals in zip(positions, values):
            ax.scatter(np.full(len(vals), pos), vals, s=16, color="#C44E52",
                       zorder=3, label="_")

        ax.set_xticks(positions, [k.replace("_", "\n") for k in keys], fontsize=7)
        ax.set_title(f"{title} (barra = média ± desvio, pontos = folds)", fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def redundancy_sweep_png(weights: list[float], summaries: list[dict], path: Path) -> None:
    """Curva do trade-off: especialização sobe, legibilidade do hipervetor cai."""

    def series(key):
        return ([s[key]["mean"] for s in summaries], [s[key]["std"] for s in summaries])

    positions = np.arange(len(weights))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    panels = [
        (axes[0], "especialização", [("edge_pair_gain", "ganho do par", "#55A868"),
                                     ("nmi_between_groups", "NMI entre grupos", "#C44E52")]),
        (axes[1], "legibilidade VSA", [("vsa_cleanup_accuracy", "cleanup", "#4C72B0"),
                                       ("nmi_code_label_mean", "NMI código-rótulo", "#8172B2")]),
        (axes[2], "tarefa", [("accuracy", "acurácia", "#4C72B0"),
                             ("auc_macro", "AUC macro", "#55A868"),
                             ("f1_macro", "F1 macro", "#CCB974")]),
    ]
    for ax, title, curves in panels:
        for key, label, color in curves:
            mean, std = series(key)
            ax.errorbar(positions, mean, yerr=std, marker="o", capsize=3,
                        label=label, color=color)
        ax.set_xticks(positions, [str(w) for w in weights])
        ax.set_xlabel("peso da penalidade de redundância")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[2].set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
