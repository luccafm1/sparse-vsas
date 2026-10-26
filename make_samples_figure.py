"""Gera a figura de amostras dos quatro datasets usados na validação.

Uma linha por dataset, quatro classes distintas por linha, todas em 28x28 e na
mesma escala — é a figura que acompanha a tabela de generalidade no artigo.

    python make_samples_figure.py [saida.pdf]
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets import load_dataset

DATASETS = ["mnist", "fashionmnist", "bloodmnist", "dermamnist"]
LABELS = {
    "mnist": "MNIST",
    "fashionmnist": "F-MNIST",
    "bloodmnist": "BloodMNIST",
    "dermamnist": "DermaMNIST",
}
N_COLS = 4


def pick(ds, rng):
    """Uma amostra de cada uma de N_COLS classes distintas."""
    classes = np.unique(ds.y)
    chosen = rng.choice(classes, size=min(N_COLS, len(classes)), replace=False)
    out = []
    for c in sorted(chosen):
        idx = np.flatnonzero(ds.y == c)
        out.append((ds.x[rng.choice(idx)], ds.class_names[c]))
    return out


def main(path: str) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for name in DATASETS:
        ds = load_dataset(name, sample_cap=4000, seed=0)
        rows.append((LABELS[name], ds.image_shape, pick(ds, rng)))

    # dois datasets por linha, quatro classes cada: grid 2x8, largo e baixo,
    # para caber como figura de página inteira sem comer uma coluna de texto
    fig, axes = plt.subplots(2, 2 * N_COLS, figsize=(10.2, 1.9))
    for r, (label, shape, samples) in enumerate(rows):
        row, block = divmod(r, 2)
        c_, h, w = shape
        for c, (flat, class_name) in enumerate(samples):
            ax = axes[row, block * N_COLS + c]
            # datasets.py achata em ordem HWC, embora reporte image_shape como
            # (C, H, W) — é a mesma convenção que o ConvEncoder desfaz.
            img = flat.reshape(h, w, c_)
            ax.imshow(img.squeeze(), cmap="gray" if c_ == 1 else None,
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            name = class_name if len(class_name) <= 15 else class_name[:13] + "…"
            ax.set_title(name, fontsize=6, pad=1.6)
            if c == 0:
                ax.set_ylabel(label, fontsize=7.5)

    fig.subplots_adjust(left=0.04, right=0.995, top=0.88, bottom=0.01,
                        wspace=0.08, hspace=0.55)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    print("escrito:", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset_samples.pdf")
