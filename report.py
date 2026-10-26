from __future__ import annotations

"""Transforma os JSON da validação em tabelas LaTeX e figuras para o artigo.

    python report.py                 # tudo que houver em results/validation/
    python report.py --latex-only

Saídas em `results/validation/paper/`: arquivos .tex prontos para \\input{} e
.pdf das figuras (vetorial, como o template do artigo pede).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validation import RESULTS_DIR, compare

PAPER_DIR = RESULTS_DIR / "paper"

# Baselines publicados do MedMNIST v2 (Yang et al., 2023), resolução 28.
MEDMNIST_BASELINES = {
    "bloodmnist": {"ResNet-18": (0.958, 0.998), "auto-sklearn": (0.878, 0.984)},
    "dermamnist": {"ResNet-18": (0.735, 0.917), "auto-sklearn": (0.749, 0.902)},
    "pathmnist": {"ResNet-18": (0.907, 0.983), "auto-sklearn": (0.716, 0.934)},
    "organamnist": {"ResNet-18": (0.935, 0.997), "auto-sklearn": (0.762, 0.963)},
}

PRETTY = {
    "mnist": "MNIST", "fashionmnist": "Fashion-MNIST", "bloodmnist": "BloodMNIST",
    "dermamnist": "DermaMNIST", "pathmnist": "PathMNIST", "organamnist": "OrganAMNIST",
}


def load_all() -> dict[str, dict]:
    out = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def cell(summary: dict, key: str, decimals: int = 3) -> str:
    if key not in summary:
        return "n/a"
    s = summary[key]
    return f"{s['mean']:.{decimals}f}\\,$\\pm$\\,{s['std']:.{decimals}f}"


# ------------------------------------------------------------------ tabelas

def table_generality(runs: dict) -> str:
    rows = []
    for ds in ["mnist", "fashionmnist", "bloodmnist", "dermamnist"]:
        r = runs.get(f"{ds}_cnn")
        if not r:
            continue
        s = r["summary"]
        reference = ""
        if ds in MEDMNIST_BASELINES:
            acc, auc = MEDMNIST_BASELINES[ds]["ResNet-18"]
            reference = f"{acc:.3f} / {auc:.3f}"
        rows.append(
            f"{PRETTY[ds]} & {cell(s,'accuracy')} & {cell(s,'auc_macro')} & "
            f"{cell(s,'balanced_accuracy')} & {cell(s,'f1_macro')} & "
            f"{cell(s,'edge_pair_gain')} & {reference or 'n/a'} \\\\")

    return "\n".join([
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{The same core and the same protocol across four domains. "
        r"Mean $\pm$ standard deviation over 5 stratified folds. The last column "
        r"gives the published ResNet-18 reference (ACC\,/\,AUC) for the MedMNIST "
        r"datasets at resolution 28.}",
        r"\label{tab:generality}", r"\vspace{-1.5mm}",
        r"\resizebox{0.92\textwidth}{!}{%",
        r"\begin{tabular}{l c c c c c c}", r"\toprule",
        r"\textbf{Dataset} & \textbf{Accuracy} & \textbf{AUC} & "
        r"\textbf{Bal.\ acc.} & \textbf{F1} & \textbf{Pair gain} & "
        r"\textbf{ResNet-18 ref.} \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}",
    ])


def table_encoder(runs: dict) -> str:
    rows = []
    for ds in ["fashionmnist", "bloodmnist"]:
        for label, key in [("dense encoder", f"{ds}_linear"),
                           ("CNN encoder", f"{ds}_cnn"),
                           ("CNN, no VSA core", f"{ds}_cnn_sem-core")]:
            r = runs.get(key)
            if not r:
                continue
            s = r["summary"]
            rows.append(f"{PRETTY[ds]} & {label} & {cell(s,'accuracy')} & "
                        f"{cell(s,'auc_macro')} & {cell(s,'balanced_accuracy')} \\\\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()

    return "\n".join([
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Ablation of the encoder and of the bottleneck itself. "
        r"The third row per dataset removes the sparse VSA core, keeping the "
        r"same encoder and the same program dimensionality, and therefore "
        r"measures the cost of discretisation.}",
        r"\label{tab:encoder}", r"\vspace{-1.5mm}",
        r"\resizebox{0.99\linewidth}{!}{%",
        r"\begin{tabular}{l l c c c}", r"\toprule",
        r"\textbf{Dataset} & \textbf{Variant} & \textbf{Accuracy} & "
        r"\textbf{AUC} & \textbf{Bal.\ acc.} \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}",
    ])


def table_specialization(runs: dict) -> str:
    rows = []
    for ds in ["fashionmnist", "bloodmnist"]:
        for w in (0.0, 0.1, 0.25, 0.5, 1.0):
            key = f"{ds}_cnn" if w == 0 else f"{ds}_cnn_w{w:g}"
            r = runs.get(key)
            if not r:
                continue
            s = r["summary"]
            rows.append(f"{PRETTY[ds]} & {w:g} & {cell(s,'accuracy')} & "
                        f"{cell(s,'nmi_between_groups')} & {cell(s,'edge_pair_gain')} & "
                        f"{cell(s,'vsa_cleanup_accuracy')} \\\\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()

    return "\n".join([
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Redundancy penalty sweep. Lower inter-group NMI means less "
        r"duplicated information; higher pair gain means connected groups carry "
        r"more together than either carries alone. VSA recovery is the "
        r"bundle\,$\rightarrow$\,unbind\,$\rightarrow$\,cleanup round trip.}",
        r"\label{tab:specialization}", r"\vspace{-1.5mm}",
        r"\resizebox{0.99\linewidth}{!}{%",
        r"\begin{tabular}{l c c c c c}", r"\toprule",
        r"\textbf{Dataset} & $\lambda$ & \textbf{Accuracy} & "
        r"\textbf{Inter-group NMI} & \textbf{Pair gain} & \textbf{VSA recovery} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}",
    ])


def table_statistics(runs: dict) -> str:
    """Testes pareados por fold das comparações que sustentam o texto."""

    comparisons = [
        ("fashionmnist_linear", "fashionmnist_cnn", "accuracy", "CNN vs dense (F-MNIST)"),
        ("bloodmnist_linear", "bloodmnist_cnn", "accuracy", "CNN vs dense (Blood)"),
        ("fashionmnist_cnn", "fashionmnist_cnn_sem-core", "accuracy", "no core vs core (F-MNIST)"),
        ("bloodmnist_cnn", "bloodmnist_cnn_sem-core", "accuracy", "no core vs core (Blood)"),
        ("fashionmnist_cnn", "fashionmnist_cnn_w0.25", "edge_pair_gain", "$\\lambda{=}0.25$ pair gain (F-MNIST)"),
        ("bloodmnist_cnn", "bloodmnist_cnn_w0.25", "edge_pair_gain", "$\\lambda{=}0.25$ pair gain (Blood)"),
        ("fashionmnist_cnn", "fashionmnist_cnn_w0.25", "accuracy", "$\\lambda{=}0.25$ accuracy cost (F-MNIST)"),
        ("bloodmnist_cnn", "bloodmnist_cnn_w0.25", "accuracy", "$\\lambda{=}0.25$ accuracy cost (Blood)"),
    ]
    rows = []
    for base_key, var_key, metric, label in comparisons:
        if base_key not in runs or var_key not in runs:
            continue
        c = compare(runs[base_key], runs[var_key], metric)
        rows.append(f"{label} & {c['baseline_mean']:.3f} & {c['variant_mean']:.3f} & "
                    f"{c['delta']:+.3f} & {c['cohens_d']:+.2f} & "
                    f"{c['folds_improved']}/{c['n_folds']} & {c['wilcoxon_p']:.3f} \\\\")

    return "\n".join([
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Fold-paired comparisons. All configurations share the same "
        r"fold partitions, so each fold yields a legitimate pair. With five "
        r"folds the Wilcoxon test has little power, so effect size is the "
        r"primary reading and $p$ is secondary.}",
        r"\label{tab:stats}", r"\vspace{-1.5mm}",
        r"\resizebox{0.99\linewidth}{!}{%",
        r"\begin{tabular}{l c c c c c c}", r"\toprule",
        r"\textbf{Comparison} & \textbf{Base} & \textbf{Variant} & "
        r"$\Delta$ & \textbf{Cohen's }$d$ & \textbf{Folds} & $p$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}",
    ])


# ------------------------------------------------------------------ figuras

def figure_tradeoff(runs: dict, path: Path) -> None:
    """A curva central do artigo: especialização sobe, legibilidade cai."""

    weights = [0.0, 0.1, 0.25, 0.5, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    colors = {"fashionmnist": "#4C72B0", "bloodmnist": "#C44E52"}

    panels = [("edge_pair_gain", "pair gain (compositionality)"),
              ("nmi_between_groups", "inter-group NMI (redundancy)"),
              ("vsa_cleanup_accuracy", "VSA recovery (readability)")]

    for ax, (key, title) in zip(axes, panels):
        for ds in ["fashionmnist", "bloodmnist"]:
            means, stds, xs = [], [], []
            for w in weights:
                run_key = f"{ds}_cnn" if w == 0 else f"{ds}_cnn_w{w:g}"
                if run_key not in runs or key not in runs[run_key]["summary"]:
                    continue
                s = runs[run_key]["summary"][key]
                means.append(s["mean"]); stds.append(s["std"]); xs.append(w)
            if xs:
                ax.errorbar(range(len(xs)), means, yerr=stds, marker="o", capsize=3,
                            label=PRETTY[ds], color=colors[ds])
                ax.set_xticks(range(len(xs)), [f"{w:g}" for w in xs])
        ax.set_xlabel(r"redundancy penalty $\lambda$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def figure_generality(runs: dict, path: Path) -> None:
    """Acurácia e acurácia balanceada lado a lado: onde o desbalanceamento morde."""

    order = ["mnist", "fashionmnist", "bloodmnist", "dermamnist"]
    present = [d for d in order if f"{d}_cnn" in runs]
    if not present:
        return

    acc = [runs[f"{d}_cnn"]["summary"]["accuracy"]["mean"] for d in present]
    acc_e = [runs[f"{d}_cnn"]["summary"]["accuracy"]["std"] for d in present]
    bal = [runs[f"{d}_cnn"]["summary"]["balanced_accuracy"]["mean"] for d in present]
    bal_e = [runs[f"{d}_cnn"]["summary"]["balanced_accuracy"]["std"] for d in present]

    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.bar(x - 0.2, acc, 0.38, yerr=acc_e, capsize=3, label="accuracy", color="#4C72B0")
    ax.bar(x + 0.2, bal, 0.38, yerr=bal_e, capsize=3, label="balanced accuracy",
           color="#DD8452")
    for i, d in enumerate(present):
        ref = MEDMNIST_BASELINES.get(d, {}).get("ResNet-18")
        if ref:
            ax.hlines(ref[0], i - 0.42, i + 0.42, color="black", linestyles="--", lw=1.2)
    ax.set_xticks(x, [PRETTY[d] for d in present], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("same core, same protocol, four domains "
                 "(dashed = published ResNet-18)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex-only", action="store_true")
    args = parser.parse_args()

    runs = load_all()
    if not runs:
        raise SystemExit(f"nenhum resultado em {RESULTS_DIR}/ — rode validation.py antes")
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    tables = {
        "table_generality.tex": table_generality(runs),
        "table_encoder.tex": table_encoder(runs),
        "table_specialization.tex": table_specialization(runs),
        "table_stats.tex": table_statistics(runs),
    }
    for filename, content in tables.items():
        (PAPER_DIR / filename).write_text(content + "\n", encoding="utf-8")
        print(f"escrito {PAPER_DIR / filename}")

    if not args.latex_only:
        figure_tradeoff(runs, PAPER_DIR / "fig_tradeoff.pdf")
        figure_generality(runs, PAPER_DIR / "fig_generality.pdf")
        print(f"figuras em {PAPER_DIR}/")

    print(f"\n{len(runs)} configurações encontradas:")
    for name, r in sorted(runs.items()):
        s = r["summary"]
        print(f"  {name:<34} acc={s['accuracy']['mean']:.4f} auc={s['auc_macro']['mean']:.4f}")


if __name__ == "__main__":
    main()
