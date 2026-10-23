from __future__ import annotations

"""Protocolo único de teste e validação da arquitetura sparse-joint.

Motivação: antes desta unificação havia três caminhos de teste incompatíveis.
O `sj_image_classifier.py` rodava MNIST/Fashion-MNIST com `layers=2, units=32,
hv_dim=64` e split único; o `kfold_medmnist_test.py` rodava MedMNIST com
`layers=3, units=128, hv_dim=256` e k-fold. Os números não eram comparáveis
entre si, o que invalida qualquer afirmação do tipo "o core é agnóstico ao
domínio" baseada em comparar um com o outro.

Aqui a arquitetura é fixa (`BASE_ARCH`) e só variam os eixos declarados como
ablação. Todo o resto do protocolo é idêntico entre execuções:

- k-fold estratificado com semente fixa, as mesmas partições para toda
  configuração do mesmo dataset (comparação pareada por fold);
- padronização ajustada só no fold de treino;
- subamostragem estratificada opcional, para igualar o orçamento de treino
  entre datasets de tamanhos diferentes;
- o mesmo conjunto de métricas, de classificação e de arquitetura;
- proveniência gravada junto do resultado (commit, semente, versões).

Uso:
    python validation.py --suite smoke          # checagem rápida
    python validation.py --suite generality     # o core em 4 domínios
    python validation.py --suite encoder        # densa vs CNN vs sem core
    python validation.py --suite specialization # varredura da penalidade
    python validation.py --suite all
"""

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.model_selection import StratifiedKFold

import datasets as dataset_module
from base import SJConfig, build_model, encode, vsa_cleanup_accuracy
from kfold_medmnist_test import (
    ConvEncoder,
    alphabet_metrics,
    classification_metrics,
    predict_proba,
    specialization_metrics,
    standardize,
    train_supervised,
)

RESULTS_DIR = Path("results/validation")

# Arquitetura de referência. Qualquer desvio disto é uma ablação declarada e
# aparece no nome da configuração.
BASE_ARCH = dict(groups=8, card=16, units=128, layers=3, topk=4,
                 hv_dim=256, frame_embed=128, unit_width=8)

# Orçamento de treino de referência.
BASE_TRAIN = dict(epochs=120, batch=1024, lr=2e-3)

# Teto de amostras por dataset: iguala o orçamento entre MNIST (70k) e os
# MedMNIST (10-17k), condição para comparar resultados entre domínios.
SAMPLE_CAP = 17000


# ------------------------------------------------------------------ config

@dataclass(frozen=True)
class RunConfig:
    dataset: str
    encoder: str = "cnn"              # "cnn" | "linear"
    core: str = "sparse_joint"        # "sparse_joint" | "none" (controle)
    redundancy_weight: float = 0.0
    folds: int = 5
    seed: int = 0
    sample_cap: int = SAMPLE_CAP
    arch: dict = field(default_factory=lambda: dict(BASE_ARCH))
    train: dict = field(default_factory=lambda: dict(BASE_TRAIN))

    @property
    def name(self) -> str:
        bits = [self.dataset, self.encoder]
        if self.core == "none":
            bits.append("sem-core")
        if self.redundancy_weight:
            bits.append(f"w{self.redundancy_weight:g}")
        return "_".join(bits)


# ------------------------------------------------------------------ controle

class NoCoreBaseline(torch.nn.Module):
    """Controle sem o gargalo VSA: encoder -> densa -> cabeça.

    Mantém a mesma dimensão de "programa" (2*hv_dim) que o core exporta, para
    que a diferença medida seja a discretização em grupos, e não a capacidade
    do vetor que chega na cabeça de classificação.
    """

    def __init__(self, encoder: torch.nn.Module, frame_embed: int,
                 program_dim: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Linear(frame_embed, program_dim), torch.nn.GELU())
        self.head = torch.nn.Linear(program_dim, n_classes)

    def forward(self, x: torch.Tensor) -> dict:
        program = self.bottleneck(self.encoder(x))
        return {"program": program, "factor_logits": [self.head(program)]}


def _train_no_core(model, x, z, *, device, epochs, batch, lr, seed):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    xt = torch.from_numpy(x).to(device)
    zt = torch.from_numpy(z).long().to(device).reshape(-1)
    generator = torch.Generator(device="cpu").manual_seed(seed + 7717)
    for _ in range(epochs):
        order = torch.randperm(len(xt), generator=generator).to(device)
        for start in range(0, len(xt), batch):
            index = order[start:start + batch]
            logits = model(xt[index])["factor_logits"][0]
            loss = torch.nn.functional.cross_entropy(logits, zt[index])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model.eval()


# ------------------------------------------------------------------ execução

def _build(cfg: RunConfig, fold: int, data, device):
    arch, train = cfg.arch, cfg.train
    sj_cfg = SJConfig(
        cardinalities=tuple([arch["card"]] * arch["groups"]),
        n_factors=1,
        input_dim=data.input_dim,
        units=arch["units"], layers=arch["layers"], topk=arch["topk"],
        unit_width=arch["unit_width"], hv_dim=arch["hv_dim"],
        frame_embed=arch["frame_embed"],
        epochs=train["epochs"], batch=train["batch"], lr=train["lr"],
        seed=cfg.seed + fold,
    )
    torch.manual_seed(sj_cfg.seed)

    if cfg.core == "none":
        encoder = (ConvEncoder(data.image_shape, sj_cfg.frame_embed) if cfg.encoder == "cnn"
                   else torch.nn.Sequential(
                       torch.nn.Linear(data.input_dim, sj_cfg.frame_embed), torch.nn.GELU()))
        model = NoCoreBaseline(encoder, sj_cfg.frame_embed, 2 * sj_cfg.hv_dim,
                               data.n_classes)
        return sj_cfg, model

    model = build_model(sj_cfg, (data.n_classes,))
    if cfg.encoder == "cnn":
        model.encoder = ConvEncoder(data.image_shape, sj_cfg.frame_embed)
    return sj_cfg, model


def run_fold(cfg: RunConfig, fold: int, data, train_idx, test_idx, device) -> dict:
    x_tr, x_te = standardize(data.x[train_idx], data.x[test_idx])
    z_tr, z_te = data.y[train_idx], data.y[test_idx]

    sj_cfg, model = _build(cfg, fold, data, device)

    started = time.time()
    if cfg.core == "none":
        model = _train_no_core(model, x_tr, z_tr, device=device,
                               epochs=sj_cfg.epochs, batch=sj_cfg.batch,
                               lr=sj_cfg.lr, seed=sj_cfg.seed)
    else:
        model = train_supervised(model, x_tr, z_tr.reshape(-1, 1), device=device,
                                 redundancy_weight=cfg.redundancy_weight)
    train_seconds = time.time() - started

    probs = predict_proba(model, x_te, device)
    metrics = classification_metrics(z_te, probs)

    if cfg.core != "none":
        codes = encode(model, x_te, device)
        metrics.update(specialization_metrics(codes, z_te, model.sj.edges, sj_cfg.max_card))
        metrics.update(alphabet_metrics(codes, sj_cfg.max_card))
        metrics["vsa_cleanup_accuracy"] = vsa_cleanup_accuracy(model, codes, device)

    metrics["train_seconds"] = train_seconds
    metrics["_predictions"] = probs.argmax(axis=1).tolist()
    metrics["_truth"] = z_te.tolist()
    return metrics


def run(cfg: RunConfig, *, device=None, quiet: bool = False) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataset_module.load_dataset(cfg.dataset, sample_cap=cfg.sample_cap, seed=cfg.seed)

    # Mesmas partições para toda configuração do mesmo dataset: é o que permite
    # o teste pareado entre configurações.
    splitter = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)

    per_fold, predictions, truth = [], [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(data.x, data.y)):
        metrics = run_fold(cfg, fold, data, train_idx, test_idx, device)
        predictions.extend(metrics.pop("_predictions"))
        truth.extend(metrics.pop("_truth"))
        per_fold.append(metrics)
        if not quiet:
            print(f"    fold {fold}: acc={metrics['accuracy']:.4f} "
                  f"auc={metrics['auc_macro']:.4f} ({metrics['train_seconds']:.0f}s)")

    return {
        "config": {**asdict(cfg), "name": cfg.name},
        "dataset_info": data.describe(),
        "per_fold": per_fold,
        "summary": aggregate(per_fold),
        "predictions": predictions,
        "truth": truth,
        "class_names": data.class_names,
        "provenance": provenance(),
    }


def aggregate(per_fold: list[dict]) -> dict:
    keys = [k for k in per_fold[0] if not k.startswith("_")]
    return {
        k: {
            "mean": float(np.mean([f[k] for f in per_fold])),
            "std": float(np.std([f[k] for f in per_fold])),
            "folds": [float(f[k]) for f in per_fold],
        }
        for k in keys
    }


def provenance() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except Exception:
            return "desconhecido"

    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ------------------------------------------------------------------ estatística

def compare(baseline: dict, variant: dict, metric: str) -> dict:
    """Teste pareado por fold entre duas configurações.

    As partições são as mesmas (mesma semente), então cada fold dá um par
    legítimo. Com 5 folds o Wilcoxon tem pouco poder, então reportamos também o
    tamanho de efeito; a leitura honesta é pela magnitude, com o p-valor como
    sinal secundário.
    """

    a = np.array(baseline["summary"][metric]["folds"])
    b = np.array(variant["summary"][metric]["folds"])
    difference = b - a

    try:
        _, p_value = stats.wilcoxon(a, b)
    except ValueError:                      # todas as diferenças iguais
        p_value = 1.0

    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return {
        "metric": metric,
        "baseline_mean": float(a.mean()),
        "variant_mean": float(b.mean()),
        "delta": float(difference.mean()),
        "delta_std": float(difference.std(ddof=1)),
        "cohens_d": float(difference.mean() / pooled) if pooled > 0 else 0.0,
        "wilcoxon_p": float(p_value),
        "folds_improved": int((difference > 0).sum()),
        "n_folds": len(a),
    }


# ------------------------------------------------------------------ suítes

MEDICAL = ["bloodmnist", "dermamnist", "pathmnist", "organamnist"]
NATURAL = ["mnist", "fashionmnist"]


def suite(name: str) -> list[RunConfig]:
    if name == "smoke":
        quick = dict(arch={**BASE_ARCH}, train=dict(epochs=4, batch=1024, lr=2e-3))
        return [RunConfig(dataset="bloodmnist", folds=2, sample_cap=2000, **quick),
                RunConfig(dataset="bloodmnist", folds=2, sample_cap=2000,
                          core="none", **quick)]

    if name == "generality":
        # O mesmo core, a mesma config, em quatro domínios.
        return [RunConfig(dataset=d) for d in ["mnist", "fashionmnist",
                                               "bloodmnist", "dermamnist"]]

    if name == "encoder":
        # Isola o encoder e o próprio gargalo.
        out = []
        for d in ["fashionmnist", "bloodmnist"]:
            out.append(RunConfig(dataset=d, encoder="linear"))
            out.append(RunConfig(dataset=d, encoder="cnn"))
            out.append(RunConfig(dataset=d, encoder="cnn", core="none"))
        return out

    if name == "specialization":
        out = []
        for d in ["fashionmnist", "bloodmnist"]:
            for w in (0.0, 0.1, 0.25, 0.5, 1.0):
                out.append(RunConfig(dataset=d, redundancy_weight=w))
        return out

    if name == "all":
        seen, out = set(), []
        for s in ("generality", "encoder", "specialization"):
            for cfg in suite(s):
                if cfg.name not in seen:
                    seen.add(cfg.name)
                    out.append(cfg)
        return out

    raise SystemExit(f"suíte desconhecida: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--only", default="", help="filtra configs por substring do nome")
    parser.add_argument("--force", action="store_true", help="reexecuta o que já tem resultado")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [c for c in suite(args.suite) if args.only in c.name]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"dispositivo: {device} | suíte '{args.suite}' | {len(configs)} configurações\n")
    for i, cfg in enumerate(configs, 1):
        path = RESULTS_DIR / f"{cfg.name}.json"
        if path.exists() and not args.force:
            print(f"[{i}/{len(configs)}] {cfg.name}: já existe, pulando")
            continue
        print(f"[{i}/{len(configs)}] {cfg.name}")
        result = run(cfg, device=device)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        s = result["summary"]
        print(f"    -> acc={s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f} "
              f"auc={s['auc_macro']['mean']:.4f}\n")

    print(f"resultados em {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
