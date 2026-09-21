from __future__ import annotations

"""K-fold da transferência de códigos via Glue.

Mesma validação da `kfold_medmnist_test.py` — `StratifiedKFold` sobre os splits
oficiais unidos, padronização ajustada só no fold de treino — mas cada fold
treina **duas** redes independentes (seeds diferentes) e mede se a cabeça de B
consegue ler os códigos de A.

Por fold, quatro leituras da cabeça de B sobre o mesmo fold de teste:

  teto (B->B)             códigos próprios de B — o melhor possível
  sem cola                códigos crus de A — os índices de filler de A não
                          significam nada no espaço de B
  fillers embaralhados    roteamento de grupos da cola, permutações de valor
                          aleatórias — isola quanto do ganho vem de alinhar
                          fillers e quanto vem só de descartar grupos não
                          resolvidos (que viram -1)
  com cola                códigos de A traduzidos por `SJGlue.import_program`

O número que importa é `com cola` menos `fillers embaralhados`. Comparar contra
`sem cola` engana: códigos estrangeiros crus caem em regiões arbitrárias da
cabeça de B e essa linha de base oscila muito entre folds.

NOTA sobre a migração para FHRR: a flag `--pairs` saiu junto com o
`base.train(mode="acquisition")` — ver a nota no `glue_transfer_test.py`. As duas
redes de cada fold são treinadas de forma puramente supervisionada.

Uso:
    python kfold_glue_test.py --dataset bloodmnist --encoder cnn --folds 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from glue import fit_sj_glue
from glue_transfer_test import (ACTION_SETS, accuracy_from_program, build_and_train,
                                 translate_with)
from kfold_medmnist_test import (aggregate, alphabet_metrics, classification_metrics,
                                  encode, load_medmnist, predict_proba,
                                  specialization_metrics, vsa_cleanup_accuracy)


def run_fold(fold, x_tr_raw, y_tr, x_te_raw, y_te, *, args, class_names,
             image_shape, device, action_set):
    channels, height, width = image_shape
    mean = x_tr_raw.mean(axis=0, keepdims=True)
    std = x_tr_raw.std(axis=0, keepdims=True) + 1e-6
    prep = lambda flat: ((flat - mean) / std).astype(np.float32)
    x_tr, x_te = prep(x_tr_raw), prep(x_te_raw)

    common = dict(n_classes=len(class_names), image_shape=image_shape,
                  encoder=args.encoder, epochs=args.epochs,
                  redundancy_weight=args.redundancy_weight, device=device,
                  card=args.card, groups=args.groups, units=args.units,
                  layers=args.layers, topk=args.topk, hv_dim=args.hv_dim,
                  frame_embed=args.frame_embed, batch=args.batch, lr=args.lr)

    started = time.time()
    model_a, _ = build_and_train(x_tr, y_tr, seed=args.seed + fold, **common)
    model_b, _ = build_and_train(x_tr, y_tr, seed=args.seed + fold + 1000, **common)
    train_seconds = time.time() - started

    ev_raw = x_tr_raw[:args.evidence]
    images = ev_raw.reshape(len(ev_raw), height, width, channels)
    base_a, base_b = encode(model_a, prep(ev_raw), device), encode(model_b, prep(ev_raw), device)
    actions_a, actions_b = {}, {}
    for name, fn in action_set.items():
        moved = prep(fn(images).reshape(len(ev_raw), -1))
        actions_a[name] = encode(model_a, moved, device)
        actions_b[name] = encode(model_b, moved, device)

    signature = {}
    for tag, base, actions in (("A", base_a, actions_a), ("B", base_b, actions_b)):
        rates = np.stack([(actions[n] != base).mean(axis=0) for n in action_set])
        signature[tag] = float(rates.std(axis=1).mean())

    names = list(action_set)
    glue = fit_sj_glue(base_a, actions_a, model_a.sj, base_b, actions_b, model_b.sj,
                       train_actions=names[:3], validation_actions=names[3:],
                       permutation_nulls=args.permutation_nulls,
                       bootstrap_replicates=args.bootstrap_replicates,
                       seed=args.seed + fold)
    cert = glue.certificate

    codes_a_np, codes_b_np = encode(model_a, x_te, device), encode(model_b, x_te, device)
    codes_a = torch.from_numpy(codes_a_np).long().to(device)
    codes_b = torch.from_numpy(codes_b_np).long().to(device)

    metrics = {}
    for tag, model, codes_np in (("a", model_a, codes_a_np), ("b", model_b, codes_b_np)):
        probs = predict_proba(model, x_te, device)
        m = classification_metrics(y_te, probs)
        m.update(specialization_metrics(codes_np, y_te, model.sj.edges, args.card))
        m.update(alphabet_metrics(codes_np, args.card))
        m["vsa_cleanup_accuracy"] = vsa_cleanup_accuracy(model, codes_np, device)
        metrics.update({f"{tag}_{k}": v for k, v in m.items()})

    with torch.no_grad():
        ceiling = accuracy_from_program(model_b, model_b.sj.program_feature_from_codes(codes_b), y_te)
        no_glue = accuracy_from_program(model_b, model_b.sj.program_feature_from_codes(codes_a), y_te)
        empty = torch.full_like(codes_b, -1)
        no_codes = accuracy_from_program(model_b, model_b.sj.program_feature_from_codes(empty), y_te)
        with_glue = accuracy_from_program(
            model_b, glue.import_program(model_b.sj, codes_a, allow_candidate=True), y_te)
        rng = np.random.default_rng(args.seed + fold)
        shuffled = float(np.mean([
            accuracy_from_program(
                model_b,
                model_b.sj.program_feature_from_codes(translate_with(
                    codes_a, cert.group_map,
                    tuple(None if v is None else tuple(rng.permutation(args.card))
                          for v in cert.value_maps),
                    glue.target_groups)),
                y_te)
            for _ in range(8)]))

    metrics.update({
        "transfer_ceiling": ceiling,
        "transfer_no_glue": no_glue,
        "transfer_no_codes": no_codes,
        "transfer_shuffled_fillers": shuffled,
        "transfer_with_glue": with_glue,
        "glue_gain_vs_shuffled": with_glue - shuffled,
        "glue_gain_vs_no_glue": with_glue - no_glue,
        "glue_source_coverage": glue.source_coverage,
        "glue_is_executable": float(glue.is_executable),
        "glue_train_score": float(cert.train_score or 0.0),
        "group_action_signature_a": signature["A"],
        "group_action_signature_b": signature["B"],
        "train_seconds": train_seconds,
    })
    return metrics, glue.status, cert.reason


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="bloodmnist")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--encoder", choices=["linear", "cnn"], default="cnn")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--card", type=int, default=8)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--hv-dim", type=int, default=128)
    p.add_argument("--frame-embed", type=int, default=128)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--redundancy-weight", type=float, default=0.0)
    p.add_argument("--actions", choices=sorted(ACTION_SETS), default="photometric")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--evidence", type=int, default=1200)
    p.add_argument("--permutation-nulls", type=int, default=39)
    p.add_argument("--bootstrap-replicates", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default="results/kfold_glue")
    args = p.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, class_names, image_shape = load_medmnist(args.dataset)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(x), size=min(args.limit, len(x)), replace=False)
        x, y = x[keep], y[keep]
    action_set = ACTION_SETS[args.actions]

    print(f"dataset={args.dataset} n={len(x)} classes={len(class_names)} "
          f"folds={args.folds} encoder={args.encoder} ações={args.actions}")

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    per_fold, statuses = [], []
    for fold, (tr, te) in enumerate(splitter.split(x, y)):
        metrics, status, reason = run_fold(fold, x[tr], y[tr], x[te], y[te], args=args,
                                            class_names=class_names, image_shape=image_shape,
                                            device=device, action_set=action_set)
        per_fold.append(metrics)
        statuses.append((status, reason))
        print(f"fold {fold + 1}/{args.folds}  accA={metrics['a_accuracy']:.4f} "
              f"teto={metrics['transfer_ceiling']:.4f} "
              f"cola={metrics['transfer_with_glue']:.4f} "
              f"embaralhado={metrics['transfer_shuffled_fillers']:.4f} "
              f"ganho={metrics['glue_gain_vs_shuffled']:+.4f} "
              f"cobertura={metrics['glue_source_coverage']:.0%} "
              f"({metrics['train_seconds']:.0f}s)", flush=True)

    summary = aggregate(per_fold)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"=== k-fold da Glue — {args.dataset} ===",
        f"folds={args.folds} encoder={args.encoder} epochs={args.epochs} "
        f"groups={args.groups} card={args.card} ações={args.actions} "
        f"redundancy_weight={args.redundancy_weight}",
        "",
        "-- qualidade das redes (média ± desvio entre folds) --",
    ]
    quality = [("a_accuracy", "acurácia A"), ("b_accuracy", "acurácia B"),
               ("a_auc_macro", "AUC macro A"), ("a_f1_macro", "F1 macro A"),
               ("a_nmi_code_label_mean", "NMI código/rótulo A"),
               ("a_nmi_between_groups", "NMI entre grupos A"),
               ("a_edge_pair_gain", "ganho de par em aresta A"),
               ("a_code_perplexity_mean", "perplexidade A"),
               ("a_vsa_cleanup_accuracy", "recuperação VSA A")]
    transfer = [("transfer_ceiling", "teto B->B"),
                ("transfer_no_glue", "sem cola (A->B cru)"),
                ("transfer_no_codes", "controle: nenhum código"),
                ("transfer_shuffled_fillers", "controle: fillers embaralhados"),
                ("transfer_with_glue", "COM cola"),
                ("glue_gain_vs_shuffled", "ganho real (cola - embaralhado)"),
                ("glue_source_coverage", "cobertura da cola"),
                ("glue_is_executable", "certificado executável"),
                ("group_action_signature_a", "assinatura por grupo A")]
    for key, label in quality:
        s = summary[key]
        lines.append(f"  {label:<32} {s['mean']:+.4f} ± {s['std']:.4f}")
    lines += ["", "-- transferência na cabeça de B --"]
    for key, label in transfer:
        s = summary[key]
        lines.append(f"  {label:<32} {s['mean']:+.4f} ± {s['std']:.4f}  "
                      f"[{s['min']:+.4f}, {s['max']:+.4f}]")
    lines += ["", "-- status do certificado por fold --"]
    for i, (status, reason) in enumerate(statuses):
        lines.append(f"  fold {i + 1}: {status} — {reason}")

    report = "\n".join(lines)
    print("\n" + report)
    tag = f"rw{args.redundancy_weight:g}"
    stem = f"{args.dataset}_{args.encoder}_{args.actions}_{tag}"
    (out_dir / f"{stem}_report.txt").write_text(report + "\n", encoding="utf-8")
    (out_dir / f"{stem}_metrics.json").write_text(
        json.dumps({"config": vars(args), "summary": summary, "per_fold": per_fold,
                     "certificate_status": statuses}, indent=2, default=str),
        encoding="utf-8")
    print(f"\nsalvo {out_dir / (stem + '_report.txt')}")


if __name__ == "__main__":
    main()
