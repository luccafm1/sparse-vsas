from __future__ import annotations

"""Transferência de códigos entre duas redes treinadas independentemente — sem e com a Glue.

A bateria k-fold mede uma rede sozinha e nunca toca no `glue.py`. Este teste
fecha essa lacuna: treina A e B separadamente (mesmos dados, seeds diferentes,
portanto scaffolds e codebooks diferentes) e pergunta se a cabeça de B consegue
ler os códigos discretos de A.

Três leituras da mesma cabeça de B, sobre o mesmo conjunto de teste:

  teto (B->B)   códigos de B na cabeça de B — o melhor possível
  sem cola      códigos crus de A na cabeça de B — controle; os índices de
                filler de A não significam nada no espaço de B
  com cola      códigos de A traduzidos por `SJGlue.import_program` antes

A cola só é útil se "com cola" ficar claramente acima de "sem cola". Se empatar,
o alinhamento não encontrou estrutura — e isso é um resultado, não um erro.

A Glue deste branch é por *operadores de ação*: ela não vê pares rotulados, só
como cada rede reage às mesmas transformações de imagem (rotação, espelhamento,
inversão, deslocamento). Cada ação vira um operador sobre os códigos, e o
alinhamento é o que casa os operadores das duas redes.

NOTA sobre a migração para FHRR: este teste nasceu sobre o `base.py` bipolar, que
tinha `train(mode="acquisition")` e os termos `response`/`invariance`. Esse
caminho não existe mais — o objetivo do `base.py` hoje é só a task loss — então a
flag `--pairs` foi removida em vez de virar um no-op silencioso. As redes A e B
são treinadas de forma puramente supervisionada.

Uso:
    python glue_transfer_test.py --dataset bloodmnist --encoder cnn
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from base import SJConfig, build_model
from glue import fit_sj_glue
from kfold_medmnist_test import (ConvEncoder, alphabet_metrics, classification_metrics,
                                  encode, load_medmnist, predict_proba,
                                  specialization_metrics, train_supervised,
                                  vsa_cleanup_accuracy)

def _occlude(im):
    out = im.copy()
    h, w = im.shape[1], im.shape[2]
    out[:, h // 3:2 * h // 3, w // 3:2 * w // 3] = 0.0
    return out


def _tint(im, channel):
    out = im.copy()
    out[..., channel % im.shape[-1]] *= 0.3
    return out


# Ações aplicadas à imagem crua [N, H, W, C]; a cola estima como os códigos de
# cada grupo respondem a cada uma. Dois regimes: as geométricas mal movem os
# códigos de um classificador bem treinado (que aprende a ser invariante a
# elas), enquanto as fotométricas movem quase todos. Medido neste repositório,
# nenhum dos dois extremos produz assinatura distinguível por grupo.
ACTION_SETS = {
    "geometric": {
        "rot90": lambda im: np.ascontiguousarray(np.rot90(im, 1, axes=(1, 2))),
        "hflip": lambda im: np.ascontiguousarray(im[:, :, ::-1]),
        "vflip": lambda im: np.ascontiguousarray(im[:, ::-1]),
        "invert": lambda im: np.ascontiguousarray(1.0 - im),
        "shift": lambda im: np.ascontiguousarray(np.roll(im, 4, axis=2)),
    },
    "photometric": {
        "invert": lambda im: np.ascontiguousarray(1.0 - im),
        "occlude": _occlude,
        "darken": lambda im: np.ascontiguousarray(im * 0.35),
        "tint0": lambda im: _tint(im, 0),
        "tint2": lambda im: _tint(im, 2),
    },
}


def build_and_train(x, z, *, seed, n_classes, image_shape, encoder, epochs,
                    redundancy_weight, device, card, groups, units, layers, topk,
                    hv_dim, frame_embed, batch, lr):
    """Mesmo treino da bateria k-fold: supervisionado, uma cabeça de classe.

    O que separa A de B é só a seed — ela decide o scaffold (`make_rigid_scaffold`)
    e as fases de papéis e fillers do FHRR. Logo os dois espaços de código são
    incomparáveis por construção, que é exatamente o que a cola precisa resolver.
    """
    cfg = SJConfig(
        cardinalities=tuple([card] * groups),
        input_dim=x.shape[1],
        units=units, layers=layers, topk=topk, hv_dim=hv_dim,
        frame_embed=frame_embed, epochs=epochs, batch=batch, lr=lr, seed=seed,
    )
    torch.manual_seed(cfg.seed)
    model = build_model(
        cfg, task="classification", num_outputs=n_classes,
        encoder=ConvEncoder(image_shape, cfg.frame_embed) if encoder == "cnn" else None)
    started = time.time()
    model = train_supervised(model, x, z, device=device,
                             redundancy_weight=redundancy_weight)
    return model, time.time() - started


@torch.no_grad()
def accuracy_from_program(model, program, y):
    """A cabeça da rede aplicada a um programa que pode não ser o dela."""

    logits = model.head(program)
    return float((logits.argmax(-1).cpu().numpy() == y).mean())


def translate_with(codes, group_map, value_maps, target_groups):
    """Mesma rota de `SJGlue.translate_codes`, mas com mapas de filler injetados.

    Serve ao controle: manter o roteamento de grupos da cola e trocar só a
    permutação de valores isola quanto do ganho vem do alinhamento de fillers e
    quanto vem apenas de descartar os grupos não resolvidos (que viram -1).
    """
    out = torch.full((len(codes), target_groups), -1, dtype=codes.dtype, device=codes.device)
    for source_group, (target_group, values) in enumerate(zip(group_map, value_maps)):
        if target_group < 0 or values is None:
            continue
        table = torch.as_tensor(values, dtype=torch.long, device=codes.device)
        source = codes[:, source_group].long()
        active = source >= 0
        out[active, target_group] = table[source[active]].to(out.dtype)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="bloodmnist")
    p.add_argument("--encoder", choices=["linear", "cnn"], default="cnn")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--card", type=int, default=8)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--hv-dim", type=int, default=128)
    p.add_argument("--frame-embed", type=int, default=128)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--redundancy-weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evidence", type=int, default=4000,
                   help="amostras usadas para estimar os operadores de ação")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--actions", choices=sorted(ACTION_SETS), default="geometric")
    p.add_argument("--permutation-nulls", type=int, default=119,
                   help="nulos de permutação do certificado; o custo dominante")
    p.add_argument("--bootstrap-replicates", type=int, default=16)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default="results/glue")
    args = p.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_raw, y, class_names, image_shape = load_medmnist(args.dataset)
    if args.limit:
        x_raw, y = x_raw[:args.limit], y[:args.limit]
    channels, height, width = image_shape

    x_tr_raw, x_te_raw, y_tr, y_te = train_test_split(
        x_raw, y, test_size=0.25, random_state=args.seed, stratify=y)

    mean = x_tr_raw.mean(axis=0, keepdims=True)
    std = x_tr_raw.std(axis=0, keepdims=True) + 1e-6
    prep = lambda flat: ((flat - mean) / std).astype(np.float32)
    x_tr, x_te = prep(x_tr_raw), prep(x_te_raw)

    action_set = ACTION_SETS[args.actions]

    common = dict(n_classes=len(class_names), image_shape=image_shape,
                  encoder=args.encoder, epochs=args.epochs,
                  redundancy_weight=args.redundancy_weight, device=device,
                  card=args.card, groups=args.groups, units=args.units,
                  layers=args.layers, topk=args.topk, hv_dim=args.hv_dim,
                  frame_embed=args.frame_embed, batch=args.batch, lr=args.lr)

    print(f"dataset={args.dataset} treino={len(x_tr)} teste={len(x_te)} "
          f"classes={len(class_names)} encoder={args.encoder} "
          f"ações={args.actions} ({len(action_set)})")
    print("treinando rede A...", flush=True)
    model_a, secs_a = build_and_train(x_tr, y_tr, seed=args.seed, **common)
    print(f"  {secs_a:.0f}s")
    print("treinando rede B (independente)...", flush=True)
    model_b, secs_b = build_and_train(x_tr, y_tr, seed=args.seed + 1000, **common)
    print(f"  {secs_b:.0f}s")

    # Evidência de transição: mesmas imagens, mesmas ações, as duas redes.
    ev_raw = x_tr_raw[:args.evidence]
    images = ev_raw.reshape(len(ev_raw), height, width, channels)
    base_a = encode(model_a, prep(ev_raw), device)
    base_b = encode(model_b, prep(ev_raw), device)
    actions_a, actions_b = {}, {}
    for name, fn in action_set.items():
        moved = prep(fn(images).reshape(len(ev_raw), -1))
        actions_a[name] = encode(model_a, moved, device)
        actions_b[name] = encode(model_b, moved, device)
        print(f"  ação '{name}': códigos mudaram em A "
              f"{float((actions_a[name] != base_a).mean()):.1%}, B "
              f"{float((actions_b[name] != base_b).mean()):.1%}")

    # A cola casa o grupo i de A com o grupo j de B comparando como cada um
    # responde a cada ação. Se todos os grupos respondem igual, não existe
    # assinatura para casar e nenhum alinhamento pode ser encontrado.
    signature = {}
    for tag, base, actions in (("A", base_a, actions_a), ("B", base_b, actions_b)):
        rates = np.stack([(actions[n] != base).mean(axis=0) for n in action_set])
        signature[tag] = float(rates.std(axis=1).mean())
        print(f"  assinatura por grupo em {tag}: desvio médio entre grupos "
              f"{signature[tag]:.4f}  (perto de 0 = grupos indistinguíveis)")

    names = list(action_set)
    train_actions, validation_actions = names[:3], names[3:]
    print(f"\najustando a cola (treino={train_actions} validação={validation_actions})...",
          flush=True)
    glue = fit_sj_glue(base_a, actions_a, model_a.sj, base_b, actions_b, model_b.sj,
                       train_actions=train_actions,
                       validation_actions=validation_actions,
                       permutation_nulls=args.permutation_nulls,
                       bootstrap_replicates=args.bootstrap_replicates,
                       seed=args.seed)
    cert = glue.certificate
    print(f"  status={glue.status} executável={glue.is_executable} tipo={glue.kind}")
    print(f"  motivo={cert.reason}")
    print(f"  group_map={cert.group_map}")
    print(f"  cobertura fonte={glue.source_coverage:.0%} alvo={glue.target_coverage:.0%}")
    print(f"  score treino={cert.train_score} validação={cert.validation_score} "
          f"p-valor={cert.action_pvalue}")
    print(f"  estabilidade bootstrap grupo={cert.bootstrap_group_stability} "
          f"filler={cert.bootstrap_filler_stability}")

    codes_a_np = encode(model_a, x_te, device)
    codes_b_np = encode(model_b, x_te, device)
    codes_a = torch.from_numpy(codes_a_np).long().to(device)
    codes_b = torch.from_numpy(codes_b_np).long().to(device)

    # Mesmas métricas que a bateria k-fold reporta, por rede.
    quality = {}
    for tag, model, codes_np in (("A", model_a, codes_a_np), ("B", model_b, codes_b_np)):
        probs = predict_proba(model, x_te, device)
        m = classification_metrics(y_te, probs)
        m.update(specialization_metrics(codes_np, y_te, model.sj.edges, args.card))
        m.update(alphabet_metrics(codes_np, args.card))
        m["vsa_cleanup_accuracy"] = vsa_cleanup_accuracy(model, codes_np, device)
        quality[tag] = m
    print("\n== qualidade de cada rede (mesmas métricas do k-fold) ==")
    keys = ["accuracy", "auc_macro", "balanced_accuracy", "f1_macro", "cohen_kappa",
            "mcc", "nmi_code_label_mean", "nmi_between_groups", "edge_pair_gain",
            "code_perplexity_mean", "alphabet_utilization", "vsa_cleanup_accuracy"]
    print(f"  {'métrica':<24} {'A':>9} {'B':>9}")
    for k in keys:
        if k in quality["A"]:
            print(f"  {k:<24} {quality['A'][k]:>9.4f} {quality['B'][k]:>9.4f}")

    with torch.no_grad():
        ceiling_b = accuracy_from_program(model_b, model_b.sj.program_feature_from_codes(codes_b), y_te)
        ceiling_a = accuracy_from_program(model_a, model_a.sj.program_feature_from_codes(codes_a), y_te)
        no_glue = accuracy_from_program(model_b, model_b.sj.program_feature_from_codes(codes_a), y_te)
        # allow_candidate deixa medir mesmo quando o certificado não é "grounded";
        # sem isso um certificado recusado impediria qualquer número.
        with_glue = accuracy_from_program(
            model_b, glue.import_program(model_b.sj, codes_a, allow_candidate=True), y_te)

        # Controle 1: nenhum código — o que a cabeça responde sem informação.
        empty = torch.full_like(codes_b, -1)
        no_codes = accuracy_from_program(
            model_b, model_b.sj.program_feature_from_codes(empty), y_te)

        # Controle 2: roteamento de grupos da cola, permutações de filler
        # aleatórias. Se empatar com "com cola", o ganho não veio do alinhamento.
        rng = np.random.default_rng(args.seed)
        accuracies = []
        for _ in range(8):
            fake = tuple(None if v is None else tuple(rng.permutation(args.card))
                          for v in cert.value_maps)
            codes_fake = translate_with(codes_a, cert.group_map, fake, glue.target_groups)
            accuracies.append(accuracy_from_program(
                model_b, model_b.sj.program_feature_from_codes(codes_fake), y_te))
        shuffled = float(np.mean(accuracies))

    chance = float(np.bincount(y_te).max() / len(y_te))
    print("\n== acurácia na cabeça de B, conjunto de teste ==")
    print(f"  classe majoritária (chance)      : {chance:.4f}")
    print(f"  teto  B->B                       : {ceiling_b:.4f}")
    print(f"  (referência) A->A                : {ceiling_a:.4f}")
    print(f"  SEM cola  (A->B cru)             : {no_glue:.4f}")
    print(f"  controle: nenhum código (-1)     : {no_codes:.4f}")
    print(f"  controle: fillers embaralhados   : {shuffled:.4f}  (média de 8)")
    print(f"  COM cola  (A->B traduzido)       : {with_glue:.4f}")
    print(f"  ganho vs sem cola                : {with_glue - no_glue:+.4f}")
    print(f"  ganho vs fillers embaralhados    : {with_glue - shuffled:+.4f}  <- o que a cola de fato alinhou")
    recovered = (with_glue - no_glue) / max(ceiling_b - no_glue, 1e-9)
    print(f"  fração do teto recuperada        : {recovered:.1%}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": args.dataset, "encoder": args.encoder, "epochs": args.epochs,
        "action_set": args.actions,
        "redundancy_weight": args.redundancy_weight,
        "group_action_signature": signature,
        "grammar_signature": list(model_a.sj.grammar_signature),
        "network_quality": quality,
        "groups": args.groups, "card": args.card,
        "train_actions": train_actions, "validation_actions": validation_actions,
        "accuracy": {"chance": chance, "ceiling_b": ceiling_b, "ceiling_a": ceiling_a,
                      "no_glue": no_glue, "no_codes": no_codes,
                      "shuffled_fillers": shuffled, "with_glue": with_glue,
                      "glue_gain_vs_no_glue": with_glue - no_glue,
                      "glue_gain_vs_shuffled": with_glue - shuffled,
                      "ceiling_fraction_recovered": recovered},
        "glue": glue.summary(),
    }
    tag = f"rw{args.redundancy_weight:g}"
    path = out_dir / f"{args.dataset}_{args.encoder}_{args.actions}_{tag}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"\nsalvo {path}")


if __name__ == "__main__":
    main()
