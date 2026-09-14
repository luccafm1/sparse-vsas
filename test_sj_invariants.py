from __future__ import annotations

"""Teste de contrato do `sj_invariants`.

Escrito a partir do que o `base.py` assume nos call sites. Vale pra duas coisas:

1. validar a reconstrução temporária do `sj_invariants.py`;
2. quando o Lucca mandar o arquivo original, rodar isto contra ele. Se passar,
   a semântica bate com o que o resto do `base.py` espera. Se falhar, o ponto
   exato da divergência aparece aqui.

    python test_sj_invariants.py
"""

import numpy as np
import torch

from base import SJConfig, build_model, encode, make_rigid_scaffold, vsa_cleanup_accuracy
from sj_invariants import code_probabilities, full_path_locality

GROUPS, CARD = 8, 16


def _core(alphabet=None):
    # hv_dim precisa ser folgado em relação a GROUPS: a recuperação do cleanup é
    # limitada pela capacidade do bundle (8 papéis superpostos em D dimensões).
    cfg = SJConfig(cardinalities=tuple([CARD] * GROUPS), n_factors=1, input_dim=24,
                   units=64, layers=3, topk=4, hv_dim=256, frame_embed=32, seed=3)
    model = build_model(cfg, (4,))
    if alphabet is not None:
        model.sj.install_alphabet(alphabet)
    return model


def check(name: str, condition: bool) -> bool:
    print(f"  [{'ok ' if condition else 'FALHA'}] {name}")
    return condition


def main() -> None:
    torch.manual_seed(0)
    model = _core()
    core = model.sj
    results = []

    print("códigos duros -> one-hot")
    codes = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]])
    probs = code_probabilities(core, codes)
    results.append(check("shape [B, G, C]", tuple(probs.shape) == (2, GROUPS, CARD)))
    results.append(check("cada linha soma 1", torch.allclose(probs.sum(-1), torch.ones(2, GROUPS))))
    results.append(check("one-hot no índice certo",
                         bool((probs.argmax(-1) == codes).all())))

    print("\nátomo ausente (código -1)")
    codes_absent = torch.tensor([[-1, 1, 2, 3, 4, 5, 6, 7]])
    probs_absent = code_probabilities(core, codes_absent)
    results.append(check("linha do -1 fica toda zero",
                         float(probs_absent[0, 0].sum()) == 0.0))
    results.append(check("demais grupos intactos",
                         torch.allclose(probs_absent[0, 1:].sum(-1), torch.ones(GROUPS - 1))))
    results.append(check("program_feature_from_codes aceita ausente",
                         core.program_feature_from_codes(codes_absent).shape[-1]
                         == core.program_dim))

    print("\nalfabeto por grupo (install_alphabet)")
    restricted = _core(alphabet=tuple([3] * GROUPS)).sj
    over = torch.full((1, GROUPS), 9)                      # fora do alfabeto de 3
    probs_over = code_probabilities(restricted, over)
    results.append(check("filler fora do alfabeto é zerado",
                         float(probs_over.sum()) == 0.0))
    soft = torch.rand(4, GROUPS, CARD)
    probs_soft = code_probabilities(restricted, soft)
    results.append(check("massa suave zera fillers mortos",
                         float(probs_soft[:, :, 3:].sum()) == 0.0))
    results.append(check("massa suave renormaliza para 1",
                         torch.allclose(probs_soft.sum(-1), torch.ones(4, GROUPS), atol=1e-5)))

    print("\nhard_only")
    hard = code_probabilities(core, soft, hard_only=True)
    results.append(check("vira one-hot", bool(((hard == 0) | (hard == 1)).all())))
    results.append(check("escolhe o argmax da massa suave",
                         bool((hard.argmax(-1) == soft.argmax(-1)).all())))

    print("\nintegração com o core")
    filler, unary = core.codes_to_vsa(codes)
    results.append(check("codes_to_vsa devolve [B, G, D]",
                         tuple(filler.shape) == (2, GROUPS, core.D)
                         and tuple(unary.shape) == (2, GROUPS, core.D)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = np.random.randn(64, 24).astype(np.float32)
    codes_np = encode(model, x, device)
    acc = vsa_cleanup_accuracy(model, codes_np, device)
    results.append(check(f"vsa_cleanup_accuracy roda e recupera ({acc:.3f})", acc > 0.99))

    print("\nfull_path_locality")
    diag = full_path_locality(model, x, device, rows=16)
    results.append(check("devolve dict com outside_fraction",
                         isinstance(diag, dict) and "outside_fraction" in diag))
    results.append(check(f"outside_fraction em [0, 1] ({diag['outside_fraction']:.3f})",
                         0.0 <= diag["outside_fraction"] <= 1.0))

    total, passed = len(results), sum(results)
    print(f"\n{passed}/{total} verificações passaram")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
