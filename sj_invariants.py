from __future__ import annotations

"""RECONSTRUÇÃO TEMPORÁRIA - substituir pelo arquivo original do Lucca.

O `base.py` da branch `ver-real` importa `code_probabilities` e
`full_path_locality` deste módulo, mas ele não foi commitado. Este arquivo é uma
reconstrução feita a partir dos pontos de uso, só para destravar a execução da
bateria de testes. Quando o original aparecer, apague este arquivo.

Contratos inferidos dos call sites em `base.py`:

    code_probabilities(core, codes_or_probs, hard_only=False) -> [B, G, C]
        - `codes_to_vsa` (linha ~388) e `program_feature_from_codes` (~417).
        - `program_feature_from_probs` valida `ndim == 3` e trata
          `probs.sum(-1) == 0` como átomo ausente, então código -1 vira linha
          zerada.
        - alfabeto por grupo vem de `core.filler_live` / `core.filler_mask`
          (ver `install_alphabet`).

    full_path_locality(model, x, device, rows) -> dict
        - `off_subgraph_influence` lê a chave "outside_fraction".
        - modelado sobre `legacy_off_subgraph_influence`, que já está no
          `base.py` e mostra o padrão de gradiente on-units vs off-units.

NOTA: nenhuma destas funções é usada no `forward()` do core, nem em
`acquisition_losses`/`oracle_losses`. O caminho de treino não passa por aqui,
então uma divergência em relação ao original afeta apenas diagnósticos e as
features de programa usadas pelo Glue.
"""

import numpy as np
import torch
import torch.nn.functional as F


def code_probabilities(core, codes_or_probs: torch.Tensor,
                       hard_only: bool = False) -> torch.Tensor:
    """Normaliza códigos duros ou massa suave para ``[batch, groups, fillers]``.

    Uma linha inteiramente zerada significa átomo ausente (código -1), que é o
    que `program_feature_from_probs` interpreta via ``probs.sum(-1) > 0``.
    """

    if not torch.is_tensor(codes_or_probs):
        raise TypeError("codes_or_probs deve ser um torch.Tensor")

    live = core.filler_live.to(codes_or_probs.device)              # (G, C)

    # ---- códigos duros [B, G] -------------------------------------------
    if codes_or_probs.ndim == 2:
        codes = codes_or_probs.long()
        if codes.shape[1] != core.G:
            raise ValueError(f"esperado [batch, {core.G}], veio {tuple(codes.shape)}")
        present = (codes >= 0) & (codes < core.C)
        probs = F.one_hot(codes.clamp(0, core.C - 1), core.C).to(live.dtype)
        probs = probs * present[..., None].to(live.dtype)
        return probs * live[None]

    # ---- massa suave [B, G, C] ------------------------------------------
    if codes_or_probs.ndim != 3:
        raise ValueError("esperado [batch, groups] ou [batch, groups, fillers]")
    if codes_or_probs.shape[1] != core.G or codes_or_probs.shape[2] != core.C:
        raise ValueError(
            f"esperado [batch, {core.G}, {core.C}], veio {tuple(codes_or_probs.shape)}")

    probs = codes_or_probs * live[None]

    if hard_only:
        mass = probs.sum(-1)
        hard = F.one_hot(probs.argmax(-1), core.C).to(probs.dtype)
        return hard * (mass > 0)[..., None].to(probs.dtype)

    total = probs.sum(-1, keepdim=True)
    return torch.where(total > 0, probs / total.clamp_min(1e-12), probs)


def full_path_locality(model, x: np.ndarray, device: torch.device,
                       rows: int = 128) -> dict[str, float]:
    """Sensibilidade do caminho real: quanto do gradiente de cada grupo vem de
    unidades que ele *não* possui.

    ``outside_fraction`` em 0 significa que cada grupo é calculado só a partir
    das unidades que possui; perto de 1 significa que o scaffold não restringe
    nada.
    """

    model.eval()
    core = model.sj
    masks = core.routing_masks().detach()                          # (L, G, U)
    owned = (masks.sum(0) > 0).float()                             # (G, U)

    chunk = torch.from_numpy(x[:rows]).to(device)
    h0 = core.input(model.encoder(chunk)).view(len(chunk), core.U, core.W)
    h0.requires_grad_(True)

    inside_all, outside_all, ratios = [], [], []
    for group in range(core.G):
        hs = core.stack_from_h0(h0, masks)
        fragments = core._fragments(hs, masks)
        values = F.normalize(core.to_value(fragments.mean(1)), dim=-1)
        chosen = core._cleanup(values)[:, group].max(-1).values.sum()

        grad = torch.autograd.grad(chosen, h0, retain_graph=True)[0].abs().sum(-1)
        influence = grad.mean(0)                                   # (U,)

        on, off = owned[group], 1.0 - owned[group]
        inside = float((influence * on).sum())
        outside = float((influence * off).sum())
        inside_mean = inside / float(on.sum().clamp_min(1.0))
        outside_mean = outside / float(off.sum().clamp_min(1.0))

        inside_all.append(inside)
        outside_all.append(outside)
        ratios.append(outside_mean / max(inside_mean, 1e-12))

    inside_total, outside_total = float(np.sum(inside_all)), float(np.sum(outside_all))
    return {
        "outside_fraction": outside_total / max(inside_total + outside_total, 1e-12),
        "outside_over_inside": float(np.mean(ratios)),
        "inside_mass": inside_total,
        "outside_mass": outside_total,
    }
