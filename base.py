from __future__ import annotations

"""Base sparse hypervector network architecture"""

from dataclasses import dataclass
import itertools
import math
import random
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SJConfig:
    seed: int = 11
    groups: int = 8
    card: int = 16
    layers: int = 3
    units: int = 96
    topk: int = 12
    unit_width: int = 16
    hv_dim: int = 256
    video_frames_obs: int = 12
    video_frames_future: int = 12
    video_size: int = 64
    max_objects: int = 10
    state_dim: int = 13
    dyn_dim: int = 6
    frame_embed: int = 160
    state_embed: int = 192
    batch: int = 24
    lr: float = 8e-4
    weight_decay: float = 1e-5


def bipolar(gen: torch.Generator, dim: int) -> torch.Tensor:
    return torch.where(
        torch.rand(dim, generator=gen) > 0.5,
        torch.ones(dim),
        -torch.ones(dim),
    )


def physical_edges(masks: torch.Tensor) -> tuple[tuple[int, int], ...]:
    """Edges are literal same-layer support intersections."""
    _, groups, _ = masks.shape
    return tuple(
        (i, j)
        for i, j in itertools.combinations(range(groups), 2)
        if bool(((masks[:, i] * masks[:, j]).sum() > 0).item())
    )


def architecture_intersection_tensor(masks: torch.Tensor) -> torch.Tensor:
    """Layerwise literal intersection multiplicities [G,G,L]."""
    if masks.ndim != 3:
        raise ValueError("masks must be [layers, groups, units]")
    return torch.einsum("lgu,lhu->ghl", masks.long(), masks.long())


def _edge_code(index: int, layers: int) -> tuple[int, ...]:
    x = index + 1
    out = []
    for _ in range(layers):
        out.append(1 + (x % 4))
        x //= 4
    return tuple(out)


def make_rigid_scaffold(cfg: SJConfig, seed: int = 0) -> torch.Tensor:
    """Build a sparse, connected, structurally distinctive layered scaffold.

    The architecture itself is intended to be a Glue compatibility certificate.
    For G=8 a fixed sparse graph is used; other G use a ring-plus-chords graph.
    """
    G, L, U, K = cfg.groups, cfg.layers, cfg.units, cfg.topk
    if G == 8:
        edges = [
            (0, 1), (0, 3), (1, 2), (1, 4), (2, 5),
            (3, 4), (4, 5), (4, 6), (5, 7), (6, 7),
        ]
    else:
        edges = {(i, i + 1) for i in range(G - 1)}
        edges |= {(i, i + 2) for i in range(max(0, G - 2)) if i % 2 == 0}
        edges = sorted(edges)

    masks = torch.zeros(L, G, U)
    next_unit = [0] * L
    for edge_idx, (i, j) in enumerate(edges):
        for layer, count in enumerate(_edge_code(edge_idx, L)):
            for _ in range(count):
                if next_unit[layer] >= U:
                    raise ValueError("units too small for requested sparse scaffold")
                u = next_unit[layer]
                next_unit[layer] += 1
                masks[layer, i, u] = 1.0
                masks[layer, j, u] = 1.0

    rng = random.Random(seed)
    for layer in range(L):
        for group in range(G):
            while int(masks[layer, group].sum()) < K:
                free = torch.where(masks[layer].sum(0) == 0)[0].tolist()
                if next_unit[layer] < U:
                    u = next_unit[layer]
                    next_unit[layer] += 1
                elif free:
                    u = rng.choice(free)
                else:
                    raise ValueError("units/topk configuration leaves no private capacity")
                masks[layer, group, u] = 1.0
    return masks


def permute_scaffold(masks: torch.Tensor, seed: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Hide group/unit labels while preserving exact scaffold isomorphism."""
    layers, groups, units = masks.shape
    rng = random.Random(seed)
    group_perm = list(range(groups))
    rng.shuffle(group_perm)
    out = torch.zeros_like(masks)
    for layer in range(layers):
        unit_perm = list(range(units))
        rng.shuffle(unit_perm)
        for src_group, dst_group in enumerate(group_perm):
            idx = torch.where(masks[layer, src_group] > 0)[0].tolist()
            out[layer, dst_group, torch.tensor([unit_perm[u] for u in idx])] = 1.0
    return out, tuple(group_perm)


class SparseVQCore(nn.Module):
    """VSA-native distributed sparse bottleneck.

    One group is a sparse population distributed across layers.  Unary VSA
    messages are written to all units owned by the group.  Bound pair messages
    are written only to literal shared units.  The exported program is graded:
    [unary bundle ; pair bundle].
    """

    def __init__(self, in_dim: int, cfg: SJConfig, masks: torch.Tensor, seed: int):
        super().__init__()
        self.cfg = cfg
        G, C, L, U = cfg.groups, cfg.card, cfg.layers, cfg.units
        W, D = cfg.unit_width, cfg.hv_dim
        self.G, self.C, self.L, self.U, self.W, self.D = G, C, L, U, W, D

        self.register_buffer("masks", masks.float().clone())
        self.edges = physical_edges(masks)
        eidx = torch.tensor(self.edges, dtype=torch.long)
        self.register_buffer("eidx", eidx)

        if len(self.edges):
            pair_masks = torch.stack([masks[:, i] * masks[:, j] for i, j in self.edges], dim=1)
        else:
            pair_masks = torch.zeros(L, 0, U)
        self.register_buffer("pair_masks", pair_masks)
        self.register_buffer("group_den", masks.sum(-1).clamp_min(1.0))
        self.register_buffer("unary_counts", masks.sum(1).clamp_min(1.0))
        self.register_buffer("pair_counts", pair_masks.sum(1).clamp_min(1.0))

        self.input = nn.Sequential(nn.Linear(in_dim, U * W), nn.GELU())
        self.transitions = nn.ModuleList([
            nn.Sequential(nn.Linear(U * W, U * W), nn.GELU())
            for _ in range(L - 1)
        ])
        self.to_value = nn.Linear(W, D, bias=False)
        self.unary_write = nn.Linear(D, W, bias=False)
        self.pair_write = nn.Linear(D, W, bias=False)
        self.norms = nn.ModuleList([nn.LayerNorm(W) for _ in range(L)])

        gen = torch.Generator().manual_seed(seed + 991)
        self.register_buffer("roles", torch.stack([bipolar(gen, D) for _ in range(G)]))
        self.codebook = nn.Parameter(torch.randn(G, C, D, generator=gen) / math.sqrt(D))
        self.denoise = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.embed_decoder = nn.Sequential(
            nn.Linear(2 * D, 2 * in_dim), nn.GELU(), nn.Linear(2 * in_dim, in_dim)
        )
        self.temp = 0.10

    @property
    def program_dim(self) -> int:
        return 2 * self.D

    def _base(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.input(x).view(len(x), self.U, self.W)
        hs = [h]
        for transition in self.transitions:
            h = transition(h.flatten(1)).view(len(x), self.U, self.W)
            hs.append(h)
        return hs

    def _fragments(self, hs: Sequence[torch.Tensor]) -> torch.Tensor:
        h = torch.stack(tuple(hs), dim=1)
        return (
            torch.einsum("bluw,lgu->blgw", h, self.masks)
            / self.group_den[None, :, :, None]
        )

    def _cleanup(self, values: torch.Tensor) -> torch.Tensor:
        values = F.normalize(values, dim=-1)
        bank = F.normalize(self.codebook, dim=-1)
        return torch.einsum("bgd,gcd->bgc", values, bank) / self.temp

    def codes_to_vsa(self, codes_or_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bank = F.normalize(self.codebook, dim=-1)
        if codes_or_probs.ndim == 2:
            probs = F.one_hot(codes_or_probs.long(), self.C).float()
        else:
            probs = codes_or_probs
        filler = torch.einsum("bgc,gcd->bgd", probs, bank)
        unary = self.roles[None] * filler
        return filler, unary

    def program_feature_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        _, unary = self.codes_to_vsa(probs)
        unary_bundle = unary.sum(1) / math.sqrt(self.G)
        if len(self.edges):
            ei, ej = self.eidx[:, 0], self.eidx[:, 1]
            pair_bundle = (unary[:, ei] * unary[:, ej]).sum(1) / math.sqrt(len(self.edges))
        else:
            pair_bundle = torch.zeros_like(unary_bundle)
        return torch.cat((unary_bundle, pair_bundle), dim=-1)

    def program_feature_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self.program_feature_from_probs(F.one_hot(codes.long(), self.C).float())

    def forward(self, x: torch.Tensor, pair_writes: bool = True) -> dict[str, torch.Tensor]:
        hs = self._base(x)
        fragments = self._fragments(hs)
        values = F.normalize(self.to_value(fragments.mean(1)), dim=-1)
        unary = self.roles[None] * values

        # Decode each unary/pair message once, then route it to every layer.
        unary_messages = self.unary_write(unary)
        pair_messages = None
        if pair_writes and len(self.edges):
            ei, ej = self.eidx[:, 0], self.eidx[:, 1]
            pair_messages = self.pair_write(unary[:, ei] * unary[:, ej])

        updated = []
        for layer, h in enumerate(hs):
            unary_write = (
                torch.einsum("bgw,gu->buw", unary_messages, self.masks[layer])
                / self.unary_counts[layer][None, :, None]
            )
            z = h + 0.20 * unary_write
            if pair_messages is not None:
                pair_write = (
                    torch.einsum("bew,eu->buw", pair_messages, self.pair_masks[layer])
                    / self.pair_counts[layer][None, :, None]
                )
                z = z + 0.20 * pair_write
            updated.append(self.norms[layer](z))

        values2 = F.normalize(self.to_value(self._fragments(updated).mean(1)), dim=-1)

        # Algebraic-only read path: bundle -> unbind -> cleanup -> rebind.
        bundle = (self.roles[None] * values2).sum(1) / math.sqrt(self.G)
        retrieved = bundle[:, None] * self.roles[None] * math.sqrt(self.G)
        denoised = F.normalize(self.denoise(retrieved), dim=-1)
        logits = self._cleanup(denoised)
        probs = logits.softmax(-1)
        one_hot = F.one_hot(probs.argmax(-1), self.C).to(probs.dtype)
        straight_through = one_hot + probs - probs.detach()
        quantized = torch.einsum(
            "bgc,gcd->bgd", straight_through, F.normalize(self.codebook, dim=-1)
        )
        quantized_st = denoised + (quantized - denoised).detach()

        unary_q = self.roles[None] * quantized_st
        unary_bundle = unary_q.sum(1) / math.sqrt(self.G)
        if len(self.edges):
            ei, ej = self.eidx[:, 0], self.eidx[:, 1]
            pair_bundle = (unary_q[:, ei] * unary_q[:, ej]).sum(1) / math.sqrt(len(self.edges))
        else:
            pair_bundle = torch.zeros_like(unary_bundle)
        program = torch.cat((unary_bundle, pair_bundle), dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "codes": logits.argmax(-1),
            "den": denoised,
            "quantized": quantized,
            "program": program,
            "recon_embed": self.embed_decoder(program),
        }

    @torch.inference_mode()
    def encode_codes(self, x: torch.Tensor, pair_writes: bool = True) -> torch.Tensor:
        return self.forward(x, pair_writes=pair_writes)["codes"]

    def vq_losses(self, out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        vq = ((out["quantized"] - out["den"].detach()) ** 2).mean()
        commit = ((out["den"] - out["quantized"].detach()) ** 2).mean()
        p = out["probs"]
        marginal = p.mean((0, 1))
        kl = (marginal * (marginal.clamp_min(1e-8).log() + math.log(self.C))).sum()
        entropy = -(p * p.clamp_min(1e-8).log()).sum(-1).mean()
        return vq, commit, kl, entropy
