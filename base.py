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


class _HardForwardSoftBackward(torch.autograd.Function):
    """Return the exact hard tensor while differentiating through the soft one."""

    @staticmethod
    def forward(ctx, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return hard

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, gradient


SJ_GRAMMAR_VERSION = "BASED-sparse-joint"
SJ_BINDING = "hadamard"
SJ_PROGRAM_GRADES = (1, 2)
SJ_JOINT_RULE = "literal-neural-intersection"
SJ_MISSING_ATOM = "absent-zero-term"


def sj_grammar_signature() -> tuple:
    """Public compatibility signature for native SJ program composition."""
    return (
        SJ_GRAMMAR_VERSION,
        SJ_BINDING,
        SJ_PROGRAM_GRADES,
        SJ_JOINT_RULE,
        SJ_MISSING_ATOM,
    )


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
    denoise: str = "identity" #<------- !!


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
    # i think cuda doesnt implement integer einsum/baddbmm so Hard masks are exactly
    # binary, so the floating product is integral and can be cast afterward
    return torch.einsum("lgu,lhu->ghl", masks.float(), masks.float()).round().long()


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
        if cfg.denoise == "identity":
            self.denoise: nn.Module = nn.Identity()
        elif cfg.denoise == "mlp":
            self.denoise = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        else:
            raise ValueError("cfg.denoise must be 'identity' or 'mlp'")
        self.embed_decoder = nn.Sequential(
            nn.Linear(2 * D, 2 * in_dim), nn.GELU(), nn.Linear(2 * in_dim, in_dim)
        )
        self.temp = 0.10
        self.pair_write_scale = math.sqrt(D)

    @property
    def program_dim(self) -> int:
        return 2 * self.D

    @property
    def grammar_signature(self) -> tuple:
        return sj_grammar_signature()

    def routing_masks(self) -> torch.Tensor:
        """Return the scaffold used by the current hard forward pass."""

        return self.masks

    def routing_pair_masks(self, masks: torch.Tensor) -> torch.Tensor:
        if not len(self.edges):
            return masks.new_zeros(self.L, 0, self.U)
        ei, ej = self.eidx[:, 0], self.eidx[:, 1]
        return masks[:, ei] * masks[:, ej]

    def routing_edge_activity(self, pair_masks: torch.Tensor) -> torch.Tensor:
        if not len(self.edges):
            return pair_masks.new_zeros(0)
        return (pair_masks.sum((0, 2)) > 0).to(pair_masks.dtype)

    def routing_regularization(self) -> dict[str, torch.Tensor]:
        zero = self.codebook.new_zeros(())
        return {"total": zero}

    def state_regularization(
        self, out: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        zero = self.codebook.new_zeros(())
        return {"total": zero}

    def scaffold_diagnostics(self) -> dict[str, object]:
        masks = self.routing_masks().detach()
        intersections = architecture_intersection_tensor(masks)
        adjacency = intersections.sum(-1) > 0
        adjacency.fill_diagonal_(False)
        return {
            "learned": False,
            "frozen": True,
            "hard_masks": masks.int().cpu().tolist(),
            "capacity_per_layer_group": masks.sum(-1).int().cpu().tolist(),
            "unit_load_mean": float(masks.sum(1).mean()),
            "unit_load_max": int(masks.sum(1).max()),
            "edge_count": int(adjacency.triu(1).sum()),
            "intersection_tensor": intersections.cpu().tolist(),
        }

    def _base(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.input(x).view(len(x), self.U, self.W)
        hs = [h]
        for transition in self.transitions:
            h = transition(h.flatten(1)).view(len(x), self.U, self.W)
            hs.append(h)
        return hs

    def _fragments(
        self,
        hs: Sequence[torch.Tensor],
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        masks = self.routing_masks() if masks is None else masks
        h = torch.stack(tuple(hs), dim=1)
        return (
            torch.einsum("bluw,lgu->blgw", h, masks)
            / masks.sum(-1).clamp_min(1.0)[None, :, :, None]
        )

    def _cleanup(self, values: torch.Tensor) -> torch.Tensor:
        values = F.normalize(values, dim=-1)
        bank = F.normalize(self.codebook, dim=-1)
        return torch.einsum("bgd,gcd->bgc", values, bank) / self.temp

    def _bind(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """MAP-bind unit vectors without shrinking their grade by sqrt(D)."""
        return left * right * math.sqrt(self.D)

    def _bind_for_write(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        """Bind for recurrent writes, with legacy-checkpoint compatibility."""
        return left * right * self.pair_write_scale

    def codes_to_vsa(self, codes_or_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bank = F.normalize(self.codebook, dim=-1)
        if codes_or_probs.ndim == 2:
            codes = codes_or_probs.long()
            if bool((codes >= self.C).any()):
                raise ValueError("SJ code is outside the filler vocabulary")
            valid = codes >= 0
            safe = codes.clamp_min(0)
            probs = F.one_hot(safe, self.C).float()
            probs = probs * valid[..., None].to(probs.dtype)
        else:
            probs = codes_or_probs
        filler = torch.einsum("bgc,gcd->bgd", probs, bank)
        unary = self.roles[None] * filler
        return filler, unary

    def program_feature_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        _, unary = self.codes_to_vsa(probs)
        active = probs.sum(-1) > 0
        unary_den = active.sum(1).clamp_min(1).to(unary.dtype).sqrt()[:, None]
        unary_bundle = unary.sum(1) / unary_den
        if len(self.edges):
            masks = self.routing_masks()
            pair_masks = self.routing_pair_masks(masks)
            edge_activity = self.routing_edge_activity(pair_masks)
            ei, ej = self.eidx[:, 0], self.eidx[:, 1]
            pair_active = active[:, ei] & active[:, ej]
            pair_terms = self._bind(unary[:, ei], unary[:, ej])
            pair_active = pair_active.to(pair_terms.dtype) * edge_activity[None]
            pair_terms = pair_terms * pair_active[..., None]
            pair_den = pair_active.sum(1).clamp_min(1).sqrt()[:, None]
            pair_bundle = pair_terms.sum(1) / pair_den
        else:
            pair_bundle = torch.zeros_like(unary_bundle)
        return torch.cat((unary_bundle, pair_bundle), dim=-1)

    def program_feature_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.long()
        if bool((codes >= self.C).any()):
            raise ValueError("SJ code is outside the filler vocabulary")
        valid = codes >= 0
        safe = codes.clamp_min(0)
        probs = F.one_hot(safe, self.C).float()
        probs = probs * valid[..., None].to(probs.dtype)
        return self.program_feature_from_probs(probs)

    def forward(self, x: torch.Tensor, pair_writes: bool = True) -> dict[str, torch.Tensor]:
        masks = self.routing_masks()
        pair_masks = self.routing_pair_masks(masks)
        edge_activity = self.routing_edge_activity(pair_masks)
        hs = self._base(x)
        fragments = self._fragments(hs, masks)
        values = F.normalize(self.to_value(fragments.mean(1)), dim=-1)
        unary = self.roles[None] * values

        # Decode each unary/pair message once, then route it to every layer
        unary_messages = self.unary_write(unary)
        pair_messages = None
        if pair_writes and len(self.edges):
            ei, ej = self.eidx[:, 0], self.eidx[:, 1]
            pair_messages = self.pair_write(
                self._bind_for_write(unary[:, ei], unary[:, ej])
            )

        updated = []
        for layer, h in enumerate(hs):
            unary_write = (
                torch.einsum("bgw,gu->buw", unary_messages, masks[layer])
                / masks[layer].sum(0).clamp_min(1.0)[None, :, None]
            )
            z = h + 0.20 * unary_write
            if pair_messages is not None:
                pair_write = (
                    torch.einsum("bew,eu->buw", pair_messages, pair_masks[layer])
                    / pair_masks[layer].sum(0).clamp_min(1.0)[None, :, None]
                )
                z = z + 0.20 * pair_write
            updated.append(self.norms[layer](z))

        values2 = F.normalize(
            self.to_value(self._fragments(updated, masks).mean(1)), dim=-1
        )

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
            pair_terms = self._bind(unary_q[:, ei], unary_q[:, ej])
            pair_terms = pair_terms * edge_activity[None, :, None]
            pair_bundle = pair_terms.sum(1) / edge_activity.sum().clamp_min(1.0).sqrt()
        else:
            pair_bundle = torch.zeros_like(unary_bundle)
        program = torch.cat((unary_bundle, pair_bundle), dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "codes": logits.argmax(-1),
            "den": denoised,
            "quantized": quantized,
            # hard-forward / soft-backward filler vectors (public)
            "quantized_st": quantized_st,
            "program": program,
            "recon_embed": self.embed_decoder(program),
            "routing_masks": masks,
            "routing_edge_activity": edge_activity,
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


class LearnedScaffoldSparseVQCore(SparseVQCore):
    """SparseVQCore whose overlapping population scaffold is learned.

    Every layer/group selects exactly ``topk`` units in the hard forward pass.
    Straight-through soft assignments carry task, reconstruction, VQ, VSA, and
    mechanism gradients into ``mask_logits``.  Groups may overlap; a literal
    pair is active exactly when their hard supports intersect in some layer.
    """

    def __init__(
        self,
        in_dim: int,
        cfg: SJConfig,
        seed: int,
        *,
        mask_temperature: float = 0.75,
        target_mean_degree: float = 2.0,
    ) -> None:
        if cfg.topk <= 0 or cfg.topk > cfg.units:
            raise ValueError("learned scaffold requires 0 < topk <= units")
        generator = torch.Generator().manual_seed(seed + 41_003)
        logits = 0.15 * torch.randn(
            cfg.layers, cfg.groups, cfg.units, generator=generator
        )
        initial = torch.zeros_like(logits)
        initial.scatter_(-1, logits.topk(cfg.topk, dim=-1).indices, 1.0)
        super().__init__(in_dim, cfg, initial, seed)

        # Every group pair is a candidate literal joint.  Hard support overlap
        # decides whether it exists; inactive candidates contribute exactly 0.
        self.edges = tuple(itertools.combinations(range(cfg.groups), 2))
        self.eidx = torch.tensor(self.edges, dtype=torch.long)
        self.mask_logits = nn.Parameter(logits)
        self.mask_temperature = float(mask_temperature)
        self.target_mean_degree = float(
            min(max(target_mean_degree, 0.0), max(cfg.groups - 1, 0))
        )
        self.register_buffer("initial_routing_masks", initial)
        self.register_buffer("frozen_routing_masks", torch.zeros_like(initial))
        self.register_buffer("scaffold_is_frozen", torch.tensor(False))

    def soft_routing_masks(self) -> torch.Tensor:
        temperature = max(self.mask_temperature, 1e-4)
        return F.softmax(self.mask_logits / temperature, dim=-1) * self.cfg.topk

    def hard_routing_masks(self) -> torch.Tensor:
        hard = torch.zeros_like(self.mask_logits)
        hard.scatter_(-1, self.mask_logits.topk(self.cfg.topk, dim=-1).indices, 1.0)
        return hard

    def routing_masks(self) -> torch.Tensor:
        if bool(self.scaffold_is_frozen.item()):
            return self.frozen_routing_masks
        hard = self.hard_routing_masks()
        soft = self.soft_routing_masks()
        return _HardForwardSoftBackward.apply(hard, soft)

    def routing_edge_activity(self, pair_masks: torch.Tensor) -> torch.Tensor:
        if not len(self.edges):
            return pair_masks.new_zeros(0)
        overlap = pair_masks.sum((0, 2))
        hard = (overlap.detach() > 0).to(overlap.dtype)
        soft = 1.0 - torch.exp(-overlap)
        return _HardForwardSoftBackward.apply(hard, soft)

    def routing_regularization(self) -> dict[str, torch.Tensor]:
        masks = self.routing_masks()
        soft = self.soft_routing_masks()
        flat = masks.permute(1, 0, 2).reshape(self.G, -1)
        normalized = F.normalize(flat, dim=-1)
        similarity = normalized @ normalized.T
        off_diagonal = ~torch.eye(self.G, dtype=torch.bool, device=masks.device)
        duplicate = F.relu(similarity[off_diagonal] - 0.55).square().mean()

        if self.L > 1:
            continuity = 1.0 - F.cosine_similarity(
                masks[:-1], masks[1:], dim=-1
            ).mean()
        else:
            continuity = masks.new_zeros(())

        intersections = torch.einsum("lgu,lhu->gh", masks, masks)
        eye = torch.eye(self.G, dtype=masks.dtype, device=masks.device)
        adjacency = (1.0 - torch.exp(-intersections)) * (1.0 - eye)
        degree = adjacency.sum(-1)
        laplacian = torch.diag(degree) - adjacency
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        algebraic_connectivity = (
            eigenvalues[1] if self.G > 1 else masks.new_tensor(1.0)
        )
        connectivity = F.relu(0.20 - algebraic_connectivity).square()
        edge_budget = (degree.mean() - self.target_mean_degree).square()

        load = masks.sum(1)
        expected_load = self.G * self.cfg.topk / self.U
        load_balance = ((load - expected_load) / max(expected_load, 1e-6)).square().mean()
        overload = F.relu(load - 2.0).square().mean()
        confidence = (soft * (1.0 - soft.clamp(max=1.0))).abs().mean()

        total = (
            0.10 * confidence
            + 0.75 * duplicate
            + 0.20 * continuity
            + 1.00 * connectivity
            + 0.08 * edge_budget
            + 0.08 * load_balance
            + 0.10 * overload
        )
        return {
            "total": total,
            "confidence": confidence,
            "duplicate": duplicate,
            "layer_continuity": continuity,
            "connectivity": connectivity,
            "algebraic_connectivity": algebraic_connectivity,
            "edge_budget": edge_budget,
            "load_balance": load_balance,
            "overload": overload,
            "mean_degree": degree.mean(),
        }

    def state_regularization(
        self, out: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        probabilities = out["probs"]
        marginal = probabilities.mean(0)
        per_group_usage_kl = (
            marginal
            * (marginal.clamp_min(1e-8).log() + math.log(self.C))
        ).sum(-1).mean()

        centered = probabilities - probabilities.mean(0, keepdim=True)
        flattened = centered.permute(1, 0, 2).reshape(self.G, -1)
        normalized = F.normalize(flattened, dim=-1)
        similarity = normalized @ normalized.T
        off_diagonal = ~torch.eye(
            self.G, dtype=torch.bool, device=probabilities.device
        )
        state_redundancy = similarity[off_diagonal].square().mean()
        total = 0.10 * per_group_usage_kl + 0.10 * state_redundancy
        return {
            "total": total,
            "per_group_usage_kl": per_group_usage_kl,
            "state_redundancy": state_redundancy,
        }

    @torch.no_grad()
    def freeze_scaffold(self) -> torch.Tensor:
        hard = self.hard_routing_masks()
        self.frozen_routing_masks.copy_(hard)
        self.scaffold_is_frozen.fill_(True)
        self.mask_logits.requires_grad_(False)
        # Retain the fixed-size candidate-pair buffer
        return hard.clone()

    def scaffold_diagnostics(self) -> dict[str, object]:
        masks = self.routing_masks().detach()
        intersections = architecture_intersection_tensor(masks)
        adjacency = intersections.sum(-1) > 0
        adjacency.fill_diagonal_(False)
        visited: set[int] = set()
        components = 0
        for start in range(self.G):
            if start in visited:
                continue
            components += 1
            stack = [start]
            visited.add(start)
            while stack:
                node = stack.pop()
                for neighbor in torch.where(adjacency[node])[0].tolist():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
        flat = F.normalize(masks.permute(1, 0, 2).reshape(self.G, -1), dim=-1)
        similarity = flat @ flat.T
        off_diagonal = similarity[~torch.eye(self.G, dtype=torch.bool, device=masks.device)]
        return {
            "learned": True,
            "frozen": bool(self.scaffold_is_frozen.item()),
            "hard_masks": masks.int().cpu().tolist(),
            "capacity_per_layer_group": masks.sum(-1).int().cpu().tolist(),
            "unit_load_mean": float(masks.sum(1).mean()),
            "unit_load_max": int(masks.sum(1).max()),
            "edge_count": int(adjacency.triu(1).sum()),
            "connected_components": components,
            "max_group_mask_cosine": float(off_diagonal.max()) if len(off_diagonal) else 0.0,
            "mean_group_mask_cosine": float(off_diagonal.mean()) if len(off_diagonal) else 0.0,
            "fraction_memberships_changed_from_initial": float(
                (masks != self.initial_routing_masks).float().mean()
            ),
            "intersection_tensor": intersections.cpu().tolist(),
            "hard_masks": masks.int().cpu().tolist(),
        }


class _PopulationRoutedTransitions:
    """Make the sparse population scaffold constrain neural computation.

    A cross-layer unit connection exists exactly when at least one SJ group
    contains both endpoint units in the corresponding layers.  Overlapping
    group memberships therefore create literal shared paths.  The hard forward
    graph stays auditable while its soft surrogate sends gradients to learned
    membership logits.
    """

    def transition_connectivity(
        self, masks: torch.Tensor, layer: int
    ) -> torch.Tensor:
        overlap = masks[layer].T @ masks[layer + 1]
        hard = (overlap.detach() > 0).to(overlap.dtype)
        soft = 1.0 - torch.exp(-overlap)
        return _HardForwardSoftBackward.apply(hard, soft)

    def _base(self, x: torch.Tensor) -> list[torch.Tensor]:
        masks = self.routing_masks()
        h = self.input(x).view(len(x), self.U, self.W)
        hs = [h]
        for layer, transition in enumerate(self.transitions):
            connectivity = self.transition_connectivity(masks, layer)
            # nn.Linear uses [output, input]  
            # every channel of a connected source unit may reach every channel of its connected target.
            block = connectivity.T.repeat_interleave(self.W, 0).repeat_interleave(
                self.W, 1
            )
            fan_in = connectivity.sum(0).clamp_min(1.0)
            scale = (self.U / fan_in).sqrt().repeat_interleave(self.W)[:, None]
            linear = transition[0]
            h = F.linear(
                h.flatten(1), linear.weight * block * scale, linear.bias
            )
            h = transition[1](h).view(len(x), self.U, self.W)
            hs.append(h)
        return hs

    def routed_transition_diagnostics(self) -> dict[str, object]:
        masks = self.routing_masks().detach()
        densities = []
        fan_ins = []
        for layer in range(self.L - 1):
            connectivity = self.transition_connectivity(masks, layer).detach()
            densities.append(float(connectivity.mean()))
            fan_ins.append(float(connectivity.sum(0).mean()))
        return {
            "population_routed_transitions": True,
            "transition_unit_density": densities,
            "mean_transition_unit_fan_in": fan_ins,
        }


class RoutedSparseVQCore(_PopulationRoutedTransitions, SparseVQCore):
    """Fixed-scaffold SJ core with scaffold-constrained neural transitions."""

    def scaffold_diagnostics(self) -> dict[str, object]:
        record = super().scaffold_diagnostics()
        record.update(self.routed_transition_diagnostics())
        return record


class RoutedLearnedScaffoldSparseVQCore(
    _PopulationRoutedTransitions, LearnedScaffoldSparseVQCore
):
    """Learned sparse overlapping subgraphs that route the acting network."""

    def scaffold_diagnostics(self) -> dict[str, object]:
        record = super().scaffold_diagnostics()
        record.update(self.routed_transition_diagnostics())
        return record
