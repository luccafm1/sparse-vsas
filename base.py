from __future__ import annotations

"""Base sparse hypervector network architecture

# What lets the whole architecture behave both neural and symbolically, natively in both? 

We call the mechanism group or circuit overlaps.

Let group overlaps be defined by the following relation:

    R_ij = F(C_i (intersec) C_j, z_i, z_j),   # regularized toward 1[C_i (intersec) C_j != null set]

In this file, group overlaps are called physical overlaps or physical edges, because we can't settle on the terminology.

# And what is a group?

A group is a sparse neural population distributed across layers.

Neural transitions are routed, and a unit-to-unit edge exists iff some group contains both endpoints. Each group emits one discrete 
filler from its own alphabet by a vector-symbolic, and the exported program is the tuple

    (unary , pair).

Usage:
    cfg = SJModConfig(groups=8, ..., relation="learned", relation_prior_weight=1.0)
    core = RelationalSparseVQCore(in_dim, cfg, masks, seed)
    loss = task_loss + core.relation_regularization()["total"]

#######
Obs.: 'SJ' was the working name of 'sparse-joint networks', we havent really settled on a name so that's what we'll call it
for the time being
"""

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sj_invariants import code_probabilities, full_path_locality

MASKED_FILLER = -1.0e4

SJ_GRAMMAR_VERSION = "true-sparse-joint"
SJ_BINDING = "hadamard"
SJ_PROGRAM_GRADES = (1, 2)
SJ_JOINT_RULE = "literal-neural-intersection"
SJ_MISSING_ATOM = "absent-zero-term"


def sj_grammar_signature() -> tuple:
    """public compatibility signature for native SJ program composition"""

    return (SJ_GRAMMAR_VERSION, SJ_BINDING, SJ_PROGRAM_GRADES,
            SJ_JOINT_RULE, SJ_MISSING_ATOM)


# ============================================================ configuration

@dataclass(frozen=True)
class SJConfig:
    """Architecture and objective"""

    # ---- world-facing shape
    cardinalities: tuple[int, ...]      # per-group filler counts, length G
    n_factors: int                      # K intervention families
    input_dim: int

    # ---- architecture
    units: int = 128
    layers: int = 3
    topk: int = 4
    unit_width: int = 8
    hv_dim: int = 256
    frame_embed: int = 64
    routed: bool = True              

    # ---- alphabet
    # "shared"    every group gets max C and the support term pulls its
    #             effective alphabet toward the size its own learned ownership
    #             implies. Required whenever the assignment is learned.
    # "per_group" group g is told its alphabet is C_g.  
    alphabet: str = "shared"

    # ---- optimization
    lr: float = 2.0e-3
    batch: int = 256
    epochs: int = 40
    seed: int = 0

    # ---- objective weights
    w_sufficiency: float = 1.0
    w_response: float = 2.0
    w_invariance: float = 2.0
    w_support: float = 2.0
    w_vq: float = 1.0
    w_oracle: float = 1.0               
    response_target: float = 0.95

    # ---- ownership
    one_to_one: bool = True             # sinkhorn balancing 
    sinkhorn_iters: int = 20
    sinkhorn_temp: float = 0.5

    # ---- support target
    support_cardinalities: tuple[int, ...] = ()

    @property
    def groups(self) -> int:
        return len(self.cardinalities)

    @property
    def max_card(self) -> int:
        return max(self.cardinalities)


# ============================================================ VSA primitives

class _HardForwardSoftBackward(torch.autograd.Function):
    """exact hard tensors and differentiable soft tensors"""

    @staticmethod
    def forward(ctx, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return hard

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, gradient


def bipolar(gen: torch.Generator, dim: int) -> torch.Tensor:
    """A random +/-1 role vector (bipolar hypervecs)"""

    return torch.where(
        torch.rand(dim, generator=gen) > 0.5,
        torch.ones(dim),
        -torch.ones(dim),
    )


def physical_edges(masks: torch.Tensor) -> tuple[tuple[int, int], ...]:
    """Group pairs that share at least one unit in some layer"""

    groups = masks.shape[1]
    out = []
    for i in range(groups):
        for j in range(i + 1, groups):
            if bool((masks[:, i] * masks[:, j]).sum() > 0):
                out.append((i, j))
    return tuple(out)


def architecture_intersection_tensor(masks: torch.Tensor) -> torch.Tensor:
    groups = masks.shape[1]
    out = torch.zeros(groups, groups, masks.shape[0])
    for i in range(groups):
        for j in range(groups):
            out[i, j] = (masks[:, i] * masks[:, j]).sum(-1)
    return out


def _edge_code(index: int, layers: int) -> tuple[int, ...]:
    x = index + 1
    out = []
    for _ in range(layers):
        out.append(1 + (x % 4))
        x //= 4
    return tuple(out)


def make_rigid_scaffold(cfg: SJConfig, seed: int = 0) -> torch.Tensor:
    """Build a sparse, connected, structurally distinctive layered scaffold""" # ainda n sei oq o chat quer dizer com scaffold mas blz

    G, L, U, K = cfg.groups, cfg.layers, cfg.units, cfg.topk
    if G < 1 or L < 1 or U < 1 or not 0 < K <= U:
        raise ValueError("scaffold requires positive dimensions and 0 < topk <= units")
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
                    raise ValueError("units or topk configuration leaves no private capacity")
                masks[layer, group, u] = 1.0
    return masks


def permute_scaffold(masks: torch.Tensor, seed: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Hide group/unit labels while preserving exact scaffold isomorphism"""

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


# ============================================================ the core

class SparseJointCore(nn.Module):
    """VSA-native distributed sparse bottleneck with heterogeneous alphabets"""

    def __init__(self, in_dim: int, cfg: SJConfig, masks: torch.Tensor, seed: int,
                 alphabet: tuple[int, ...] | None = None):
        super().__init__()
        self.cfg = cfg
        self.routed = cfg.routed
        G, C, L, U = cfg.groups, cfg.max_card, cfg.layers, cfg.units
        W, D = cfg.unit_width, cfg.hv_dim
        self.G, self.C, self.L, self.U, self.W, self.D = G, C, L, U, W, D

        self.register_buffer("masks", masks.float().clone())
        self.edges = physical_edges(masks)
        self.register_buffer("eidx", torch.tensor(self.edges, dtype=torch.long))

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
            nn.Sequential(nn.Linear(U * W, U * W), nn.GELU()) for _ in range(L - 1)
        ])
        self.to_value = nn.Linear(W, D, bias=False)
        self.unary_write = nn.Linear(D, W, bias=False)
        self.pair_write = nn.Linear(D, W, bias=False)
        self.norms = nn.ModuleList([nn.LayerNorm(W) for _ in range(L)])

        gen = torch.Generator().manual_seed(seed + 991)
        self.register_buffer("roles", torch.stack([bipolar(gen, D) for _ in range(G)]))
        self.codebook = nn.Parameter(torch.randn(G, C, D, generator=gen) / math.sqrt(D))
        self.embed_decoder = nn.Sequential(
            nn.Linear(2 * D, 2 * in_dim), nn.GELU(), nn.Linear(2 * in_dim, in_dim)
        )
        self.temp = 0.10
        self.pair_write_scale = math.sqrt(D)

        alphabet = tuple(alphabet) if alphabet is not None else tuple([C] * G)
        self.install_alphabet(alphabet)

    # ---------------------------------------------------------------- alphabet

    def install_alphabet(self, cardinalities: tuple[int, ...]) -> None:
        """Restrict group ``g`` to its first ``C_g`` fillers"""

        if len(cardinalities) != self.G:
            raise ValueError("one alphabet size per group is required")
        mask = torch.zeros(self.G, self.C)
        for group, cardinality in enumerate(cardinalities):
            if not 1 <= cardinality <= self.C:
                raise ValueError(f"alphabet {cardinality} outside 1..{self.C}")
            mask[group, cardinality:] = MASKED_FILLER
        self.register_buffer("filler_mask", mask)
        self.register_buffer("filler_live", (mask == 0).float())
        self.alphabet = tuple(cardinalities)

    # -------------------------------------------------------------- properties

    @property
    def program_dim(self) -> int:
        return 2 * self.D

    @property
    def grammar_signature(self) -> tuple:
        return sj_grammar_signature()

    def routing_masks(self) -> torch.Tensor:
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

    # ---------------------------------------------------------------- routing

    def transition_connectivity(self, masks: torch.Tensor, layer: int) -> torch.Tensor:
        """A unit-to-unit edge exists iff some group holds both endpoints"""

        overlap = masks[layer].T @ masks[layer + 1]
        hard = (overlap.detach() > 0).to(overlap.dtype)
        soft = 1.0 - torch.exp(-overlap)
        return _HardForwardSoftBackward.apply(hard, soft)

    def _base(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.stack_from_h0(self.input(x).view(len(x), self.U, self.W))

    def stack_from_h0(self, h0: torch.Tensor,
                      masks: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Run the transition stack from a given first hidden state.

        if we take h0 as an argument it's what lets off_subgraph_influence
        differentiate the chosen filler with respect to unit activations
        """

        masks = self.routing_masks() if masks is None else masks
        hs, h = [h0], h0
        for layer, transition in enumerate(self.transitions):
            if self.routed:
                connectivity = self.transition_connectivity(masks, layer)
                block = connectivity.T.repeat_interleave(self.W, 0).repeat_interleave(
                    self.W, 1)
                fan_in = connectivity.sum(0).clamp_min(1.0)
                scale = (self.U / fan_in).sqrt().repeat_interleave(self.W)[:, None]
                linear = transition[0]
                h = F.linear(h.flatten(1), linear.weight * block * scale, linear.bias)
                h = transition[1](h).view(len(h0), self.U, self.W)
            else:
                h = transition(h.flatten(1)).view(len(h0), self.U, self.W)
            hs.append(h)
        return hs

    def _fragments(self, hs, masks: torch.Tensor | None = None) -> torch.Tensor:
        masks = self.routing_masks() if masks is None else masks
        h = torch.stack(tuple(hs), dim=1)
        return (
            torch.einsum("bluw,lgu->blgw", h, masks)
            / masks.sum(-1).clamp_min(1.0)[None, :, :, None]
        )

    # ------------------------------------------------------------- VSA algebra

    def _cleanup(self, values: torch.Tensor) -> torch.Tensor:
        """cosine-based cleanup"""

        values = F.normalize(values, dim=-1)
        bank = F.normalize(self.codebook, dim=-1)
        logits = torch.einsum("bgd,gcd->bgc", values, bank) / self.temp
        return logits + self.filler_mask[None]

    def _bind(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """MAP-bind unit vectors without shrinking their grade by sqrt(D)."""

        return left * right * math.sqrt(self.D)

    def _bind_for_write(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left * right * self.pair_write_scale

    def codes_to_vsa(self, codes_or_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = code_probabilities(self, codes_or_probs)
        bank = F.normalize(self.codebook, dim=-1)
        filler = torch.einsum("bgc,gcd->bgd", probs, bank)
        return filler, self.roles[None] * filler

    def program_feature_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Graded [unary ; pair] program from a filler distribution
        """

        if not isinstance(probs, torch.Tensor) or probs.ndim != 3:
            raise ValueError("expected soft atom mass with shape [batch, groups, fillers]")
        _, unary = self.codes_to_vsa(probs)
        active = probs.sum(-1) > 0
        unary_den = active.sum(1).clamp_min(1).to(unary.dtype).sqrt()[:, None]
        unary_bundle = unary.sum(1) / unary_den
        if len(self.edges):
            masks = self.routing_masks()
            edge_activity = self.routing_edge_activity(self.routing_pair_masks(masks))
            ei, ej = self.eidx[:, 0], self.eidx[:, 1]
            pair_active = active[:, ei] & active[:, ej]
            pair_terms = self._bind(unary[:, ei], unary[:, ej])
            pair_active = pair_active.to(pair_terms.dtype) * edge_activity[None]
            pair_terms = pair_terms * pair_active[..., None]
            pair_bundle = pair_terms.sum(1) / pair_active.sum(1).clamp_min(1).sqrt()[:, None]
        else:
            pair_bundle = torch.zeros_like(unary_bundle)
        return torch.cat((unary_bundle, pair_bundle), dim=-1)

    def program_feature_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self.program_feature_from_probs(code_probabilities(self, codes, hard_only=True))

    # ---------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor, pair_writes: bool = True) -> dict[str, torch.Tensor]:
        masks = self.routing_masks()
        pair_masks = self.routing_pair_masks(masks)
        edge_activity = self.routing_edge_activity(pair_masks)
        hs = self._base(x)
        fragments = self._fragments(hs, masks)
        values = F.normalize(self.to_value(fragments.mean(1)), dim=-1)
        unary = self.roles[None] * values

        # decode each unary/pair message once, then route it to every layer
        unary_messages = self.unary_write(unary)
        pair_messages = None
        if pair_writes and len(self.edges):
            ei, ej = self.eidx[:, 0], self.eidx[:, 1]
            pair_messages = self.pair_write(self._bind_for_write(unary[:, ei], unary[:, ej]))

        updated = []
        for layer, h in enumerate(hs):
            unary_write = (
                torch.einsum("bgw,gu->buw", unary_messages, masks[layer])
                / masks[layer].sum(0).clamp_min(1.0)[None, :, None]
            )
            z = h + 0.20 * unary_write
            if pair_messages is not None:
                z = z + 0.20 * (
                    torch.einsum("bew,eu->buw", pair_messages, pair_masks[layer])
                    / pair_masks[layer].sum(0).clamp_min(1.0)[None, :, None]
                )
            updated.append(self.norms[layer](z))

        values2 = F.normalize(self.to_value(self._fragments(updated, masks).mean(1)), dim=-1)

        # bundle -> unbind -> normalize -> cleanup -> rebind
        bundle = (self.roles[None] * values2).sum(1) / math.sqrt(self.G)
        retrieved = bundle[:, None] * self.roles[None] * math.sqrt(self.G)
        denoised = F.normalize(retrieved, dim=-1)
        logits = self._cleanup(denoised)
        probs = logits.softmax(-1)
        one_hot = F.one_hot(probs.argmax(-1), self.C).to(probs.dtype)
        straight_through = one_hot + probs - probs.detach()
        quantized = torch.einsum(
            "bgc,gcd->bgd", straight_through, F.normalize(self.codebook, dim=-1))
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
            "logits": logits, "probs": probs, "codes": logits.argmax(-1),
            "den": denoised, "quantized": quantized, "quantized_st": quantized_st,
            "program": program, "recon_embed": self.embed_decoder(program),
            "routing_masks": masks, "routing_edge_activity": edge_activity,
        }

    @torch.inference_mode()
    def encode_codes(self, x: torch.Tensor, pair_writes: bool = True) -> torch.Tensor:
        return self.forward(x, pair_writes=pair_writes)["codes"]

    # ----------------------------------------------------------- VQ objectives

    def vq_losses(self, out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """commitment terms plus a per-group uniform prior over fillers"""

        vq = ((out["quantized"] - out["den"].detach()) ** 2).mean()
        commit = ((out["den"] - out["quantized"].detach()) ** 2).mean()
        p = out["probs"].mean(0)                                  # (G, C)
        live = self.filler_live
        target = live / live.sum(-1, keepdim=True)
        kl = (p.clamp_min(1e-8) * (p.clamp_min(1e-8).log() - target.clamp_min(1e-8).log()))
        kl = (kl * live).sum(-1).mean()
        entropy = -(out["probs"] * out["probs"].clamp_min(1e-8).log()).sum(-1).mean()
        return vq, commit, kl, entropy

    # -------------------------------------------------------------- diagnostics

    def scaffold_diagnostics(self) -> dict[str, object]:
        masks = self.routing_masks().detach()
        intersections = architecture_intersection_tensor(masks)
        adjacency = intersections.sum(-1) > 0
        adjacency.fill_diagonal_(False)
        record = {
            "frozen": True,
            "alphabet": list(self.alphabet),
            "hard_masks": masks.int().cpu().tolist(),
            "capacity_per_layer_group": masks.sum(-1).int().cpu().tolist(),
            "unit_load_mean": float(masks.sum(1).mean()),
            "unit_load_max": int(masks.sum(1).max()),
            "edge_count": int(adjacency.triu(1).sum()),
        }
        if self.routed:
            densities, fan_ins = [], []
            for layer in range(self.L - 1):
                connectivity = self.transition_connectivity(masks, layer).detach()
                densities.append(float(connectivity.mean()))
                fan_ins.append(float(connectivity.sum(0).mean()))
            record["population_routed_transitions"] = True
            record["transition_unit_density"] = densities
            record["mean_transition_unit_fan_in"] = fan_ins
        return record


# ============================================================ the model

class FactorSJ(nn.Module):
    """encoder and the sparse hypervector core"""

    def __init__(self, cfg: SJConfig, factor_cardinalities: tuple[int, ...]):
        super().__init__()
        self.cfg = cfg
        self.factor_cardinalities = tuple(factor_cardinalities)
        if len(self.factor_cardinalities) != cfg.n_factors:
            raise ValueError("one sufficiency head per factor is required")

        torch.manual_seed(cfg.seed)
        masks = make_rigid_scaffold(cfg, seed=cfg.seed)
        self.encoder = nn.Sequential(nn.Linear(cfg.input_dim, cfg.frame_embed), nn.GELU())
        alphabet = (cfg.cardinalities if cfg.alphabet == "per_group"
                    else tuple([cfg.max_card] * cfg.groups))
        self.sj = SparseJointCore(cfg.frame_embed, cfg, masks, cfg.seed, alphabet)
        self.heads = nn.ModuleList(
            nn.Linear(self.sj.program_dim, c) for c in self.factor_cardinalities
        )
        # Which intervention family owns which group.  Learned; never supplied.
        self.ownership_logits = nn.Parameter(torch.zeros(cfg.n_factors, cfg.groups))

    def forward(self, x: torch.Tensor) -> dict:
        out = self.sj(self.encoder(x), pair_writes=True)
        out["factor_logits"] = [head(out["program"]) for head in self.heads]
        return out

    def ownership(self) -> torch.Tensor:
        """group assignment through sinkhorn balancing [1]
        ----
        [1] https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem
        """

        cfg = self.cfg
        log_a = self.ownership_logits / cfg.sinkhorn_temp
        if not cfg.one_to_one:
            return log_a.softmax(-1)
        col_target = float(log_a.shape[0]) / log_a.shape[1]
        for _ in range(cfg.sinkhorn_iters):
            log_a = log_a - log_a.logsumexp(-1, keepdim=True)
            log_a = log_a - log_a.logsumexp(0, keepdim=True) + math.log(col_target)
        return (log_a - log_a.logsumexp(-1, keepdim=True)).exp()

    def hard_ownership(self) -> list[int]:
        """assignment permutation"""

        from scipy.optimize import linear_sum_assignment

        with torch.no_grad():
            assign = self.ownership().cpu().numpy()
        rows, cols = linear_sum_assignment(-assign)
        owner = [-1] * assign.shape[0]
        for family, group in zip(rows, cols):
            owner[family] = int(group)
        return owner


def build_model(cfg: SJConfig, factor_cardinalities: tuple[int, ...]) -> FactorSJ: return FactorSJ(cfg, factor_cardinalities)


# ============================================================ the objective

def disagreement(p_base: torch.Tensor, p_changed: torch.Tensor) -> torch.Tensor:
    """Differentiable Pr[code changes] per group: 1 - sum_c p_a(c) p_b(c)"""

    return 1.0 - (p_base * p_changed).sum(-1)


def total_variation(p_base: torch.Tensor, p_changed: torch.Tensor) -> torch.Tensor:
    """total variation between categorical posteriors"""

    return 0.5 * (p_base - p_changed).abs().sum(-1)


def acquisition_losses(
    model: FactorSJ,
    out: dict,
    z: torch.Tensor,
    pair_out: list[tuple[dict, dict]] | None,
) -> dict[str, torch.Tensor]:
    """This **never** consumes a factor value assigned to a group or a factor-to-group assignment.
    """

    cfg = model.cfg
    losses: dict[str, torch.Tensor] = {}

    losses["sufficiency"] = cfg.w_sufficiency * sum(
        F.cross_entropy(logits, z[:, i]) for i, logits in enumerate(out["factor_logits"])
    )

    vq, commit, kl, _ = model.sj.vq_losses(out)
    losses["vq"] = cfg.w_vq * (vq + 0.25 * commit)

    assign = model.ownership()                                    # (K, G)
    if cfg.alphabet == "shared":
        declared = cfg.support_cardinalities or model.factor_cardinalities
        cards = torch.tensor(declared, dtype=assign.dtype, device=assign.device)
        mass = assign.sum(0)                                      # (G,)
        target = ((assign * cards[:, None]).sum(0) / mass.clamp_min(1e-6)).clamp_min(1.0)
        p_group = out["probs"].mean(0)                            # (G, C)
        entropy = -(p_group * p_group.clamp_min(1e-8).log()).sum(-1)
        weight = mass / mass.sum().clamp_min(1e-6) * len(mass)
        losses["support"] = cfg.w_support * (
            weight * (entropy.exp() - target) ** 2).mean()
    else:
        losses["support"] = cfg.w_support * kl

    if pair_out:
        response_terms, invariance_terms = [], []
        for family, (base, changed) in enumerate(pair_out):
            delta = disagreement(base["probs"], changed["probs"]).mean(0)   # (G,)
            owned = (assign[family] * delta).sum()
            response_terms.append(F.relu(cfg.response_target - owned))
            weight = 1.0 - assign[family]
            invariance_terms.append((weight * delta).sum() / weight.sum().clamp_min(1e-6))
        losses["response"] = cfg.w_response * torch.stack(response_terms).mean()
        losses["invariance"] = cfg.w_invariance * torch.stack(invariance_terms).mean()
    else:
        zero = out["program"].new_zeros(())
        losses["response"], losses["invariance"] = zero, zero
    return losses


def oracle_losses(model: FactorSJ, out: dict, z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Group ``i`` is told to encode factor ``i``, so the filler permutation is the
    identity by construction.  This is an upper bound on what the architecture
    can represent and must never be reported as an acquisition result."""

    cfg = model.cfg
    losses = {
        "oracle": cfg.w_oracle * sum(
            F.cross_entropy(out["logits"][:, i, :], z[:, i]) for i in range(z.shape[1])),
        "sufficiency": cfg.w_sufficiency * sum(
            F.cross_entropy(logits, z[:, i])
            for i, logits in enumerate(out["factor_logits"])),
    }
    vq, commit, kl, _ = model.sj.vq_losses(out)
    losses["vq"] = cfg.w_vq * (vq + 0.25 * commit)
    losses["support"] = cfg.w_support * kl
    return losses


# ============================================================ training

def train(
    model: FactorSJ,
    x: np.ndarray,
    z: np.ndarray,
    *,
    mode: str,
    device: torch.device,
    pairs: dict | None = None,
) -> FactorSJ:

    cfg = model.cfg
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    xt = torch.from_numpy(x).to(device)
    zt = torch.from_numpy(z).long().to(device)

    pair_tensors = None
    if pairs is not None:
        pair_tensors = [
            (torch.from_numpy(pairs["base"]).to(device), torch.from_numpy(c).to(device))
            for c in pairs["changed"]
        ]

    n = len(xt)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 7717)
    for _ in range(cfg.epochs):
        order = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, cfg.batch):
            index = order[start:start + cfg.batch]
            out = model(xt[index])
            if mode == "oracle":
                losses = oracle_losses(model, out, zt[index])
            elif mode == "acquisition":
                pair_out = None
                if pair_tensors is not None:
                    m = len(pair_tensors[0][0])
                    sub = torch.randint(0, m, (min(cfg.batch, m),),
                                        generator=generator).to(device)
                    pair_out = [(model(base[sub]), model(changed[sub]))
                                for base, changed in pair_tensors]
                losses = acquisition_losses(model, out, zt[index], pair_out)
            else:
                raise ValueError(f"unknown training mode: {mode}")
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model.eval()


@torch.no_grad()
def encode(model: FactorSJ, x: np.ndarray, device: torch.device,
           batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(x), batch):
        chunk = torch.from_numpy(x[start:start + batch]).to(device)
        out.append(model(chunk)["codes"].cpu().numpy())
    return np.concatenate(out, axis=0)


# ============================================================ diagnostics

@torch.no_grad()
def vsa_cleanup_accuracy(model: FactorSJ, codes: np.ndarray,
                         device: torch.device) -> float:
    """bundle the G fillers, unbind by role, clean up, and check recovery"""

    core = model.sj
    c = torch.from_numpy(codes[: min(len(codes), 4096)]).long().to(device)
    _, unary = core.codes_to_vsa(c)
    bundle = unary.sum(1) / math.sqrt(core.G)
    retrieved = bundle[:, None] * core.roles[None] * math.sqrt(core.G)
    recovered = core._cleanup(F.normalize(retrieved, dim=-1)).argmax(-1)
    return float((recovered == c).float().mean())


def locality_diagnostics(model, x, device, rows=128):
    """Actual-path sensitivity"""
    return full_path_locality(model, x, device, rows)


def off_subgraph_influence(model, x, device, rows=128):
    """Actual-path outside gradient fraction"""
    return locality_diagnostics(model, x, device, rows)["outside_fraction"]


def legacy_off_subgraph_influence(model: FactorSJ, x: np.ndarray, device: torch.device,
                           rows: int = 128) -> float:
    """Mean ``|d logit(chosen filler of g) / d h|`` off g's units over on g's units.

    0 means each group is computed only from units it owns such that 1 means the scaffold
    carries no computational meaning.  
    """

    model.eval()
    core = model.sj
    masks = core.routing_masks().detach()              # (L, G, U)
    owned = (masks.sum(0) > 0).float()                 # (G, U)
    chunk = torch.from_numpy(x[:rows]).to(device)
    h0 = core.input(model.encoder(chunk)).view(len(chunk), core.U, core.W)

    ratios = []
    for group in range(core.G):
        fragments = core._fragments(core.stack_from_h0(h0, masks), masks)
        values = F.normalize(core.to_value(fragments.mean(1)), dim=-1)
        chosen = core._cleanup(values)[:, group].max(-1).values.sum()
        grad = torch.autograd.grad(chosen, h0, retain_graph=True)[0].abs().sum(-1)
        influence = grad.mean(0)                       # (U,)
        on, off = owned[group], 1.0 - owned[group]
        on_mean = (influence * on).sum() / on.sum().clamp_min(1.0)
        off_mean = (influence * off).sum() / off.sum().clamp_min(1.0)
        ratios.append(float(off_mean / on_mean.clamp_min(1e-12)))
    return float(np.mean(ratios))
