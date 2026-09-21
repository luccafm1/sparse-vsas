
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
filler from its own alphabet by a vector-symbolic cleanup, and the exported program is the tuple

    (unary , pair).

The current implementation uses Fourier Holographic Reduced Representation (FHRR) phasors for roles and fillers. 

For glue.py, the following are strictly necessary:

    G, C, grammar_signature, routing_masks(), program_feature_from_codes(...)

where hard code -1 means an absent/unresolved atom. 

Usage:
    cfg = SJConfig(input_dim=256, cardinalities=(6,) * 8)
    model = SJClassifier(cfg, num_classes=40)
    details = model.forward_details(x)

#######
Obs.: 'SJ' was the working name of 'sparse-joint networks', we havent really settled on a name so that's what we'll call it
for the time being.
"""

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


MASKED_FILLER = -1.0e4
TaskType = Literal["classification", "regression"]

SJ_GRAMMAR_VERSION = "sparse-joint-fhrr-v1"
SJ_BINDING = "complex-hadamard"
SJ_PROGRAM_GRADES = (1, 2)
SJ_JOINT_RULE = "literal-neural-intersection"
SJ_MISSING_ATOM = "absent-zero-term"


def sj_grammar_signature() -> tuple:
    """public compatibility signature for native SJ program composition"""

    return (
        SJ_GRAMMAR_VERSION,
        SJ_BINDING,
        SJ_PROGRAM_GRADES,
        SJ_JOINT_RULE,
        SJ_MISSING_ATOM,
    )


def code_probabilities(core, value, *, hard_only=False):
    """Validate hard codes or nonnegative atom mass; zero mass means absence."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("SJ codes/probabilities must be a torch.Tensor")
    if value.device != core.codebook.device:
        raise ValueError("SJ codes and core must be on the same device")

    if value.ndim == 2:
        if value.shape[1] != core.G:
            raise ValueError("expected codes with shape [batch, groups]")
        if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("hard SJ codes must have a signed integer dtype")
        codes = value.long()
        if bool(((codes < -1) | (codes >= core.C)).any()):
            raise ValueError("SJ code is outside the live filler vocabulary (only -1 denotes absence)")
        valid = codes >= 0
        probs = F.one_hot(codes.clamp_min(0), core.C).to(core.filler_mask.dtype)
        probs = probs * valid[..., None]
    else:
        if hard_only or value.ndim != 3 or value.shape[1:] != (core.G, core.C):
            raise ValueError("expected probabilities with shape [batch, groups, fillers]")
        if not value.is_floating_point():
            raise TypeError("soft SJ atom mass must be floating point")
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError("soft SJ atom mass must be finite and nonnegative")
        probs = value.to(core.filler_mask.dtype)

    if not bool(torch.isfinite(probs).all()) or not bool(torch.isfinite(probs.sum(-1)).all()):
        raise ValueError("soft SJ atom mass overflows the core dtype")
    if bool((probs * (1.0 - core.filler_live)[None]).any()):
        raise ValueError("SJ code/probability uses a disabled group filler")
    return probs


# ============================================================ configuration

@dataclass(frozen=True)
class SJConfig:
    """Architecture and optimization configuration."""

    # ---- world-facing shape
    input_dim: int
    cardinalities: tuple[int, ...] = (8, 8, 8, 8, 8, 8, 8, 8)

    # ---- architecture
    units: int = 128
    layers: int = 3
    topk: int = 4
    unit_width: int = 8
    hv_dim: int = 256
    frame_embed: int = 64
    cleanup_temp: float = 0.02

    # ---- optimization
    lr: float = 2.0e-3
    batch: int = 256
    epochs: int = 40
    seed: int = 0
    weight_decay: float = 1.0e-5
    grad_clip: float = 5.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not self.cardinalities or any(c <= 0 for c in self.cardinalities):
            raise ValueError("cardinalities must contain positive integers")
        if self.units <= 0 or self.layers <= 0 or self.unit_width <= 0 or self.hv_dim <= 0:
            raise ValueError("architecture dimensions must be positive")
        if not 0 < self.topk <= self.units:
            raise ValueError("topk must satisfy 0 < topk <= units")
        if self.cleanup_temp <= 0:
            raise ValueError("cleanup_temp must be positive")

    @property
    def groups(self) -> int:
        return len(self.cardinalities)

    @property
    def max_card(self) -> int:
        return max(self.cardinalities)


# ============================================================ VSA primitives

def physical_edges(masks: torch.Tensor) -> tuple[tuple[int, int], ...]:
    """Group pairs that share at least one unit in some layer."""

    groups = masks.shape[1]
    out = []
    for i in range(groups):
        for j in range(i + 1, groups):
            if bool((masks[:, i] * masks[:, j]).sum() > 0):
                out.append((i, j))
    return tuple(out)


def architecture_intersection_tensor(masks: torch.Tensor) -> torch.Tensor:
    """Per-layer literal intersection counts for every pair of groups."""

    groups = masks.shape[1]
    out = torch.zeros(groups, groups, masks.shape[0], device=masks.device)
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


def make_rigid_scaffold(cfg: SJConfig, seed: int | None = None) -> torch.Tensor:
    """Build the fixed sparse, connected, structurally distinctive layered scaffold."""

    G, L, U, K = cfg.groups, cfg.layers, cfg.units, cfg.topk
    seed = cfg.seed if seed is None else seed

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

    # Shared units define physical group overlaps.
    for edge_idx, (i, j) in enumerate(edges):
        for layer, count in enumerate(_edge_code(edge_idx, L)):
            for _ in range(count):
                if next_unit[layer] >= U:
                    raise ValueError("units too small for requested sparse scaffold")
                u = next_unit[layer]
                next_unit[layer] += 1
                masks[layer, i, u] = 1.0
                masks[layer, j, u] = 1.0

    # Add private units until every group has at least topk members.
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


# ============================================================ the core

class SparseJointCore(nn.Module):
    """FHRR-native distributed sparse symbolic bottleneck."""

    def __init__(self, in_dim: int, cfg: SJConfig, masks: torch.Tensor):
        super().__init__()
        self.cfg = cfg
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

        self.input = nn.Sequential(nn.Linear(in_dim, U * W), nn.GELU())
        self.transitions = nn.ModuleList([
            nn.Sequential(nn.Linear(U * W, U * W), nn.GELU()) for _ in range(L - 1)
        ])
        self.to_value = nn.Linear(W, D, bias=False)
        self.unary_write = nn.Linear(2 * D, W, bias=False)
        self.pair_write = nn.Linear(2 * D, W, bias=False)
        self.norms = nn.ModuleList([nn.LayerNorm(W) for _ in range(L)])

        # Roles and fillers define a fixed FHRR coordinate system.
        gen = torch.Generator().manual_seed(cfg.seed + 991)
        role_phase = torch.rand(G, D, generator=gen) * 2 * math.pi
        filler_phase = torch.rand(G, C, D, generator=gen) * 2 * math.pi
        self.register_buffer("roles", torch.exp(1j * role_phase))
        self.register_buffer("codebook", filler_phase)

        filler_mask = torch.zeros(G, C)
        for group, cardinality in enumerate(cfg.cardinalities):
            filler_mask[group, cardinality:] = MASKED_FILLER
        self.register_buffer("filler_mask", filler_mask)

    # -------------------------------------------------------------- properties

    @property
    def filler_live(self) -> torch.Tensor:
        return (self.filler_mask == 0).float()

    @property
    def program_dim(self) -> int:
        # Real and imaginary parts of unary and pair bundles.
        return 4 * self.D

    @property
    def grammar_signature(self) -> tuple:
        return sj_grammar_signature()

    @property
    def scaffold_is_frozen(self) -> bool:
        # glue.py checks this before fingerprinting a learned scaffold.
        return True

    def routing_masks(self) -> torch.Tensor:
        return self.masks

    def routing_pair_masks(self, masks: torch.Tensor | None = None) -> torch.Tensor:
        masks = self.masks if masks is None else masks
        if not len(self.edges):
            return masks.new_zeros(self.L, 0, self.U)
        ei, ej = self.eidx[:, 0], self.eidx[:, 1]
        return masks[:, ei] * masks[:, ej]

    # ------------------------------------------------------------- VSA algebra

    def filler_bank(self) -> torch.Tensor:
        return torch.exp(1j * self.codebook)

    @staticmethod
    def _as_real(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x.real, x.imag), dim=-1)

    @staticmethod
    def _normalize_hv(x: torch.Tensor) -> torch.Tensor:
        return x / x.abs().clamp_min(1.0e-8)

    @staticmethod
    def _bind(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left * right

    def _to_hv(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.exp(1j * raw)

    def program_feature_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Graded [unary ; pair] program from filler mass; zero mass means absence."""

        probs = code_probabilities(self, probs)
        bank = self.filler_bank()
        filler = torch.einsum("bgc,gcd->bgd", probs.to(bank.dtype), bank)
        active = probs.sum(-1) > 0

        unary = self.roles[None] * filler
        unary_den = active.sum(1).clamp_min(1).to(unary.real.dtype).sqrt()[:, None]
        unary_bundle = unary.sum(1) / unary_den

        if len(self.edges):
            ei, ej = self.eidx[:, 0], self.eidx[:, 1]
            pair_active = active[:, ei] & active[:, ej]
            pair_terms = self._bind(unary[:, ei], unary[:, ej])
            pair_terms = pair_terms * pair_active[..., None]
            pair_den = pair_active.sum(1).clamp_min(1).to(pair_terms.real.dtype).sqrt()[:, None]
            pair_bundle = pair_terms.sum(1) / pair_den
        else:
            pair_bundle = torch.zeros_like(unary_bundle)

        return self._as_real(torch.cat((unary_bundle, pair_bundle), dim=-1))

    def program_feature_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Native SJ program from hard codes; -1 denotes an absent atom."""

        return self.program_feature_from_probs(code_probabilities(self, codes, hard_only=True))

    # ---------------------------------------------------------------- routing

    def transition_connectivity(self, layer: int) -> torch.Tensor:
        """A unit-to-unit edge exists iff some group holds both endpoints."""

        overlap = self.masks[layer].T @ self.masks[layer + 1]
        return (overlap > 0).to(overlap.dtype)

    def _base(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.input(x).view(len(x), self.U, self.W)
        hs = [h]
        for layer, transition in enumerate(self.transitions):
            connectivity = self.transition_connectivity(layer)
            block = connectivity.T.repeat_interleave(self.W, 0).repeat_interleave(self.W, 1)
            fan_in = connectivity.sum(0).clamp_min(1.0)
            scale = (self.U / fan_in).sqrt().repeat_interleave(self.W)[:, None]
            linear = transition[0]
            h = F.linear(h.flatten(1), linear.weight * block * scale, linear.bias)
            h = transition[1](h).view(len(x), self.U, self.W)
            hs.append(h)
        return hs

    def _fragments(self, hs: list[torch.Tensor]) -> torch.Tensor:
        h = torch.stack(hs, dim=1)
        return (
            torch.einsum("bluw,lgu->blgw", h, self.masks)
            / self.masks.sum(-1).clamp_min(1.0)[None, :, :, None]
        )

    def _cleanup(self, values: torch.Tensor) -> torch.Tensor:
        """FHRR cleanup against the fixed filler bank."""

        values = self._normalize_hv(values)
        bank = self.filler_bank()
        similarity = torch.real(torch.einsum("bgd,gcd->bgc", values, torch.conj(bank)))
        return similarity / self.D / self.cfg.cleanup_temp + self.filler_mask[None]

    # ---------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hs = self._base(x)
        fragments = self._fragments(hs)

        # First symbolic readout is written back into the neural populations.
        values = self._to_hv(self.to_value(fragments.mean(1)))
        unary = self.roles[None] * values
        unary_messages = self.unary_write(self._as_real(unary))

        pair_messages = None
        if len(self.edges):
            ei, ej = self.eidx[:, 0], self.eidx[:, 1]
            pair_messages = self.pair_write(
                self._as_real(self._bind(unary[:, ei], unary[:, ej]))
            )

        updated = []
        for layer, h in enumerate(hs):
            unary_write = (
                torch.einsum("bgw,gu->buw", unary_messages, self.masks[layer])
                / self.masks[layer].sum(0).clamp_min(1.0)[None, :, None]
            )
            z = h + 0.20 * unary_write
            if pair_messages is not None:
                z = z + 0.20 * (
                    torch.einsum("bew,eu->buw", pair_messages, self.pair_masks[layer])
                    / self.pair_masks[layer].sum(0).clamp_min(1.0)[None, :, None]
                )
            updated.append(self.norms[layer](z))

        # bundle -> unbind -> cleanup -> straight-through hard filler -> rebind
        values2 = self._to_hv(self.to_value(self._fragments(updated).mean(1)))
        bundle = (self.roles[None] * values2).sum(1) / math.sqrt(self.G)
        retrieved = bundle[:, None] * torch.conj(self.roles[None])
        denoised = self._normalize_hv(retrieved)
        logits = self._cleanup(denoised)
        probs = logits.softmax(-1)
        codes = logits.argmax(-1)

        one_hot = F.one_hot(codes, self.C).to(self.codebook.dtype)
        bank = self.filler_bank()
        hard_quantized = torch.einsum("bgc,gcd->bgd", one_hot.to(bank.dtype), bank)
        quantized_st = denoised + (hard_quantized - denoised).detach()

        unary_q = self.roles[None] * quantized_st
        unary_bundle = unary_q.sum(1) / math.sqrt(self.G)

        if len(self.edges):
            ei, ej = self.eidx[:, 0], self.eidx[:, 1]
            pair_terms = self._bind(unary_q[:, ei], unary_q[:, ej])
            pair_bundle = pair_terms.sum(1) / math.sqrt(len(self.edges))
        else:
            pair_bundle = torch.zeros_like(unary_bundle)

        program = self._as_real(torch.cat((unary_bundle, pair_bundle), dim=-1))
        return {
            "logits": logits,
            "probs": probs,
            "codes": codes,
            "denoised": denoised,
            "quantized": hard_quantized,
            "program": program,
        }

    @torch.inference_mode()
    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["codes"]

    # -------------------------------------------------------------- diagnostics

    def scaffold_diagnostics(self) -> dict[str, Any]:
        masks = self.routing_masks().detach()
        intersections = architecture_intersection_tensor(masks)
        adjacency = intersections.sum(-1) > 0
        adjacency.fill_diagonal_(False)
        return {
            "frozen": True,
            "hard_masks": masks.int().cpu().tolist(),
            "groups": self.G,
            "edges": self.edges,
            "edge_count": int(adjacency.triu(1).sum()),
            "capacity_per_layer_group": masks.sum(-1).int().cpu().tolist(),
            "unit_load_mean": float(masks.sum(1).mean()),
            "unit_load_max": int(masks.sum(1).max()),
        }


# ============================================================ the model

class SparseJointNetwork(nn.Module):
    """Encoder, sparse-joint core, and one task head."""

    def __init__(
        self,
        cfg: SJConfig,
        *,
        task: TaskType,
        num_outputs: int,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        if num_outputs <= 0:
            raise ValueError("num_outputs must be positive")

        self.cfg = cfg
        self.task = task
        self.num_outputs = int(num_outputs)
        self._default_encoder = encoder is None

        torch.manual_seed(cfg.seed)
        self.encoder = encoder or nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(cfg.input_dim, cfg.frame_embed),
            nn.GELU(),
        )
        masks = make_rigid_scaffold(cfg, seed=cfg.seed)
        self.sj = SparseJointCore(cfg.frame_embed, cfg, masks)
        self.head = nn.Linear(self.sj.program_dim, self.num_outputs)

    def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._default_encoder and not x.is_floating_point():
            x = x.float()
        encoded = self.encoder(x)
        if encoded.ndim != 2 or encoded.shape[-1] != self.cfg.frame_embed:
            raise ValueError(
                f"encoder must return [batch, {self.cfg.frame_embed}], got {tuple(encoded.shape)}"
            )
        return encoded

    def forward_details(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self._encode_input(x)
        details = self.sj(encoded)
        details["encoded"] = encoded
        details["prediction"] = self.head(details["program"])
        return details

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_details(x)["prediction"]

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self(x)
        return prediction.argmax(-1) if self.task == "classification" else prediction

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        if self.task != "classification":
            raise TypeError("predict_proba is only available for classification")
        return self(x).softmax(-1)

    @torch.inference_mode()
    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_details(x)["codes"]

    @torch.inference_mode()
    def program(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_details(x)["program"]


class SJClassifier(SparseJointNetwork):
    def __init__(self, cfg: SJConfig, num_classes: int, *, encoder: nn.Module | None = None):
        super().__init__(cfg, task="classification", num_outputs=num_classes, encoder=encoder)


class SJRegressor(SparseJointNetwork):
    def __init__(self, cfg: SJConfig, num_targets: int = 1, *, encoder: nn.Module | None = None):
        super().__init__(cfg, task="regression", num_outputs=num_targets, encoder=encoder)


def build_model(
    cfg: SJConfig,
    *,
    task: TaskType,
    num_outputs: int,
    encoder: nn.Module | None = None,
) -> SparseJointNetwork:
    return SparseJointNetwork(cfg, task=task, num_outputs=num_outputs, encoder=encoder)


# ============================================================ the objective

def _task_loss(
    model: SparseJointNetwork,
    prediction: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module | None = None,
) -> torch.Tensor:
    if model.task == "classification":
        targets = targets.long().squeeze(-1) if targets.ndim > 1 else targets.long()
        return criterion(prediction, targets) if criterion else F.cross_entropy(prediction, targets)

    targets = targets.to(prediction.dtype)
    if model.num_outputs == 1 and targets.ndim == 1:
        targets = targets[:, None]
    return criterion(prediction, targets) if criterion else F.mse_loss(prediction, targets)


def supervised_losses(
    model: SparseJointNetwork,
    details: dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    criterion: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Current SJ baseline objective: task loss only."""

    return {"task": _task_loss(model, details["prediction"], targets, criterion)}


# ============================================================ training / evaluation

def _as_dataset(data: Dataset | tuple[Any, Any]) -> Dataset:
    if isinstance(data, Dataset):
        return data
    if not isinstance(data, (tuple, list)) or len(data) != 2:
        raise TypeError("data must be a Dataset or (x, y) pair")
    x, y = data
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    y = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
    return TensorDataset(x, y)


def _unpack(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("each batch must contain inputs and targets")


def _device_of(model: nn.Module, device: torch.device | str | None) -> torch.device:
    return torch.device(device) if device is not None else next(model.parameters()).device


@torch.no_grad()
def evaluate(
    model: SparseJointNetwork,
    data: Dataset | tuple[Any, Any] | DataLoader,
    *,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    device = _device_of(model, device)
    loader = data if isinstance(data, DataLoader) else DataLoader(
        _as_dataset(data), batch_size=batch_size or model.cfg.batch, shuffle=False
    )

    was_training = model.training
    model.to(device).eval()
    total = 0
    loss_sum = 0.0
    correct = 0
    abs_sum = 0.0
    sq_sum = 0.0

    for batch in loader:
        x, y = _unpack(batch)
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        loss = _task_loss(model, prediction, y, criterion)
        n = len(x)
        total += n
        loss_sum += float(loss) * n

        if model.task == "classification":
            target = y.long().squeeze(-1) if y.ndim > 1 else y.long()
            correct += int((prediction.argmax(-1) == target).sum())
        else:
            target = y.to(prediction.dtype)
            if model.num_outputs == 1 and target.ndim == 1:
                target = target[:, None]
            error = prediction - target
            abs_sum += float(error.abs().sum())
            sq_sum += float(error.pow(2).sum())

    if was_training:
        model.train()
    if total == 0:
        raise ValueError("cannot evaluate an empty dataset")

    metrics = {"loss": loss_sum / total}
    if model.task == "classification":
        metrics["accuracy"] = correct / total
    else:
        count = total * model.num_outputs
        mse = sq_sum / count
        metrics.update({"mse": mse, "rmse": math.sqrt(mse), "mae": abs_sum / count})
    return metrics


def fit(
    model: SparseJointNetwork,
    train_data: Dataset | tuple[Any, Any] | DataLoader,
    *,
    val_data: Dataset | tuple[Any, Any] | DataLoader | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    device: torch.device | str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    criterion: nn.Module | None = None,
    restore_best: bool = True,
) -> list[dict[str, float]]:
    """Train the SJ network; if validation data is supplied, restore its best checkpoint."""

    cfg = model.cfg
    epochs = cfg.epochs if epochs is None else int(epochs)
    batch_size = cfg.batch if batch_size is None else int(batch_size)
    device = _device_of(model, device)
    model.to(device)

    loader = train_data if isinstance(train_data, DataLoader) else DataLoader(
        _as_dataset(train_data), batch_size=batch_size, shuffle=True
    )
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr if lr is None else lr,
            weight_decay=cfg.weight_decay,
        )

    history: list[dict[str, float]] = []
    best_state = None
    best_score = -math.inf if model.task == "classification" else math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        seen = 0
        loss_sum = 0.0

        for batch in loader:
            x, y = _unpack(batch)
            x, y = x.to(device), y.to(device)
            details = model.forward_details(x)
            loss = supervised_losses(model, details, y, criterion=criterion)["task"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            n = len(x)
            seen += n
            loss_sum += float(loss.detach()) * n

        row = {"epoch": float(epoch), "train_loss": loss_sum / seen}
        if val_data is not None:
            metrics = evaluate(
                model, val_data, batch_size=batch_size, device=device, criterion=criterion
            )
            row.update({f"val_{k}": v for k, v in metrics.items()})

            score = metrics["accuracy"] if model.task == "classification" else metrics["loss"]
            improved = score > best_score if model.task == "classification" else score < best_score
            if improved:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

        history.append(row)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history


def test(
    model: SparseJointNetwork,
    test_data: Dataset | tuple[Any, Any] | DataLoader,
    **kwargs: Any,
) -> dict[str, float]:
    return evaluate(model, test_data, **kwargs)
