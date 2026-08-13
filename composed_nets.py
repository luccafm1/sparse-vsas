from __future__ import annotations

"""Composed sparse hypervector networks"""

from dataclasses import asdict, dataclass, fields
import itertools
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import SJConfig, SparseVQCore, architecture_intersection_tensor, physical_edges


# Composed specialist networks

class VideoEncoder(nn.Module):
    def __init__(self, cfg: SJConfig):
        super().__init__()
        self.cfg = cfg
        self.frame = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2), nn.GELU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.GELU(),
            nn.Conv2d(48, 80, 3, 2, 1), nn.GELU(),
            nn.Conv2d(80, 112, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gru = nn.GRU(112, cfg.frame_embed, batch_first=True)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        f = self.frame(video.reshape(B * T, C, H, W)).flatten(1).view(B, T, -1)
        _, h = self.gru(f)
        return h[-1]


class SmallFrameDecoder(nn.Module):
    def __init__(self, program_dim: int, n_out: int = 4, size: int = 32):
        super().__init__()
        self.n_out, self.size = n_out, size
        self.net = nn.Sequential(
            nn.Linear(program_dim, 512), nn.GELU(),
            nn.Linear(512, n_out * 3 * size * size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z).view(len(z), self.n_out, 3, self.size, self.size))


class StateSequenceEncoder(nn.Module):
    def __init__(self, cfg: SJConfig):
        super().__init__()
        self.cfg = cfg
        per_frame = cfg.max_objects * (cfg.state_dim + 1)
        self.inp = nn.Sequential(
            nn.Linear(per_frame, 256), nn.GELU(),
            nn.Linear(256, cfg.state_embed), nn.GELU(),
        )
        self.gru = nn.GRU(cfg.state_embed, cfg.state_embed, batch_first=True)

    def forward(self, state: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        B, T, O, _ = state.shape
        m = object_mask[:, None, :, None].expand(B, T, O, 1).float()
        x = torch.cat((state * m, m), dim=-1).flatten(2)
        _, h = self.gru(self.inp(x))
        return h[-1]


class VisionSJSpecialist(nn.Module):
    def __init__(self, cfg: SJConfig, masks: torch.Tensor, seed: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = VideoEncoder(cfg)
        self.sj = SparseVQCore(cfg.frame_embed, cfg, masks, seed)
        self.future_frames = SmallFrameDecoder(2 * cfg.hv_dim, n_out=4, size=32)

    def forward(self, video_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        embed = self.encoder(video_obs)
        out = self.sj(embed, pair_writes=True)
        out["embed"] = embed
        out["future_rgb"] = self.future_frames(out["program"])
        return out

    @torch.inference_mode()
    def codes(self, video_obs: torch.Tensor) -> torch.Tensor:
        return self.forward(video_obs)["codes"]


class CodeDynamics(nn.Module):
    def __init__(self, cfg: SJConfig):
        super().__init__()
        self.cfg = cfg
        d = 2 * cfg.hv_dim
        self.net = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, cfg.groups * cfg.card),
        )

    def forward(self, program: torch.Tensor) -> torch.Tensor:
        return self.net(program).view(len(program), self.cfg.groups, self.cfg.card)


class PhysicsSJSpecialist(nn.Module):
    def __init__(self, cfg: SJConfig, masks: torch.Tensor, seed: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = StateSequenceEncoder(cfg)
        self.sj = SparseVQCore(cfg.state_embed, cfg, masks, seed)
        self.dynamics = CodeDynamics(cfg)
        out_dim = cfg.video_frames_future * cfg.max_objects * cfg.dyn_dim
        self.future_state = nn.Sequential(
            nn.Linear(2 * cfg.hv_dim, 512), nn.GELU(), nn.Linear(512, out_dim)
        )

    def encode_state(self, state: torch.Tensor, object_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        embed = self.encoder(state, object_mask)
        out = self.sj(embed, pair_writes=True)
        out["embed"] = embed
        return out

    def rollout_from_codes(self, codes: torch.Tensor) -> dict[str, torch.Tensor]:
        present_program = self.sj.program_feature_from_codes(codes)
        future_logits = self.dynamics(present_program)
        future_probs = future_logits.softmax(-1)
        future_program = self.sj.program_feature_from_probs(future_probs)
        future_state = self.future_state(future_program).view(
            len(codes), self.cfg.video_frames_future, self.cfg.max_objects, self.cfg.dyn_dim
        )
        return {
            "future_logits": future_logits,
            "future_probs": future_probs,
            "future_program": future_program,
            "future_state": future_state,
        }

    @torch.inference_mode()
    def codes(self, state: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        return self.encode_state(state, object_mask)["codes"]


class OutcomeDecoderSJ(nn.Module):
    def __init__(self, cfg: SJConfig, masks: torch.Tensor, seed: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = StateSequenceEncoder(cfg)
        self.sj = SparseVQCore(cfg.state_embed, cfg, masks, seed)
        self.readout = nn.Sequential(
            nn.Linear(2 * cfg.hv_dim, 192), nn.GELU(), nn.Linear(192, 6)
        )

    def forward(self, future_state: torch.Tensor, object_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        embed = self.encoder(future_state, object_mask)
        out = self.sj(embed, pair_writes=True)
        out["embed"] = embed
        out["outcome_pred"] = self.readout(out["program"])
        return out

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self.readout(self.sj.program_feature_from_codes(codes))

    @torch.inference_mode()
    def codes(self, future_state: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        return self.forward(future_state, object_mask)["codes"]


# Architecture compatibility

@dataclass(frozen=True)
class StructuralCertificate:
    compatible: bool
    group_maps: tuple[tuple[int, ...], ...]
    ambiguity: int
    best_cost: float


def structural_isomorphisms(sig_a: np.ndarray, sig_b: np.ndarray, atol: float = 0.0) -> StructuralCertificate:
    A, B = np.asarray(sig_a), np.asarray(sig_b)
    if A.shape != B.shape or A.ndim != 3 or A.shape[0] != A.shape[1]:
        return StructuralCertificate(False, (), 0, float("inf"))
    G = A.shape[0]
    best, solutions = float("inf"), []
    layers = np.arange(B.shape[2])
    for perm in itertools.permutations(range(G)):
        cost = float(np.abs(A - B[np.ix_(perm, perm, layers)]).sum())
        if cost < best - 1e-12:
            best, solutions = cost, [tuple(map(int, perm))]
        elif abs(cost - best) <= 1e-12:
            solutions.append(tuple(map(int, perm)))
    compatible = best <= atol
    return StructuralCertificate(compatible, tuple(solutions) if compatible else (), len(solutions) if compatible else 0, best)


def scaffold_signature(masks: torch.Tensor) -> np.ndarray:
    return architecture_intersection_tensor(masks.cpu()).numpy()


# Unpaired sparse-hv Glue

@dataclass(frozen=True)
class SJGlueResult:
    compatible: bool
    group_map: tuple[int, ...] | None
    value_maps: tuple[tuple[int, ...] | None, ...]
    used_joints: tuple[tuple[int, ...], ...]
    rejected_joints: tuple[tuple[int, ...], ...]
    resolved_slots: tuple[int, ...]
    structural_ambiguity: int
    objective: float
    margin: float


def _empirical_logpmf(codes: np.ndarray, C: int, alpha: float) -> np.ndarray:
    x = np.asarray(codes, dtype=np.int64)
    G = x.shape[1]
    idx = np.zeros(len(x), dtype=np.int64)
    for g in range(G):
        idx = idx * C + x[:, g]
    counts = np.bincount(idx, minlength=C ** G).reshape((C,) * G).astype(np.float64)
    counts += alpha
    return np.log(counts / counts.sum())


def _marginalized(logp: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
    complement = tuple(i for i in range(logp.ndim) if i not in subset)
    table = logp.mean(axis=complement) if complement else logp.copy()
    surviving = tuple(i for i in range(logp.ndim) if i in subset)
    if surviving != subset:
        table = np.transpose(table, [surviving.index(i) for i in subset])
    return table


def _pure_component(logp: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
    base = _marginalized(logp, subset).copy()
    if len(subset) == 1:
        return base - base.mean()
    for r in range(1, len(subset)):
        for lower in itertools.combinations(subset, r):
            comp = _pure_component(logp, lower)
            shape = [1] * len(subset)
            for ax, slot in enumerate(subset):
                if slot in lower:
                    shape[ax] = comp.shape[lower.index(slot)]
            base -= comp.reshape(shape)
    return base


def interaction_tables(codes: np.ndarray, C: int, joints: Iterable[Sequence[int]], alpha: float = 0.5):
    logp = _empirical_logpmf(codes, C, alpha)
    return {tuple(map(int, j)): _pure_component(logp, tuple(map(int, j))) for j in joints}


def _joint_cost_table(a: np.ndarray, b: np.ndarray, C: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    perms = list(itertools.permutations(range(C)))
    out = np.empty((len(perms),) * a.ndim, dtype=np.float64)
    for choice in itertools.product(range(len(perms)), repeat=a.ndim):
        idx = np.ix_(*[np.asarray(perms[k]) for k in choice])
        out[choice] = float(((a - b[idx]) ** 2).mean())
    return out, perms


def fit_sj_glue(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
    card: int,
    *,
    joints: Sequence[Sequence[int]] | None = None,
    trim: int = 0,
    alpha: float = 0.5,
    min_margin: float = 0.0,
) -> SJGlueResult:
    A, B = np.asarray(codes_a, dtype=np.int64), np.asarray(codes_b, dtype=np.int64)
    G = A.shape[1]
    sig_a, sig_b = scaffold_signature(masks_a), scaffold_signature(masks_b)
    cert = structural_isomorphisms(sig_a, sig_b)
    if not cert.compatible:
        return SJGlueResult(False, None, tuple([None] * G), (), (), (), 0, float("inf"), 0.0)

    source_joints = [tuple(map(int, j)) for j in (joints or physical_edges(masks_a))]
    best_global = None

    for group_map in cert.group_maps:
        target_joints, axis_orders = [], []
        for joint in source_joints:
            mapped = tuple(group_map[i] for i in joint)
            sorted_slots = tuple(sorted(mapped))
            target_joints.append(sorted_slots)
            axis_orders.append(tuple(sorted_slots.index(v) for v in mapped))

        ta = interaction_tables(A, card, source_joints, alpha)
        tb_raw = interaction_tables(B, card, target_joints, alpha)
        specs = []
        perms = None
        for joint, target_joint, axes in zip(source_joints, target_joints, axis_orders):
            tb = tb_raw[target_joint]
            if axes != tuple(range(len(axes))):
                tb = np.transpose(tb, axes)
            costs, perms = _joint_cost_table(ta[joint], tb, card)
            specs.append((joint, costs))
        assert perms is not None

        # Components of the informative joint graph. Singleton components abstain.
        adjacency = [set() for _ in range(G)]
        for joint, _ in specs:
            for u, v in itertools.combinations(joint, 2):
                adjacency[u].add(v); adjacency[v].add(u)
        seen, components = set(), []
        for root in range(G):
            if root in seen:
                continue
            stack, comp = [root], []
            seen.add(root)
            while stack:
                u = stack.pop(); comp.append(u)
                for v in adjacency[u]:
                    if v not in seen:
                        seen.add(v); stack.append(v)
            components.append(tuple(sorted(comp)))

        maps: list[tuple[int, ...] | None] = [None] * G
        used, rejected, margins = [], [], []
        total_obj = 0.0

        for comp in components:
            if len(comp) == 1:
                continue
            local = {u: k for k, u in enumerate(comp)}
            local_specs = [(j, c) for j, c in specs if all(u in local for u in j)]
            P = len(perms)
            shape = (P,) * len(comp)
            arrays, names = [], []
            for joint, costs in local_specs:
                order = np.argsort([local[u] for u in joint])
                joint_sorted = tuple(joint[k] for k in order)
                c2 = np.transpose(costs, tuple(order)) if len(joint) > 1 else costs
                broadcast_shape = [1] * len(comp)
                for u in joint_sorted:
                    broadcast_shape[local[u]] = P
                arrays.append(np.broadcast_to(c2.reshape(broadcast_shape), shape))
                names.append(joint)

            stacked = np.stack(arrays, axis=0)
            keep = max(1, len(arrays) - int(trim))
            objective = (
                np.partition(stacked, keep - 1, axis=0)[:keep].sum(axis=0)
                if keep < len(arrays)
                else stacked.sum(axis=0)
            )
            flat = objective.ravel()
            k2 = min(2, len(flat))
            winners = np.argpartition(flat, k2 - 1)[:k2]
            winners = winners[np.argsort(flat[winners])]
            index = np.unravel_index(int(winners[0]), shape)
            best = float(flat[winners[0]])
            second = float(flat[winners[1]]) if len(winners) > 1 else float("inf")
            margin = second - best
            if margin < min_margin:
                continue

            for u in comp:
                maps[u] = tuple(map(int, perms[index[local[u]]]))
            residuals = np.array([float(edge[index]) for edge in stacked])
            order = np.argsort(residuals)
            keep_idx = set(map(int, order[:keep]))
            used.extend(names[k] for k in keep_idx)
            rejected.extend(names[k] for k in range(len(names)) if k not in keep_idx)
            total_obj += best
            margins.append(margin)

        resolved = tuple(i for i, p in enumerate(maps) if p is not None)
        margin = min(margins) if margins else 0.0
        candidate = (total_obj, -len(resolved), -margin, group_map, tuple(maps), tuple(used), tuple(rejected), resolved, margin)
        if best_global is None or candidate[:3] < best_global[:3]:
            best_global = candidate

    assert best_global is not None
    obj, _, _, gm, maps, used, rejected, resolved, margin = best_global
    return SJGlueResult(True, tuple(gm), maps, used, rejected, resolved, cert.ambiguity, float(obj), float(margin))


def apply_glue_codes(codes: torch.Tensor, glue: SJGlueResult) -> torch.Tensor:
    if glue.group_map is None:
        raise RuntimeError("incompatible sparse-joint architectures")
    out = torch.full_like(codes, -1)
    for src, value_map in enumerate(glue.value_maps):
        if value_map is None:
            continue
        mapping = torch.as_tensor(value_map, dtype=torch.long, device=codes.device)
        out[:, glue.group_map[src]] = mapping[codes[:, src]]
    return out


#checkpoints

def save_checkpoint(path: str | Path, model: nn.Module, masks: torch.Tensor, *, kind: str, cfg: SJConfig, seed: int, metadata: dict | None = None):
    payload = {
        "format": "sj-model-checkpoint-v1",
        "kind": kind,
        "config": asdict(cfg),
        "masks": masks.detach().cpu(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "seed": int(seed),
        "metadata": metadata or {},
    }
    torch.save(payload, Path(path))


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> nn.Module:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    allowed = {f.name for f in fields(SJConfig)}
    cfg = SJConfig(**{k: v for k, v in ckpt["config"].items() if k in allowed})
    masks = ckpt["masks"].to(map_location)
    kind = ckpt["kind"]
    seed = int(ckpt.get("seed", cfg.seed))
    if kind == "vision":
        model = VisionSJSpecialist(cfg, masks, seed)
    elif kind == "physics":
        model = PhysicsSJSpecialist(cfg, masks, seed)
    elif kind == "decoder":
        model = OutcomeDecoderSJ(cfg, masks, seed)
    else:
        raise ValueError(f"unknown checkpoint kind: {kind}")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(map_location).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
