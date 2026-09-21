from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ACTION_NULL_STATISTIC = "support_then_quality_lexicographic_v1"

_CONFLICT_POLICIES = {"require_equal", "prefer_source", "prefer_target"}


def physical_edges(masks: torch.Tensor) -> tuple[tuple[int, int], ...]:
    if masks.ndim != 3:
        raise ValueError("masks must be [layers, groups, units]")
    groups = masks.shape[1]
    return tuple(
        (i, j)
        for i, j in itertools.combinations(range(groups), 2)
        if bool((masks[:, i] * masks[:, j]).sum() > 0)
    )


def _as_probs(x, C):
    x = np.asarray(x)
    if not isinstance(C, (int, np.integer)) or C < 1:
        raise ValueError("cardinality must be a positive integer")
    if x.ndim not in (2, 3) or min(x.shape[:2], default=0) < 1:
        raise ValueError(
            "operator evidence must be a nonempty [rows,groups] or "
            "[rows,groups,fillers] array"
        )
    if x.ndim == 2:
        if x.dtype.kind not in "iu" or np.any((x < 0) | (x >= C)):
            raise ValueError(
                "operator evidence requires observed integer fillers; "
                "missing codes are not evidence"
            )
        return np.eye(C, dtype=np.float64)[x.astype(np.int64)]
    if x.shape[-1] != C or x.dtype.kind not in "fiu":
        raise ValueError("invalid probability evidence shape or dtype")
    y = x.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        mass = y.sum(-1, keepdims=True)
    if (
        not np.isfinite(y).all()
        or not np.isfinite(mass).all()
        or np.any(y < 0)
        or np.any(mass <= 0)
    ):
        raise ValueError(
            "operator probabilities must have finite nonnegative positive mass"
        )
    return y / mass


def _graph_snapshot(core, require_frozen=False):
    if (
        require_frozen
        and hasattr(core, "scaffold_is_frozen")
        and not bool(core.scaffold_is_frozen)
    ):
        raise ValueError("freeze the learned scaffold before certifying Glue")
    masks = (
        core.routing_masks()
        if callable(getattr(core, "routing_masks", None))
        else core.masks
    )
    masks = torch.as_tensor(masks).detach().clone()
    if (
        masks.ndim != 3
        or masks.shape[1] != core.G
        or min(masks.shape) < 1
        or not bool(torch.isfinite(masks).all())
        or bool(((masks != 0) & (masks != 1)).any())
    ):
        raise ValueError("Glue requires a finite hard [layers,groups,units] graph")
    raw = masks.to(device="cpu", dtype=torch.uint8).numpy()
    fingerprint = hashlib.sha256(
        str(tuple(raw.shape)).encode() + raw.tobytes()
    ).hexdigest()
    return SimpleNamespace(
        G=core.G,
        C=core.C,
        grammar_signature=core.grammar_signature,
        masks=masks,
        graph_hash=fingerprint,
    )


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(np.maximum(p, 1e-300))).sum())


def _nmi(j):
    p = j / np.maximum(j.sum(), 1e-12)
    px = p.sum(1)
    py = p.sum(0)
    mi = float(
        (
            p
            * np.log(
                np.maximum(p, 1e-300)
                / (np.maximum(px[:, None] * py[None, :], 1e-300))
            )
        ).sum()
    )
    h = math.sqrt(max(_entropy(px) * _entropy(py), 1e-16))
    return mi / h if h else 0.0


@dataclass(frozen=True)
class ActionOperator:
    name: str
    blocks: np.ndarray
    strength: np.ndarray


@dataclass
class OperatorGlueResult:
    status: str
    compatible: bool
    group_map: tuple[int, ...]
    value_maps: tuple[tuple[int, ...] | None, ...]
    resolved_slots: tuple[int, ...]
    target_groups: int
    reason: str
    train_score: float | None = None
    validation_score: float | None = None
    action_pvalue: float | None = None
    group_costs: tuple[float, ...] = ()
    group_margins: tuple[float, ...] = ()
    filler_margins: tuple[float, ...] = ()
    bootstrap_group_stability: float | None = None
    bootstrap_filler_stability: float | None = None
    bootstrap_exact_fraction: float | None = None
    bootstrap_unique_maps: int | None = None
    used_joints: tuple[tuple[int, int], ...] = ()
    rejected_joints: tuple[tuple[int, int], ...] = ()
    action_null_mode: str | None = None
    action_null_count: int = 0
    source_graph_hash: str | None = None
    target_graph_hash: str | None = None
    action_null_statistic: str | None = None


def estimate_action_operator(base, after, C, name, alpha=0.05):
    p0 = _as_probs(base, C)
    p1 = _as_probs(after, C)
    if p0.shape != p1.shape:
        raise ValueError("before/after evidence must have identical aligned shapes")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("operator pseudocount must be finite and nonnegative")
    G = p0.shape[1]
    B = np.zeros((G, G, C, C), float)
    S = np.zeros((G, G), float)
    for i in range(G):
        for j in range(G):
            z = np.full((C, C), alpha, float) + p0[:, i, :].T @ p1[:, j, :]
            S[i, j] = _nmi(z)
            B[i, j] = z / np.maximum(z.sum(1, keepdims=True), 1e-12)
    return ActionOperator(str(name), B, S)


def estimate_operators(base, actions, C, names, alpha=0.05):
    out = {}
    for a in names:
        observation = actions[a]
        if isinstance(observation, (tuple, list)) and len(observation) == 2:
            before, after = observation
        else:
            before, after = base, observation
        ident = estimate_action_operator(before, before, C, "__id__", alpha)
        raw = estimate_action_operator(before, after, C, a, alpha)
        d = raw.blocks - ident.blocks
        strength = np.linalg.norm(
            d.reshape(d.shape[0], d.shape[1], -1), axis=-1
        ) / math.sqrt(C * C)
        out[a] = ActionOperator(str(a), d, strength)
    return out


def _spec(M):
    sv = np.linalg.svd(M, compute_uv=False)
    diag = np.sort(np.diag(M))
    flat = np.sort(M.ravel())
    return np.concatenate([sv, diag, [flat.mean(), flat.std(), flat[-1], flat[0]]])


def _stats(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return np.zeros(6)
    return np.array(
        [
            x.mean(),
            x.std(),
            x.max(),
            np.quantile(x, 0.25),
            np.quantile(x, 0.5),
            np.quantile(x, 0.75),
        ]
    )


def _neighbors(masks, G):
    nei = [[] for _ in range(G)]
    for i, j in set(physical_edges(masks)):
        nei[i].append(j)
        nei[j].append(i)
    return nei


def _group_descriptors(
    ops: Mapping[str, ActionOperator],
    masks: torch.Tensor,
    C: int,
    names: Sequence[str],
):
    G = next(iter(ops.values())).blocks.shape[0]
    nei = _neighbors(masks, G)
    out = []
    for i in range(G):
        f = []
        for a in names:
            op = ops[a]
            f.extend(_spec(op.blocks[i, i]))
            f.extend(_stats([op.strength[i, j] for j in nei[i]]))
            f.extend(_stats([op.strength[j, i] for j in nei[i]]))
            cand = sorted(nei[i], key=lambda j: op.strength[i, j], reverse=True)[:2]
            for k in range(2):
                f.extend(
                    _spec(op.blocks[i, cand[k]])
                    if k < len(cand)
                    else np.zeros(2 * C + 4)
                )
        v = np.asarray(f, float)
        v = v / (np.linalg.norm(v) + 1e-12)
        out.append(v)
    return np.stack(out)


def _group_operator_energy(
    ops: Mapping[str, ActionOperator], masks: torch.Tensor, names: Sequence[str]
):
    G = next(iter(ops.values())).blocks.shape[0]
    nei = _neighbors(masks, G)
    e = np.zeros(G, float)
    for i in range(G):
        vals = []
        for a in names:
            op = ops[a]
            vals.append(np.linalg.norm(op.blocks[i, i]))
            vals.extend(np.linalg.norm(op.blocks[i, j]) for j in nei[i])
            vals.extend(np.linalg.norm(op.blocks[j, i]) for j in nei[i])
        e[i] = float(np.mean(vals)) if vals else 0.0
    return e


def _group_match(DA, DB, max_cost=0.35, min_margin=0.01):
    cost = np.sqrt(np.maximum(((DA[:, None, :] - DB[None, :, :]) ** 2).mean(-1), 0))
    ri, cj = linear_sum_assignment(cost)
    gm = [-1] * len(DA)
    gc = [1.0] * len(DA)
    marg = [0.0] * len(DA)
    for i, j in zip(ri, cj):
        row = np.sort(cost[i])
        second = row[1] if len(row) > 1 else 1.0
        gc[i] = float(cost[i, j])
        marg[i] = float(second - cost[i, j])
        if cost[i, j] <= max_cost and marg[i] >= min_margin:
            gm[i] = int(j)
    return gm, gc, marg, cost


def _shared_pairs(masksA, masksB, gm):
    ea = set(physical_edges(masksA))
    eb = set(physical_edges(masksB))
    pairs = []
    rej = []
    for i, j in ea:
        if gm[i] >= 0 and gm[j] >= 0 and tuple(sorted((gm[i], gm[j]))) in eb:
            pairs.append((i, j))
        elif gm[i] >= 0 and gm[j] >= 0:
            rej.append((i, j))
    return pairs, rej


def _init_filler_cost(i, opsA, opsB, gm, pairs, names, C):
    j = gm[i]
    cost = np.zeros((C, C), float)
    related = {i}
    for u, v in pairs:
        if u == i:
            related.add(v)
        if v == i:
            related.add(u)
    for a in names:
        A = opsA[a].blocks
        B = opsB[a].blocks
        for c in range(C):
            ar = np.sort(A[i, i, c])
            ac = np.sort(A[i, i, :, c])
            for d in range(C):
                cost[c, d] += np.mean((ar - np.sort(B[j, j, d])) ** 2) + np.mean(
                    (ac - np.sort(B[j, j, :, d])) ** 2
                )
        for k in related:
            if k == i or gm[k] < 0:
                continue
            m = gm[k]
            for c in range(C):
                ar = np.sort(A[i, k, c])
                ac = np.sort(A[k, i, :, c])
                for d in range(C):
                    cost[c, d] += np.mean((ar - np.sort(B[j, m, d])) ** 2) + np.mean(
                        (ac - np.sort(B[m, j, :, d])) ** 2
                    )
    return cost


def _hungarian_perm(cost):
    r, c = linear_sum_assignment(cost)
    p = np.empty(len(r), int)
    p[r] = c
    return p, float(cost[r, c].sum())


def _assignment_margin(cost, p):
    C = len(p)
    r = np.arange(C)
    best = float(cost[r, p].sum())
    second = float("inf")
    for c, d in enumerate(p):
        z = cost.copy()
        z[c, d] = np.inf
        rr, cc = linear_sum_assignment(z)
        second = min(second, float(z[rr, cc].sum()))
    return second - best if np.isfinite(second) else float("inf")


def _conditional_permutation_cost(i, p, opsA, opsB, gm, P, related, names):
    j = gm[i]
    p = np.asarray(p, dtype=int)
    cost = 0.0
    terms = 0
    for a in names:
        A = opsA[a].blocks
        B = opsB[a].blocks
        cost += float(np.mean((A[i, i] - B[j, j][np.ix_(p, p)]) ** 2))
        terms += 1
        for k in related[i]:
            if k == i or gm[k] < 0 or P[k] is None:
                continue
            m = gm[k]
            pk = np.asarray(P[k], dtype=int)
            cost += float(np.mean((A[i, k] - B[j, m][np.ix_(p, pk)]) ** 2))
            terms += 1
            cost += float(np.mean((A[k, i] - B[m, j][np.ix_(pk, p)]) ** 2))
            terms += 1
    return cost / max(terms, 1)


def _solve_fillers(opsA, opsB, gm, pairs, names, C, iters=12):
    P = [
        None
        if j < 0
        else _hungarian_perm(_init_filler_cost(i, opsA, opsB, gm, pairs, names, C))[0]
        for i, j in enumerate(gm)
    ]
    margins = [0.0] * len(gm)
    related = [set([i]) for i in range(len(gm))]
    for u, v in pairs:
        related[u].add(v)
        related[v].add(u)
    if C <= 6:
        candidates = tuple(itertools.permutations(range(C)))
        for _ in range(iters):
            changed = False
            for i, j in enumerate(gm):
                if j < 0:
                    continue
                scored = sorted(
                    (
                        _conditional_permutation_cost(
                            i, p, opsA, opsB, gm, P, related, names
                        ),
                        p,
                    )
                    for p in candidates
                )
                best, p = scored[0]
                margins[i] = (
                    float(scored[1][0] - best) if len(scored) > 1 else float("inf")
                )
                p = np.asarray(p, dtype=int)
                if P[i] is None or not np.array_equal(p, P[i]):
                    P[i] = p
                    changed = True
            if not changed:
                break
        return [None if p is None else tuple(map(int, p)) for p in P], margins
    for _ in range(iters):
        changed = False
        for i, j in enumerate(gm):
            if j < 0:
                continue
            cost = np.zeros((C, C), float)
            for a in names:
                A = opsA[a].blocks
                B = opsB[a].blocks
                for k in related[i]:
                    if k == i or gm[k] < 0 or P[k] is None:
                        continue
                    m = gm[k]
                    pk = np.asarray(P[k])
                    for c in range(C):
                        av = A[i, k, c, :]
                        for d in range(C):
                            cost[c, d] += np.mean((av - B[j, m, d, pk]) ** 2)
                    for c in range(C):
                        av = A[k, i, :, c]
                        for d in range(C):
                            cost[c, d] += np.mean((av - B[m, j, pk, d]) ** 2)
                init = _init_filler_cost(
                    i, {a: opsA[a]}, {a: opsB[a]}, gm, [], [a], C
                )
                cost += 0.25 * init
            p, _ = _hungarian_perm(cost)
            margins[i] = _assignment_margin(cost, p)
            if P[i] is None or not np.array_equal(p, P[i]):
                P[i] = p
                changed = True
        if not changed:
            break
    return [None if p is None else tuple(map(int, p)) for p in P], margins


def _score(opsA, opsB, gm, P, pairs, names):
    vals = []
    weights = []
    use = [
        (i, i) for i, j in enumerate(gm) if j >= 0 and P[i] is not None
    ] + list(pairs)
    for a in names:
        A = opsA[a]
        B = opsB[a]
        for i, k in use:
            if gm[i] < 0 or gm[k] < 0 or P[i] is None or P[k] is None:
                continue
            ai = A.blocks[i, k].ravel()
            bb = B.blocks[gm[i], gm[k]][
                np.ix_(np.asarray(P[i]), np.asarray(P[k]))
            ].ravel()
            na = np.linalg.norm(ai)
            nb = np.linalg.norm(bb)
            if max(na, nb) <= 1e-12:
                continue
            cos = float(ai @ bb / (na * nb + 1e-12))
            w = max(0.05, 0.5 * (A.strength[i, k] + B.strength[gm[i], gm[k]]))
            vals.append(cos * w)
            weights.append(w)
    return float(np.sum(vals) / np.sum(weights)) if weights else 0.0


def _solve_given_support(OA, OB, gm, coreA, coreB, names, C, min_filler_margin):
    coreA, coreB = _graph_snapshot(coreA), _graph_snapshot(coreB)
    gm = list(gm)
    pairs, rej = _shared_pairs(coreA.masks, coreB.masks, gm)
    P, fmar = _solve_fillers(OA, OB, gm, pairs, names, C)
    for i in range(len(gm)):
        if gm[i] >= 0 and (P[i] is None or fmar[i] < min_filler_margin):
            gm[i] = -1
            P[i] = None
    pairs, rej = _shared_pairs(coreA.masks, coreB.masks, gm)
    sc = _score(OA, OB, gm, P, pairs, names)
    return gm, P, fmar, pairs, rej, sc


def _fit_once(
    baseA,
    actionsA,
    coreA,
    baseB,
    actionsB,
    coreB,
    names,
    alpha,
    max_group_cost,
    min_group_margin,
    min_filler_margin,
    min_group_energy,
    support_target_score=None,
):
    coreA, coreB = _graph_snapshot(coreA), _graph_snapshot(coreB)
    OA = estimate_operators(baseA, actionsA, coreA.C, names, alpha)
    OB = estimate_operators(baseB, actionsB, coreB.C, names, alpha)
    DA = _group_descriptors(OA, coreA.masks, coreA.C, names)
    DB = _group_descriptors(OB, coreB.masks, coreB.C, names)
    EA = _group_operator_energy(OA, coreA.masks, names)
    EB = _group_operator_energy(OB, coreB.masks, names)
    gm, gc, gmar, _ = _group_match(DA, DB, max_group_cost, min_group_margin)
    for i, j in enumerate(gm):
        if j >= 0 and min(EA[i], EB[j]) < min_group_energy:
            gm[i] = -1
    gm, P, fmar, pairs, rej, sc = _solve_given_support(
        OA, OB, gm, coreA, coreB, names, coreA.C, min_filler_margin
    )
    if support_target_score is not None:
        while sc < support_target_score:
            active = [i for i, j in enumerate(gm) if j >= 0 and P[i] is not None]
            if len(active) <= 2:
                break
            best = None
            for drop in active:
                trial = [j if i != drop else -1 for i, j in enumerate(gm)]
                tg, tP, tf, tp, tr, ts = _solve_given_support(
                    OA, OB, trial, coreA, coreB, names, coreA.C, min_filler_margin
                )
                if len(tp) == 0:
                    continue
                cand = (ts, -drop, tg, tP, tf, tp, tr)
                if best is None or cand[:2] > best[:2]:
                    best = cand
            if best is None or best[0] <= sc + 1e-12:
                break
            sc, _, gm, P, fmar, pairs, rej = best
    return {
        "opsA": OA,
        "opsB": OB,
        "gm": gm,
        "P": P,
        "gc": gc,
        "gmar": gmar,
        "fmar": fmar,
        "pairs": pairs,
        "rej": rej,
        "score": sc,
    }


def _map_key(f):
    return (
        tuple(f["gm"]),
        tuple(None if p is None else tuple(p) for p in f["P"]),
    )


def _bootstrap(
    baseA,
    actionsA,
    coreA,
    baseB,
    actionsB,
    coreB,
    names,
    ref,
    *,
    alpha,
    max_group_cost,
    min_group_margin,
    min_filler_margin,
    min_group_energy,
    support_target_score,
    replicates,
    fraction,
    seed,
):
    if replicates <= 0:
        return None, None, None, None
    rng = np.random.default_rng(seed)
    gacc = []
    facc = []
    keys = []

    def sample_system(base, actions):
        sampled = {}
        shared_idx = None
        for name in names:
            observation = actions[name]
            if isinstance(observation, (tuple, list)) and len(observation) == 2:
                before, after = observation
                n = max(16, int(len(before) * fraction))
                idx = rng.integers(0, len(before), n)
                sampled[name] = (np.asarray(before)[idx], np.asarray(after)[idx])
            else:
                if shared_idx is None:
                    n = max(16, int(len(base) * fraction))
                    shared_idx = rng.integers(0, len(base), n)
                sampled[name] = np.asarray(observation)[shared_idx]
        return (
            base if shared_idx is None else np.asarray(base)[shared_idx]
        ), sampled

    for _ in range(replicates):
        ba, aa = sample_system(baseA, actionsA)
        bb0, bb = sample_system(baseB, actionsB)
        f = _fit_once(
            ba,
            aa,
            coreA,
            bb0,
            bb,
            coreB,
            names,
            alpha,
            max_group_cost,
            min_group_margin,
            min_filler_margin,
            min_group_energy,
            support_target_score,
        )
        keys.append(_map_key(f))
        gs = []
        fs = []
        for i, j in enumerate(ref["gm"]):
            if j < 0:
                continue
            gs.append(f["gm"][i] == j)
            rp = ref["P"][i]
            fp = f["P"][i]
            fs.append(fp is not None and rp is not None and tuple(fp) == tuple(rp))
        gacc.append(float(np.mean(gs)) if gs else 0.0)
        facc.append(float(np.mean(fs)) if fs else 0.0)
    exact = float(np.mean([k == _map_key(ref) for k in keys]))
    return float(np.mean(gacc)), float(np.mean(facc)), exact, len(set(keys))


def _fit_operator_sj_glue(
    base_a,
    actions_a,
    core_a,
    base_b,
    actions_b,
    core_b,
    *,
    train_actions,
    validation_actions=(),
    alpha=0.05,
    max_group_cost=0.35,
    min_group_margin=0.005,
    min_filler_margin=1e-5,
    min_group_energy=0.02,
    min_train_score=0.82,
    min_validation_score=0.80,
    permutation_nulls=119,
    exact_action_null=True,
    p_threshold=0.05,
    bootstrap_replicates=16,
    bootstrap_fraction=0.80,
    min_group_stability=0.80,
    min_filler_stability=0.70,
    min_exact_fraction=0.50,
    seed=0,
):
    GA, GB = core_a.G, core_b.G
    empty = tuple([-1] * GA)
    none = tuple([None] * GA)
    if core_a.C != core_b.C or core_a.grammar_signature != core_b.grammar_signature:
        return OperatorGlueResult(
            "no_match", False, empty, none, (), GB, "SJ grammar/cardinality mismatch"
        )
    tr = tuple(train_actions)
    va = tuple(validation_actions)
    if len(set(tr)) != len(tr) or len(set(va)) != len(va) or set(tr) & set(va):
        raise ValueError(
            "training and validation action names must be unique and disjoint"
        )
    if any(
        not np.isfinite(v)
        for v in (
            alpha,
            max_group_cost,
            min_group_margin,
            min_filler_margin,
            min_group_energy,
            min_train_score,
            min_validation_score,
            min_group_stability,
            min_filler_stability,
            min_exact_fraction,
        )
    ):
        raise ValueError("operator thresholds must be finite")
    if not isinstance(permutation_nulls, (int, np.integer)) or not isinstance(
        bootstrap_replicates, (int, np.integer)
    ):
        raise ValueError("null/bootstrap counts must be integers")
    if (
        not 0 <= p_threshold <= 1
        or permutation_nulls < 0
        or bootstrap_replicates < 0
        or not np.isfinite(bootstrap_fraction)
        or bootstrap_fraction <= 0
    ):
        raise ValueError("invalid null/bootstrap configuration")
    if not tr:
        return OperatorGlueResult(
            "algebraic_candidate",
            False,
            empty,
            none,
            (),
            GB,
            "no grounding interventions supplied",
        )
    f = _fit_once(
        base_a,
        actions_a,
        core_a,
        base_b,
        actions_b,
        core_b,
        tr,
        alpha,
        max_group_cost,
        min_group_margin,
        min_filler_margin,
        min_group_energy,
        min_train_score,
    )
    gm = f["gm"]
    P = f["P"]
    resolved = tuple(i for i, j in enumerate(gm) if j >= 0 and P[i] is not None)
    if not resolved:
        return OperatorGlueResult(
            "algebraic_candidate",
            False,
            tuple(gm),
            tuple(P),
            (),
            GB,
            "operator algebra leaves no identifiable shared SJ atoms",
            f["score"],
            group_costs=tuple(f["gc"]),
            group_margins=tuple(f["gmar"]),
            filler_margins=tuple(f["fmar"]),
        )
    vscore = None
    if va:
        OAv = estimate_operators(base_a, actions_a, core_a.C, va, alpha)
        OBv = estimate_operators(base_b, actions_b, core_b.C, va, alpha)
        vscore = _score(OAv, OBv, gm, P, f["pairs"], va)
    rng = np.random.default_rng(seed + 71)
    identity = tuple(range(len(tr)))
    if exact_action_null and len(tr) <= 6:
        seen = [p for p in itertools.permutations(range(len(tr))) if p != identity]
        null_mode = "exact"
    else:
        seen_set = set()
        target = min(
            int(permutation_nulls),
            math.factorial(len(tr)) - 1 if len(tr) <= 9 else int(permutation_nulls),
        )
        while len(seen_set) < target:
            p = tuple(rng.permutation(len(tr)).tolist())
            if p != identity:
                seen_set.add(p)
        seen = list(seen_set)
        null_mode = "monte_carlo"
    null = []
    for p in seen:
        bad = {a: actions_b[tr[p[k]]] for k, a in enumerate(tr)}
        z = _fit_once(
            base_a,
            actions_a,
            core_a,
            base_b,
            bad,
            core_b,
            tr,
            alpha,
            max_group_cost,
            min_group_margin,
            min_filler_margin,
            min_group_energy,
            min_train_score,
        )
        null.append(
            (
                sum(j >= 0 and q is not None for j, q in zip(z["gm"], z["P"])),
                z["score"],
            )
        )
    extreme = sum(
        n > len(resolved) or (n == len(resolved) and s >= f["score"] - 1e-12)
        for n, s in null
    )
    pval = (1 + extreme) / (1 + len(null)) if len(null) else 1.0
    gs, fs, ex, uniq = _bootstrap(
        base_a,
        actions_a,
        core_a,
        base_b,
        actions_b,
        core_b,
        tr,
        f,
        alpha=alpha,
        max_group_cost=max_group_cost,
        min_group_margin=min_group_margin,
        min_filler_margin=min_filler_margin,
        min_group_energy=min_group_energy,
        support_target_score=min_train_score,
        replicates=bootstrap_replicates,
        fraction=bootstrap_fraction,
        seed=seed + 991,
    )
    ground = bool(
        f["score"] >= min_train_score
        and pval <= p_threshold
        and (vscore is None or vscore >= min_validation_score)
        and len(f["pairs"]) > 0
        and gs is not None
        and gs >= min_group_stability
        and fs is not None
        and fs >= min_filler_stability
        and ex is not None
        and ex >= min_exact_fraction
    )
    return OperatorGlueResult(
        "grounded" if ground else "algebraic_candidate",
        ground,
        tuple(gm),
        tuple(P),
        resolved,
        GB,
        "interventional SJ operators admit a stable native intertwiner"
        if ground
        else "operator evidence does not certify an executable partial intertwiner",
        f["score"],
        vscore,
        float(pval),
        tuple(f["gc"]),
        tuple(f["gmar"]),
        tuple(f["fmar"]),
        gs,
        fs,
        ex,
        uniq,
        tuple(f["pairs"]),
        tuple(f["rej"]),
        null_mode,
        len(null),
    )


def fit_operator_sj_glue(
    base_a, actions_a, core_a, base_b, actions_b, core_b, *, train_actions, **options
):
    a, b = _graph_snapshot(core_a, True), _graph_snapshot(core_b, True)
    result = _fit_operator_sj_glue(
        base_a, actions_a, a, base_b, actions_b, b, train_actions=train_actions, **options
    )
    result.source_graph_hash = a.graph_hash
    result.target_graph_hash = b.graph_hash
    result.action_null_statistic = ACTION_NULL_STATISTIC
    return result


def apply_glue_codes(codes, glue, allow_ungrounded=False):
    if (
        not isinstance(codes, torch.Tensor)
        or codes.ndim != 2
        or codes.shape[1] != len(glue.group_map)
        or codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError(
            "expected signed integer source codes with the certified group count"
        )
    if bool((codes < -1).any()):
        raise ValueError("only -1 denotes a missing code")
    if glue.status != "grounded" and not allow_ungrounded:
        raise RuntimeError(f"refusing {glue.status}: {glue.reason}")
    out = torch.full(
        (len(codes), glue.target_groups), -1, dtype=codes.dtype, device=codes.device
    )
    for i, (j, p) in enumerate(zip(glue.group_map, glue.value_maps)):
        if j < 0 or p is None:
            continue
        m = torch.as_tensor(p, dtype=torch.long, device=codes.device)
        src = codes[:, i].long()
        valid = src >= 0
        if bool((src[valid] >= len(m)).any()):
            raise ValueError("source SJ code is outside the filler vocabulary")
        out[valid, j] = m[src[valid]].to(out.dtype)
    return out


def _core_signature(core: Any) -> tuple:
    signature = getattr(core, "grammar_signature", None)
    if signature is None:
        raise TypeError("core does not expose an SJ grammar_signature")
    return tuple(signature)


def _validate_codes(codes: torch.Tensor, groups: int, card: int, name: str) -> None:
    if not torch.is_tensor(codes):
        raise TypeError(f"{name} must be a torch.Tensor")
    if codes.ndim != 2 or codes.shape[1] != groups:
        raise ValueError(f"{name} must have shape [batch, {groups}]")
    if torch.is_floating_point(codes) or codes.dtype == torch.bool:
        raise TypeError(f"{name} must contain integer hard SJ codes")
    if bool(((codes < -1) | (codes >= card)).any()):
        raise ValueError(f"{name} contains a code outside [-1, {card - 1}]")


@dataclass(frozen=True)
class SJTransitionEvidence:
    base: np.ndarray
    actions: Mapping[str, Any]


@dataclass(frozen=True)
class SJGlue:
    certificate: OperatorGlueResult
    source_groups: int
    target_groups: int
    cardinality: int
    grammar_signature: tuple
    confirmation_certificate: OperatorGlueResult | None = None

    def __post_init__(self) -> None:
        group_map = self.certificate.group_map
        value_maps = self.certificate.value_maps
        if (
            len(group_map) != self.source_groups
            or len(value_maps) != self.source_groups
        ):
            raise ValueError("certificate has the wrong number of source groups")
        image = [j for j in group_map if j >= 0]
        if len(image) != len(set(image)):
            raise ValueError("an SJ Glue group map must be injective")
        if any(j >= self.target_groups for j in image):
            raise ValueError("certificate maps outside the target SJ groups")
        for group, values in zip(group_map, value_maps):
            if (group < 0) != (values is None):
                raise ValueError("group and filler maps must have identical support")
            if values is not None and sorted(values) != list(range(self.cardinality)):
                raise ValueError("every resolved filler map must be a permutation")

    @property
    def status(self) -> str:
        if self.confirmation_certificate is not None and not self.is_executable:
            return "independently_unconfirmed"
        return self.certificate.status

    @property
    def is_executable(self) -> bool:
        primary = bool(
            self.certificate.compatible and self.certificate.status == "grounded"
        )
        if self.confirmation_certificate is None:
            return primary
        confirmation = self.confirmation_certificate
        same_map = (
            self.certificate.group_map == confirmation.group_map
            and self.certificate.value_maps == confirmation.value_maps
            and self.certificate.source_graph_hash == confirmation.source_graph_hash
            and self.certificate.target_graph_hash == confirmation.target_graph_hash
        )
        return bool(
            primary
            and confirmation.compatible
            and confirmation.status == "grounded"
            and same_map
        )

    @property
    def resolved_source_groups(self) -> tuple[int, ...]:
        return tuple(
            i
            for i, (j, values) in enumerate(
                zip(self.certificate.group_map, self.certificate.value_maps)
            )
            if j >= 0 and values is not None
        )

    @property
    def resolved_target_groups(self) -> tuple[int, ...]:
        return tuple(self.certificate.group_map[i] for i in self.resolved_source_groups)

    @property
    def source_coverage(self) -> float:
        return len(self.resolved_source_groups) / max(self.source_groups, 1)

    @property
    def target_coverage(self) -> float:
        return len(self.resolved_target_groups) / max(self.target_groups, 1)

    @property
    def kind(self) -> str:
        if (
            len(self.resolved_source_groups) == self.source_groups
            and len(self.resolved_target_groups) == self.target_groups
        ):
            return "total"
        return "partial"

    def _require_executable(self, allow_candidate: bool) -> None:
        if not self.is_executable and not allow_candidate:
            raise RuntimeError(
                f"refusing non-grounded SJ Glue ({self.status}): "
                f"{self.certificate.reason}"
            )

    def translate_codes(
        self, source_codes: torch.Tensor, *, allow_candidate: bool = False
    ) -> torch.Tensor:
        self._require_executable(allow_candidate)
        _validate_codes(
            source_codes, self.source_groups, self.cardinality, "source_codes"
        )
        translated = torch.full(
            (len(source_codes), self.target_groups),
            -1,
            dtype=source_codes.dtype,
            device=source_codes.device,
        )
        for source_group in self.resolved_source_groups:
            target_group = self.certificate.group_map[source_group]
            values = torch.as_tensor(
                self.certificate.value_maps[source_group],
                dtype=torch.long,
                device=source_codes.device,
            )
            source = source_codes[:, source_group].long()
            active = source >= 0
            translated[active, target_group] = values[source[active]].to(
                translated.dtype
            )
        return translated

    def merge_codes(
        self,
        source_codes: torch.Tensor,
        target_codes: torch.Tensor,
        *,
        conflict: str = "require_equal",
        allow_candidate: bool = False,
    ) -> torch.Tensor:
        if conflict not in _CONFLICT_POLICIES:
            raise ValueError(f"conflict must be one of {sorted(_CONFLICT_POLICIES)}")
        _validate_codes(
            target_codes, self.target_groups, self.cardinality, "target_codes"
        )
        imported = self.translate_codes(source_codes, allow_candidate=allow_candidate)
        if len(imported) != len(target_codes):
            raise ValueError(
                "source_codes and target_codes must have the same batch size"
            )
        merged = target_codes.clone()
        fill = (merged < 0) & (imported >= 0)
        merged[fill] = imported[fill]
        disagree = (merged >= 0) & (imported >= 0) & (merged != imported)
        if bool(disagree.any()):
            if conflict == "require_equal":
                rows, groups = torch.where(disagree)
                sample = list(zip(rows[:4].tolist(), groups[:4].tolist()))
                raise ValueError(f"conflicting shared SJ atoms at {sample}")
            if conflict == "prefer_source":
                merged[disagree] = imported[disagree]
        return merged

    def import_program(
        self,
        target_core: Any,
        source_codes: torch.Tensor,
        *,
        allow_candidate: bool = False,
    ) -> torch.Tensor:
        self._validate_target_core(target_core)
        translated = self.translate_codes(source_codes, allow_candidate=allow_candidate)
        return target_core.program_feature_from_codes(translated)

    def compose_program(
        self,
        target_core: Any,
        source_codes: torch.Tensor,
        target_codes: torch.Tensor,
        *,
        conflict: str = "require_equal",
        allow_candidate: bool = False,
    ) -> torch.Tensor:
        self._validate_target_core(target_core)
        merged = self.merge_codes(
            source_codes,
            target_codes,
            conflict=conflict,
            allow_candidate=allow_candidate,
        )
        return target_core.program_feature_from_codes(merged)

    def _validate_target_core(self, target_core: Any) -> None:
        if (
            int(target_core.G) != self.target_groups
            or int(target_core.C) != self.cardinality
        ):
            raise ValueError(
                "target core shape does not match the SJ Glue certificate"
            )
        if _core_signature(target_core) != self.grammar_signature:
            raise ValueError("target core uses a different SJ program grammar")
        fingerprint = self.certificate.target_graph_hash
        if (
            fingerprint is not None
            and _graph_snapshot(target_core, True).graph_hash != fingerprint
        ):
            raise ValueError("target executing graph changed after Glue certification")

    def summary(self) -> dict[str, Any]:
        record = asdict(self.certificate)
        record.update(
            {
                "status": self.status,
                "compatible": self.is_executable,
                "kind": self.kind,
                "source_groups": self.source_groups,
                "target_groups": self.target_groups,
                "cardinality": self.cardinality,
                "source_coverage": self.source_coverage,
                "target_coverage": self.target_coverage,
                "grammar_signature": list(self.grammar_signature),
                "is_executable": self.is_executable,
                "independent_confirmation": (
                    asdict(self.confirmation_certificate)
                    if self.confirmation_certificate is not None
                    else None
                ),
                "independent_map_agreement": (
                    self.certificate.group_map
                    == self.confirmation_certificate.group_map
                    and self.certificate.value_maps
                    == self.confirmation_certificate.value_maps
                    if self.confirmation_certificate is not None
                    else None
                ),
            }
        )
        return record


def fit_sj_glue(
    base_a: np.ndarray,
    actions_a: Mapping[str, np.ndarray],
    core_a: Any,
    base_b: np.ndarray,
    actions_b: Mapping[str, np.ndarray],
    core_b: Any,
    *,
    train_actions: Sequence[str],
    validation_actions: Sequence[str] = (),
    **certificate_options: Any,
) -> SJGlue:
    signature_a = _core_signature(core_a)
    signature_b = _core_signature(core_b)
    result = fit_operator_sj_glue(
        base_a,
        actions_a,
        core_a,
        base_b,
        actions_b,
        core_b,
        train_actions=train_actions,
        validation_actions=validation_actions,
        **certificate_options,
    )
    signature = signature_a if signature_a == signature_b else signature_a
    return SJGlue(
        certificate=result,
        source_groups=int(core_a.G),
        target_groups=int(core_b.G),
        cardinality=int(core_a.C),
        grammar_signature=signature,
    )


def certify_sj_glue(
    fit_a: SJTransitionEvidence,
    fit_b: SJTransitionEvidence,
    confirmation_a: SJTransitionEvidence,
    confirmation_b: SJTransitionEvidence,
    core_a: Any,
    core_b: Any,
    *,
    train_actions: Sequence[str],
    validation_actions: Sequence[str] = (),
    **certificate_options: Any,
) -> SJGlue:
    fit_options = dict(certificate_options)
    confirmation_options = dict(certificate_options)
    if "seed" in confirmation_options:
        confirmation_options["seed"] = int(confirmation_options["seed"]) + 100_003
    primary = fit_sj_glue(
        fit_a.base,
        fit_a.actions,
        core_a,
        fit_b.base,
        fit_b.actions,
        core_b,
        train_actions=train_actions,
        validation_actions=validation_actions,
        **fit_options,
    )
    confirmation = fit_sj_glue(
        confirmation_a.base,
        confirmation_a.actions,
        core_a,
        confirmation_b.base,
        confirmation_b.actions,
        core_b,
        train_actions=train_actions,
        validation_actions=validation_actions,
        **confirmation_options,
    )
    return SJGlue(
        certificate=primary.certificate,
        source_groups=primary.source_groups,
        target_groups=primary.target_groups,
        cardinality=primary.cardinality,
        grammar_signature=primary.grammar_signature,
        confirmation_certificate=confirmation.certificate,
    )


__all__ = [
    "ACTION_NULL_STATISTIC",
    "ActionOperator",
    "OperatorGlueResult",
    "SJGlue",
    "SJTransitionEvidence",
    "apply_glue_codes",
    "certify_sj_glue",
    "estimate_action_operator",
    "estimate_operators",
    "fit_operator_sj_glue",
    "fit_sj_glue",
    "physical_edges",
]
