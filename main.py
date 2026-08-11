"""
A network's hidden units are organised into GROUPS, which may span layers and
may OVERLAP. Every group owns a random bipolar hypervector (its role). Where
groups intersect, their roles BIND (elementwise product) to form a composite
role. Each group's current state is cleaned up to a FILLER drawn from that
group's bank. The whole network state is exported as one BUNDLED hypervector:
the superposition of role <bind> filler over active groups.

Two networks trained independently, with different group layouts, different
role vectors and different filler banks, are GLUED by 

(1) discovering each side's active role expressions from its bundles, 
(2) searching the base-role correspondence that makes the two discovered inventories agree, and 
(3) reading off the filler bijections. 

A handful of paired observations (anchors) suffices; the requirement scales with filler-bank cardinality, not with the
number of modules.

BUNDLING VS TUPLES: a tuple is a declared schema. The bundle lets a receiver discover structure the sender never declared,
including conjunctive facts arising from group intersection (DEMO). Partial morphisms largely descend from that commitment
"""

from __future__ import annotations
import math, random, itertools, collections
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# We assume bipolar hypervectors. Test FHRR maybe

def bipolar(gen: torch.Generator, D: int) -> torch.Tensor:
    return torch.where(torch.rand(D, generator=gen) > 0.5,
                       torch.ones(D), -torch.ones(D))


def bind(*xs: torch.Tensor) -> torch.Tensor:
    out = xs[0]
    for x in xs[1:]:
        out = out * x
    return out


unbind = bind  # self-inverse


def bundle(xs: Sequence[torch.Tensor], D: int | None = None) -> torch.Tensor:
    if not xs:
        return torch.zeros(D)
    return torch.stack(list(xs)).sum(0) / math.sqrt(len(xs))


def cosine(x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return (M.float() @ x.float()) / (M.float().norm(dim=1) * x.float().norm() + 1e-9)


def cleanup(x: torch.Tensor, bank: torch.Tensor) -> tuple[int, float]:
    """nearest codeword in a bank, with its similarity."""
    z = cosine(x, bank)
    i = int(z.argmax())
    return i, float(z[i])


# How to build and structure groups

@dataclass
class GroupLayout:
    """n_units hidden units partitioned into overlapping groups.

    `members[g]` is the unit index set of group g. Two groups that share a unit
    have an intersection, and the fact carried by that intersection gets the
    BOUND role role_g1 * role_g2.
    """
    n_units: int
    members: list[list[int]]

    @property
    def n_groups(self) -> int:
        return len(self.members)

    def intersections(self) -> list[tuple[int, int]]:
        out = []
        for i in range(self.n_groups):
            for j in range(i + 1, self.n_groups):
                if set(self.members[i]) & set(self.members[j]):
                    out.append((i, j))
        return out

    @staticmethod
    def random(n_units: int, n_groups: int, group_size: int, seed: int = 0) -> "GroupLayout":
        r = random.Random(seed)
        return GroupLayout(n_units,
                           [sorted(r.sample(range(n_units), group_size))
                            for _ in range(n_groups)])


# Role-vectors

class Codebook:
    """Private to one module. Two modules never share these; glue recovers the
    correspondence between them."""

    def __init__(self, layout: GroupLayout, exprs: Sequence[tuple[int, ...]],
                 cards: Sequence[int], seed: int = 0, D: int = 1024):
        self.D = D
        self.layout = layout
        self.exprs = [tuple(sorted(e)) for e in exprs]   # (g,) or (g1, g2)
        self.cards = list(cards)
        g = torch.Generator().manual_seed(seed)
        r = random.Random(seed + 19)

        # one role hypervector per group, in a private (scrambled) order
        self.roles = torch.stack([bipolar(g, D) for _ in range(layout.n_groups)])
        order = list(range(layout.n_groups)); r.shuffle(order)
        self.role_perm = order

        # one filler bank per fact, each with a private value ordering
        self.banks = [torch.stack([bipolar(g, D) for _ in range(c)]) for c in cards]
        self.filler_perm = [r.sample(range(c), c) for c in cards]

    def role_of(self, expr: tuple[int, ...]) -> torch.Tensor:
        """unary group -> its role; intersecting pair -> the BOUND composite role."""
        vs = [self.roles[self.role_perm[g]] for g in expr]
        return bind(*vs)

    def encode(self, facts: Sequence[int]) -> torch.Tensor:
        """export the module state as ONE bundled hypervector.
        facts[k] is the value of fact k, or -1 if that fact is absent."""
        terms = []
        for k, (e, v) in enumerate(zip(self.exprs, facts)):
            if v < 0:
                continue
            terms.append(bind(self.role_of(e), self.banks[k][self.filler_perm[k][v]]))
        return bundle(terms, self.D)

    def decode(self, prog: torch.Tensor) -> list[int]:
        """inverse of encode, for a receiver that already knows this codebook."""
        inv = [{p: i for i, p in enumerate(fp)} for fp in self.filler_perm]
        out = []
        for k, e in enumerate(self.exprs):
            pid, _ = cleanup(unbind(prog, self.role_of(e)), self.banks[k])
            out.append(inv[k][pid])
        return out


# Sparse heads

class SparseHead(nn.Module):
    """One group's readout: a HARD top-k support over the joint hidden state
    (straight-through), then low-rank. The support is what makes the group
    sparse and inspectable; scrambling it destroys the module's behaviour even
    with the weights untouched."""

    def __init__(self, d_in: int, n_out: int, rank: int = 12, k: int = 16):
        super().__init__()
        self.support = nn.Parameter(torch.zeros(d_in))
        self.k = min(k, d_in)
        self.U = nn.Parameter(torch.randn(d_in, rank) * 0.05)
        self.W = nn.Parameter(torch.randn(rank, n_out) * 0.05)

    def mask(self) -> torch.Tensor:
        soft = torch.sigmoid(self.support)
        hard = torch.zeros_like(soft)
        hard[torch.topk(soft, self.k).indices] = 1.0
        return hard + soft - soft.detach()          # straight-through

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh((h * self.mask()) @ self.U) @ self.W

    def active(self) -> list[int]:
        """which hidden units this group actually reads."""
        return torch.topk(torch.sigmoid(self.support), self.k).indices.tolist()

    def mdl(self) -> torch.Tensor:
        s = torch.sigmoid(self.support)
        return self.U.abs().mean() * s.mean() + 0.05 * s.mean()


class SparseVSANet(nn.Module):
    """A module. Trained on its own data, exports bundled hypervectors."""

    def __init__(self, d_in: int, layout: GroupLayout, exprs: Sequence[tuple[int, ...]],
                 cards: Sequence[int], hidden: int = 96, seed: int = 0, D: int = 1024):
        super().__init__()
        torch.manual_seed(seed)
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.heads = nn.ModuleList([SparseHead(hidden, c + 1) for c in cards])
        self.code = Codebook(layout, exprs, cards, seed=seed + 7, D=D)
        self.cards = list(cards)

    def hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.hidden_state(x)
        return [hd(h) for hd in self.heads]

    @torch.no_grad()
    def facts(self, x: torch.Tensor) -> list[list[int]]:
        outs = self.forward(x)
        return [[int(o[i].argmax()) - 1 for o in outs] for i in range(x.shape[0])]

    @torch.no_grad()
    def export(self, x: torch.Tensor) -> list[torch.Tensor]:
        """the module's interface: one bundled hypervector per input."""
        return [self.code.encode(f) for f in self.facts(x)]

    def mdl(self) -> torch.Tensor:
        return sum(hd.mdl() for hd in self.heads)


def train(net: SparseVSANet, X: torch.Tensor, Y: torch.Tensor,
          steps: int = 800, batch: int = 64, lr: float = 3e-3,
          w_mdl: float = 1e-3, seed: int = 0) -> SparseVSANet:
    """Y[:, k] is fact k's value + 1 (0 means absent)."""
    op = torch.optim.AdamW(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ix = torch.randint(0, len(X), (batch,), generator=g)
        outs = net(X[ix])
        loss = sum(F.cross_entropy(o, Y[ix][:, k]) for k, o in enumerate(outs))
        loss = loss + w_mdl * net.mdl()
        op.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        op.step()
    net.eval()
    return net


# Sparse GLUE

def candidate_exprs(n_groups: int) -> list[tuple[int, ...]]:
    return ([(i,) for i in range(n_groups)] +
            [(i, j) for i in range(n_groups) for j in range(i + 1, n_groups)])


def discover(code: Codebook, progs: Sequence[torch.Tensor], top_k: int | None = None,
             gap_cut: float = 0.5) -> list[dict]:
    """Recover a module's active role structure from its bundles ALONE.

    For each candidate expression: unbind it from every program and see whether
    the residue cleans up coherently into a single bank. Real expressions score
    far above the rest, so `top_k` need not be supplied — the score gap gives the
    cutoff. Composite (conjunctive) facts are found here without ever having been
    declared, which is the property a declared factor inventory cannot provide.
    """
    rows = []
    for e in candidate_exprs(code.layout.n_groups):
        role = code.role_of(e)
        ids, banks, scores = [], [], []
        for p in progs:
            q = unbind(p, role)
            best = (-9.0, None, None)
            for bi, M in enumerate(code.banks):
                i, v = cleanup(q, M)
                if v > best[0]:
                    best = (v, bi, i)
            scores.append(best[0]); banks.append(best[1]); ids.append(best[2])
        dom_bank, dom_n = collections.Counter(banks).most_common(1)[0]
        consistency = dom_n / max(1, len(progs))
        rows.append({'expr': e, 'bank': dom_bank, 'ids': ids,
                     'consistency': consistency,
                     'score': float(np.mean(scores)) * consistency})
    rows.sort(key=lambda z: -z['score'])
    if top_k is not None:
        return rows[:top_k]
    # self-determining cutoff
    for i in range(1, len(rows)):                    
        if rows[i]['score'] < gap_cut * rows[i - 1]['score']:
            return rows[:i]
    return rows


def _filler_match(a: Sequence[int], b: Sequence[int]) -> tuple[float, dict]:
    C = collections.Counter(zip(a, b))
    va, vb = sorted(set(a)), sorted(set(b))
    M = np.zeros((len(va), len(vb)))
    for i, x in enumerate(va):
        for j, y in enumerate(vb):
            M[i, j] = C[(x, y)]
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].sum() / max(1, len(a))), {va[i]: vb[j] for i, j in zip(r, c) if M[i, j] > 0}


@dataclass
class Glue:
    """A partial morphism from module A to module B, fitted from paired bundles."""
    role_perm: tuple[int, ...] | None = None
    facts: list[dict] = field(default_factory=list)   # {'a','b','fillers'}
    score: float = -1.0

    def transform(self, prog_a: torch.Tensor, code_a: Codebook,
                  code_b: Codebook) -> torch.Tensor | None:
        """re-express A's bundle in B's private code. Returns None (ABSTAIN) if
        any fact falls outside the recovered correspondence."""
        terms = []
        for f in self.facts:
            a, b, fm = f['a'], f['b'], f['fillers']
            pid, _ = cleanup(unbind(prog_a, code_a.role_of(a['expr'])),
                             code_a.banks[a['bank']])
            if pid not in fm:
                return None
            terms.append(bind(code_b.role_of(b['expr']), code_b.banks[b['bank']][fm[pid]]))
        return bundle(terms, code_b.D)


def fit_glue(code_a: Codebook, progs_a: Sequence[torch.Tensor],
             code_b: Codebook, progs_b: Sequence[torch.Tensor],
             top_k: int | None = None, min_filler_score: float = 0.0) -> Glue:
    """Discover both inventories, then find the base-role correspondence that
    makes them agree. Anchors are the paired programs; the number required
    scales with filler-bank cardinality, not with the number of modules."""
    ra = discover(code_a, progs_a, top_k)
    rb = {r['expr']: r for r in discover(code_b, progs_b, None)}
    n = code_a.layout.n_groups
    best = Glue()
    for perm in itertools.permutations(range(n)):
        total, det, ok = 0.0, [], True
        for a in ra:
            e = tuple(sorted(perm[x] for x in a['expr']))
            b = rb.get(e)
            if b is None or len(code_a.banks[a['bank']]) != len(code_b.banks[b['bank']]):
                ok = False; break
            sc, fm = _filler_match(a['ids'], b['ids'])
            if sc < min_filler_score:
                ok = False; break
            total += sc
            det.append({'a': a, 'b': b, 'fillers': fm})
        if ok and total > best.score:
            best = Glue(role_perm=perm, facts=det, score=total)
    return best


def compose(g_ah: Glue, g_hb: Glue) -> Glue:
    """Φ_AH then Φ_HB. Composition is exact where defined; the domain is the
    INTERSECTION, so coverage can shrink even though accuracy does not."""
    by_expr = {tuple(sorted(f['a']['expr'])): f for f in g_hb.facts}
    out = []
    for f in g_ah.facts:
        mid = tuple(sorted(f['b']['expr']))
        h = by_expr.get(mid)
        if h is None:
            continue
        fm = {u: h['fillers'][w] for u, w in f['fillers'].items() if w in h['fillers']}
        if fm:
            out.append({'a': f['a'], 'b': h['b'], 'fillers': fm})
    perm = None
    if g_ah.role_perm and g_hb.role_perm:
        perm = tuple(g_hb.role_perm[i] for i in g_ah.role_perm)
    return Glue(role_perm=perm, facts=out, score=min(g_ah.score, g_hb.score))




# DEMO (Claude)
if __name__ == "__main__":
    N_GROUPS, HID = 6, 96
    layout = GroupLayout.random(HID, N_GROUPS, group_size=24, seed=0)
    inter = layout.intersections()
# FATOS UNARIOS e conjuntivos
    exprs = [(0,), (1,), (2,)] + inter[:4]           
    cards = [4, 3, 2, 4, 3, 2, 2][:len(exprs)]

    # dois mundos aleatórios 
    # observados independentemente
    rng = np.random.default_rng(0)
    r = random.Random(0)
    S = [[r.randrange(c) for c in cards] for _ in range(3000)]
    din = sum(c + 1 for c in cards)
    proj = rng.normal(0, 1 / math.sqrt(din), (48, din)).astype(np.float32)

    def render(s):
        v = np.zeros(din, dtype=np.float32); o = 0
        for c, val in zip(cards, s):
            v[o + val + 1] = 1; o += c + 1
        return proj @ v

    X = torch.tensor(np.stack([render(s) for s in S]))
    Y = torch.tensor([[v + 1 for v in s] for s in S])

    A = train(SparseVSANet(48, layout, exprs, cards, HID, seed=11), X, Y, seed=1)
    B = train(SparseVSANet(48, layout, exprs, cards, HID, seed=23), X, Y, seed=2)
    print(f"two modules trained independently; group intersections: {inter[:4]}")
    print(f"group 0 reads hidden units {A.heads[0].active()[:8]} ...")

    pa, pb = A.export(X[:150]), B.export(X[:150])
    g = fit_glue(A.code, pa, B.code, pb)
    print(f"discovered {len(g.facts)} facts without being told the arity")
    print(f"  expressions: {[f['a']['expr'] for f in g.facts]}")
    print(f"  true:        {sorted(exprs)}")

    ok = 0
    pa_t, pb_t = A.export(X[200:400]), B.export(X[200:400])
    for p, q in zip(pa_t, pb_t):
        t = g.transform(p, A.code, B.code)
        ok += t is not None and B.code.decode(t) == B.code.decode(q)
    print(f"translation A->B on held-out inputs: {ok}/{len(pa_t)}")
