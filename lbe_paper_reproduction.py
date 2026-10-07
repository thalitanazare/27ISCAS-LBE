"""Reproduction script for "Generalised Lower Bound Error for Multiple
Pseudo-Orbits in Nonlinear Systems" (T. Nazaré, A. M. Lima, E. Nepomuceno,
ISCAS 2027).

This single file contains every calculation used in the Methodology and
Results sections of the paper, and nothing else. It is a simplified,
self-contained version of the authors' working module: the numerical kernel
(the six pseudo-orbit realisations, the exact minimum enclosing ball and the
high-precision reference) is copied without any change to the order of
floating-point operations, so the published crossings are reproduced bit for
bit.

-------------------------------------------------------------------------------
WHAT THE PAPER COMPUTES
-------------------------------------------------------------------------------
1. Pseudo-orbits. Each system (Logistic, Hénon, Lorenz/RK4) is implemented by
   six algebraically equivalent expressions labelled G, H, K, L, M, N. In exact
   arithmetic they are the same map; in finite precision their rounding differs,
   so they produce six different pseudo-orbits p_{i,n} from the same initial
   condition (IC).

2. Generalised LBE. For the nested set K_k = {first k labels}, k = 2..6,

       R_{k,n} = min_c max_{i in K_k} || p_{i,n} - c ||_2 ,

   the radius of the minimum enclosing Euclidean ball. For k = 2 it equals half
   the separation of the pair (G, H), i.e. the original LBE of Nepomuceno and
   Martins (2016). Because the sets are nested, R_{k,n} <= R_{k+1,n}.
   The ball is solved EXACTLY on the stored floating-point states using
   rational arithmetic (fractions.Fraction), so every threshold comparison is a
   certified predicate on the stored inputs.

3. Reference error. A high-precision trajectory q_n (mpmath, 100 and 200
   decimal digits, inputs lifted exactly) gives

       E^ref_n = max_{i=1..6} || p_{i,n} - q_n ||_2 ,

   always over ALL six realisations, so the event to be detected is the same
   for every k. For Lorenz the reference evaluates the same fixed-step RK4 map;
   it is NOT the exact solution of the Lorenz ODE.

4. Detection delay. With the first strict crossing
   tau_eps(u) = min{n : u_n > eps},

       D_k = tau_eps(R_k) - tau_eps(E^ref).

   A missing crossing is right-censored (missing), never zero. Delays are used
   only when the reference crossing is identical at 100 and 200 digits.

5. Tightness. rho_{k,n} = E^ref_n / R_{k,n} >= 1, summarised per IC by its
   median over the steps with E^ref_n inside the tolerance range studied
   (R_{k,n} = 0 < E^ref_n counts as rho = +inf).

6. Robustness design ("design A"). 100 ICs per system, sampled at fixed
   intervals along the binary64 attractor of realisation G after a burn-in.
   They are declared before any outcome is computed and are never replaced.
   Medians at different k always use the same eligible IC cohort, each IC with
   equal weight. The proposed bound is the full ensemble k = 6. The saturation
   size k_sat (smallest k reaching the minimum median delay at eps = 1e-3) is
   descriptive only.

7. Arithmetic formats. binary64 (Python float, 45 tolerances in
   [1e-12, 1e-1]) and binary32 (numpy.float32, 26 tolerances in [1e-6, 1e-1]).
   In binary32 the parameters, IC and step are rounded to binary32 first and the
   reference uses exactly those rounded values.

8. Subsets and orderings (Table II). Exact crossings for all 57 subsets of
   {G,...,N} with at least two elements, and k_sat for all 720 orderings.

9. Cost. Software timings (one warm-up, seven repetitions, rotated k order) of
   generating k realisations plus the binary64 pairwise maximum, compared with
   one 100-digit reference step. These are interpreted-Python timings on the
   machine that runs this script, not hardware estimates; they will differ from
   the paper's Apple M2 / CPython 3.13 values.

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
    python lbe_paper_reproduction.py                  # full paper reproduction
    python lbe_paper_reproduction.py --ics 5          # quick check, 5 ICs/system
    python lbe_paper_reproduction.py --out my_folder  # choose the output folder
    python lbe_paper_reproduction.py --help

All outputs are written to ONE folder, relative to the directory where the
script is run (default: ./lbe_outputs). Nothing is written anywhere else.

Requirements: Python >= 3.10, numpy >= 2.0, pandas, mpmath, matplotlib.
(numpy >= 2 is needed so that float32 scalars are not promoted when combined
with Python integers; the script checks this and stops otherwise.)

Run time. Exact rational geometry and 200-digit references are slower than
ordinary floating point; Lorenz (5500 RK4 steps) dominates at a few seconds
per IC. RUNTIME_PLACEHOLDER Each IC is stored in <out>/cases/ as soon as it
finishes, so an interrupted run resumes where it stopped. Use --ics for a fast
first check.

-------------------------------------------------------------------------------
OUTPUTS (inside the output folder)
-------------------------------------------------------------------------------
cases/                         one .npz per (format, system, IC): cache
detection_<format>.csv         every crossing and delay (IC x k x tolerance)
registry_<format>.csv          completed/failed status of every declared IC
tightness_<format>.csv         per-IC median rho_k
summary_<format>.csv           Table I quantities for each system
subsets_summary.csv            Table II: median-delay range per subset size
orderings_summary.csv          Table II: k_sat over the 720 orderings
cost_by_k.csv                  software timings (Fig. 3, bottom row)
fast_detector_audit.csv        binary64 pairwise detector vs exact predicates
paper_values.csv               every number quoted in the text, by macro name
fig2_representative_crossings.pdf / .svg
fig3_tightness.pdf / .svg

Figure 1 of the paper is an illustrative schematic and is not generated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from bisect import bisect_left
from dataclasses import asdict, dataclass, replace
from fractions import Fraction as Q
from itertools import combinations, permutations
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')  # file output only; no display is needed
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

SCRIPT_VERSION = '1.0'

# =============================================================================
# 1. EXPERIMENTAL SETTINGS (as declared in the paper)
# =============================================================================

LABELS = ('G', 'H', 'K', 'L', 'M', 'N')   # the six realisations, recorded order
K_VALUES = (2, 3, 4, 5, 6)                 # nested sets K_k = LABELS[:k]
SYSTEMS = ('Logistic', 'Henon', 'Lorenz')
DISPLAY = {'Logistic': 'Logistic', 'Henon': 'Hénon', 'Lorenz': 'Lorenz'}

PAPER_EPSILON = 1e-3          # tolerance used for headline numbers
REFERENCE_DIGITS = (100, 200)  # two precisions; crossings must agree
N_ICS = 100                    # ICs per system (design A)

# binary64: 45 log-spaced tolerances in [1e-12, 1e-1]; 1e-3 is a grid point.
TOLERANCES_BINARY64 = tuple(float(v) for v in np.unique(np.r_[np.logspace(-12, -1, 45), PAPER_EPSILON]))
# binary32: 26 log-spaced tolerances in [1e-6, 1e-1].
TOLERANCES_BINARY32 = tuple(float(10.0 ** e) for e in np.round(np.linspace(-6, -1, 26), 10))
TOLERANCES = {'binary64': TOLERANCES_BINARY64, 'binary32': TOLERANCES_BINARY32}

# Attractor IC design: (burn-in, stride) of the binary64 orbit of G.
ATTRACTOR_SAMPLING = {'Logistic': (1000, 101), 'Henon': (1000, 101), 'Lorenz': (5000, 1009)}

# Semantic colours used in every figure.
COLOURS = {'original': '#000000', 'proposed': '#0072B2', 'tolerance': '#FF7427',
           'reference': '#999999', 'binary32': '#999999'}


@dataclass(frozen=True)
class Config:
    """One system: parameters, initial state, step h (1 for maps) and horizon."""
    name: str
    parameters: tuple
    x0: tuple
    h: float
    steps: int

    @property
    def dimension(self):
        return len(self.x0)


def baseline_configs():
    """Representative settings of the paper (Section IV-A)."""
    return {
        'Logistic': Config('Logistic', (3.9,), (0.1,), 1.0, 110),
        'Henon': Config('Henon', (1.4, 0.3), (0.0, 0.0), 1.0, 160),
        'Lorenz': Config('Lorenz', (10.0, 28.0, 8.0 / 3.0), (1.0, 1.0, 1.0), 0.01, 5500),
    }


# =============================================================================
# 2. THE SIX REALISATIONS G, H, K, L, M, N
# =============================================================================
# WARNING: the written order of every expression IS the experiment. Python
# evaluates scalar expressions left to right, so e.g. r*x*(1-x) means
# (r*x)*(1-x). Do not simplify, reassociate or vectorise these lines: any such
# change alters the rounding and therefore the pseudo-orbits.
# The same functions accept Python floats (binary64), numpy.float32 (binary32)
# and mpmath numbers (high-precision reference, which always uses label 'G').

def lorenz_field(state, p, label='G'):
    """Lorenz vector field; M and N use the field of G (they differ in RK4)."""
    x, y, z = state
    s, r, b = p
    if label == 'H':
        return (s * y - s * x, r * x - x * z - y, y * x - z * b)
    if label == 'K':
        return ((y - x) * s, (r - z) * x - y, x * y - b * z)
    if label == 'L':
        return (s * (y - x), r * x - (x * z + y), -(b * z - x * y))
    return (s * (y - x), x * (r - z) - y, x * y - b * z)


def rk4_stages(state, p, h, label='G'):
    """Four classical RK4 stages (a, b, c, d) with the field of `label`."""
    a = lorenz_field(state, p, label)
    u2 = tuple((x + h / 2 * v for x, v in zip(state, a)))
    b = lorenz_field(u2, p, label)
    u3 = tuple((x + h / 2 * v for x, v in zip(state, b)))
    c = lorenz_field(u3, p, label)
    u4 = tuple((x + h * v for x, v in zip(state, c)))
    d = lorenz_field(u4, p, label)
    return ((a, b, c, d), (state, u2, u3, u4))


def step(name, state, p, h, label='G'):
    """One step of system `name` evaluated with realisation `label`."""
    if name == 'Logistic':
        x, = state
        r, = p
        if label == 'G':
            value = r * x * (1 - x)
        elif label == 'H':
            value = r * (x * (1 - x))
        elif label == 'K':
            value = r * x - r * (x * x)
        elif label == 'L':
            value = r * (x - x * x)
        elif label == 'M':
            value = (r - r * x) * x
        else:
            value = r * x - r * x * x
        return (value,)
    if name == 'Henon':
        x, y = state
        a, b = p
        if label == 'G':
            value = 1 - a * (x * x) + y
        elif label == 'H':
            value = 1 + y - a * x * x
        elif label == 'K':
            value = 1 + (y - a * (x * x))
        elif label == 'L':
            value = y + (1 - a * x * x)
        elif label == 'M':
            value = 1 + y - a * (x * x)
        else:
            value = 1 - (a * x * x - y)
        return (value, b * x)
    # Lorenz: one fixed-step RK4 step. M and N change the order of the RK4 sum.
    stages, _ = rk4_stages(state, p, h, label)
    out = []
    for x, a, b, c, d in zip(state, *stages):
        if label == 'M':
            total = a + d + 2 * (b + c)
        elif label == 'N':
            total = a + 2 * b + (2 * c + d)
        else:
            total = a + 2 * b + 2 * c + d
        out.append(x + h / 6 * total)
    return tuple(out)


# =============================================================================
# 3. PSEUDO-ORBITS IN binary64 AND binary32
# =============================================================================

def simulate_binary64(cfg):
    """Return states[n, i, :] for n = 0..steps and realisation i = G..N."""
    states = np.empty((cfg.steps + 1, 6, cfg.dimension))
    p = tuple(map(float, cfg.parameters))
    for i, label in enumerate(LABELS):
        x = tuple(map(float, cfg.x0))
        states[0, i] = x
        for n in range(1, cfg.steps + 1):
            x = step(cfg.name, x, p, float(cfg.h), label)
            if not all((math.isfinite(v) for v in x)):
                raise FloatingPointError(f'{cfg.name}/{label}: nonfinite state at n={n}.')
            states[n, i] = x
    return states


def simulate_binary32(cfg):
    """Same expressions with every operation rounded to IEEE binary32.

    Values are stored in a float64 array, which represents binary32 exactly.
    A type check guarantees that no operation was silently promoted.
    """
    states = np.empty((cfg.steps + 1, 6, cfg.dimension))
    p = tuple(np.float32(v) for v in cfg.parameters)
    h = np.float32(cfg.h)
    for i, label in enumerate(LABELS):
        x = tuple(np.float32(v) for v in cfg.x0)
        states[0, i] = x
        for n in range(1, cfg.steps + 1):
            x = step(cfg.name, x, p, h, label)
            if not all(type(v) is np.float32 for v in x):
                raise TypeError(f'{cfg.name}/{label}: binary32 evaluation was promoted.')
            if not all(np.isfinite(v) for v in x):
                raise FloatingPointError(f'{cfg.name}/{label}: nonfinite binary32 state at n={n}.')
            states[n, i] = x
    return states


def binary32_config(cfg):
    """Parameters, IC and step rounded to binary32 (the reference uses these)."""
    to32 = lambda v: float(np.float32(v))
    return replace(cfg, parameters=tuple(map(to32, cfg.parameters)),
                   x0=tuple(map(to32, cfg.x0)), h=to32(cfg.h))


# =============================================================================
# 4. EXACT MINIMUM ENCLOSING EUCLIDEAN BALL
# =============================================================================
# Every stored state is converted exactly to a rational number. All squared
# distances, squared radii and squared tolerances are then exact rationals, so
# "R_{k,n} > eps" is decided without rounding. Only the plotted radii are
# rounded to float.
#
# Method: the minimum ball is unique and is determined by a support set of at
# most d+1 points (Welzl, 1991). The solver first tries the farthest pair
# (ball with that pair as diameter); if all points are inside, R^2 = diam^2/4.
# Otherwise it searches supports of size 3..d+1 and accepts the first one whose
# circumcentre has non-negative barycentric weights and contains every point.

def to_rationals(frame):
    """Exact rational copy of one frame of states (unscaled Euclidean metric)."""
    return [[Q.from_float(float(x)) / Q.from_float(1.0) for x in row] for row in frame]


def squared_distances(points):
    """Exact matrix of squared Euclidean distances between points."""
    size = len(points)
    D = [[Q(0) for _ in range(size)] for _ in range(size)]
    for i, j in combinations(range(size), 2):
        D[i][j] = D[j][i] = sum(((x - y) ** 2 for x, y in zip(points[i], points[j])))
    return D


def solve_exact(A, b):
    """Small exact Gaussian elimination; returns None for a singular system."""
    size = len(b)
    aug = [list(row) + [v] for row, v in zip(A, b)]
    for j in range(size):
        pivot = next((i for i in range(j, size) if aug[i][j] != 0), None)
        if pivot is None:
            return None
        aug[j], aug[pivot] = (aug[pivot], aug[j])
        den = aug[j][j]
        aug[j] = [v / den for v in aug[j]]
        for i in range(size):
            if i != j and aug[i][j] != 0:
                fac = aug[i][j]
                aug[i] = [u - fac * v for u, v in zip(aug[i], aug[j])]
    return [row[-1] for row in aug]


def try_support(D, ids, k):
    """Circumscribed ball of points `ids`, if it is the minimum ball of all k.

    Works only with squared distances (Gram matrix relative to an anchor).
    Returns (R^2, ids, barycentric weights) or None.
    """
    anchor, rest = (ids[0], ids[1:])
    gram = [[(D[anchor][i] + D[anchor][j] - D[i][j]) / 2 for j in rest] for i in rest]
    alpha = solve_exact(gram, [D[anchor][i] / 2 for i in rest])
    if alpha is None:
        return None
    w = [1 - sum(alpha), *alpha]
    if any((v < 0 for v in w)):
        return None
    radius2 = sum((w[i] * w[j] * D[ids[i]][ids[j]] for i, j in combinations(range(len(ids)), 2)))
    if any((sum((v * D[l][i] for i, v in zip(ids, w))) > 2 * radius2 for l in range(k))):
        return None
    return (radius2, tuple(ids), tuple(w))


def minimum_ball(D, k, dimension, warm=None):
    """Exact minimum enclosing ball of the first k points.

    Returns (R^2, support ids, support weights, diam^2), where diam^2 is the
    largest squared pairwise distance, so diam^2/4 is the best-pair bound.
    `warm` (support of the previous step) only speeds up the search.
    """
    pair = max(combinations(range(k), 2), key=lambda ij: D[ij[0]][ij[1]])
    i, j = pair
    diameter2 = D[i][j]
    if diameter2 == 0:
        return (Q(0), (0,), (Q(1),), diameter2)
    if all((D[l][i] + D[l][j] <= diameter2 for l in range(k))):
        return (diameter2 / 4, pair, (Q(1, 2), Q(1, 2)), diameter2)
    if warm and len(warm) > 2:
        hit = try_support(D, warm, k)
        if hit is not None:
            return (*hit, diameter2)
    for size in range(3, min(dimension + 1, k) + 1):
        for ids in combinations(range(k), size):
            hit = try_support(D, ids, k)
            if hit is not None:
                return (*hit, diameter2)
    raise ArithmeticError('Exact minimum-ball support was not found.')


def exact_sqrt(q):
    """Float square root of a non-negative rational (for plotting/ratios only)."""
    if q == 0:
        return 0.0
    try:
        value = float(q)
        if value > 0 and math.isfinite(value):
            return math.sqrt(value)
    except OverflowError:
        pass
    with mp.workdps(40):
        return float(mp.sqrt(mp.mpf(q.numerator) / q.denominator))


def nested_bounds(states, tolerances, dimension):
    """R_{k,n} and B^pair_{k,n} for k = 2..6, with their first strict crossings.

    Returns radius[n, j], pair[n, j] (float, j = k-2) and cross_ball[j, t],
    cross_pair[j, t] (first n with bound > eps_t, or -1 if never). All
    comparisons use exact rationals. Three invariants are checked at every step:
    R_2 = B^pair_2, R_k >= B^pair_k and R_{k+1} >= R_k.
    """
    count, width = len(states), len(tolerances)
    radius, pair = np.zeros((count, 5)), np.zeros((count, 5))
    cross_ball = np.full((5, width), -1, dtype=int)
    cross_pair = np.full((5, width), -1, dtype=int)
    eps2 = [Q.from_float(float(e)) ** 2 for e in tolerances]   # increasing
    warm = [None] * 5
    for n, frame in enumerate(states):
        D = squared_distances(to_rationals(frame))
        previous = None
        for j, k in enumerate(K_VALUES):
            q, ids, _, diameter2 = minimum_ball(D, k, dimension, warm[j])
            warm[j] = ids
            pairq = diameter2 / 4
            if q < pairq or (previous is not None and q < previous):
                raise ArithmeticError('Exact geometric inequality failed.')
            if j == 0 and q != pairq:
                raise ArithmeticError('k=2 must recover half the pair separation.')
            radius[n, j], pair[n, j] = exact_sqrt(q), exact_sqrt(pairq)
            # bisect_left counts the tolerances with eps^2 < bound^2 (strict >).
            for crossings, bound2 in ((cross_ball, q), (cross_pair, pairq)):
                prefix = bisect_left(eps2, bound2)
                row = crossings[j]
                row[:prefix] = np.where(row[:prefix] < 0, n, row[:prefix])
            previous = q
    return radius, pair, cross_ball, cross_pair


# =============================================================================
# 5. HIGH-PRECISION REFERENCE ERROR
# =============================================================================

def reference_error(cfg, states, tolerances, digits):
    """E^ref_n = max over the six realisations of ||p_{i,n} - q_n||_2.

    q_n iterates the same mathematical map (label G) at `digits` decimal digits,
    starting from the exactly lifted binary64/binary32 inputs. The subtraction
    is done at high precision. Returns (errors[n] as float, crossing[t]).
    For Lorenz this is the same RK4 map, not the exact ODE solution.
    """
    errors = np.zeros(len(states))
    crossing = np.full(len(tolerances), -1, dtype=int)
    with mp.workdps(digits):
        state = tuple((mp.mpf(float(x)) for x in cfg.x0))
        p = tuple((mp.mpf(float(v)) for v in cfg.parameters))
        h = mp.mpf(float(cfg.h))
        eps = [mp.mpf(e) for e in tolerances]
        for n in range(len(states)):
            values = []
            for row in states[n]:
                delta = [mp.mpf(float(x)) - y for x, y in zip(row, state)]
                values.append(mp.sqrt(mp.fsum((v * v for v in delta))))
            value = max(values)
            errors[n] = float(value)
            for a, threshold in enumerate(eps):
                if crossing[a] < 0 and value > threshold:
                    crossing[a] = n
            if n < cfg.steps:
                state = step(cfg.name, state, p, h)
    return errors, crossing


# =============================================================================
# 6. AUDIT OF THE FAST (binary64) PAIRWISE DETECTOR
# =============================================================================

def fast_detector_audit(states, cross_pair, tolerances, dimension):
    """Compare the cheap binary64 pairwise-maximum detector with exact crossings.

    For each k and tolerance, the float detector's first crossing must equal
    the exact one; in addition, the exact and float predicates are compared at
    each crossing sample and its predecessor. Returns (comparisons,
    crossing disagreements, boundary samples, boundary disagreements).
    """
    maximum = np.zeros(len(states))
    comparisons = crossing_bad = samples_total = boundary_bad = 0
    for k in K_VALUES:
        for i in range(k - 1):
            diff = states[:, i] - states[:, k - 1]
            distance = np.zeros(len(states))
            for axis in range(dimension):
                distance = distance + diff[:, axis] * diff[:, axis]
            maximum = np.maximum(maximum, distance)
        bound2 = maximum / 4
        for t, e in enumerate(tolerances):
            hits = np.flatnonzero(bound2 > float(e) * float(e))
            fast = int(hits[0]) if len(hits) else -1
            exact = int(cross_pair[k - 2, t])
            samples = sorted({n for c in (fast, exact) for n in (c - 1, c) if 0 <= n < len(states)})
            for n in samples:
                D = squared_distances(to_rationals(states[n, :k]))
                q = max(D[i][j] for i, j in combinations(range(k), 2)) / 4
                boundary_bad += int(bool(bound2[n] > float(e) * float(e)) != bool(q > Q.from_float(float(e)) ** 2))
            comparisons += 1
            crossing_bad += int(fast != exact)
            samples_total += len(samples)
    return comparisons, crossing_bad, samples_total, boundary_bad


# =============================================================================
# 7. ONE CASE = ONE SYSTEM, ONE IC, ONE ARITHMETIC FORMAT
# =============================================================================

def run_case(cfg, precision, case_id, out_dir):
    """Simulate, compute exact bounds, references and the detector audit.

    The result is cached in out_dir/cases/; a cached file is reused only if its
    signature (configuration, tolerances, digits, script version) matches.
    """
    if precision == 'binary32':
        cfg = binary32_config(cfg)
    tolerances = TOLERANCES[precision]
    signature = json.dumps({'config': asdict(cfg), 'precision': precision, 'tolerances': list(tolerances),
                            'digits': list(REFERENCE_DIGITS), 'version': SCRIPT_VERSION}, sort_keys=True)
    token = hashlib.sha256(signature.encode()).hexdigest()[:16]
    path = out_dir / 'cases' / precision / cfg.name / f'{case_id}_{token}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z['signature']) == signature:
                return {key: z[key].copy() for key in z.files} | {'config': cfg, 'case_id': case_id}
    states = simulate_binary64(cfg) if precision == 'binary64' else simulate_binary32(cfg)
    radius, pair, cross_ball, cross_pair = nested_bounds(states, tolerances, cfg.dimension)
    low_err, low_cross = reference_error(cfg, states, tolerances, REFERENCE_DIGITS[0])
    high_err, high_cross = reference_error(cfg, states, tolerances, REFERENCE_DIGITS[1])
    audit = np.array(fast_detector_audit(states, cross_pair, tolerances, cfg.dimension))
    result = dict(radius=radius, pair=pair, cross_ball=cross_ball, cross_pair=cross_pair,
                  reference=high_err, ref_cross_low=low_cross, ref_cross_high=high_cross,
                  audit=audit, signature=np.array(signature))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return result | {'config': cfg, 'case_id': case_id}


def detection_rows(case, precision):
    """Crossings and delays D_k for every k and tolerance of one case.

    D_k = n_ball - n_ref only when the reference crossing exists and is identical
    at both precisions and the ball crosses; otherwise NaN (missing/censored).
    """
    cfg, rows = case['config'], []
    low, high = case['ref_cross_low'], case['ref_cross_high']
    for j, k in enumerate(K_VALUES):
        for t, eps in enumerate(TOLERANCES[precision]):
            nb, npair, nr = case['cross_ball'][j, t], case['cross_pair'][j, t], high[t]
            stable = bool(low[t] == high[t])
            ok = stable and nr >= 0
            rows.append({'model': cfg.name, 'precision': precision, 'design_id': case['case_id'],
                         'x0': json.dumps(list(cfg.x0)), 'k': k, 'epsilon': float(eps),
                         'n_ball': nb if nb >= 0 else np.nan, 'n_pair': npair if npair >= 0 else np.nan,
                         'n_ref': nr if nr >= 0 else np.nan, 'reference_stable': stable,
                         'D_ball': nb - nr if ok and nb >= 0 else np.nan,
                         'D_pair': npair - nr if ok and npair >= 0 else np.nan})
    return rows


def tightness_rows(case, precision, window):
    """Per-IC median of rho_{k,n} = E^ref_n / R_{k,n} over steps with E^ref in `window`.

    Arithmetic median of the ratios; R = 0 < E^ref gives rho = +inf (kept).
    """
    R, E = case['radius'], case['reference']
    mask = (E >= window[0]) & (E <= window[1])
    rows = []
    for j, k in enumerate(K_VALUES):
        with np.errstate(divide='ignore'):
            rho = E[mask] / R[mask, j]
        rows.append({'model': case['config'].name, 'precision': precision, 'design_id': case['case_id'], 'k': k,
                     'n_steps': int(mask.sum()), 'n_zero_radius': int((R[mask, j] == 0).sum()),
                     'median_rho': float(np.median(rho)) if mask.any() else np.nan,
                     'min_rho': float(rho.min()) if mask.any() else np.nan})
    return rows


# =============================================================================
# 8. INITIAL-CONDITION DESIGN
# =============================================================================

def attractor_initial_conditions(configs, count):
    """ICs spread along each attractor, declared before any outcome.

    The binary64 orbit of G from the representative IC is advanced for a
    burn-in and then sampled every `stride` steps. No detection result is used
    to choose, keep or replace a point.
    """
    design = {}
    for name, cfg in configs.items():
        burn_in, stride = ATTRACTOR_SAMPLING[name]
        p, x, points = tuple(map(float, cfg.parameters)), tuple(map(float, cfg.x0)), []
        for n in range(1, burn_in + stride * count + 1):
            x = step(name, x, p, float(cfg.h), 'G')
            if n > burn_in and (n - burn_in) % stride == 0:
                points.append(tuple(map(float, x)))
        design[name] = points[:count]
    return design


def run_design(configs, design, precision, out_dir):
    """Run every declared IC; failures are recorded, never replaced."""
    detection, tightness, registry, audit = [], [], [], []
    window = (TOLERANCES[precision][0], TOLERANCES[precision][-1])
    for name, cfg in configs.items():
        for i, x0 in enumerate(design[name]):
            case_id = f'attractor_{i:03d}'
            info = {'model': name, 'precision': precision, 'design_id': case_id,
                    'x0': json.dumps(list(x0)), 'status': 'failed', 'error': ''}
            try:
                case = run_case(replace(cfg, x0=tuple(map(float, x0))), precision, case_id, out_dir)
                detection += detection_rows(case, precision)
                tightness += tightness_rows(case, precision, window)
                audit.append({'model': name, 'precision': precision, 'design_id': case_id,
                              **dict(zip(('comparisons', 'crossing_disagreements', 'boundary_samples',
                                          'boundary_disagreements'), map(int, case['audit'])))})
                info['status'] = 'completed'
            except (FloatingPointError, ArithmeticError, ValueError, OverflowError, TypeError) as exc:
                info['error'] = f'{type(exc).__name__}: {exc}'
                warnings.warn(f'{precision}/{name}/{case_id} kept as failed: {exc}')
            registry.append(info)
            print(f'  {precision} {name}: IC {i + 1}/{len(design[name])}', end='\r', flush=True)
        print(f'  {precision} {name}: {len(design[name])} ICs done.          ', flush=True)
    return (pd.DataFrame(detection), pd.DataFrame(tightness), pd.DataFrame(registry), pd.DataFrame(audit))


# =============================================================================
# 9. SUMMARIES (TABLE I, TOLERANCE ROBUSTNESS, TIGHTNESS)
# =============================================================================

def cohort(detection, model, epsilon):
    """D_k pivot (IC x k) on the common eligible cohort: ICs with D_k for all k."""
    rows = detection.loc[(detection['model'] == model) & np.isclose(detection['epsilon'], epsilon, rtol=1e-12, atol=0)]
    return rows.pivot(index='design_id', columns='k', values='D_ball').reindex(columns=list(K_VALUES)).dropna()


def smallest_k_at_minimum(medians):
    """Smallest k attaining the minimum median delay (k_sat, descriptive)."""
    return int(medians.index[np.isclose(medians, medians.min(), rtol=0, atol=1e-12)].min())


def summarise(detection, tightness, epsilon=PAPER_EPSILON, k_full=6):
    """Table I and the text values for one arithmetic format.

    Percentage reduction is the MEDIAN OF PER-IC percentages 100(D2-D6)/D2 over
    ICs with D2 > 0; it is not computed from the two medians.
    """
    rows = []
    for model in SYSTEMS:
        if model not in set(detection['model']):
            continue
        pivot = cohort(detection, model, epsilon)
        d2, d6 = pivot[2], pivot[k_full]
        valid = d2 > 0
        pct = 100 * (d2[valid] - d6[valid]) / d2[valid]
        medians = pivot.median()
        grid = sorted(detection.loc[detection['model'] == model, 'epsilon'].unique())
        tol_better = sum(int(cohort(detection, model, e)[k_full].median() < cohort(detection, model, e)[2].median())
                         for e in grid if len(cohort(detection, model, e)))
        rho = tightness.loc[tightness['model'] == model].groupby('k')['median_rho'].median()
        rows.append({'model': model, 'n_eligible': len(pivot),
                     **{f'median_D{k}': float(medians[k]) for k in K_VALUES},
                     'q25_D2': float(d2.quantile(.25)), 'q75_D2': float(d2.quantile(.75)),
                     'q25_D6': float(d6.quantile(.25)), 'q75_D6': float(d6.quantile(.75)),
                     'median_reduction_pct': float(pct.median()) if len(pct) else np.nan,
                     'n_reduction': int(len(pct)), 'median_abs_reduction': float((d2 - d6).median()),
                     'n_better': int((d6 < d2).sum()), 'n_equal': int((d6 == d2).sum()),
                     'n_D2_zero': int((d2 == 0).sum()), 'k_sat': smallest_k_at_minimum(medians),
                     'tolerances': len(grid), 'tolerances_median_better': tol_better,
                     **{f'median_rho_k{k}': float(rho[k]) for k in K_VALUES},
                     'rho_factor_2_to_6': float(rho[2] / rho[k_full]),
                     'min_rho_any_step': float(tightness.loc[tightness['model'] == model, 'min_rho'].min()),
                     'ball_pair_crossings_identical': bool(np.array_equal(
                         detection.loc[detection['model'] == model, 'n_ball'].fillna(-1),
                         detection.loc[detection['model'] == model, 'n_pair'].fillna(-1)))})
    return pd.DataFrame(rows)


def ball_versus_pair(detection, model='Lorenz'):
    """Cases (k >= 3) where the exact ball crosses before the same-set best pair."""
    rows = detection.loc[(detection['model'] == model) & (detection['k'] >= 3)]
    diff = rows.loc[rows['n_ball'].fillna(-1) != rows['n_pair'].fillna(-1)]
    early = diff['n_pair'] - diff['n_ball']
    return {'cases': len(diff), 'total': len(rows),
            'min_steps': int(early.min()) if len(diff) else 0, 'max_steps': int(early.max()) if len(diff) else 0,
            'max_epsilon': float(diff['epsilon'].max()) if len(diff) else np.nan}


# =============================================================================
# 10. SUBSETS AND ORDERINGS (TABLE II, binary64)
# =============================================================================

def censored_median(values):
    """Median with right-censored values (NaN) as +inf; NaN if the middle is censored."""
    v = np.sort(np.where(np.isnan(values), np.inf, values))
    m = len(v)
    middle = v[m // 2] if m % 2 else 0.5 * (v[m // 2 - 1] + v[m // 2])
    return float(middle) if np.isfinite(middle) else np.nan


def subset_crossings(cfg, states, cross_ball, ref_low, ref_high, case_id, tolerances):
    """Exact ball crossings for all 57 subsets of the six realisations.

    Pair crossings are exact rational predicates. The ball can never cross
    later than the pairwise maximum of the same subset, and by Jung's theorem
    R^2 <= m/(2(m+1)) diam^2 with m = min(d, |S|-1). The exact ball is therefore
    evaluated only on earlier frames where that bound (screened in binary64 with
    a 0.999 margin) can exceed eps^2. Nested subsets must reproduce the crossings
    already computed by nested_bounds().
    """
    pairs = list(combinations(range(6), 2))
    subsets = [S for k in K_VALUES for S in combinations(range(6), k)]
    thresholds = [Q.from_float(float(e)) ** 2 for e in tolerances]
    pair_cross = np.full((15, len(tolerances)), -1, dtype=int)
    frames = {}
    for n, frame in enumerate(states):
        if (pair_cross >= 0).all():
            break
        D = squared_distances(to_rationals(frame))
        frames[n] = D
        for p, (i, j) in enumerate(pairs):
            stop = bisect_left(thresholds, D[i][j] / 4)
            missing = pair_cross[p, :stop] < 0
            pair_cross[p, :stop][missing] = n
    pair_cross = np.where(pair_cross < 0, np.nan, pair_cross.astype(float))
    diam2 = {(i, j): np.sum((states[:, i] - states[:, j]) ** 2, axis=1) for i, j in pairs}
    exact = {}

    def exact_ball(n, S):
        if (n, S) not in exact:
            if n not in frames:
                frames[n] = squared_distances(to_rationals(states[n]))
            D = frames[n]
            exact[n, S] = minimum_ball([[D[i][j] for j in S] for i in S], len(S), cfg.dimension)[0]
        return exact[n, S]

    stable = ref_low == ref_high
    rows = []
    for S in subsets:
        first = np.vstack([pair_cross[pairs.index((i, j))] for i, j in combinations(S, 2)])
        first = np.where(np.isnan(first), np.inf, first).min(axis=0)
        first[~np.isfinite(first)] = np.nan
        ball = first.copy()
        m = min(cfg.dimension, len(S) - 1)
        if m > 1:
            upper = m / (2 * (m + 1)) * np.max(np.vstack([diam2[p] for p in combinations(S, 2)]), axis=0)
            for t, e in enumerate(tolerances):
                stop = int(first[t]) if np.isfinite(first[t]) else len(states)
                for n in np.flatnonzero(upper[:stop] > 0.999 * e * e):
                    if exact_ball(int(n), S) > thresholds[t]:
                        ball[t] = n
                        break
        if S == tuple(range(len(S))):
            saved = np.where(cross_ball[len(S) - 2] < 0, np.nan, cross_ball[len(S) - 2])
            if not np.array_equal(ball, saved, equal_nan=True):
                raise ValueError(f'{cfg.name}/{case_id}: nested subset disagrees with stored crossings.')
        name = ''.join(LABELS[i] for i in S)
        for t, e in enumerate(tolerances):
            ok = bool(stable[t]) and ref_high[t] >= 0
            rows.append({'model': cfg.name, 'design_id': case_id, 'epsilon': e, 'subset': name, 'k': len(S),
                         'n_ball': ball[t], 'n_pair_max': first[t],
                         'delay': ball[t] - ref_high[t] if ok and np.isfinite(ball[t]) else np.nan,
                         'censored': bool(ok and not np.isfinite(ball[t]))})
    return rows


def subsets_and_orderings(configs, design, out_dir, epsilon=PAPER_EPSILON):
    """Table II: median-delay range over subsets of each size and k_sat over 720 orders."""
    tolerances = TOLERANCES['binary64']
    rows = []
    for name, cfg in configs.items():
        for i, x0 in enumerate(design[name]):
            case_id = f'attractor_{i:03d}'
            c = replace(cfg, x0=tuple(map(float, x0)))
            case = run_case(c, 'binary64', case_id, out_dir)     # cached: crossings only
            states = simulate_binary64(c)                        # deterministic re-simulation
            rows += subset_crossings(c, states, case['cross_ball'], case['ref_cross_low'],
                                     case['ref_cross_high'], case_id, tolerances)
            print(f'  subsets {name}: IC {i + 1}/{len(design[name])}', end='\r', flush=True)
        print(f'  subsets {name}: done.                 ', flush=True)
    frame = pd.DataFrame(rows)
    summary, orders = [], []
    for model in configs:
        rows_e = frame.loc[(frame['model'] == model) & np.isclose(frame['epsilon'], epsilon, rtol=1e-12, atol=0)]
        pivot = rows_e.pivot(index='design_id', columns='subset', values='delay')
        medians = {S: censored_median(pivot[S].to_numpy(float)) for S in pivot.columns}
        for k in K_VALUES:
            values = np.array([medians[S] for S in medians if len(S) == k])
            summary.append({'model': model, 'epsilon': epsilon, 'k': k, 'n_ics': len(pivot), 'n_subsets': len(values),
                            'n_undefined': int(np.isnan(values).sum()),
                            'min_median': np.nanmin(values) if np.isfinite(values).any() else np.nan,
                            'max_median': np.nanmax(values) if np.isfinite(values).any() else np.nan})
        counts = np.zeros(7, dtype=int)
        for order in permutations(LABELS):
            curve = np.array([medians[''.join(sorted(order[:k], key=LABELS.index))] for k in K_VALUES])
            ranked = np.where(np.isnan(curve), np.inf, curve)
            counts[int(np.flatnonzero(ranked == ranked.min())[0] + 2)] += 1
        orders.append({'model': model, 'epsilon': epsilon, 'n_orders': 720,
                       **{f'k_sat_{k}': int(counts[k]) for k in K_VALUES}})
    return pd.DataFrame(summary), pd.DataFrame(orders)


# =============================================================================
# 11. SOFTWARE COST (FIG. 3 BOTTOM ROW AND SECTION IV-F)
# =============================================================================

def measure_cost(configs, repeats=7, geometry_samples=256, timing_steps=1000, exact_repeats=5, exact_samples=64):
    """Per-step software cost of generating k realisations plus pairwise evaluation.

    On the representative baseline only. Each repetition times, for every k
    (in rotated order), the generation of k binary64 realisations and the
    binary64 pairwise maximum on the same frames; the total is their sum. One
    warm-up repetition is excluded; medians and quartiles are reported. A
    100-digit reference step is timed in the same repetitions. Separately, the
    exact rational ball for k = 6 is compared with the exact pairwise maximum.
    Machine-dependent: compare ratios, not absolute values, with the paper.
    """
    rows = []
    for model, cfg in configs.items():
        states = simulate_binary64(cfg)
        frames = [tuple(tuple(map(float, row)) for row in frame) for frame in
                  states[np.unique(np.linspace(0, cfg.steps, geometry_samples, dtype=int))]]
        steps = min(cfg.steps, timing_steps)
        measurements, refs = {k: [] for k in K_VALUES}, []
        for repeat in range(repeats + 1):
            with mp.workdps(100):
                x = tuple(mp.mpf(float(v)) for v in cfg.x0)
                params = tuple(mp.mpf(float(v)) for v in cfg.parameters)
                h = mp.mpf(float(cfg.h))
                start = perf_counter()
                for _ in range(steps):
                    x = step(model, x, params, h)
                reference = (perf_counter() - start) / steps
            ks = list(K_VALUES)
            ks = ks[repeat % 5:] + ks[:repeat % 5]   # rotate to avoid position effects
            for k in ks:
                start = perf_counter()
                for label in LABELS[:k]:
                    x = tuple(map(float, cfg.x0))
                    for _ in range(steps):
                        x = step(model, x, cfg.parameters, cfg.h, label)
                generation = (perf_counter() - start) / steps
                start = perf_counter()
                for frame in frames:
                    value = max(sum((a - b) * (a - b) for a, b in zip(frame[i], frame[j]))
                                for i, j in combinations(range(k), 2)) / 4
                geometry = (perf_counter() - start) / len(frames)
                if repeat:
                    measurements[k].append((generation, geometry, generation + geometry))
            if repeat:
                refs.append(reference)
        # Exact audit cost at k = 6: exact pairwise maximum versus exact ball.
        exact_frames = states[np.unique(np.linspace(0, cfg.steps, exact_samples, dtype=int))]
        timings = {'pairs': [], 'ball': []}
        for repeat in range(exact_repeats + 1):
            for key in ('pairs', 'ball'):
                start = perf_counter()
                for frame in exact_frames:
                    D = squared_distances(to_rationals(frame))
                    value = (minimum_ball(D, 6, cfg.dimension)[0] if key == 'ball'
                             else max(D[i][j] for i, j in combinations(range(6), 2)) / 4)
                if repeat:
                    timings[key].append((perf_counter() - start) / len(exact_frames))
        ball_overhead = 100 * (np.median(timings['ball']) / np.median(timings['pairs']) - 1)
        baseline = float(np.median(np.array(measurements[2])[:, 2]))
        for k, values in measurements.items():
            a = np.array(values) * 1e6
            rows.append({'model': model, 'k': k, 'generation_us': np.median(a[:, 0]),
                         'pairwise_us': np.median(a[:, 1]), 'total_us': np.median(a[:, 2]),
                         'q25_us': np.percentile(a[:, 2], 25), 'q75_us': np.percentile(a[:, 2], 75),
                         'ratio_to_pair': np.median(a[:, 2]) / (baseline * 1e6),
                         'reference_us': np.median(refs) * 1e6,
                         'ratio_to_reference': np.median(a[:, 2]) / (np.median(refs) * 1e6),
                         'exact_ball_overhead_pct_k6': ball_overhead,
                         'repeats': repeats, 'python': sys.version.split()[0], 'platform': platform.platform()})
        print(f'  cost {model}: done.', flush=True)
    return pd.DataFrame(rows)


# =============================================================================
# 12. FIGURES (vector PDF and SVG, 9 pt serif, IEEE column widths)
# =============================================================================

def figure_style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                         'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'path',
                         'svg.hashsalt': 'generalised-lbe-paper', 'legend.frameon': False})


def save_vector(fig, out_dir, stem):
    for ext in ('pdf', 'svg'):
        fig.savefig(out_dir / f'{stem}.{ext}', format=ext, facecolor='white')
    plt.close(fig)


def figure_crossings(baselines, out_dir, epsilon=PAPER_EPSILON, k=6):
    """Fig. 2: reference error, original pair (G,H) and k = 6 for the representative ICs."""
    figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))
    fig.subplots_adjust(left=.073, right=.99, bottom=.23, top=.73, wspace=.41)
    for letter, model, ax in zip('abc', SYSTEMS, axes):
        case = baselines[model]
        t = TOLERANCES['binary64'].index(epsilon)
        n_ref, n2, nk = (int(case['ref_cross_high'][t]), int(case['cross_ball'][0, t]), int(case['cross_ball'][k - 2, t]))
        size = len(case['reference'])
        x = np.arange(size) * case['config'].h if model == 'Lorenz' else np.arange(size)
        crossings = [n_ref, n2, nk]
        if min(crossings) < 0:
            start, stop = 0, size - 1
        else:
            span = max(crossings) - min(crossings)
            start = max(0, min(crossings) - max(10, math.ceil(.6 * span)))
            stop = min(size - 1, max(crossings) + max(8, math.ceil(.4 * span)))
        series = [(case['reference'], n_ref, COLOURS['reference'], 'D'),
                  (case['radius'][:, 0], n2, COLOURS['original'], 'o'),
                  (case['radius'][:, k - 2], nk, COLOURS['proposed'], 's')]
        visible = [epsilon]
        for y, n, colour, marker in series:
            y = np.asarray(y, dtype=float)
            part = y[start:stop + 1]
            visible.extend(part[np.isfinite(part) & (part > 0)])
            ax.plot(x, np.where(y > 0, y, np.nan), color=colour, linewidth=1.25)
            if 0 <= n and start <= n <= stop:
                ax.axvline(x[n], color=colour, linewidth=.6, alpha=.55)
                ax.plot(x[n], y[n], marker=marker, color=colour, markeredgecolor='white', markeredgewidth=.45, zorder=6)
        ax.axhline(epsilon, color=COLOURS['tolerance'], linestyle='--', linewidth=1)
        ax.set(yscale='log', xlim=(x[start], x[stop]), ylim=(max(np.finfo(float).tiny, min(visible) / 2), max(visible) * 2),
               xlabel='Model time $t=nh$' if model == 'Lorenz' else 'Iteration $n$')
        ax.set_ylabel('Lower bound / error' if letter == 'a' else '')
        ax.text(0, 1.055, f'({letter}) {DISPLAY[model]}', transform=ax.transAxes, ha='left', va='bottom')
        ax.grid(axis='y', color='0.9', linewidth=.4)
    handles = [Line2D([], [], color=COLOURS['original'], label='Original LBE, pair $(G,H)$'),
               Line2D([], [], color=COLOURS['proposed'], label=f'Generalised LBE, $k={k}$'),
               Line2D([], [], color=COLOURS['tolerance'], linestyle='--', label='Tolerance'),
               Line2D([], [], color=COLOURS['reference'], label='Reference error')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .998), ncol=4)
    save_vector(fig, out_dir, 'fig2_representative_crossings')


def _tightness_boxes(ax, tight64, tight32, model, ks, widths=.30, line=1.0):
    for tight, offset, colour in ((tight64, -.17, COLOURS['proposed']), (tight32, .17, COLOURS['binary32'])):
        if tight is None or not len(tight):
            continue
        groups = [tight.loc[(tight.model == model) & (tight.k == k), 'median_rho'].to_numpy() for k in ks]
        box = ax.boxplot(groups, positions=np.asarray(ks) + offset, widths=widths, whis=(5, 95), patch_artist=True,
                         showfliers=False, medianprops={'color': '#000000', 'linewidth': line},
                         whiskerprops={'color': colour, 'linewidth': line * .8}, capprops={'color': colour, 'linewidth': line * .8})
        for patch in box['boxes']:
            patch.set(facecolor=colour, edgecolor=colour, alpha=.30)


def figure_tightness(tight64, tight32, detection64, cost, out_dir, epsilon=PAPER_EPSILON):
    """Fig. 3. Top: rho_k over ICs (box = IQR, whiskers = 5th-95th percentiles).
    Bottom: binary64 median delay (Q1-Q3 bars) versus cost relative to (G,H);
    if cost was skipped, versus k."""
    figure_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.2))
    fig.subplots_adjust(left=.08, right=.985, bottom=.14, top=.87, wspace=.30, hspace=1.35)
    for col, model in enumerate(SYSTEMS):
        ax = axes[0, col]
        _tightness_boxes(ax, tight64, tight32, model, list(K_VALUES))
        ax.axhline(1, color=COLOURS['reference'], linestyle='--', linewidth=.8)
        ax.set(yscale='log', xticks=list(K_VALUES), xlim=(1.4, 6.6), xlabel='Number of pseudo-orbits $k$',
               title=f'({"abc"[col]}) {DISPLAY[model]}')
        ax.set_xticklabels([str(k) for k in K_VALUES])
        if col == 0:
            ax.set_ylabel(r'$\rho_k=E^{\mathrm{ref}}/R_k$')
            # Linear enlargement of the Logistic boxes near the floor rho = 1.
            zoom = ax.inset_axes([.43, .40, .54, .56])
            _tightness_boxes(zoom, tight64, tight32, model, [4, 5, 6], line=.8)
            zoom.set(ylim=(.95, 4.15), xlim=(3.5, 6.5), xticks=(4, 5, 6), yticks=(1, 2, 4))
            zoom.set_xticklabels(['4', '5', '6'])
            zoom.tick_params(labelsize=6.5, pad=1, length=2)
            zoom.axhline(1, color=COLOURS['reference'], linestyle='--', linewidth=.5)
            ax.add_patch(Rectangle((3.5, .95), 3.0, 3.2, facecolor='#DEEAF2', edgecolor='#66889E',
                                   alpha=.55, linewidth=.6, zorder=.5))
            for source, target in (((3.5, 4.15), (0, 0)), ((6.5, 4.15), (1, 0))):
                ax.add_artist(ConnectionPatch(source, target, coordsA=ax.transData, coordsB=zoom.transAxes,
                                              color='#66889E', linewidth=.5, alpha=.7, zorder=1))
        ax = axes[1, col]
        p = cohort(detection64, model, epsilon)
        med, q1, q3 = p.median(), p.quantile(.25), p.quantile(.75)
        if cost is not None:
            xs = cost.loc[cost.model == model].sort_values('k')['ratio_to_pair'].to_numpy()
            xlabel = 'Cost / original-pair cost'
        else:
            xs, xlabel = np.array(K_VALUES, dtype=float), 'Number of pseudo-orbits $k$'
        ax.errorbar(xs, med, yerr=[med - q1, q3 - med], fmt='o-', color=COLOURS['proposed'],
                    capsize=2, linewidth=1, markersize=3)
        upper = max(float(q3.max()), 1.0)
        ax.set_xlim(float(xs.min()) - .5, float(xs.max()) + .5)
        ax.set_ylim(-.12 * upper, 1.18 * upper)
        if cost is not None:
            # k is shown on an auxiliary top axis so that no label covers the data.
            top = ax.secondary_xaxis('top')
            top.set_xticks(xs, labels=[str(k) for k in K_VALUES])
            top.tick_params(labelsize=8, pad=1, length=2, colors='#A85400')
            top.set_xlabel('$k$', color='#A85400', labelpad=1, fontsize=8)
            for position in xs:
                ax.axvline(position, color='#DBA36D', linestyle=(0, (3, 3)), linewidth=.45, alpha=.65, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_title(f'({"def"[col]}) {DISPLAY[model]}', loc='left', y=1.0, pad=17)
        if col == 0:
            ax.set_ylabel('Detection delay (steps)')
        ax.grid(alpha=.25)
    fig.legend(handles=[Patch(facecolor=COLOURS['proposed'], alpha=.30, label='binary64'),
                        Patch(facecolor=COLOURS['binary32'], alpha=.30, label='binary32')],
               loc='upper center', ncol=2)
    save_vector(fig, out_dir, 'fig3_tightness')


# =============================================================================
# 13. NUMBERS QUOTED IN THE PAPER
# =============================================================================

def fmt(value):
    """Integers without decimals, otherwise one decimal (as in the paper)."""
    return '--' if not np.isfinite(value) else f'{value:.1f}'.rstrip('0').rstrip('.')


def paper_values(summaries, representative, ball32, subsets, orders, cost, audit):
    """Every quantity cited in the text, named after the LaTeX macro it fills."""
    rows = []
    add = lambda macro, value, meaning: rows.append({'macro': macro, 'value': value, 'meaning': meaning})
    for model in SYSTEMS:
        for tag, fmt_name in (('B', 'binary64'), ('S', 'binary32')):
            s = summaries.get(fmt_name)
            if s is None or model not in set(s['model']):
                continue
            r = s.set_index('model').loc[model]
            add(f'A{tag}Red{model}', f"{r['median_reduction_pct']:.1f}", f'{fmt_name}: median per-IC % reduction D2->D6')
            add(f'A{tag}Better{model}', int(r['n_better']), f'{fmt_name}: ICs with D6 < D2')
            add(f'A{tag}Equal{model}', int(r['n_equal']), f'{fmt_name}: ICs with D6 = D2')
            add(f'A{tag}Zero{model}', int(r['n_D2_zero']), f'{fmt_name}: ICs with D2 = 0')
            add(f'A{tag}Abs{model}', fmt(r['median_abs_reduction']), f'{fmt_name}: median of D2 - D6')
            add(f'A{tag}Tol{model}', int(r['tolerances']), f'{fmt_name}: tolerances tested')
            add(f'A{tag}TolBetter{model}', int(r['tolerances_median_better']), f'{fmt_name}: tolerances with median D6 < D2')
            add(f'A{tag}Sat{model}', int(r['k_sat']), f'{fmt_name}: k_sat at 1e-3 (descriptive)')
            for k in K_VALUES:
                add(f'{fmt_name}_median_D{k}_{model}', fmt(r[f'median_D{k}']), f'{fmt_name}: Table I median D_{k}')
            add(f'A{tag}RhoTwo{model}', f"{r['median_rho_k2']:.1f}", f'{fmt_name}: median rho, k=2')
            add(f'A{tag}RhoSix{model}', f"{r['median_rho_k6']:.2f}", f'{fmt_name}: median rho, k=6')
            add(f'A{tag}RhoFactor{model}', f"{r['rho_factor_2_to_6']:.1f}", f'{fmt_name}: tightening factor')
            add(f'A{tag}RhoMin{model}', f"{r['min_rho_any_step']:.3f}", f'{fmt_name}: smallest rho at any step (>= 1)')
        if model in representative:
            add(f'ARepTwo{model}', fmt(representative[model][0]), 'representative IC: D2 at 1e-3')
            add(f'ARepSix{model}', fmt(representative[model][1]), 'representative IC: D6 at 1e-3')
        if cost is not None:
            c = cost.loc[cost['model'] == model].set_index('k')
            add(f'Time{model}H', f"{c.loc[2, 'total_us']:.2f}", 'us per step, k=2 (this machine)')
            add(f'Time{model}N', f"{c.loc[6, 'total_us']:.2f}", 'us per step, k=6 (this machine)')
            add(f'TimeReference{model}', f"{c.loc[2, 'reference_us']:.2f}", 'us per 100-digit reference step')
            add(f'ACostBall{model}', f"{c.loc[6, 'exact_ball_overhead_pct_k6']:.0f}", '% extra cost: exact ball vs exact pairs, k=6')
        if subsets is not None:
            s = subsets.loc[subsets['model'] == model].set_index('k')
            for k in K_VALUES:
                lo, hi = s.loc[k, 'min_median'], s.loc[k, 'max_median']
                text = fmt(lo) if lo == hi else f'{fmt(lo)}--{fmt(hi)}'
                add(f'TableII_{model}_k{k}', text + (' (dagger)' if s.loc[k, 'n_undefined'] else ''),
                    'Table II: range of subset medians at 1e-3')
            o = orders.loc[orders['model'] == model].iloc[0]
            attained = [k for k in K_VALUES if o[f'k_sat_{k}'] > 0]
            add(f'TableII_{model}_ksat_range', f'{min(attained)}--{max(attained)}', 'Table II: k_sat over 720 orderings')
    if ball32 is not None:
        add('ABallCases', ball32['cases'], 'binary32 Lorenz, k>=3: ball crosses before pair')
        add('ABallTotal', ball32['total'], 'binary32 Lorenz, k>=3: cases compared')
        add('ABallMin', ball32['min_steps'], 'binary32 Lorenz: smallest advance (steps)')
        add('ABallMax', ball32['max_steps'], 'binary32 Lorenz: largest advance (steps)')
        add('ABallMaxEps', f"{ball32['max_epsilon']:.3g}", 'binary32 Lorenz: largest tolerance with an advance')
    if audit is not None and len(audit):
        add('FastAuditComparisons', int(audit['comparisons'].sum()), 'fast vs exact pairwise crossings compared')
        add('FastAuditDisagreements', int(audit['crossing_disagreements'].sum() + audit['boundary_disagreements'].sum()),
            'disagreements found (expected 0)')
    return pd.DataFrame(rows)


# =============================================================================
# 14. MAIN PROGRAMME
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description='Reproduce the generalised k-pseudo-orbit LBE results (ISCAS 2027).')
    parser.add_argument('--out', default='lbe_outputs', help='output folder, relative to the current directory')
    parser.add_argument('--ics', type=int, default=N_ICS, help='ICs per system (paper: 100); the first N of the declared design')
    parser.add_argument('--formats', nargs='+', default=['binary64', 'binary32'], choices=['binary64', 'binary32'])
    parser.add_argument('--skip-subsets', action='store_true', help='skip Table II (57 subsets, 720 orderings)')
    parser.add_argument('--skip-cost', action='store_true', help='skip the software timings')
    args = parser.parse_args(argv)

    if int(np.__version__.split('.')[0]) < 2:
        sys.exit('numpy >= 2 is required for exact binary32 evaluation (no scalar promotion).')
    if len(TOLERANCES_BINARY64) != 45 or len(TOLERANCES_BINARY32) != 26 or PAPER_EPSILON not in TOLERANCES_BINARY64:
        sys.exit('Tolerance grids differ from the paper.')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = baseline_configs()
    design = attractor_initial_conditions(configs, N_ICS)
    design = {m: v[:args.ics] for m, v in design.items()}   # a prefix of the declared design
    print(f'Output folder: {out_dir.resolve()}')

    # Representative ICs (Fig. 2 and the representative delays in the text).
    print('Representative ICs (binary64)...', flush=True)
    baselines = {m: run_case(cfg, 'binary64', 'baseline', out_dir) for m, cfg in configs.items()}
    t = TOLERANCES['binary64'].index(PAPER_EPSILON)
    representative = {m: (int(c['cross_ball'][0, t]) - int(c['ref_cross_high'][t]),
                          int(c['cross_ball'][4, t]) - int(c['ref_cross_high'][t])) for m, c in baselines.items()}

    # Attractor design in each arithmetic format (Table I, Fig. 3 top).
    detections, tightness, summaries, audits = {}, {}, {}, []
    for precision in args.formats:
        print(f'Design A, {precision}: {args.ics} ICs per system...', flush=True)
        det, tight, registry, audit = run_design(configs, design, precision, out_dir)
        det.to_csv(out_dir / f'detection_{precision}.csv', index=False)
        tight.to_csv(out_dir / f'tightness_{precision}.csv', index=False)
        registry.to_csv(out_dir / f'registry_{precision}.csv', index=False)
        summaries[precision] = summarise(det, tight)
        summaries[precision].to_csv(out_dir / f'summary_{precision}.csv', index=False)
        detections[precision], tightness[precision] = det, tight
        audits.append(audit)
    audit = pd.concat(audits, ignore_index=True)
    audit.to_csv(out_dir / 'fast_detector_audit.csv', index=False)
    ball32 = ball_versus_pair(detections['binary32']) if 'binary32' in detections else None

    subsets = orders = None
    if not args.skip_subsets and 'binary64' in detections:
        print('Table II: subsets and orderings (binary64)...', flush=True)
        subsets, orders = subsets_and_orderings(configs, design, out_dir)
        subsets.to_csv(out_dir / 'subsets_summary.csv', index=False)
        orders.to_csv(out_dir / 'orderings_summary.csv', index=False)

    cost = None
    if not args.skip_cost:
        print('Software cost (keep the machine idle)...', flush=True)
        cost = measure_cost(configs)
        cost.to_csv(out_dir / 'cost_by_k.csv', index=False)

    print('Figures...', flush=True)
    figure_crossings(baselines, out_dir)
    if 'binary64' in detections:
        figure_tightness(tightness['binary64'], tightness.get('binary32'), detections['binary64'], cost, out_dir)

    values = paper_values(summaries, representative, ball32, subsets, orders, cost, audit)
    values.to_csv(out_dir / 'paper_values.csv', index=False)
    with pd.option_context('display.max_rows', None, 'display.width', 160, 'display.max_colwidth', 60):
        print(values.to_string(index=False))
    print(f'\nDone. All outputs are in {out_dir.resolve()}')


if __name__ == '__main__':
    main()
