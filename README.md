# Generalised Lower Bound Error for Multiple Pseudo-Orbits in Nonlinear Systems

Reproducibility repository for the paper submitted to the
**IEEE International Symposium on Circuits and Systems (ISCAS 2027)**.

**Authors**

- Thalita Nazaré — Hamilton Institute, Dept. of Electronic Engineering, Maynooth University, Ireland
- Arthur M. Lima — Hamilton Institute, Dept. of Electronic Engineering, Maynooth University, Ireland
- Erivelton Nepomuceno — Hamilton Institute, Centre for Ocean Energy Research, Dept. of Electronic Engineering, Maynooth University, Ireland

**Contact:** Thalita Nazaré — thalita.nazare@mu.ie

> Status: submitted to ISCAS 2027 and under review. Citation details will be added here once they are available.

---

## Contents

| File | Purpose |
|---|---|
| `lbe_paper_reproduction.py` | Self-contained script that reproduces every number, table and data figure in the Methodology and Results sections |
| `README.md` | This document: the complete experiment specification |

The script contains only the code used in the paper. It writes all outputs to a single folder relative to the directory in which it is run, and contains no machine-specific paths.

## Quick start

```bash
pip install "numpy>=2" pandas mpmath matplotlib
python lbe_paper_reproduction.py            # full reproduction (writes ./lbe_outputs)
python lbe_paper_reproduction.py --ics 5    # quick check: first 5 ICs per system
python lbe_paper_reproduction.py --help     # options (--out, --formats, --skip-subsets, --skip-cost)
```

Run time on the machine below: RUNTIME_PLACEHOLDER Each IC is cached in `lbe_outputs/cases/` as soon as it finishes, so an interrupted run resumes where it stopped.

### Outputs (in `lbe_outputs/`)

| File | Content |
|---|---|
| `detection_<format>.csv` | Crossings and delays for every IC, \(k\) and tolerance |
| `registry_<format>.csv` | Completed/failed status of every declared IC |
| `tightness_<format>.csv` | Per-IC median tightness ratio \(\rho_k\) |
| `summary_<format>.csv` | Table I quantities per system |
| `subsets_summary.csv`, `orderings_summary.csv` | Table II |
| `cost_by_k.csv` | Software timings (Fig. 3, bottom row) |
| `fast_detector_audit.csv` | binary64 pairwise detector checked against exact predicates |
| `paper_values.csv` | Every number quoted in the text, labelled by its LaTeX macro name |
| `fig2_representative_crossings.pdf/.svg` | Figure 2 |
| `fig3_tightness.pdf/.svg` | Figure 3 |

Figure 1 is an illustrative schematic of the method and is not generated from data.

## Computational environment

All results in the paper were obtained with:

| Item | Value |
|---|---|
| Computer | MacBook Air, Apple M2, 8 GB RAM |
| Operating system | macOS 27.0.1 (arm64) |
| Python | CPython 3.13.9 |
| Libraries | NumPy 2.3.4, pandas 3.0.6, mpmath 1.4.1, matplotlib 3.11.2 |
| Execution | Interpreted, single-threaded Python |

Requirements for reproduction are Python ≥ 3.10 and NumPy ≥ 2.0. NumPy 2 is needed because its scalar rules keep `numpy.float32` results in binary32 when they are combined with Python integers; the script checks for promotion and stops if it occurs.

The numerical results (pseudo-orbits, exact crossings, references, delays, tightness ratios) are deterministic and should be identical on any IEEE 754 platform running CPython. Timings depend on the machine and will differ in absolute value.

---

## 1. Models, parameters and horizons

All pseudo-orbits use IEEE 754 arithmetic with each scalar operation rounded separately by CPython (no fused or vectorised operations).

| System | Map | Parameters | Step | Horizon | Representative IC |
|---|---|---|---|---|---|
| Logistic | \(x_{n+1}=r x_n(1-x_n)\) | \(r=3.9\) | 1 | 110 iterations | \(x_0=0.1\) |
| Hénon | \(x_{n+1}=1-a x_n^2+y_n,\; y_{n+1}=b x_n\) | \(a=1.4,\ b=0.3\) | 1 | 160 iterations | \((0,0)\) |
| Lorenz | \(\dot x=\sigma(y-x),\ \dot y=x(\varrho-z)-y,\ \dot z=xy-\beta z\) | \(\sigma=10,\ \varrho=28,\ \beta=8/3\) (binary64 `8.0/3.0`) | \(h=0.01\), classical RK4 | 5500 steps | \((1,1,1)\) |

For Lorenz, the studied evolution \(F\) is the fixed-step RK4 map, not the continuous-time flow. Delays are reported in iterations (maps) or RK4 steps (Lorenz); model time is \(t=nh\).

## 2. The six realisations G, H, K, L, M, N

All six are mathematically equivalent in exact arithmetic and differ only in algebraic form or evaluation order. Python evaluates `*`, `+`, `-` from left to right; the parentheses below make the resulting order explicit. All realisations start from the same stored initial state and parameters.

### Logistic (\(x_{n+1}\))

| Label | Python expression | Explicit order |
|---|---|---|
| G | `r * x * (1 - x)` | `(r*x)*(1-x)` |
| H | `r * (x * (1 - x))` | `r*(x*(1-x))` |
| K | `r * x - r * (x * x)` | `(r*x)-(r*(x*x))` |
| L | `r * (x - x * x)` | `r*(x-(x*x))` |
| M | `(r - r * x) * x` | `(r-(r*x))*x` |
| N | `r * x - r * x * x` | `(r*x)-((r*x)*x)` |

### Hénon (\(x_{n+1}\); \(y_{n+1}\) = `b * x` for all labels)

| Label | Python expression | Explicit order |
|---|---|---|
| G | `1 - a * (x * x) + y` | `(1-(a*(x*x)))+y` |
| H | `1 + y - a * x * x` | `(1+y)-((a*x)*x)` |
| K | `1 + (y - a * (x * x))` | `1+(y-(a*(x*x)))` |
| L | `y + (1 - a * x * x)` | `y+(1-((a*x)*x))` |
| M | `1 + y - a * (x * x)` | `(1+y)-(a*(x*x))` |
| N | `1 - (a * x * x - y)` | `1-(((a*x)*x)-y)` |

### Lorenz (vector field and RK4 weighted sum)

Vector field \((\dot x,\dot y,\dot z)\) with `s, r, b` = \(\sigma,\varrho,\beta\):

| Label | Vector field |
|---|---|
| G, M, N | `s*(y-x)`, `x*(r-z)-y`, `x*y-b*z` |
| H | `s*y-s*x`, `r*x-x*z-y`, `y*x-z*b` |
| K | `(y-x)*s`, `(r-z)*x-y`, `x*y-b*z` |
| L | `s*(y-x)`, `r*x-(x*z+y)`, `-(b*z-x*y)` |

RK4 stages (all labels, using the label's vector field \(f\)):

```
a = f(u);  u2 = u + h/2*a;  b = f(u2);  u3 = u + h/2*b;  c = f(u3);  u4 = u + h*c;  d = f(u4)
```

(`h/2*a` is evaluated as `(h/2)*a`, component-wise.) The update is `u_next = u + h/6 * total`, component-wise, with

| Label | `total` |
|---|---|
| G, H, K, L | `a + 2*b + 2*c + d` = `((a+2b)+2c)+d` |
| M | `a + d + 2*(b + c)` = `(a+d)+2(b+c)` |
| N | `a + 2*b + (2*c + d)` = `(a+2b)+(2c+d)` |

## 3. Nested sets, subsets and orderings

- Recorded order: G, H, K, L, M, N. Original pair: (G, H).
- Nested sets: \(\mathcal K_k\) = the first \(k\) labels, \(k=2,\dots,6\).
- Proposed bound: the full ensemble, \(k=6\). No \(k\) is selected from the outcomes.
- Order sensitivity (Table II): all \(2^6-7=57\) subsets with \(|S|\ge2\) and all \(6!=720\) orderings (nested sets formed by the first \(k\) labels of each ordering). Neither the recorded order nor the original pair was optimised.

## 4. Initial-condition design

- 100 ICs per system, sampled along each attractor: the binary64 orbit of realisation G from the representative IC is advanced for a burn-in and then sampled once every `stride` steps.

  | System | Burn-in | Stride |
  |---|---|---|
  | Logistic | 1000 | 101 |
  | Hénon | 1000 | 101 |
  | Lorenz | 5000 RK4 steps | 1009 RK4 steps |

- The design is deterministic (no random sampling, no seed) and was declared before any delay was computed. Failed cases would be recorded, never replaced; all 300 binary64 and 300 binary32 cases completed, with stable reference crossings at every tolerance.
- The ICs lie on a single orbit and are therefore not statistically independent samples.
- The representative IC of each system (Section 1) is used for Figure 2 and the representative delays in the text. It is not one of the 100 design ICs.

## 5. Arithmetic formats and tolerances

| Format | Arithmetic | Tolerances |
|---|---|---|
| binary64 | Python `float` | 45 values, `numpy.logspace(-12, -1, 45)`, including \(10^{-3}\) exactly |
| binary32 | `numpy.float32` scalars, one binary32 rounding per operation (promotion is checked) | 26 values, `10**linspace(-6, -1, 26)` |

In binary32, the parameters, IC and step are rounded to binary32 first, and the reference uses exactly those rounded values. The paper tolerance is \(\varepsilon=10^{-3}\). All tolerances reuse the same trajectories and are not independent replications.

## 6. Bounds and geometry

- Generalised LBE: \(R_{k,n}=\min_{\mathbf c}\max_{i\in\mathcal K_k}\|\mathbf p_{i,n}-\mathbf c\|_2\) (minimum enclosing Euclidean ball; unscaled Euclidean metric).
- Same-set pairwise bound: \(B^{\rm pair}_{k,n}=\tfrac12\max_{i,j\in\mathcal K_k}\|\mathbf p_{i,n}-\mathbf p_{j,n}\|_2\).
- Invariants checked at every stored step: \(R_{2,n}=B^{\rm pair}_{2,n}\), \(B^{\rm pair}_{k,n}\le R_{k,n}\), \(R_{k,n}\le R_{k+1,n}\).
- Jung's theorem: \(R_{k,n}\le\sqrt{2m/(m+1)}\,B^{\rm pair}_{k,n}\), \(m=\min(d,k-1)\); the factor is \(\sqrt{4/3}\approx1.1547\) for \(d=2\) and \(\sqrt{3/2}\approx1.2247\) for \(d=3\).
- Exact computation: stored states are converted exactly to rationals (`fractions.Fraction`). Squared distances, the support-set solution (at most \(d+1\) points) and comparisons with squared tolerances are all exact, so every threshold crossing is a certified predicate on the stored states.
- Subset analysis: the ball crossing cannot be later than the pairwise-maximum crossing. The exact ball is therefore evaluated only on earlier frames where Jung's bound (screened in binary64 with a 0.999 margin) can exceed \(\varepsilon^2\). Nested subsets must reproduce the crossings of the main computation.

## 7. High-precision reference

- `mpmath` at 100 and 200 decimal digits. The initial state, parameters and step are lifted exactly, and the reference uses expression G.
- For Lorenz, the reference evaluates the same fixed-step RK4 map. It measures finite-precision disagreement with that map, not integration error. It is a numerical approximation, not a rigorous enclosure.
- Full-set reference error: \(E^{\rm ref}_n=\max_{1\le i\le 6}\|\mathbf p_{i,n}-\mathbf q_n\|_2\), computed at high precision, always over all six realisations. The event to be detected is therefore the same for every \(k\).
- Stability check: delays are used only when the reference crossings at 100 and 200 digits agree. They agree for every IC and tolerance in both formats.

## 8. Crossings, delays, tightness and summaries

- First strict crossing: \(\tau_\varepsilon(u)=\min\{n:u_n>\varepsilon\}\).
- Detection delay: \(D_k=\tau_\varepsilon(R_k)-\tau_\varepsilon(E^{\rm ref})\), using the same full-set reference for every \(k\), subset and ordering.
- A missing crossing is right-censored, never zero. In the subset and ordering analysis, censored values enter medians as \(+\infty\), and a median is undefined when its middle order statistic is censored.
- Medians and IQRs (`numpy` linear interpolation) use the same eligible IC cohort for every \(k\) at each tolerance, with equal weight per IC.
- Percentage reduction: per IC, \(100(D_2-D_6)/D_2\), defined only for \(D_2>0\). The paper reports the **median of these per-IC values**, not the reduction between two medians.
- Saturation size \(k_{\rm sat}\): the smallest \(k\) that reaches the minimum median delay at \(\varepsilon=10^{-3}\). It is descriptive only and does not define the proposed bound.
- Tightness: \(\rho_{k,n}=E^{\rm ref}_n/R_{k,n}\ge1\); \(R_{k,n}=0<E^{\rm ref}_n\) counts as \(\rho=\infty\) and is never excluded. Per IC, the arithmetic median of \(\rho_{k,n}\) is taken over the steps with \(E^{\rm ref}_n\) in the tolerance range: \([10^{-12},10^{-1}]\) (binary64) or \([10^{-6},10^{-1}]\) (binary32). Figure 3 shows its distribution over ICs, and the text reports the median over ICs. The reference error uses the 200-digit run.

## 9. Computational cost

All timings use `time.perf_counter` on the representative ICs, in interpreted single-threaded Python.

- **Cost per \(k\) (Fig. 3, bottom; per-step times in the text):** for each \(k=2,\dots,6\), the time to generate the first \(k\) binary64 realisations (full horizon for the maps, first 1000 steps for Lorenz) plus the binary64 pairwise maximum, evaluated on up to 256 equally spaced stored frames. The total is the sum of these two separately timed parts, not a streaming benchmark. Protocol: one warm-up followed by seven repetitions, with the order of \(k\) rotated in each repetition; medians are reported. Cost is expressed relative to the original pair \(k=2\).
- **Reference:** one step of the 100-digit map, timed in the same repetitions.
- **Exact audit overhead:** exact minimum ball versus exact pairwise maximum for \(k=6\) on 64 equally spaced frames (one warm-up, five repetitions). This includes the rational conversion and the squared distances.

These are software timings that depend on the machine. They are not hardware throughput estimates.

## 10. Audit of the fast pairwise detector

The cheap binary64 pairwise maximum is compared with the exact pairwise crossings for every IC, \(k\) and tolerance in both formats (106,500 comparisons). The comparison checks the first crossings, and also the exact and binary64 predicates at each crossing and its predecessor. There were zero disagreements. binary32 states are represented exactly in binary64 for this audit, which does not benchmark binary32 hardware arithmetic. Agreement on these samples does not certify future inputs.

## 11. Expected results

Values reported in the paper. With `lbe_paper_reproduction.py`, all numerical values below are reproduced exactly; timings are reproduced only approximately.

### Representative ICs, \(\varepsilon=10^{-3}\), binary64 (Fig. 2)

| | Logistic | Hénon | Lorenz |
|---|---|---|---|
| \(D_2\) → \(D_6\) | 6 → 0 iterations | 8 → 2 iterations | 156 → 5 steps |

### Design A, \(\varepsilon=10^{-3}\) (Table I)

| | Logistic | Hénon | Lorenz |
|---|---|---|---|
| binary64 median \(D_k\), \(k=2..6\) | 5, 1, 1, 0, 0 | 2, 2, 1, 1, 1 | 159, 159, 19.5, 11.5, 10 |
| binary64 median per-IC reduction (\(k=6\)) | 100% | 50% | 53.9% |
| binary64 ICs improved / unchanged / \(D_2=0\) | 92 / 8 / 3 | 58 / 42 / 14 | 93 / 7 / 0 |
| binary64 \(k_{\rm sat}\) | 5 | 4 | 6 |
| binary32 median \(D_k\), \(k=2..6\) | 5, 1, 1, 0, 0 | 2, 2, 1, 1, 1 | 167.5, 167.5, 17, 9.5, 5.5 |
| binary32 median per-IC reduction (\(k=6\)) | 100% | 50% | 64.6% |
| Tolerances with smaller \(k=6\) median | 45/45 (b64), 26/26 (b32) | 45/45, 26/26 | 45/45, 26/26 |

### Tightness (Fig. 3, top): median \(\rho_k\) over ICs, \(k=2\) → \(k=6\)

| | Logistic | Hénon | Lorenz |
|---|---|---|---|
| binary64 | 9.6 → 1.29 | 3.1 → 1.57 | 4.2 → 1.69 |
| binary32 | 11.1 → 1.38 | 2.9 → 1.60 | 4.5 → 1.60 |

No step of any IC gave \(\rho_{k,n}<1\).

### Subsets and orderings, binary64 (Table II)

| | \(k=2\) | 3 | 4 | 5 | 6 | \(k_{\rm sat}\) range (720 orders) |
|---|---|---|---|---|---|---|
| Logistic | 1–5 | 1–2 | 0–1 | 0–1 | 0 | 4–6 |
| Hénon | 2–5 | 2–2.5 | 1–2 | 1 | 1 | 4–5 |
| Lorenz | 148.5–211† | 17–211 | 11.5–108 | 10–13 | 10 | 5–6 |

† Lorenz \((G,K)\) is excluded: its binary64 states coincide bitwise throughout the horizon for all 100 ICs, which gives a zero bound and no crossing. Its delay is undefined, not zero.

### Geometry

- binary64: the ball and the pairwise maximum have identical crossings for all 57 subsets, ICs and tolerances.
- binary32 Lorenz (\(k\ge3\)): the ball crossed 1–3 steps before the pairwise maximum in 9 of 10,400 cases, all at \(\varepsilon<10^{-5}\).

### Cost (paper machine, µs per step)

| | Logistic | Hénon | Lorenz |
|---|---|---|---|
| \(k=2\) | 0.78 | 0.80 | 5.46 |
| \(k=6\) | 5.48 | 5.94 | 20.42 |
| \(k=6\) relative to \(k=2\) | 7.0× | 7.5× | 3.7× |
| One 100-digit reference step | 3.13 | 4.75 | 67.14 |
| Exact ball vs exact pairs (\(k=6\)) | +13% | +26% | +22% |

## 12. Scope and limitations

- The high-precision Lorenz reference evaluates the RK4 map; it is not the exact solution of the Lorenz ODE.
- The 100 ICs are sampled from one orbit and the tolerances share trajectories. The robustness results are descriptive, not independent replications.
- A small bound cannot certify accuracy, because common errors shared by all realisations are not detected.
- The paper discusses finite-window largest-Lyapunov-exponent estimation from \(\ln R_{k,n}\) conceptually, but reports no numerical LLE results. That analysis is therefore not part of this repository.
- Hardware (e.g. FPGA) execution was not measured.

## How to cite

Citation details will be added after the review process.
