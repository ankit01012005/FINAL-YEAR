# Handover to the paper author

Everything needed to write up this project, in one place. **Every number here has been
re-derived from the data and is machine-checked** — `python paper/results/verify_numbers.py`
exits non-zero if any figure in the paper has drifted from the CSVs. It currently passes.

A draft already exists at `paper/main.tex` (~10–11 pages, IEEEtran). This document is the
source of truth to verify it against, not a replacement for it.

- [1. What the system does](#1-what-the-system-does)
- [2. The objective, formally](#2-the-objective-formally)
- [3. Why the obvious methods fail — the paper's contribution](#3-why-the-obvious-methods-fail--the-papers-contribution)
- [4. The flow](#4-the-flow)
- [5. The data](#5-the-data)
- [6. Results — every verified number](#6-results--every-verified-number)
- [7. Outputs](#7-outputs)
- [8. What must NOT be claimed](#8-what-must-not-be-claimed)
- [9. Limitations to state, not hide](#9-limitations-to-state-not-hide)
- [10. How to regenerate every number](#10-how-to-regenerate-every-number)
- [11. Open items before submission](#11-open-items-before-submission)

---

## 1. What the system does

Given a dataset of employees and teams, it **finds the under-performing teams by itself**
and recommends **which employees to exchange between them**, so that the weakest team
improves without damaging the others — using each employee's role, performance record and
observed sentiment, under a hard role-match constraint and an explicit protective guard.

One question, precisely:

> *Here is the company. Which teams are failing, whom should we move between them, and what
> is the evidence for each move?*

The output is a **recommendation for HR review, not an instruction.** The system makes no
claim about whether a move will suit an employee personally.

**Title of the existing draft:** *Performance Analysis of Company Teams and Employee
Reassignment Using Sentiment Analysis: A Provably Exact Redistribution Framework.*

**Framing:** methodological, with negative results. That is what the evidence supports, and
the draft says so in the abstract rather than burying it.

---

## 2. The objective, formally

Each employee carries one **intrinsic** value — no team argument:

```
v(e) = perf_norm(e) · φ(s(e))          φ(s) = 0.5 + 1.0·s
```

`perf_norm` is min–max scaled `Performance Score`; `s(e)` is the observed sentiment in
`[0,1]`. So `φ(0) = 0.5` (a wholly disengaged person contributes half) and `φ(1) = 1.5`.

A team's score is the sum of its members' values (×100 for readability). The pool objective
is **Rawlsian, not average**:

```
J(P) = min_t TeamScore(t)  −  λ · std_t TeamScore(t)          λ = 0.5
```

**Hard constraints, enforced by construction rather than by penalty:**

| Constraint | Enforcement |
|---|---|
| Role match | a swap exchanges employees of the same *derived* role class |
| Team size | every exchange is symmetric k-for-k, so sizes never change |
| Leadership continuity | `Is_Team_Lead` employees never move |
| Protective floor | employees below `s = 0.35` are never moved |
| Churn budget | at most 2 swaps per role class |
| Disjointness | recommendations never share a team, so all can be actioned together |

---

## 3. Why the obvious methods fail — the paper's contribution

This is the spine of the paper. Four results, each forced by a measurement rather than a
preference.

### 3.1 Performance-only reassignment is *exactly* zero-sum

Because `v(e)` has no team argument, the pool total is invariant under every legal swap.
Measured on TEAM_0098 × TEAM_0202 (23 employees):

```
pool value total before : 13.3630905400
pool value total after  : 13.3630905400
difference              : +0.0000000000
```

**Consequence, and it is not commonly drawn:** an optimiser built on a *mean* team score is
not weak, it is **vacuous** — the quantity it optimises cannot change. The aggregator must
be non-linear. Under the same swap, with the same intrinsic `v`, the Rawlsian objective
moves from `J = −7.51` to `J = 52.29`. Nothing was created; the **minimum** moved.

### 3.2 The pairing Hungarian solve is degenerate

For a symmetric k-for-k exchange, the gain decomposes as a row constant plus a column
constant:

```
Δ(a,b) = [v(a,B) − v(a,A)] + [v(b,A) − v(b,B)] = β(a) + α(b)
```

By row/column invariance of the assignment problem, **every** perfect matching between the
same two sets scores identically, so a Hungarian solve over that matrix decides nothing.

Measured in its strongest form — using a genuinely *team-dependent* `v(e,t)` from the
Mamdani FIS, so the result cannot be dismissed as an artefact of the intrinsic
simplification. TEAM_0030 × TEAM_0245, Developer class, k = 4:

- all `4! = 24` matchings scored **−0.1008912705**, spread **exactly zero**
- separability residual `max|Δᵢⱼ − (βᵢ + αⱼ)|` = **exactly zero**
- under intrinsic value, `max|Δ| = 0` exactly

**Therefore: "who swaps with whom" is a reporting convenience, not a decision variable.**
The decision variables are the *sets* that move. No Hungarian solve appears anywhere in the
system, and that is a deliberate, proven choice.

### 3.3 The shipped role labels are random

Role is the *only* feasibility constraint, so if it were noise every swap would be
meaningless. A χ² test of the shipped `Role` against `Title` gives **p = 0.5463**, and
against `Department` **p = 0.5112**. The labels carry no information.

Roles were re-derived from evidence. A naive keyword rule produced 52% Business Analysts,
which is absurd. Measuring how strongly each title token predicts department explains why:

| token | n | lift | verdict |
|---|---|---|---|
| tester | 34 | 7.40 | occupation |
| qa | 32 | 7.05 | occupation |
| devops | 39 | 3.98 | occupation |
| developer | 314 | 1.57 | occupation |
| *— threshold: lift ≥ 1.5 —* | | | |
| engineer | 1057 | 1.00 | **grade** — appears in every department |
| consultant | 2008 | 0.94 | **grade** — 40% of the workforce; a rank, not a job |
| senior | 1462 | 1.02 | **grade** |

**"Consultant", "Senior", "Engineer", "Manager", "Analyst", "Lead" and "Associate" are rungs
on a promotion ladder, not occupations.** Only 16 tokens clear a lift of 1.5; everything
else resolves through `Department`, which is real.

| | shipped | derived |
|---|---|---|
| χ² p (Title vs Role) | 0.5463 | **9.50 × 10⁻¹⁷⁶** |
| χ² p (Department vs Role) | 0.5112 | **0.0** |
| Developer / DevOps / BA / QA | 2432 / 667 / 422 / 909 | **2579 / 961 / 851 / 609** |
| resolved from title | — | 22.04% |
| resolved from department | — | 77.86% |
| resolved by fallback | — | **0.10%** (5 employees) |
| Σ Num_* == Team_Size | 570/570 | **570/570** |

`Team Lead` is deliberately **not** a derived role — it is a structural slot, exactly one
per team, carried as a boolean and excluded from swaps.

### 3.4 Counterfactual satisfaction is unfalsifiable here

The original design predicted how satisfied an employee *would be* after a move, `ŝ(e,t)`,
via a Mamdani fuzzy inference system. **This was withdrawn, and the withdrawal is itself a
reportable result.** There are no post-reassignment outcomes anywhere in the data, so no
such prediction can be validated. Two further measurements sealed it:

- the nominal adjustment range was `±0.30`, but centroid defuzzification of a triangular
  aggregate never reaches the edges of its universe — the realised δ **saturates at ±0.15**.
  Quoting ±0.30 would have been wrong.
- the between-person spread that made δ look meaningful was, at the time, an artefact of a
  synthetic placeholder rather than of measured data.

`src/fis.py` is retained as reference material and is **not imported by the decision path**.
What replaced it: observed sentiment only, intrinsic to the person. The system now claims
less and proves more.

### 3.5 Optimality — two separate claims, deliberately not conflated

**Per pool.** At `pool_size = 2` the search enumerates every feasible set-pair under the
churn budget and evaluates them by numpy broadcasting. The result is the **proven global
optimum** for that pool. Median 18 states; **99.9%** of a 1,000-pair sweep solved
exhaustively. At `pool_size ≥ 3` it is steepest-ascent local search and `proven_optimal` is
reported as `False`.

**Across pools.** The equalisation bound gives a provable ceiling on what *any* reallocation
of a pool could achieve:

```
J = min_t mean_t − λ·std  ≤  min_t mean_t  ≤  pool mean
```

because a minimum can never exceed the mean of the set it is drawn from, and `−λ·std` only
lowers `J`. Pools are therefore solved in descending order of headroom and the search stops
as soon as every unexamined pool's ceiling falls below the weakest recommendation already
shortlisted. **That is a proof, not a heuristic cutoff.** On the shipped data it solves
**394 of 9,349 pools** (95.8% pruned) and reports
`no_unexamined_pool_can_beat_shortlist = true`.

---

## 4. The flow

Four datasets in, four steps. The system is handed **data**, not decisions.

```
  1. LOAD      read the datasets that were given
  2. CHECK     reliability and integrity of each one, and of the joins between them
  3. COMPUTE   repair roles, build v(e), discover weak teams, form and bound pools
  4. RECOMMEND solve the pools worth solving, rank, and report with evidence
```

Expanded:

```
  Employee_Team_Assignment.csv ──┐
  Team_Dataset.csv ──────────────┤
  delivered sentiment dataset ───┘
              │
              ▼
     role repair            the shipped Role column is random (χ² p = 0.55);
     roles.py               re-derive it from title tokens + department
              │
              ▼
     value                  v(e) = perf_norm(e) · φ(s(e))
     objective.py           one number per person, carried with them
              │
              ▼
     DISCOVERY              find the under-performing teams (143 of 570)
     pipeline.py            group them into pools that can legally exchange (9,349)
              │
              ▼
     bound + prune          provable ceiling per pool; solve in ceiling order
     objective.py           and stop early with a proof (394 of 9,349 solved)
              │
              ▼
     optimise               K=2: exhaustive, proven optimal
     search.py              K≥3: steepest-ascent local search
              │
              ▼
   ranked, non-overlapping recommendations  ──►  recommended_swaps.csv
```

**Step 2 is not a formality** — it has caught every real problem so far. A failure stops the
run; it does not warn and continue, because a recommendation computed from a dataset that
failed its own integrity check is worse than no recommendation.

| Invariant | Checked in |
|---|---|
| `Employee_ID` unique across the roster | `pipeline.load_context` |
| the shipped `Role` column is not random noise | `roles.py` — it *was* |
| `Σ Num_*` reconciles with `Team_Size` for every team | `roles.recompute_team_counts` |
| sentiment covers the roster 1:1, in range, no nulls | `sentiment_build`, `sentiment_io.validate` |
| sentiment is not a restatement of the star rating | `sentiment_build.audit_leakage` |
| the engine is not silently running on placeholder data | `is_placeholder` flag |

Runtime: a full discovery-and-recommend run over all 570 teams takes **≈ 2.7 s** on a
laptop CPU; a single pool solves in ≈ 8 ms. No GPU, no model download, no network access.

---

## 5. The data

| Dataset | Rows × cols | Role |
|---|---|---|
| `Employee_Team_Assignment.csv` | 5,000 × 17 | the roster |
| `Team_Dataset.csv` | 570 × 18 | which teams exist, and which are weak |
| delivered sentiment dataset | 5,000 × 21 | one engagement score per `Employee_ID` |
| `employee_reviews.csv` | 86,117 reviews | **optional** — only to audit the above for leakage |

**Only five columns actually drive a recommendation:** `Employee_ID`, `Team_ID`, `Title`,
`Department`, `Performance Score`, plus the sentiment score. State this plainly in the
paper — a reviewer will ask, and it bounds what the system can claim.

### 5.1 The sentiment input — important for how you write Methodology

**We did not compute the sentiment scores.** A separate subsystem trained and tested its own
model (BERT `[CLS]` encoding fused with fuzzy membership features over review text) and
delivered a finished dataset keyed by `Employee_ID`. Our system treats it as an input,
validates it, and consumes it.

Validation results on the delivered dataset:

| Check | Result |
|---|---|
| rows / roster coverage | 5,000 / 5,000 — **100.0%**, no duplicates, no unknown IDs |
| nulls in the score | 0 |
| class probabilities | sum to 1 for all 5,000 rows; the score is exactly their centroid |
| occupation match | `Title` **and** `Department` match the roster for all 5,000 |
| distribution | mean 0.7035, std 0.3268, range [0.0265, 0.9928] |
| **rating leakage** | r² = **0.29** against `Overall_rating` — passes |

That last row is load-bearing. Had the delivered score merely echoed the star rating
(r² > 0.95), sentiment would be a second copy of a number the engine already holds, every
employee's value would be intrinsic twice over, and the pool objective would collapse back
to exactly zero-sum (§3.1). `run.py sentiment` **refuses to write** in that case.

### 5.2 Two defects found and fixed in the sentiment notebook

Worth a sentence in the paper if Methodology-B describes the model, because both are the
kind of error a reviewer looks for:

1. **Target leakage.** Feeding rating-derived fuzzy memberships (`r_low/r_mid/r_high`) as
   model inputs reconstructs the 3-class label at **100.0% validation accuracy with the
   text ignored entirely** — BERT contributed nothing. The rating must build the *label*
   only.
2. **Null ratings defaulted to POSITIVE.** All 14,330 of them, via `fillna(3)` followed by
   an `else` branch. Now dropped explicitly.

Also fixed: the TF-IDF vectoriser, SVD and scaler were fitted on train and validation
concatenated, and the vectoriser was never persisted — so inference substituted zeros for
the three topic features the model was trained on.

---

## 6. Results — every verified number

### 6.1 Main comparison (n = 1,000 team pairs) — Table II

| Arm | mean gain | median | % improved | % **damaged** | full wins | full loses |
|---|---|---|---|---|---|---|
| **proposed** | **8.0811** | **5.7184** | **82.10** | **0.00** | — | — |
| perf-only | 4.8470 | 1.7425 | 59.00 | 19.80 | 71.10% | **0.00%** |
| sent-only | 0.2596 | 0.0000 | 39.40 | 41.10 | 81.50% | **0.00%** |
| random | 0.1365 | 0.0000 | 45.20 | 45.30 | 80.40% | **0.00%** |
| worst-for-best | 2.0021 | 0.5779 | 53.10 | 37.50 | 77.40% | **0.00%** |

Also: **99.9%** proven optimal · **71.1%** of pairs where sentiment changed the selected
movers · mean **69.06%** of the provable ceiling captured · mean **3.35** moves · median
**18** states evaluated.

**Three readings, all of which belong in the paper:**

1. The full objective **never loses** to any baseline and **damages a pool 0.00%** of the
   time. With 99.9% proven optimality, that is the expected signature of an exact optimiser
   on a correctly specified objective — not a surprise, and it should not be sold as one.
2. `sent-only` has a median gain of **exactly 0.0000** and damages 41.10% of pools.
   **Sentiment alone recommends nothing useful.** It is a modifier on performance, not a
   driver.
3. **The headline practitioner result:** the intuitive human heuristic — trade the weaker
   team's worst member for the stronger team's best — **damages the pool in 37.50% of
   cases**. This survives every modelling choice the project made.

### 6.2 Balance weight λ (n = 200) — Table IV

| λ | win vs perf-only | loss | % of U(P) |
|---|---|---|---|
| 0.00 | 51.0% | **30.0%** | 64.69 |
| 0.25 | 58.0% | 23.0% | 64.59 |
| **0.50** | **68.5%** | **0.0%** | 64.55 |
| 1.00 | 80.5% | 0.5% | 64.48 |
| 2.00 | 80.5% | 0.5% | 64.38 |

The dispersion term is **structural, not a matter of taste**. With intrinsic value §3.1
removes all first-order signal and a bare minimum has little to grip; at λ = 0 the system is
beaten by a trivial baseline on 30% of pools. Note also that raising λ is *not free*:
λ ≥ 1 wins more often but reintroduces a 0.5% loss rate, so **λ = 0.5 is the only setting
tested that is never beaten.**

### 6.3 Churn budget B (n = 200) — Table V

| B | mean gain | moves | med. states | % of U(P) |
|---|---|---|---|---|
| 1 | 6.9761 | 2.52 | 12 | 62.27 |
| **2** | **7.1006** | **3.26** | **15** | **64.55** |
| 3 | 7.1059 | 3.38 | 15 | 64.64 |

B = 2 → 3 improves mean gain by **0.0053** while adding 0.12 moves per pool, and is where
exhaustive enumeration stops being universal (99.5% proven optimal at B = 3, against 100%
at B ≤ 2). **B = 2 is the right default, and this is the evidence.**

### 6.4 Protective floor σ (n = 200) — Table VI

| σ | held | mean gain | median | win vs perf-only |
|---|---|---|---|---|
| 0.00 (off) | 0 | 8.0160 | 5.3832 | 78.5% |
| **0.35 (on)** | **791** | **7.1006** | **3.9705** | 68.5% |

Holding disengaged employees in place costs **11.42% of mean gain**. **Quote this price.**
A constraint asserted without its cost is not an engineering decision.

State the sensitivity too: σ = 0.35 is an *absolute* threshold, and its price depends on the
sentiment distribution it is applied to. Under an earlier synthetic placeholder (std 0.162)
the same threshold held 288 employees and cost 5.1%; on the delivered scores (std 0.327) it
holds 791 and costs 11.4%. A floor expressed as a *quantile* would be portable; this one is
not. That belongs in future work.

### 6.5 Case study — Table VII

TEAM_0098 (n = 3) × TEAM_0202 (n = 20), solved exhaustively over **13** states. TEAM_0098
has three members, one of whom is its lead and cannot move; it is the weakest team in the
organisation on effective contribution.

| | before | after |
|---|---|---|
| TS(TEAM_0098) | 7.14 | **53.59** |
| TS(TEAM_0202) | 65.74 | 58.78 |
| J | −7.51 | **52.29** |
| pool value total | 1336.31 | 1336.31 |

bound U(P) = 58.10 · headroom 65.61 (rank 1 of 9,349) · captured **91.14%** · one exchange,
Developer class: `308441 ↔ 772467`.

### 6.6 Dataset properties worth a table

- **The performance metrics are mutually independent**: mean |r| = **0.0109**, max 0.0272.
  There is no latent quality factor in this data — which is precisely why a learned model
  was rejected in favour of an explicit objective. This is a genuine finding, not a caveat.
- `Team_Performance` for the discovered "Low" band spans **73.70–78.09** — a 4.39-point
  window. These are the weakest teams, not failing ones.

---

## 7. Outputs

| File | Contents |
|---|---|
| `output/recommended_swaps.csv` | **the deliverable** — one row per move: rank, pool, who moves, from → to, swap partner, performance, sentiment, value, and a plain-language reason naming the role match and the value gap |
| `output/recommended_pools.csv` | one row per recommendation: n_moves, weakest team before/after, J before/after, gain, `pct_of_achievable`, `proven_optimal` |
| `output/low_teams.csv` | every flagged team, worst first, with the rule that flagged it and how many members it can actually move |
| `output/sweep_results.csv` | the evaluation log, one row per pair per arm — Table II is computed from this |

---

## 8. What must NOT be claimed

Read this section before writing the abstract.

1. **Do not claim sentiment drives the recommendation.** It is a modifier. `sent-only` has a
   median gain of exactly zero. The defensible claim is that sentiment changed the selected
   movers on 71.1% of pairs and lifts mean gain from 4.85 to 8.08 *on top of* performance.
2. **Do not claim an accuracy figure.** There is no ground truth. No reassignment was ever
   executed and measured. Evaluation is a simulation study against baselines on the system's
   own objective, and §8.4 of the draft states that circularity explicitly rather than
   leaving a reviewer to find it.
3. **Do not claim the system predicts personal fit or post-move satisfaction.** That claim
   was withdrawn on purpose (§3.4).
4. **Do not conflate the two optimality claims.** Per-pool exhaustive optimality at K = 2 is
   proven. Cross-pool pruning is proven. `pool_size ≥ 3` is local search and must not be
   quoted as optimal.
5. **Do not describe the organisational data as real.** Teams and performance metrics are
   synthetic. The *sentiment* is modelled from real reviews; the *roster* is not a real
   workforce.
6. **Do not quote ±0.30 for the FIS adjustment range.** The realised value saturates at
   ±0.15, and the FIS is out of the decision path anyway.

---

## 9. Limitations to state, not hide

1. **One review per employee.** The delivered dataset resolves exactly one review to each
   employee, so the signal is cross-sectional and carries no trajectory. `n_reviews = 1`
   everywhere, so the contract's confidence weight cannot discriminate between rows.
2. **The sentiment scores are produced outside this repository** — by design, another
   subsystem owns the model — so they cannot be re-derived here. We validate what we are
   given; we cannot regenerate it.
3. **No ground truth, therefore no accuracy figure.** See §8.2.
4. **Min–max normalisation is set by two individuals.** `perf_norm` is scaled by the
   company-wide min and max `Performance Score` (50 and 100), so an employee scoring exactly
   50 gets `v = 0.00` regardless of sentiment, and the scale is fragile to a single outlier
   hire.
5. **"Low" means bottom quartile, not crisis** — a 4.39-point window.
6. **`Team_Performance` and the value-based `team_score` disagree.** The shipped column is
   computed from `Composite_Score`; the objective uses `Performance Score` and sentiment.
   Discovery uses the former, optimisation the latter, and both are reported side by side in
   `low_teams.csv` so the gap is visible.
7. **Local search beyond two teams.** See §8.4.
8. **Fairness is unaudited.** A reassignment system affects careers. No check has been run
   on recommendation rates across demographic attributes. State this.
9. **The system is configurable, not yet general-purpose.** See `WORKFLOW.md` §13.1 — the
   optimiser, bound and search are data-independent, but role derivation carries a
   token taxonomy tuned to this organisation.

---

## 10. How to regenerate every number

```bash
pip install -r requirements.txt

python run.py doctor                        # all invariants; must show is_placeholder: false
python run.py sentiment                     # rebuild dataset/sentiment_fuzzy.csv from the delivery
python run.py recommend --top 10 --save     # the deliverable
python run.py roles --tokens                # the token-lift table in §3.3

python run.py sweep --limit 1000 --save     # Table II          (~9 min)
python paper/results/run_ablations.py       # Tables IV-VI      (~20 min)
python paper/results/make_figures.py        # Figures 1-2
python paper/results/verify_numbers.py      # MUST exit 0 before submitting
python paper/results/check_tex.py           # structural LaTeX validation
```

**`verify_numbers.py` is the guard.** It exits non-zero if any figure quoted in the paper has
drifted from the underlying data. One caveat for whoever edits the tables: **it stores the
paper's claims as literals in its own source**, so a number changed in `main.tex` must be
mirrored there too, or the check quietly keeps testing the old value.

---

## 11. Open items before submission

| # | Item | Owner |
|---|---|---|
| 1 | **The paper has never been compiled.** `check_tex.py` is a static checker, not `pdflatex`. Upload `paper/` to Overleaf, set main document to `main.tex`, recompile twice. | author |
| 2 | Institution, city and contact e-mail — `paper/main.tex:48` is still a `TODO` | team |
| 3 | Prof. Partha Ghosh appears as final author **and** in the Acknowledgment — pick one convention | team |
| 4 | Page limit: ~10–11 pages against a likely 6–8 cap. `paper/README.md` lists the compressible material in order: Related Work subsections, then Table I, then the case study | author + team |
| 5 | No `bibtex` run needed — the bibliography is a hand-written `thebibliography` block, so two `pdflatex` passes suffice | author |

---

*Generated from the state of the repository at the initial commit. Re-run
`python paper/results/verify_numbers.py` after any change to confirm this document and the
paper still match the data.*
