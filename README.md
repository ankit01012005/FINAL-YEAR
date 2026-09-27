# Team Rebalancing by Role-Matched Employee Reassignment

Given a **dataset** of employees and teams, this system finds the under-performing teams
by itself and recommends **which employees to exchange between them** so that the weakest
team improves without wrecking the others — using each employee's role, performance record
and observed sentiment, under a hard role-match constraint and an explicit protective
guard.

It answers one question, precisely:

> *Here is the company. Which teams are failing, whom should we move between them, and
> what is the evidence for each move?*

The output is a **recommendation for HR review**, not an instruction. The system does not
claim to predict whether a move will suit an employee personally — see
[Limitations](#limitations).

---

## Table of contents

- [What the system does](#what-the-system-does)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Input data](#input-data)
- [Commands](#commands)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Guarantees and constraints](#guarantees-and-constraints)
- [Current status](#current-status)
- [How sentiment reaches the system](#how-sentiment-reaches-the-system)
- [Limitations](#limitations)
- [Role derivation](#role-derivation)
- [Further reading](#further-reading)

---

## What the system does

```
  Employee_Team_Assignment.csv ──┐
  Team_Dataset.csv ──────────────┤
  sentiment_fuzzy.csv ───────────┘
              │
              ▼
     ┌──────────────────┐
     │  role repair     │  the shipped Role column is random (chi2 p = 0.55);
     │  roles.py        │  re-derive it from title tokens + department
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │  value           │  v(e) = perf_norm(e) * phi(s(e))
     │  objective.py    │  one number per person, carried with them
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │  DISCOVERY       │  find the under-performing teams (143 of 570)
     │  pipeline.py     │  group them into pools that can legally exchange
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │  bound + prune   │  provable ceiling per pool; solve in ceiling order
     │  objective.py    │  and stop early with a proof (394 of 9,349 solved)
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │  optimise        │  K=2: exhaustive, proven optimal
     │  search.py       │  K>=3: steepest-ascent local search
     └────────┬─────────┘
              ▼
   ranked, non-overlapping recommendations  ->  recommended_swaps.csv
```

Five design commitments, each forced by a measurement rather than a preference:

| Commitment | Why |
|---|---|
| Roles are re-derived, never trusted | shipped `Role` vs `Title`: χ² p = **0.5463** — random labels |
| Value is **intrinsic**, `v(e)` not `v(e,t)` | no data can validate a counterfactual satisfaction score |
| Objective is `min − λ·std`, never a mean | a mean is provably invariant under every legal swap |
| No Hungarian solve anywhere | the cost matrix is degenerate by construction — twice over |
| Pools are discovered, not supplied | the input is a dataset; naming a pair by hand is the old scope |

---

## Requirements

Python ≥ 3.10 and three libraries:

```
pandas
numpy
scipy
```

Verified against pandas 3.0.0, numpy 2.3.5, scipy 1.16.0. No GPU, no model download, no
network access. A full discovery-and-recommend run over all 570 teams takes **≈ 2.7 s** on
a laptop CPU; a single pool solves in ≈ 8 ms.

```bash
pip install pandas numpy scipy
```

---

## Quick start

```bash
# health check — reads everything, verifies every invariant, solves one pool
python run.py doctor

# THE MAIN PATH: dataset in, ranked recommendations out
python run.py recommend

# ... and write the three CSVs to output/
python run.py recommend --save
```

As a library:

```python
from src import pipeline

ctx = pipeline.load_context()               # load + repair + attach sentiment + build v(e)
out = pipeline.recommend(ctx, pool_size=2, top=10)

print(out["meta"])                                 # discovery + search statistics
print(pipeline.pools_frame(out["results"]))        # one row per recommended pool
print(pipeline.moves_frame(ctx, out["results"]))   # one row per recommended move
```

`load_context` does all the expensive work once. Everything downstream — a single pool or
a sweep over ten thousand — reuses it.

---

## Input data

### Required — shipped with the project

**`dataset/Employee_Team_Assignment.csv`** — 5,000 rows × 17 columns.

| Column | Used for |
|---|---|
| `Employee_ID` | primary key; must be unique |
| `Team_ID` | current assignment; the thing a recommendation changes |
| `Title` | **role derivation** (token lift) |
| `Department` | **role derivation** (fallback, and it is the reliable one) |
| `Role` | read only to extract `Is_Team_Lead`; the four working labels are discarded |
| `Performance Score` | the performance half of `v(e)` |
| `Peer Rating`, `Manager Feedback`, `KPI Score` | carried, not consumed by the objective |
| `Attendance (%)`, `Task Completion (%)`, `Work Hours Logged`, `Training Hours`, `Promotion Eligibility`, `Project_Size`, `Peer_Rating_100`, `Composite_Score` | carried, unused |

Only **five** columns actually drive a recommendation: `Employee_ID`, `Team_ID`, `Title`,
`Department`, `Performance Score` — plus `s_crisp` from the sentiment file. This is stated
plainly because a reviewer will ask, and because it bounds what the system can claim.

**`dataset/Team_Dataset.csv`** — 570 rows × 18 columns.

| Column | Used for |
|---|---|
| `Team_ID` | primary key |
| `Performance_Category` | **team discovery** — which teams are under-performing |
| `Team_Performance` | reported for context; fallback discovery rule if the category column is missing |
| `Team_Size` | integrity check (`Σ Num_* == Team_Size`) |
| `Num_Developer`, `Num_QA`, `Num_DevOps`, `Num_Business_Analyst`, … | recomputed from the repaired roles, then verified |

### Required — supplied by the sentiment subsystem

**`dataset/sentiment_fuzzy.csv`** — one row per employee. The loader accepts three shapes
and normalises them, so the other subsystem is not forced into one format:

1. **Full** — `Employee_ID, mu_neg, mu_neu, mu_pos, s_crisp, n_reviews, join_level`
2. **Memberships only** — `Employee_ID, mu_neg, mu_neu, mu_pos` → `s_crisp` is defuzzified
   by centroid over `[0.0, 0.5, 1.0]`
3. **Crisp only** — `Employee_ID` plus any one of `s_crisp`, `sentiment`,
   `sentiment_score`, `score`, `compound`, `polarity` → memberships are fuzzified back

Two invariants are enforced (`sentiment_io.validate`):

- `s_crisp ∈ [0, 1]` for every row
- `Employee_ID` covers the roster with no duplicates

This file is **present and validated** — see [How sentiment reaches the
system](#how-sentiment-reaches-the-system). It is generated by `run.py sentiment`
rather than edited by hand.

If it is ever absent, every run falls back to a clearly-flagged synthetic Beta(5,3)
placeholder and prints a banner; no number produced under the placeholder is valid.
`doctor` reports which of the two is in force as `sentiment_is_placeholder`.

### Optional

**`employee_reviews.csv`** — raw review text with star ratings. Only used by
`sentiment_io.audit_against_rating`, which checks whether the delivered sentiment scores
actually correlate with the stars. Not part of the recommendation path.

---

## Commands

### `python run.py recommend`

**The main path.** Discovers the under-performing teams, forms pools, solves them, and
prints a ranked list of non-overlapping recommendations.

| Flag | Default | Meaning |
|---|---|---|
| `--pool-size N` | `2` | teams per pool. `2` is exhaustive and proven optimal; `3+` uses local search |
| `--top N` | `10` | how many recommendations to return |
| `--category NAME` | `Low` | which `Performance_Category` counts as under-performing |
| `--budget N` | `2` | max swaps per role class |
| `--lam F` | `0.5` | balance weight λ |
| `--save` | off | write `recommended_swaps.csv`, `recommended_pools.csv`, `low_teams.csv` |

### `python run.py sentiment`

Builds `dataset/sentiment_fuzzy.csv` from the sentiment subsystem's delivered dataset.
It searches the paths in `config.SENTIMENT_INPUTS`, resolves the columns it needs by
alias, and reports which column it used for each field.

| Flag | Default | Meaning |
|---|---|---|
| `--scored PATH` | auto | read this dataset instead of searching |
| `--dry-run` | off | report the columns and audits, write nothing |

It refuses to write in two cases: if `s_crisp` regresses on
`Overall_rating` at r² > 0.95 (contract invariant 1 — sentiment would be restating a
number the engine already holds, and the objective collapses to zero-sum), or if
`sentiment_io.validate` reports any contract violation.

### `python run.py discover`

Shows the discovery step alone: which teams were flagged, by what rule, how many members
each can actually move, and the headroom ranking of the pools formed from them. Use it to
sanity-check the input before trusting a recommendation.

### `python run.py doctor`

End-to-end health check. Loads both datasets, applies the role repair, validates the
sentiment contract, runs discovery, and solves the highest-headroom pool as a smoke test.
Every number it prints is an invariant that should hold on any correct input.

### `python run.py roles`

The role-repair audit. `--tokens` adds the token-lift evidence table, `--save` writes the
repaired employee and team frames.

### `python run.py swap`

Manual override for one named pool — `--teams TEAM_0098 TEAM_0202 ...` or `--worst N`.
Retained for inspection and for the case where HR has already decided which teams to look
at. It bypasses discovery entirely.

### `python run.py sweep`

The evaluation harness: five arms over a sample of low-team pairs.

| Flag | Default | Meaning |
|---|---|---|
| `--limit N` | `200` | pairs to sample |
| `--ablate` | off | also run the λ, churn-budget and sentiment-floor ablations |
| `--save` | off | write `sweep_results.csv` |

---

## Outputs

### `recommended_swaps.csv` — the deliverable

One row per recommended move.

| Column | Meaning |
|---|---|
| `rank` | which recommendation this move belongs to (1 = highest gain) |
| `pool` | the teams exchanging members |
| `pool_gain` | the whole pool's improvement in `J` |
| `Employee_ID`, `Title`, `Role` | who moves, and the role class the swap matched on |
| `from_team` → `to_team` | the move |
| `swap_with` | the employee moving the other way |
| `Performance Score` | raw score, 50–100 |
| `Sentiment` | observed `s_crisp`, 0–1 |
| `Value` | `100 · v(e)`, the single number the optimiser ranks on |
| `reason` | plain-language justification naming the role match and the value gap |

### `recommended_pools.csv` — the summary

One row per recommendation: `rank`, `pool`, `n_moves`, `weakest_team`, its score before and
after, `J_before`, `J_after`, `gain`, `pct_of_achievable`, `proven_optimal`.

`pct_of_achievable` is the share of the **provable** maximum this recommendation captures
(see [Guarantees](#guarantees-and-constraints)). Values in the 95–99% range mean the churn
budget is barely binding.

### `low_teams.csv` — the discovery log

Every team the system flagged, worst first, with the rule that flagged it, its value-based
`team_score`, how many members it can actually move, and which role classes those members
occupy. A team with `movable = 0` cannot participate in any exchange.

### `sweep_results.csv` — the evaluation log

One row per evaluated pair: the gain under each of the five arms, plus `n_moves_full`,
`states_full`, `proven_optimal`, `pct_of_achievable`, `sentiment_changed_decision`,
`n_protected`. This is the file the results table in the paper is computed from.

### `employees_roles_derived.csv`, `teams_roles_recomputed.csv`

Written by `roles --save`. The repaired inputs, for inspection or for handing to another
subsystem.

---

## Configuration

Every knob lives in `src/config.py`. Nothing else in the project holds a literal path or
threshold.

```python
# Discovery — the system finds its own teams
LOW_CATEGORY = "Low"          # Performance_Category value treated as under-performing
LOW_QUANTILE = 0.25           # fallback rule if that column is absent
POOL_SIZE = 2                 # 2 => every pool proven optimal
MIN_SHARED_ROLES = 1          # a pool needs a shared role class to be feasible
TOP_RECOMMENDATIONS = 10

# Value function  v(e) = perf_norm(e) * phi(s(e)),  phi(s) = PHI_BASE + PHI_SLOPE*s
PHI_BASE = 0.5                # phi(0) = 0.5  -> a wholly disengaged person contributes half
PHI_SLOPE = 1.0               # phi(1) = 1.5  -> a wholly engaged one contributes 1.5x

# Objective  J = min_t TeamScore(t) - LAMBDA_BALANCE * std_t TeamScore(t)
LAMBDA_BALANCE = 0.5          # load-bearing: at 0 the system loses to performance-only

# Search
CHURN_BUDGET = 2              # max swaps per role class
ALLOW_LEAD_SWAPS = False      # leadership continuity
MAX_ENUM_STATES = 2_000_000   # above this, exhaustive falls back to local search

# Protective filter
SENTIMENT_FLOOR = 0.35        # below this an employee is held in place
ENFORCE_SENTIMENT_FLOOR = True
```

---

## Project layout

```
final_year/
├── run.py                    CLI: recommend · discover · roles · swap · sweep ·
│                             sentiment · doctor
├── src/
│   ├── config.py             every path and knob                          (90)
│   ├── roles.py              role repair, token-lift measurement         (278)
│   ├── sentiment_io.py       the contract with the sentiment subsystem   (265)
│   ├── sentiment_build.py   delivered dataset -> contract, with checks   (200)
│   ├── objective.py          v(e), J, the bound, the protective filter   (245)
│   ├── search.py             exhaustive (K=2) and local search (K>=3)    (284)
│   ├── pipeline.py           discovery, pool formation, pruned ranking   (372)
│   ├── evaluate.py           five arms, three ablations                  (239)
│   └── fis.py                RETIRED — reference material only           (307)
├── dataset/
│   ├── Employee_Team_Assignment.csv     5,000 × 17
│   ├── Team_Dataset.csv                   570 × 18
│   ├── <delivered sentiment dataset>    ← handed to us by the sentiment subsystem
│   └── sentiment_fuzzy.csv              ← written by `run.py sentiment`
├── output/                   all generated CSVs
├── docs/                     background papers
├── archive/                  superseded datasets and artefacts — nothing live reads these
├── README.md                 this file
├── NeuroFuzzy_BERT_Sentiment.ipynb   REFERENCE ONLY — not a pipeline stage
├── WORKFLOW.md               concept, method, and full code walkthrough
└── PROJECT.md                the design record: derivations, measurements, rejected paths
```

`src/fis.py` implements the Mamdani inference system that used to predict counterfactual
satisfaction `ŝ(e,t)`. **It is no longer in the decision path** and is not imported by
`src/__init__.py`. It is kept because the rule base, its organisational-behaviour
provenance and the leave-one-out feature construction are worth reporting as an explored
and rejected design.

---

## Guarantees and constraints

**Hard constraints — never violated, by construction, not by penalty:**

| Constraint | Enforcement |
|---|---|
| Role match | a swap exchanges employees of the same derived role class |
| Team size | every exchange is symmetric k-for-k, so sizes never change |
| Leadership continuity | `Is_Team_Lead` employees never move (`ALLOW_LEAD_SWAPS = False`) |
| Protective floor | employees below `SENTIMENT_FLOOR` are never moved |
| Churn budget | at most `CHURN_BUDGET` swaps per role class |
| Disjointness | recommendations in one shortlist never share a team, so all can be actioned together |

**Optimality — two separate claims, deliberately not conflated:**

1. **Per pool.** At `pool_size = 2` the search enumerates every feasible set-pair under the
   churn budget and evaluates them all by numpy broadcasting. The result is the **proven
   global optimum** for that pool. Median 18 states; 99.90% of a 1,000-pair sweep solved
   exhaustively. At `pool_size ≥ 3` the search is steepest-ascent local search and
   `proven_optimal` is reported as `False`.

2. **Across pools.** `objective.equalisation_bound` gives a provable ceiling on what *any*
   reallocation of a pool could achieve:

   ```
   J  =  min_t mean_t  −  λ·std  ≤  min_t mean_t  ≤  pool mean
   ```

   because a minimum can never exceed the mean of the set it is drawn from, and `−λ·std`
   only lowers `J` further. Pools are therefore solved in descending order of headroom, and
   the search stops as soon as every unexamined pool's ceiling falls below the weakest
   recommendation already on the list. That is a proof, not a heuristic cutoff: on the
   shipped data it solves **394 pools out of 9,349** and reports
   `no_unexamined_pool_can_beat_shortlist = true`.

**The zero-sum property, stated explicitly.** Because `v(e)` has no team argument, the pool
total is invariant under every legal swap:

```
pool value sum before : 13.3630905400
pool value sum after  : 13.3630905400
delta                 : +0.0000000000

TeamScore(TEAM_0098)   7.1425  ->  53.5868
TeamScore(TEAM_0202)  65.7441  ->  58.7774

J before -7.507952    J after 52.289204    gain +59.797155
```

Nothing was created. The *minimum* moved. This is the thesis of the project and the reason
the objective cannot be a mean.

---

## Current status

| Component | State |
|---|---|
| Role repair | **done** — χ² p = 9.50e-176, 0.10% fallback, 570/570 teams reconcile |
| Team discovery | **done** — 143 teams found, 0 with no movable member |
| Pool formation | **done** — 9,349 feasible pairs under the real sentiment floor |
| Value function | **done** — intrinsic `v(e)`, zero-sum verified to 10 decimal places |
| Bound + pruning | **done** — 95.8% of pools skipped with a proof |
| Optimiser | **done** — exhaustive at K=2, local search beyond |
| Protective filter | **done** — 791 employees held; measured cost 8.02 -> 7.10 mean gain |
| Evaluation harness | **done** — five arms, three ablations, 1,000-pair sweep |
| CLI + outputs | **done** — six commands, four CSVs |
| Sentiment adapter | **done** — `run.py sentiment`, column resolution + checks |
| **Real sentiment data** | **done** — delivered, validated, placeholder retired |

Results from the **1,000-pair sweep on the delivered sentiment data** (the same scale
the paper's Table II is computed at):

| Arm | mean gain | median | % improved | % **damaged** | full wins | full loses |
|---|---|---|---|---|---|---|
| **full** | **8.0811** | **5.7184** | **82.10** | **0.00** | — | — |
| perf_only | 4.8470 | 1.7425 | 59.00 | 19.80 | 71.10% | **0.00%** |
| sent_only | 0.2596 | 0.0000 | 39.40 | 41.10 | 81.50% | **0.00%** |
| random | 0.1365 | 0.0000 | 45.20 | 45.30 | 80.40% | **0.00%** |
| worst_for_best | 2.0021 | 0.5779 | 53.10 | 37.50 | 77.40% | **0.00%** |

99.90% of pairs solved to proven optimality; mean 3.35 moves; median 18 states;
mean 69.06% of the provable ceiling captured.

**The full objective never loses to any baseline, and damages a pool 0.00% of the
time.** The intuitive human heuristic — trade the weaker team's worst member for the
stronger team's best — **damages the pool in 37.50% of cases**. That result depends on
no modelling choice this project made, and it is the one a practitioner should read.

Sentiment changed the selected movers on **71.1%** of pairs — it is not decorative. But
`sent_only` has a median gain of exactly **0.0000** and damages the pool 41.10% of the
time: sentiment alone recommends nothing useful. It is a modifier on performance, not
a driver, and the paper says so.

**Ablations** (200 pairs):

| λ | wins vs perf_only | loses vs perf_only |
|---|---|---|
| **0.00** | 51.00% | **30.00%** |
| 0.25 | 58.00% | 23.00% |
| **0.50** | **68.50%** | **0.00%** |
| 1.00 | 80.50% | 0.50% |
| 2.00 | 80.50% | 0.50% |

The balance term is not a preference. At λ = 0 the system **loses to a
performance-only baseline 30% of the time**, because with intrinsic value a pure `min`
has far less to grip. λ = 0.5 is the *only* setting tested with a **0.00% loss rate** —
λ ≥ 1 wins more often but reintroduces a 0.5% loss, so it is not a free improvement.

| churn budget | mean gain | mean moves | % of achievable |
|---|---|---|---|
| 1 | 6.9761 | 2.52 | 62.27 |
| **2** | **7.1006** | **3.26** | **64.55** |
| 3 | 7.1059 | 3.38 | 64.64 |

Budget 2 → 3 buys +0.005 gain for +0.12 moves. Budget 2 remains the right default, and
it is also the last budget at which every pair solves exhaustively (B = 3 drops to
99.5% proven optimal).

| sentiment floor | held | mean gain | median | wins vs perf_only |
|---|---|---|---|---|
| 0.00 (off) | 0 | 8.0160 | 5.3832 | 78.50% |
| **0.35 (on)** | **791** | **7.1006** | **3.9705** | 68.50% |

Holding disengaged employees in place costs **11.42% of mean gain** — nearly double the
5.1% measured under the placeholder, because the delivered sentiment is far more
dispersed (std 0.327 against 0.162) and the floor therefore binds on 791 employees
rather than 288. That is the measured price of the ethical constraint, and quoting it
is the point.

> **The floor is miscalibrated, not wrong.** `SENTIMENT_FLOOR = 0.35` was chosen
> against the placeholder's distribution. On the delivered data it holds 791
> employees and strips two low teams of every movable member
> (`low_teams_with_no_movable_member` rose from 0 to 2, and feasible pools fell from
> 9,757 to 9,349). Re-expressing it as a quantile of the observed distribution is a
> modelling decision that changes a published result, so it has been left alone and
> documented in `src/config.py`.

---

## How sentiment reaches the system

**We do not compute sentiment.** A separate subsystem trains and tests its own model
over the review corpus and hands us a finished dataset keyed by `Employee_ID`.
`run.py sentiment` locates the columns it needs in that dataset, checks it against the
roster, and writes the contract file the engine reads.

```
delivered sentiment dataset          Employee_ID + a sentiment score
        |                            (+ class probabilities, if shipped)
        |  run.py sentiment  --  resolve columns, check integrity
        v
dataset/sentiment_fuzzy.csv          what the engine reads
```

Column names are not fixed, because the subsystem names its own output. Each field is
resolved by alias and the command **prints which column it chose for what**, so a
wrong guess is visible rather than silent:

```
"columns_used": {
  "id":     "Employee_ID",           "score":  "Model_Sentiment_Score",
  "mu_neg": "Model_Prob_Low",        "mu_neu": "Model_Prob_Medium",
  "mu_pos": "Model_Prob_High",       "rating": "Overall_rating"
}
```

Nothing is written until the dataset passes every integrity check:

| Check | On the delivered dataset |
|---|---|
| one score per employee, no duplicates, no unknown IDs | 5,000 / 5,000 (**100.0%**) |
| score present and finite, within `[0, 1]` | 0 nulls; range [0.027, 0.993] |
| score is not a restatement of the star rating | r² = **0.29** — passes |
| `sentiment_io.validate` contract check | clean |

That third row is the one that matters. Had the delivered score simply echoed the star
rating (r² > 0.95), sentiment would be a second copy of a number the engine already
holds, every employee's value would be intrinsic twice over, and the pool objective
would return to exactly zero-sum. `run.py sentiment` **refuses to write** in that case
rather than let it through quietly.

The delivered dataset also carries `Title` and `Department`, and both match the roster
for all 5,000 employees — the reviews were attached on occupation, not at random.

---

## Limitations

1. **One review per employee.** The delivered dataset resolves exactly one review to
   each of the 5,000 employees. A single review is a thin, cross-sectional basis for a
   person's engagement: it carries no trajectory, and `n_reviews = 1` everywhere means
   the confidence weight in the contract cannot discriminate between rows.

2. **The sentiment scores are produced outside this repository.** That is by design —
   another subsystem owns the model — but it means the scores cannot be re-derived
   here, and working convention 6 (*every reported number is reproducible from a
   committed notebook*) does not hold for the sentiment step. We validate what we are
   given rather than trusting it; we cannot regenerate it.

3. **No counterfactual satisfaction.** The system deliberately does not predict how happy
   an employee would be after a move. Nothing in the dataset could validate such a
   prediction — there are no post-reassignment outcomes — so the claim was withdrawn rather
   than defended. Recommendations rest on role, performance and *observed* sentiment only.

4. **No ground truth, therefore no accuracy figure.** There is no record of a real
   reassignment and its result, so the system cannot be validated against outcomes. It is
   evaluated against baselines on its own objective, which is weaker and is stated as such.

5. **Min-max normalisation is set by two individuals.** `perf_norm` is scaled by the
   company-wide minimum and maximum `Performance Score` (50 and 100). One consequence is
   visible in the data: an employee scoring exactly 50 gets `v = 0.00` regardless of their
   sentiment. The scale is also fragile to a single outlier hire.

6. **The performance metrics are mutually independent** (mean |r| = 0.0109, max 0.0272).
   There is no latent quality factor in this data, which is precisely why a learned model
   was rejected in favour of an explicit objective.

7. **"Low" means bottom quartile, not crisis.** The discovered band spans
   `Team_Performance` 73.70–78.09 — a 4.39-point window. These are the weakest teams, not
   failing ones.

8. **`Team_Performance` and the value-based `team_score` disagree.** The shipped column is
   computed from `Composite_Score`; the objective uses `Performance Score` and sentiment. A
   team can rank differently under each. Discovery uses the former, optimisation the
   latter, and the two are reported side by side in `low_teams.csv` so the gap is visible.

9. **Local search beyond two teams.** At `pool_size ≥ 3` the per-pool result is a local
   optimum. The cross-pool pruning proof still holds, but `proven_optimal` is `False` and
   should not be quoted as optimality.

---

## Role derivation

The shipped `Role` column carries no information: a χ² test of `Role` against `Title`
gives p = 0.5463, and against `Department` p = 0.5112. The labels are random, so they
cannot define a swap-feasibility constraint.

Roles are therefore re-derived from evidence. A naive keyword rule ("Consultant" →
Business Analyst, "Engineer" → Developer) was tried first and produced 52% Business
Analysts, which is absurd. Measuring how strongly each title token predicts department
explains why:

```
token           n     lift   verdict
tester         34     7.40   occupation
test          112     7.21   occupation
qa             32     7.05   occupation
automation     48     7.05   occupation
network        57     4.21   occupation
administrator  62     4.20   occupation
devops         39     3.98   occupation
sap           151     3.42   occupation
developer     314     1.57   occupation
--------------------------------------- threshold: lift >= 1.5
software      982     1.24   weak
manager       412     1.11   grade
senior       1462     1.02   grade
engineer     1057     1.00   grade      ← appears in every department
associate     761     0.97   grade
consultant   2008     0.94   grade      ← 40% of the workforce; a rank, not a job
lead          159     0.92   grade
analyst       755     0.91   grade
```

**"Consultant", "Senior", "Engineer", "Manager", "Analyst", "Lead" and "Associate" are
rungs on a promotion ladder, not occupations.** Only 16 tokens clear a lift of 1.5;
everything else resolves through `Department`, which is real.

| | shipped | derived |
|---|---|---|
| χ² p (Title vs Role) | 0.5463 | **9.50e-176** |
| χ² p (Department vs Role) | 0.5112 | **0.0** |
| Developer / DevOps / BA / QA | 2432 / 667 / 422 / 909 | **2579 / 961 / 851 / 609** |
| resolved from title | — | 22.04% |
| resolved from department | — | 77.86% |
| resolved by fallback | — | **0.10%** (5 employees) |
| Σ Num_* == Team_Size | 570/570 | **570/570** |

`Team Lead` is deliberately **not** a derived role. It is a structural slot — exactly one
per team — carried as the boolean `Is_Team_Lead`, and leads are excluded from swaps.

Reproduce with `python run.py roles --tokens`.

---

## Further reading

- **`WORKFLOW.md`** — the concept, the objective, and a plain-language walkthrough of the
  code path from first file read to final HR output. Read this to understand the system.
- **`PROJECT.md`** — the full design record: derivations, measurements, rejected
  alternatives, and open questions.

---

## Project

Final-year project, Computer Science & Engineering.

| | |
|---|---|
| Manshi Bachhan | 16900123057 |
| Ankit Kumar | 16900123017 |
| Nandan Kumar | 16900123060 |
| Beauty Mondal | 16900123031 |
| Gautam Kumar Mishra | 16900123043 |

Under the guidance of Prof. Partha Ghosh.
