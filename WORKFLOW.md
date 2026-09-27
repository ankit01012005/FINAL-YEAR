# Workflow — Concept, Method, and Code Walkthrough

*A single document explaining what this project is, why it works the way it does, and
exactly what happens in the code from the first line executed to the report an HR manager
finally reads.*

Read `README.md` for installation and commands. Read this to **understand** the system.

---

## Contents

**Part I — The Concept** *(read this like a paper)*
1. [Abstract](#1-abstract)
2. [The problem](#2-the-problem)
3. [Objective](#3-objective)
4. [Four traps, and why the obvious method fails](#4-four-traps-and-why-the-obvious-method-fails)
5. [Our method, in one page](#5-our-method-in-one-page)

**Part II — The Code Workflow** *(read this to understand the program)*
6. [Map of the whole journey](#6-map-of-the-whole-journey)
7. [Stage-by-stage walkthrough](#7-stage-by-stage-walkthrough)
8. [A complete worked example](#8-a-complete-worked-example)
9. [The evaluation branch](#9-the-evaluation-branch)

**Part III — The Output**
10. [What HR finally sees](#10-what-hr-finally-sees)
11. [Symbol glossary](#11-symbol-glossary)
12. [Assumptions and honest limits](#12-assumptions-and-honest-limits)
13. [Deferred work — agreed, not yet implemented](#13-deferred-work--agreed-not-yet-implemented)

---
---

# Part I — The Concept

## 1. Abstract

Organisations routinely discover that some of their teams are under-performing, and the
cheapest available remedy is not hiring but **redistribution**: moving people who are
already on the payroll between the teams that need them most. In practice this decision is
made by intuition — "swap our weakest developer for one of theirs" — and it is usually
wrong, for a reason that is provable rather than anecdotal.

This project takes a company dataset, **discovers the under-performing teams itself**,
groups them into pools that are actually able to exchange members, and recommends
**which specific employees should swap places** between them. Each recommendation names the
role class the swap matched on, the performance gap that motivates it, and the sentiment
score of everyone involved.

Three properties distinguish the method:

1. **The objective is non-linear on purpose.** We prove that any sum-or-mean team score is
   *exactly* invariant under every legal swap the system could propose — it assigns zero
   value to all of them, measured to ten decimal places. Only a Rawlsian
   `min − λ·std` aggregation responds at all.
2. **The optimiser is exact where it matters.** For a two-team pool the search enumerates
   every feasible exchange under a churn budget and returns the **proven global optimum**.
   Across the 9,757 candidate pools, a provable upper bound lets the system skip **98.7%**
   of them and still guarantee that nothing skipped could have beaten the shortlist.
3. **It refuses to claim what it cannot verify.** The system does *not* predict how
   satisfied an employee would be in a team they have never worked in. That counterfactual
   is unfalsifiable against this data, so it was withdrawn. Recommendations rest on role,
   performance, and *observed* sentiment only, and the output is explicitly advisory.

The thesis is therefore **redistributive, not generative**: reassignment does not
manufacture value, it stops a team from failing while the talent to rescue it already sits
inside the pool.

---

## 2. The problem

### 2.1 The situation

A company has 5,000 employees across 570 teams. Some teams are doing badly. Management
wants to know what to do about it, and the only lever available in the short term is who
sits on which team.

The naive framing — "identify the weak employees and remove them" — is not available and
not desirable. What *is* available is exchange: team A gives someone to team B, and team B
gives someone back. Nobody is hired, nobody is fired, no team changes size.

### 2.2 The constraints, and why they are realistic

| Constraint | Why it exists |
|---|---|
| **Role match** | You cannot fill a QA vacancy with a DevOps engineer. Every exchange is same-role-class. |
| **Size preservation** | Exchanges are symmetric k-for-k, so headcount and budget lines never move. |
| **Leadership continuity** | Team leads do not move. Losing a lead destabilises a team in ways this model cannot capture. |
| **Churn budget** | At most 2 swaps per role class. Reorganising everyone at once is not a plan; it is chaos. |
| **Protective floor** | An employee whose sentiment is already below 0.35 is not moved. Involuntary reassignment is a known churn accelerant for exactly the group already at risk. |

Each of these is enforced **by construction** — the search never generates an illegal
state — rather than by penalising it afterwards.

### 2.3 The available information

Two files ship with the project, and one arrives from the sentiment subsystem.

Of the 17 employee columns, only **five** actually reach a decision:

| Column | Role in the method |
|---|---|
| `Employee_ID` | identity |
| `Team_ID` | the current assignment — the thing being changed |
| `Title` | evidence for role derivation |
| `Department` | evidence for role derivation |
| `Performance Score` | the performance half of the value function |

plus `s_crisp` from `sentiment_fuzzy.csv` — the sentiment half. Nine columns
(`KPI Score`, `Attendance (%)`, `Task Completion (%)`, `Work Hours Logged`,
`Training Hours`, `Promotion Eligibility`, `Project_Size`, `Peer_Rating_100`,
`Composite_Score`) are carried but unused, and **no column of `Team_Dataset.csv` reaches
the objective at all** — it is used only to decide which teams are under-performing.

This is stated up front because it bounds what the project can honestly claim.

---

## 3. Objective

### 3.1 In plain words

> Make the **worst** team in the pool as good as possible, and do not achieve that by
> making the teams wildly unequal.

Not "maximise total output". Total output cannot be changed by moving people around — we
prove this below. What *can* be changed is which team ends up carrying the deficit.

### 3.2 Formally

**Employee value.** Each employee has one number, independent of where they sit:

```
    v(e)  =  perf_norm(e)  ·  φ(s(e))

    perf_norm(e) = (perf(e) − min perf) / (max perf − min perf)        ∈ [0, 1]
    φ(s)         = PHI_BASE + PHI_SLOPE · s  =  0.5 + s                ∈ [0.5, 1.5]
```

φ is the engagement multiplier: a wholly disengaged person contributes half of their raw
capability, a wholly engaged one contributes one-and-a-half times it. The functional form
is a modelling choice; its *shape* (monotone increasing, bounded away from zero) is the
part that matters.

**Team score.** The mean effective contribution of a team's members, scaled for legibility:

```
    TeamScore(t)  =  100 · mean{ v(e) : e ∈ t }
```

**Pool objective.** For a pool of teams `T`:

```
    J(T)  =  min  TeamScore(t)   −   λ · std  TeamScore(t)
             t∈T                          t∈T
```

with `λ = LAMBDA_BALANCE = 0.5`. The first term is **Rawlsian**: the pool is judged by its
worst-off member team. The second penalises dispersion, so the optimiser cannot satisfy the
first term by hollowing out one team to prop up another.

**The optimisation problem.** Over all role-matched, size-preserving, lead-respecting,
floor-respecting exchanges within a churn budget, maximise `J`.

### 3.3 Why the constraints make it tractable

The churn budget is not only an operational nicety — it is what converts an intractable
problem into an exactly solvable one. Without it, a two-team pool with 3 and 20 members
admits an astronomical number of reallocations. With a budget of 2 per role class, the
example in §8 has **28 states**, all of which are evaluated. The median across a 300-pair
sweep is **22 states**. The constraint that makes the recommendation *humane* is the same
constraint that makes it *provable*.

---

## 4. Four traps, and why the obvious method fails

This is the heart of the project. Four plausible approaches were tried and each is
demonstrably wrong. The measurements below are reproducible.

---

### Trap 1 — Performance-only reassignment is exactly zero-sum

**The intuition:** score each team by the mean of its members' performance, then swap
people to raise the score.

**Why it fails:** it cannot. A role-matched exchange removes a set of people from team A
and adds the same number back. The *sum* over the pool is untouched, so any objective built
from sums or means over the pool is **mathematically invariant**.

Measured directly on the current code:

```
pool value sum before : 13.3933597210
pool value sum after  : 13.3933597210
delta                 : +0.0000000000
```

Not "small". Not "within tolerance". **Exactly zero**, in double precision.

This is not a quirk of the performance column. It holds for **any** value function that
depends only on the employee and not on the team — which, since we deliberately removed the
counterfactual satisfaction term (Trap 4), is exactly our situation.

**What survives:** `min` and `std` are non-linear. Redistribution that holds the sum fixed
still moves them. On the same pool:

```
J before -5.583329    J after 55.303946    gain +60.887275
```

The pool created nothing. It stopped one team from failing, using talent already inside the
pool. **This is why the objective is `min − λ·std` and can never be a mean.**

It is also why λ is load-bearing rather than a taste parameter. Ablating it:

| λ | wins vs performance-only | **loses** vs performance-only |
|---|---|---|
| **0.00** | 45.00% | **50.00%** |
| 0.25 | 56.25% | 37.50% |
| 0.50 | 71.25% | **0.00%** |
| 1.00 | 93.75% | 0.00% |

At λ = 0 the system loses to a trivial baseline half the time.

---

### Trap 2 — The Hungarian algorithm is degenerate here

**The intuition:** this is an assignment problem. Build a matrix of "how much does swapping
person *a* for person *b* gain?" and run the Hungarian algorithm on it.

**Why it fails:** for **any** value function `v(e,t)` whatsoever, the gain from exchanging
`a ∈ A` with `b ∈ B` decomposes:

```
    Δ(a,b) = [v(b,A) + v(a,B)] − [v(a,A) + v(b,B)]
           = [v(a,B) − v(a,A)]  +  [v(b,A) − v(b,B)]
           =        β(a)         +        α(b)
```

A row constant plus a column constant. The assignment problem is **invariant** under adding
a constant to any row or column, so every perfect matching between the same two sets scores
identically. The Hungarian solve returns an arbitrary one and calls it optimal.

Measured on TEAM_0030 × TEAM_0245: all 6 matchings scored **−1.4055555556**, spread
**2.8 × 10⁻¹⁴**.

**The stronger statement needs no algebra at all.** After a symmetric k-for-k exchange, the
resulting team compositions depend only on *which sets moved*, never on the pairing between
them. "Who swaps with whom" is a reporting convenience, not a decision variable. The
decision variables are the **sets**.

**The second death.** A *slot*-LAP over `c(e,t) = −v(e,t)` used to be the one legitimate
Hungarian formulation in this project — the cost was a genuine function of both arguments,
so it gave a real upper bound. Once value became intrinsic (`v(e)`, no team argument), even
that collapsed: every slot assignment now has the same total. The function was **deleted**
rather than kept as decoration, and `objective.equalisation_bound` replaced it — cheaper,
and strictly valid.

---

### Trap 3 — The role labels in the data are random

**The intuition:** the dataset has a `Role` column. Use it for the role-match constraint.

**Why it fails:** the column carries no information.

```
χ²  Role vs Title      p = 0.5463
χ²  Role vs Department p = 0.5112
```

Under a p-value of 0.55 you cannot reject the hypothesis that the labels were shuffled. A
feasibility constraint built on them would be meaningless — the system would "match roles"
while pairing a tester with a network administrator.

**The repair, and the measurement that guided it.** A naive keyword rule was tried first
("Consultant" → Business Analyst, "Engineer" → Developer) and produced **52% Business
Analysts**, which is absurd. Measuring how strongly each title token predicts department
explains why:

```
token           n     lift   verdict
tester         34     7.40   OCCUPATION
qa             32     7.05   OCCUPATION
network        57     4.21   OCCUPATION
devops         39     3.98   OCCUPATION
sap           151     3.42   OCCUPATION
developer     314     1.57   OCCUPATION
------------------------------------------ threshold: lift >= 1.5
software      982     1.24   weak
manager       412     1.11   grade
senior       1462     1.02   grade
engineer     1057     1.00   grade     ← in every department equally
consultant   2008     0.94   grade     ← 40% of the workforce; a rank, not a job
analyst       755     0.91   grade
```

**"Senior", "Consultant", "Engineer", "Manager", "Analyst", "Lead" and "Associate" are rungs
on a promotion ladder, not occupations.** They describe *how senior* someone is, not *what
they do*. Only 16 tokens clear a lift of 1.5. Everything else falls through to
`Department`, which is real.

Result: χ² p goes from 0.5463 to **9.50 × 10⁻¹⁷⁶**, with only 5 employees (0.10%) needing a
fallback, and `Σ Num_* == Team_Size` for **570/570** teams.

---

### Trap 4 — Predicting counterfactual satisfaction is unfalsifiable

**The intuition — and this project's own earlier design.** Build a fuzzy inference system
that predicts `ŝ(e,t)`: how satisfied employee *e* *would be* in team *t*. Then value
becomes team-dependent, `v(e,t)`, reassignment genuinely creates value, and the zero-sum
result of Trap 1 is escaped.

It was built. `src/fis.py` is a full 12-rule Mamdani system with min t-norm, max
aggregation and centroid defuzzification, over leave-one-out relational features
(`perf_gap`, `peer_gap`, `mgr_gap`, `team_perf`, `team_size`, `role_scarcity`,
`dept_match`), with every rule cited to organisational-behaviour literature — relative
deprivation (Crosby 1976), person–environment fit (Kristof-Brown 2005), social loafing
(Karau & Williams 1993), similarity-attraction (Byrne 1971), token isolation (Kanter 1977),
LMX (Graen & Uhl-Bien 1995).

**Why it was withdrawn:** *nothing in the dataset can check it.* There are no
post-reassignment outcomes — no record of a person who moved and how they then felt. The
prediction is unfalsifiable against the available evidence, and the entire value claim of
the system would rest on it.

Two further problems surfaced under measurement:

- The nominal adjustment range is `DELTA_RANGE = 0.30`, but centroid defuzzification of a
  triangular aggregate never reaches the edges of its universe. The realised δ **saturates
  at ±0.15**. Quoting ±0.30 in a paper would have been wrong.
- The between-person sentiment spread that made δ look meaningful was, at the time, an
  artefact of the Beta(5,3) *placeholder* rather than of measured data.

**What replaced it.** Observed sentiment only, intrinsic to the person, `v(e) = perf_norm(e)
· φ(s(e))`. This walks straight back into Trap 1 — and that is *accepted*, because the
Rawlsian objective survives it. The system now claims less and proves more.

The retired half of the do-no-harm constraint went with it. That rule compared predicted
satisfaction before and after a move, which required exactly the counterfactual now gone.
Its defensible half survives as the **protective sentiment floor**: someone already below
0.35 is disengaged, and involuntary reassignment is a known churn accelerant for that group,
so they are held in place. This reads only an *observed* quantity, and its cost is measured
rather than assumed:

| floor | employees held | mean gain | median | wins vs perf_only |
|---|---|---|---|---|
| 0.00 (off) | 0 | 8.0160 | 5.3832 | 78.50% |
| **0.35 (on)** | **791** | **7.1006** | **3.9705** | 68.50% |

**11.4% of mean gain** is the price of the ethical constraint. Quoting it is the point.

The price is specific to the sentiment distribution the floor is applied to: the same
`0.35` held 288 employees and cost 5.1% under the old placeholder, whose spread was half
as wide. See §13.3 — a floor expressed as a quantile would survive that change.

---

## 5. Our method, in one page

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1   REPAIR                                                        │
 │  The Role column is random (p = 0.55). Re-derive it from title tokens   │
 │  that actually predict department, falling back to department itself.   │
 │  → every employee has a trustworthy role class                         │
 └────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2   VALUE                                                         │
 │  v(e) = perf_norm(e) · φ(s(e)).  One number per person, carried with    │
 │  them. Intrinsic by design — no unverifiable counterfactual.            │
 │  → a value vector over all 5,000 employees                             │
 └────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌────────────────────────────────────────────────────────────────────────┐
 │ STAGE 3   DISCOVER                                                      │
 │  Find the under-performing teams from the dataset itself (no pair is    │
 │  supplied by hand). Group them into pools that share a role class with  │
 │  movable members, else no legal swap exists.                            │
 │  → 143 teams → 9,757 feasible pools                                     │
 └────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌────────────────────────────────────────────────────────────────────────┐
 │ STAGE 4   BOUND & PRUNE                                                 │
 │  J ≤ 100 · (pool value total / headcount), because a minimum cannot     │
 │  exceed the mean it is drawn from. Solve pools in descending order of   │
 │  this ceiling and stop when the ceiling drops below the worst           │
 │  recommendation already found — a proof, not a heuristic.               │
 │  → 123 pools solved, 9,634 skipped with a guarantee                     │
 └────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌────────────────────────────────────────────────────────────────────────┐
 │ STAGE 5   OPTIMISE                                                      │
 │  K=2: enumerate every feasible set-pair under the churn budget and      │
 │       evaluate all of them by numpy broadcasting → PROVEN OPTIMAL       │
 │  K≥3: steepest-ascent local search over the 1-swap neighbourhood        │
 │  → the best exchange for each pool                                      │
 └────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌────────────────────────────────────────────────────────────────────────┐
 │ STAGE 6   SELECT & REPORT                                               │
 │  Greedily pack non-overlapping pools so every recommendation on the     │
 │  list can be actioned simultaneously. Emit who moves where, with whom   │
 │  they trade, and the role/performance/sentiment evidence for each.      │
 └────────────────────────────────────────────────────────────────────────┘
```

### Why an explicit objective and not machine learning?

A reviewer will ask. The answer is a measurement, not a preference.

**The eight performance metrics in this dataset are mutually independent** — mean |r| =
0.0109, max 0.0272. There is no latent "employee quality" factor to learn. A supervised
model would fit noise and report a confident, meaningless score.

And there is no target to fit. Learning "which swap is best" requires a record of swaps and
their outcomes. That record does not exist.

An explicit objective needs no ground truth, is fully inspectable, gives the same answer
twice, and can be **proved optimal** on the space it searches. Every number in the
recommendation traces back to a formula a reader can check by hand — as §8 does.

---
---

# Part II — The Code Workflow

## 5b. The inputs, and the shape of a run

The system is handed **datasets**, not decisions. Everything downstream follows from
them, and the run is always the same four steps.

| # | Dataset | What it is for | Required |
|---|---|---|---|
| 1 | **Employee** | the roster: who exists, their team, title, department, performance | yes |
| 2 | **Team** | which teams exist, and which are under-performing | yes |
| 3 | **Sentiment** | one engagement score per `Employee_ID`, delivered by the other subsystem | yes |
| 4 | **Reviews** | raw review text and star ratings — used *only* to audit dataset 3 for rating leakage | optional |

```
  1. LOAD      read the datasets that were given
  2. CHECK     reliability and integrity of each one, and of the joins between them
  3. COMPUTE   repair roles, build v(e), discover weak teams, form and bound pools
  4. RECOMMEND solve the pools worth solving, rank, and report with evidence
```

Step 2 is not a formality, and it is the step that has caught every real problem so
far. `python run.py doctor` runs it alone and prints one line per invariant:

| Invariant | Where it is checked |
|---|---|
| `Employee_ID` is unique across the roster | `pipeline.load_context` |
| the shipped `Role` column is not random noise | `roles.py` (it *was* — χ² p = 0.55) |
| `Σ Num_*` reconciles with `Team_Size` for every team | `roles.recompute_team_counts` |
| sentiment covers the roster 1:1, in range, no nulls | `sentiment_build`, `sentiment_io.validate` |
| sentiment is not a restatement of the star rating | `sentiment_build.audit_leakage` |
| the engine is not silently running on placeholder data | `is_placeholder` flag |

A failure at step 2 stops the run. It does not warn and continue, because a
recommendation computed from a dataset that failed its own integrity check is worse
than no recommendation at all.

---

## 6. Map of the whole journey

What calls what, in execution order:

```
python run.py recommend
   │
   ├─ run.py : main()                      parse arguments, dispatch
   │
   └─ run.py : cmd_recommend()
        │
        ├─ pipeline.load_context() ─────────────────────── ONE-TIME PREPARATION
        │     ├─ pd.read_csv(EMPLOYEES_CSV)                 5,000 × 17
        │     ├─ pd.read_csv(TEAMS_CSV)                       570 × 18
        │     ├─ roles.derive_roles()                       repair the role column
        │     ├─ roles.recompute_team_role_counts()         rebuild Num_* and verify
        │     ├─ roles.coverage_report()                    the audit numbers
        │     ├─ sentiment_io.load_sentiment()              real file or flagged placeholder
        │     ├─ objective.normalise_performance()          perf_norm(e)
        │     ├─ objective.sentiment_series()               s(e)
        │     ├─ objective.value_series()                   v(e) = perf_norm · φ(s)
        │     └─ objective.protected()                      who is held by the floor
        │              ↓
        │         Context  ── everything downstream reuses this
        │
        └─ pipeline.recommend()
              ├─ discover_low_teams()                       143 teams, worst first
              │     ├─ team_stats()                         n, value_sum, team_score
              │     └─ _movable_roles()                     who can actually move
              │
              ├─ form_pools()                               9,757 feasible pairs
              │
              └─ rank_pools()
                    ├─ pool_headroom()                      J_before and J_bound, vectorised
                    │      └─ (sorted descending by headroom)
                    │
                    ├─ analyse_pool()  ── repeated, in ceiling order
                    │     └─ search.recommend_swaps()
                    │           ├─ K=2 → search.search_pair()        EXHAUSTIVE
                    │           │        ├─ build_assignment()
                    │           │        ├─ movable_by_role()
                    │           │        ├─ _role_options()
                    │           │        └─ numpy broadcast over the cartesian product
                    │           └─ K≥3 → search.local_search()       STEEPEST ASCENT
                    │
                    └─ _select_disjoint()                   greedy non-overlapping packing
                           ↓
                 pipeline.pools_frame()  +  pipeline.moves_frame()
                           ↓
                    printed, and written to output/*.csv
```

---

## 7. Stage-by-stage walkthrough

Each stage below is **what goes in → what the code does → what comes out**.

---

### Stage 0 — `run.py` · the front door

**In:** a command line.
**Does:** `argparse` builds six subcommands and dispatches to one function.

| Command | Function | Purpose |
|---|---|---|
| `recommend` | `cmd_recommend` | **the main path** — discover, solve, rank, report |
| `discover` | `cmd_discover` | show the discovery step alone |
| `roles` | `cmd_roles` | the role-repair audit |
| `swap` | `cmd_swap` | manual override for one named pool |
| `sweep` | `cmd_sweep` | baselines and ablations |
| `doctor` | `cmd_doctor` | end-to-end health check |

Two helpers matter. `_placeholder_banner` prints a loud warning whenever sentiment is
synthetic — it is impossible to run this system and not notice. `_advisory_footer` prints
after every recommendation, restating that the output is for review and that the system
does not predict personal fit.

**Out:** control passes to `pipeline`.

---

### Stage 1 — `src/config.py` · the single source of truth

**In:** nothing. It is imported.
**Does:** resolves every path from `ROOT = Path(__file__).resolve().parents[1]`, so the
project runs identically from any working directory or notebook, and defines every
threshold in the system.

The four groups: **discovery** (`LOW_CATEGORY`, `LOW_QUANTILE`, `POOL_SIZE`,
`TOP_RECOMMENDATIONS`), **value** (`PHI_BASE`, `PHI_SLOPE`), **objective and search**
(`LAMBDA_BALANCE`, `CHURN_BUDGET`, `ALLOW_LEAD_SWAPS`, `MAX_ENUM_STATES`), and
**protection** (`SENTIMENT_FLOOR`, `ENFORCE_SENTIMENT_FLOOR`).

A final `retired` section keeps `DELTA_RANGE` alive so `fis.py` still imports, clearly
labelled as belonging to the withdrawn design.

**Out:** constants. No literal path or threshold appears anywhere else in the project.

---

### Stage 2 — `src/roles.py` · making the feasibility constraint real

**In:** the raw employee frame.
**Does:** repairs the role column, because the shipped one is random (Trap 3).

| Function | What it does |
|---|---|
| `measure_token_lift` | For each title token, how much more likely is one department given that token? Separates occupations from promotion grades. |
| `role_from_title` | Applies the measured tokens; returns `None` if no token clears the threshold. |
| `derive_roles` | The pipeline: measured token → semantic keyword → department → default. Records `Role_Source` for every employee and sets `Is_Team_Lead`. |
| `recompute_team_role_counts` | Rebuilds `Num_Developer`, `Num_QA`, … from the repaired roles and adds `Num_Total_Check`. |
| `coverage_report` | The audit numbers: source breakdown, χ² p-values, before/after distributions. |

**Out:** an employee frame with a trustworthy `Role` and `Is_Team_Lead`, a team frame whose
role counts reconcile (570/570), and the audit dict.

```
resolved from department  3,893  (77.86%)
resolved from title       1,102  (22.04%)
resolved by fallback          5  ( 0.10%)
χ² p (Title vs derived Role)   9.50e-176      ← was 0.5463
```

---

### Stage 3 — `src/sentiment_build.py` + `src/sentiment_io.py` · the sentiment input

**In:** the sentiment dataset delivered by the other subsystem.
**Does:** finds the columns we need, checks them, and hands the engine a clean frame.

**We do not compute sentiment.** Another subsystem trains and tests its own model over
the review corpus and gives us a finished dataset keyed by `Employee_ID`. Treating
that dataset as an input — rather than rebuilding it here — is what keeps this stage
small enough to trust.

`sentiment_build` does three things and nothing else:

1. **Resolve the columns.** Names are not fixed, because the subsystem names its own
   output. `Employee_ID`, the score, and the three class probabilities are each
   matched against a list of accepted spellings, and the command *prints the column
   it chose for every field*, so a wrong guess is visible instead of silent.
2. **Check the dataset against the roster.** One score per employee, no duplicates, no
   unknown IDs, no nulls, score within `[0, 1]` (a `[-1, 1]` convention is rescaled
   and the rescaling is reported).
3. **Check the score is not the rating in disguise.** `audit_leakage` regresses the
   delivered score on the star rating. Above r² ≈ 0.95 the model has learned to echo
   the rating instead of reading the text; sentiment would then be a second copy of
   a number the engine already holds and the pool objective would collapse back to
   zero-sum (Trap 1). **The command refuses to write** in that case.

Only after all three pass does it write `dataset/sentiment_fuzzy.csv`.

`sentiment_io` is then the reader. `load_sentiment` accepts three file shapes and
normalises all of them:

1. Full — `Employee_ID, mu_neg, mu_neu, mu_pos, s_crisp, n_reviews, join_level`
2. Memberships only — `s_crisp` is defuzzified by centroid over `[0.0, 0.5, 1.0]`
3. Crisp only — memberships are re-fuzzified from the scalar

`validate` enforces `s_crisp ∈ [0,1]` and full roster coverage without duplicates.
`make_placeholder` draws Beta(5,3) scores if the file is missing and **flags every
downstream report** with `is_placeholder = True`, so a missing input can never be
mistaken for a real one.

**Out:** a normalised frame plus a metadata dict carrying `is_placeholder`, the source,
and any contract violations.

> **Current state: delivered and validated.** 5,000 rows, 100% roster coverage, no
> duplicates, no unknown IDs, r² = 0.29 against the star rating. The placeholder is
> retired and every number in this document is real.

---

### Stage 4 — `src/objective.py` · what a person is worth, and what a pool is worth

**In:** the repaired employees and the sentiment frame.
**Does:** defines the entire value and scoring model — five functions, each a few lines.

| Function | Formula | Meaning |
|---|---|---|
| `normalise_performance` | `(p − min) / (max − min)` | performance on `[0,1]`, company-wide scale |
| `sentiment_series` | `s_crisp`, reindexed | observed sentiment, neutral 0.5 if missing |
| `value_series` | `v(e) = perf_norm(e) · φ(s(e))` | **one number per person** |
| `team_score` | `100 · mean{v(e)}` | a team on a readable 0–100 scale |
| `objective` | `min TeamScore − λ·std TeamScore` | **`J`** — the thing being maximised |

Then the two pieces that make the system exact and safe:

**`equalisation_bound`** — a provable ceiling, computed without any search:

```
    J = min_t mean_t − λ·std  ≤  min_t mean_t  ≤  pool mean
```

A minimum can never exceed the mean of the set it is drawn from, and `−λ·std` only lowers
`J` further. So `J ≤ 100 · (pool value total / pool headcount)`, attained only by a
perfectly equalised pool. This is cheap enough to evaluate for all 9,757 candidate pools —
which is what makes the pruning in Stage 5 possible.

**`protected`** — the boolean mask of employees held in place by the sentiment floor. It
reads only an observed quantity, which is precisely what distinguishes it from the
withdrawn do-no-harm rule.

Finally `SwapResult` — the dataclass every recommendation is returned in. Its
`counterparts()` pairs each mover with the person coming the other way (a reporting
convenience: the objective depends only on the sets), `capture_pct` reports the share of the
provable maximum captured, and `to_frame()` renders the HR table.

**Out:** `v(e)` for all 5,000 employees; the scoring functions; the bound; the result type.

---

### Stage 5 — `src/pipeline.py` · discovery, pooling, and the pruned ranking

This is the stage that implements the project's defining behaviour: **the system finds its
own teams.**

**In:** the `Context`.

#### 5a. `discover_low_teams` — which teams need help

Uses `Performance_Category == "Low"` when the column exists. That column was verified to be
a clean quartile split of `Team_Performance` (Low 73.70–78.09, Medium 78.10–80.86, High
80.89–86.63, with the 0.25 quantile at exactly 78.092) — unlike `Role`, it can be trusted.
When it is absent, the same split is recomputed from `LOW_QUANTILE`.

For each discovered team it also reports what actually matters operationally: the
value-based `team_score`, how many members are **movable** (not a lead, not below the
floor), and which role classes those members occupy.

**Out:** 143 teams, worst first.

```
  Team_ID  Team_Performance  Team_Size  n  value_sum  team_score  movable  movable_roles
TEAM_0098             73.70          3  3   0.260001    8.666695        2  Business Analyst,Developer
TEAM_0199             74.18          4  4   0.428050   10.701254        2  Business Analyst,Developer
TEAM_0123             74.98          3  3   0.534771   17.825712        1  Developer
```

#### 5b. `form_pools` — who can trade with whom

A pool is **infeasible** unless some role class has movable members in at least two of its
teams — otherwise no legal swap exists and solving it is wasted work.

- `pool_size = 2` enumerates every feasible pair. This keeps every pool inside the
  exhaustive, provably-optimal regime. **9,757** of the 10,153 possible pairs are feasible.
- `pool_size ≥ 3` builds pools greedily: seed with the weakest unassigned team, then add the
  teams sharing the most role classes with it, breaking ties towards the largest score gap —
  because a pool of equally-placed teams has nothing to redistribute.

#### 5c. `pool_headroom` — the ceiling, for everything at once

Fully vectorised over pools of any size. Team means and counts are gathered once, then
`J_before` and `J_bound` are computed for all 9,757 pools in a handful of numpy operations.
No search is involved.

#### 5d. `rank_pools` — solve in ceiling order, then stop with a proof

```
    sort pools by (J_bound − J_before) descending
    for each pool in that order:
        if we already hold `top` disjoint recommendations
           and this pool's ceiling ≤ the weakest of them:
               STOP — no unexamined pool can beat the shortlist
        else solve it
```

The stopping rule is a **theorem, not a heuristic**: any unvisited pool's true optimum is
bounded by its ceiling, which is bounded by the weakest recommendation already held. On the
shipped data this solves **123 pools and skips 9,634**.

#### 5e. `_select_disjoint` — recommendations that can all be actioned

Two pools sharing a team are mutually exclusive: the same person cannot move to two places,
and the second pool's arithmetic assumes the first was never applied. Without this the
shortlist degenerates into the *same* weak team paired against four different partners —
five rows that look like five recommendations but are really one.

A greedy set-packing over solved pools (best gain first, skip conflicts) fixes it. Pools
with no improving swap are dropped: "do nothing" is a valid answer but not a recommendation.

**Out:** a ranked, non-overlapping list of `SwapResult`s, plus the meta dict.

```json
{
  "low_teams_found": 143,
  "candidate_pools": 9757,
  "pools_solved": 123,
  "pools_pruned_by_bound": 9634,
  "pools_dropped_no_gain": 1,
  "pools_dropped_overlapping": 117,
  "no_unexamined_pool_can_beat_shortlist": true,
  "all_pools_proven_optimal": true
}
```

---

### Stage 6 — `src/search.py` · the optimiser

**In:** the employees, `v(e)`, the pool's team IDs, and the protected mask.

`recommend_swaps` dispatches on pool size.

#### `search_pair` — exhaustive, for two teams

The whole point is avoiding a Python loop over compositions. The trick is that **deltas are
additive across role classes**, so the cartesian product is evaluated by broadcasting.

1. `build_assignment` — team → member tuple.
2. `movable_by_role` — role → movable members, after removing leads and protected employees.
3. For each shared role class, `_role_options` enumerates every `(S_A, S_B)` set-pair of
   equal size from `k = 0` up to the churn budget.
4. For each option, precompute the value delta it induces on team A. Because value is
   intrinsic, **team B's delta is exactly the negative** — one line of code that *is* the
   zero-sum property:

   ```python
   score_a = 100.0 * (base_a + total) / n_a
   score_b = 100.0 * (base_b - total) / n_b
   ```

5. Broadcast the per-class delta arrays into one array over the full product, evaluate `J`
   everywhere at once, and take `argmax`. For two teams the standard deviation of a 2-point
   sample is exactly half the absolute difference, so:

   ```python
   j = np.minimum(score_a, score_b) - lam * np.abs(score_a - score_b) / 2.0
   ```

Every feasible state under the budget is evaluated, so the answer is the **proven global
optimum**. If the product ever exceeds `MAX_ENUM_STATES`, it falls back to local search and
says so in `method`.

#### `local_search` — steepest ascent, for three or more teams

Each iteration evaluates every feasible single swap across every team pair in the pool and
commits the best strictly-improving one, stopping at a local optimum or the churn budget.
`proven_optimal` is `False` and is reported as such. This is the honest price of allowing
pools larger than two.

**Out:** a `SwapResult` — the moves, `J` before and after, per-team scores before and after,
the bound, the state count, and whether optimality was proven.

---

### Stage 7 — back in `run.py` · rendering

`pools_frame` renders one row per recommendation; `moves_frame` renders one row per moved
person, tagged with the rank and pool it belongs to. `--save` writes three CSVs:
`recommended_swaps.csv`, `recommended_pools.csv`, `low_teams.csv`.

---

## 8. A complete worked example

The top recommendation from a real run, traced end to end. Every number below can be
recomputed by hand from the two tables.

**The pool:** TEAM_0098 (3 members) and TEAM_0202 (20 members).

**TEAM_0098 — every member:**

| ID | Title | Role | Lead? | perf | perf_norm | s | 100·v |
|---|---|---|---|---|---|---|---|
| 330480 | Sap Sd Consultant | Business Analyst | no | 51 | 0.02 | 0.8000 | **2.60** |
| 308441 | Senior Software Engineer | Developer | no | 51 | 0.02 | 0.6295 | **2.26** |
| 727077 | Associate Consultant | QA | **yes** | 58 | 0.16 | 0.8213 | **21.14** |

```
TeamScore(TEAM_0098) = (2.60 + 2.26 + 21.14) / 3 = 8.67
TeamScore(TEAM_0202) = 1313.32 / 20                = 65.67
```

**Step 1 — discovery.** Both teams carry `Performance_Category == "Low"` (73.70 and 78.09),
so both enter the candidate set of 143.

**Step 2 — movable members.** Employee 727077 is the team lead, so TEAM_0098 can move only
two people. Nobody in either team is below the sentiment floor.

```
TEAM_0098 movable: {Business Analyst: [330480], Developer: [308441]}
TEAM_0202 movable: {Business Analyst: [376693], DevOps: [5 people], Developer: [13 people]}
```

Shared role classes: **Business Analyst** and **Developer**. DevOps is present in TEAM_0202
but not TEAM_0098, so no DevOps swap is possible.

**Step 3 — the objective before.**

```
J_before = min(8.67, 65.67) − 0.5 · std(8.67, 65.67)
         = 8.67 − 0.5 · (65.67 − 8.67)/2
         = 8.67 − 14.25
         = −5.58
```

Negative, because the pool is extraordinarily unbalanced.

**Step 4 — the ceiling.**

```
J_bound  = 100 · (26.00 + 1313.32) / (3 + 20) = 58.23
headroom = 58.23 − (−5.58) = 63.82
```

This is the highest headroom of all 9,757 pools, so it is the **first pool solved**.

**Step 5 — the state count.**

```
Business Analyst : 1 movable × 1 movable, budget 2  →  k=0 (1 option) + k=1 (1 option)  =  2
Developer        : 1 movable × 13 movable, budget 2 →  k=0 (1 option) + k=1 (13 options) = 14

total states = 2 × 14 = 28
```

All 28 are evaluated by broadcasting. `method = "exhaustive (budget=2, 28 states)"`.

**Step 6 — the winner.** Two exchanges:

| out of TEAM_0098 | 100·v | ↔ | into TEAM_0098 | 100·v |
|---|---|---|---|---|
| 330480 (Business Analyst) | 2.60 | ↔ | 376693 (Business Analyst) | 38.37 |
| 308441 (Developer) | 2.26 | ↔ | 157522 (Developer) | 135.26 |

**Step 7 — the objective after.**

```
TEAM_0098 = (21.14 + 38.37 + 135.26) / 3   = 64.92
TEAM_0202 = (1313.32 − 173.63 + 4.86) / 20 = 57.23

J_after = min(64.92, 57.23) − 0.5 · (64.92 − 57.23)/2
        = 57.23 − 1.92
        = 55.30

gain = 55.30 − (−5.58) = 60.89
capture = 60.89 / 63.82 = 95.4% of the provable maximum
```

**Step 8 — the zero-sum check.**

```
before : 26.00 + 1313.32 = 1339.32
after  : 194.77 + 1144.55 = 1339.32
delta  : 0.00
```

Nothing was created. The weakest team went from 8.67 to 64.92; the stronger went from 65.67
to 57.23. **The deficit was moved, not removed.** That sentence is the project.

**Step 9 — pruning.** With this pool solved at gain 60.89, every pool whose ceiling falls
below the eventual fifth-best gain can be skipped. After 123 pools the condition holds and
the search stops, with a guarantee covering the remaining 9,634.

**Step 10 — disjointness.** TEAM_0098 also appears in the second, third and fifth
highest-gain pools. Those are dropped: employee 308441 cannot move to four different teams.
The final list spans ten distinct teams.

---

## 9. The evaluation branch

`python run.py sweep` answers the reviewer's question: *is any of this better than the
obvious thing?*

### The five arms

| Arm | Value function it optimises |
|---|---|
| **full** | `v(e) = perf_norm(e) · φ(s(e))` — the proposed system |
| perf_only | `v(e) = perf_norm(e) · φ(0.5)` — sentiment ablated |
| sent_only | `v(e) = 1 · φ(s(e))` — performance ablated |
| random | a random feasible role-matched swap |
| worst_for_best | the intuitive heuristic: the weaker team gives its worst, takes the other's best |

**Every arm is re-scored on the full objective.** Each arm picks its moves using its own
value function, but is then judged on the same yardstick. Without this the comparison would
be meaningless — each arm would simply like its own objective best.

### The headline metric

`sentiment_changed_decision_pct` — the share of pools where the sentiment-aware optimiser
picks a *different set of movers* than the performance-only one. **If this were near zero,
the sentiment component would be decorative and the project would have no contribution.**

It is **71.1%** over 1,000 pairs.

### Results (1,000 pairs)

| Arm | mean gain | median | % improved | % damaged | full wins | full loses |
|---|---|---|---|---|---|---|
| **full** | **8.0811** | **5.7184** | **82.10** | **0.00** | — | — |
| perf_only | 4.8470 | 1.7425 | 59.00 | 19.80 | 71.10% | **0.00%** |
| sent_only | 0.2596 | 0.0000 | 39.40 | 41.10 | 81.50% | **0.00%** |
| random | 0.1365 | 0.0000 | 45.20 | 45.30 | 80.40% | **0.00%** |
| worst_for_best | 2.0021 | 0.5779 | 53.10 | 37.50 | 77.40% | **0.00%** |

Three honest readings, all of which belong in the paper:

- The full system **never loses** to any baseline and **damages a pool 0.00%** of the time.
  Combined with 99.9% proven optimality, that is the expected signature of an exact
  optimiser on a correctly specified objective.
- `sent_only` has a median gain of exactly **0.0000** and damages 41.10% of pools.
  **Sentiment alone recommends nothing useful.** It is a modifier on performance, not a
  driver. Claiming otherwise would be overselling.
- The intuitive human move — trade the weak team's worst for the strong team's best —
  **damages the pool in 37.50% of cases**. This is the result that survives every
  modelling choice we made, and the one a practitioner should take away.

### Ablations

**λ (balance weight)** — see Trap 1. At λ = 0 the system loses to performance-only 30% of
the time. λ = 0.5 is the only setting tested that never loses; λ ≥ 1 wins more often but
reintroduces a 0.5% loss rate. The balance term is structural, not a preference.

**Churn budget** — 1 → 2 buys +0.12 mean gain; 2 → 3 buys +0.005 for +0.12 extra moves,
and is where exhaustive enumeration stops being universal (99.5% proven optimal at B = 3,
against 100% at B ≤ 2). Budget 2 is the right default, and this is the evidence.

**Protective floor** — costs 11.4% of mean gain to hold 791 disengaged employees in place.
That price is specific to this sentiment distribution; see §13.3.

---
---

# Part III — The Output

## 10. What HR finally sees

The console output of `python run.py recommend --top 5`, rendered as a brief:

---

> ### Reassignment recommendations
>
> **Scope.** 570 teams examined. **143** flagged as under-performing
> (`Performance_Category == "Low"`). **9,757** feasible exchange pools formed; **123**
> solved, and it is *proven* that none of the remaining 9,634 could have produced a better
> recommendation than those below.
>
> **Confidence.** Every recommendation below is the **proven global optimum** for its pool
> under a churn budget of 2 swaps per role class. No two recommendations share a team, so
> all five can be actioned together.
>
> ---
>
> #### Ranked recommendations
>
> | # | Teams | Moves | Weakest team | Before → After | Gain | % of achievable |
> |---|---|---|---|---|---|---|
> | 1 | TEAM_0098 + TEAM_0202 | 4 | TEAM_0098 | 8.67 → **64.92** | 60.89 | 95.4% |
> | 2 | TEAM_0199 + TEAM_0273 | 4 | TEAM_0199 | 10.70 → **48.84** | 49.68 | 95.7% |
> | 3 | TEAM_0123 + TEAM_0195 | 2 | TEAM_0123 | 17.83 → **53.12** | 46.01 | 98.7% |
> | 4 | TEAM_0207 + TEAM_0546 | 4 | TEAM_0207 | 22.56 → **52.18** | 38.20 | 98.6% |
> | 5 | TEAM_0212 + TEAM_0330 | 4 | TEAM_0212 | 20.74 → **50.36** | 37.38 | 99.4% |
>
> ---
>
> #### Recommendation 1 — TEAM_0098 ↔ TEAM_0202
>
> | Employee | Title | Role | Move | Perf | Sentiment | Trades with |
> |---|---|---|---|---|---|---|
> | 330480 | Sap Sd Consultant | Business Analyst | 0098 → 0202 | 51 | 0.80 | 376693 |
> | 376693 | Senior Analyst | Business Analyst | 0202 → **0098** | 68 | 0.57 | 330480 |
> | 308441 | Senior Software Engineer | Developer | 0098 → 0202 | 51 | 0.63 | 157522 |
> | 157522 | Associate Consultant | Developer | 0202 → **0098** | 97 | 0.94 | 308441 |
>
> **Why:** TEAM_0098 has three people and is the weakest team in the company on effective
> contribution (8.67 / 100). Both exchanges are strictly within role class. It receives a
> Business Analyst 17 performance points stronger and a Developer 46 points stronger, both
> with higher engagement; TEAM_0202 has twenty members and absorbs the loss, falling from
> 65.67 to 57.23 while remaining well above the pool minimum.
>
> **What this does not say.** Nothing here predicts whether employees 157522 and 376693 will
> be *happier* in TEAM_0098. The system has no evidence for that and does not model it.
> Their current sentiment scores are shown so a manager who knows them can make that call.
>
> ---
>
> *Recommendations for HR review, not instructions. Role match, performance and sentiment
> are shown for every person on the list.*

---

### Which columns of the raw output feed which part of that brief

| Brief element | Source |
|---|---|
| "570 teams examined / 143 flagged" | `meta.low_teams_found`, `len(ctx.teams)` |
| "9,757 pools / 123 solved / proven" | `meta.candidate_pools`, `pools_solved`, `no_unexamined_pool_can_beat_shortlist` |
| "proven global optimum" | `SwapResult.proven_optimal` |
| Before → After per team | `scores_before[t]`, `scores_after[t]` |
| Gain | `J_after − J_before` |
| % of achievable | `SwapResult.capture_pct` |
| Who moves, and where | `moves` → `from_team`, `to_team` |
| Trades with | `SwapResult.counterparts()` |
| Perf / Sentiment | `Performance Score`, `s_crisp` |
| "Why" paragraph | `reason` column, plus the team score deltas |
| The disclaimer | `run.py::_advisory_footer` |

---

## 11. Symbol glossary

| Symbol | Code | Meaning |
|---|---|---|
| `e` | `Employee_ID` | an employee |
| `t` | `Team_ID` | a team |
| `perf_norm(e)` | `ctx.perf_norm` | performance min-max scaled to `[0,1]` company-wide |
| `s(e)` | `ctx.s` | **observed** sentiment, `[0,1]`. Intrinsic to the person |
| `φ(s)` | `objective.phi` | engagement multiplier, `0.5 + s` |
| `v(e)` | `ctx.values` | employee value, `perf_norm · φ(s)`. **No team argument** |
| `TeamScore(t)` | `objective.team_score` | `100 · mean{v(e) : e ∈ t}` |
| `J` | `objective.objective` | `min TeamScore − λ·std TeamScore` |
| `J_bound` | `objective.equalisation_bound` | provable ceiling: `100 · pool mean` |
| headroom | `pool_headroom` | `J_bound − J_before`, the pruning key |
| λ | `LAMBDA_BALANCE` | balance weight, 0.5 |
| budget | `CHURN_BUDGET` | max swaps per role class, 2 |
| floor | `SENTIMENT_FLOOR` | protective threshold, 0.35 |
| ~~`ŝ(e,t)`~~ | ~~`fis.satisfaction_matrix`~~ | **withdrawn** — see Trap 4 |

---

## 12. Assumptions and honest limits

1. **Sentiment is delivered, not derived here.** The scores come from another
   subsystem's model; we validate them (roster coverage, range, no duplicates, and a
   leakage test against the star rating) but we cannot regenerate them. One review backs
   each employee, so the signal is cross-sectional and carries no trajectory.

2. **The system does not predict personal fit.** By design. It recommends on role,
   performance and *observed* sentiment. Whether a specific move suits a specific person is
   a judgement for people who know them, and the output says so on every run.

3. **`φ(s) = 0.5 + s` is a modelling choice.** Nothing in the data determines that a wholly
   disengaged employee contributes exactly half. The *shape* — monotone, bounded away from
   zero — is defensible; the coefficients are a stated assumption.

4. **Min-max normalisation is anchored by two individuals.** With `Performance Score`
   spanning 50–100, an employee scoring exactly 50 receives `v = 0.00` regardless of their
   sentiment — visible in TEAM_0202 (employee 231914). The scale is also fragile to a single
   outlier hire.

5. **No ground truth, therefore no accuracy figure.** There is no record of a real
   reassignment and its outcome. The system is evaluated against baselines on its own
   objective, which is weaker than outcome validation, and it is reported as such.

6. **"Low" means bottom quartile, not crisis.** The band spans `Team_Performance`
   73.70–78.09 — a 4.39-point window.

7. **Two different notions of "team performance" coexist.** The shipped `Team_Performance`
   comes from `Composite_Score`; the objective's `team_score` comes from `Performance Score`
   and sentiment. Discovery uses the first, optimisation the second, and `low_teams.csv`
   prints both so the discrepancy is visible rather than hidden.

8. **Beyond two teams, optimality is lost.** At `pool_size ≥ 3` the per-pool result is a
   local optimum. The cross-pool pruning proof still holds, but `proven_optimal` is `False`
   and must not be quoted as optimality.

9. **The performance metrics are mutually independent** (mean |r| = 0.0109). This dataset
   contains no latent quality factor. It is the reason a learned model was rejected, and it
   would need rechecking on any real dataset.

---

*Final-year project, Computer Science & Engineering — Manshi Bachhan (16900123057),
Ankit Kumar (16900123017), Nandan Kumar (16900123060), Beauty Mondal (16900123031),
Gautam Kumar Mishra (16900123043). Under the guidance of Prof. Partha Ghosh.*

---

## 13. Deferred work — agreed, not yet implemented

Recorded here so it is not lost. **None of this is built yet**; the notes exist so
that what we do build stays pointed in the right direction.

### 13.1 Make the system general-purpose

Today the code is aligned to the four datasets above, and the column names of *these*
files are assumed in `src/config.py`. The system should eventually accept any
employee / team / sentiment triple without a code change.

What already points that way, and should be preserved:

- every path and column name is declared in `config.py` and nowhere else
- `sentiment_build.resolve_columns` already matches columns by alias rather than by
  exact name, and reports what it picked — this is the pattern to extend
- `sentiment_io.load_sentiment` already accepts three different file shapes
- roles are *derived* from the data, never trusted from a label, so a new dataset with
  different job titles does not need a new taxonomy hand-written for it

What is still hard-wired and would have to move:

- employee and team column names (`Performance Score`, `Team_Size`, …) are constants
- discovery assumes a `Performance_Category` column with a `Low` value, falling back
  to a quantile split
- the four working role classes are a fixed tuple
- `perf_norm` min-max scaling is fitted on whatever roster it is given, so two runs on
  different datasets are not comparable to each other

The likely shape of the fix is a small dataset-profile (a mapping from our internal
field names to the supplied file's columns) resolved once at load time, with the
same *report what you picked* behaviour `sentiment_build` already has. **Do not
start this until the current pipeline is finished and the paper is submitted.**

### 13.2 Paper synchronisation is batched, not continuous

`paper/main.tex` and `paper/results/verify_numbers.py` hold the empirical numbers as
literals, in two places that must agree. They were brought fully into line with the
delivered sentiment data and **`verify_numbers.py` passes as of 2026-09-23**.

From here on the paper is **not** re-synced after every change. Finish the code work
first, then do one pass at the end:

```bash
python run.py sweep --limit 1000 --save     # Table II
python paper/results/run_ablations.py       # Tables IV-VI
python paper/results/make_figures.py        # Figs 1-2
python paper/results/verify_numbers.py      # must exit 0 before submitting
```

`verify_numbers.py` is the guard: it exits non-zero if any figure quoted in the paper
has drifted from the data. Note that it stores the paper's claims as its own
literals, so a table edited in `main.tex` must be mirrored there too — otherwise the
check quietly keeps testing the old number.

### 13.3 Smaller items

- **`SENTIMENT_FLOOR` is calibrated to one distribution.** At `0.35` it held 288
  employees under the placeholder and holds 791 on the delivered scores, because the
  two distributions have very different spread. Expressed as a *quantile* it would
  survive a change of sentiment source; as an absolute threshold it does not.
- **The sentiment dataset was delivered twice, under two names.**
  `Capgemini_Employee_Reviews_IT_only_with_MODEL_PREDICTIONS.csv.xls` and
  `datasetsentiment_fuzzy.csv` hold the same 5,000 rows — verified identical except for
  the CSV dtype of the rating columns (float64 vs int64). The second name is
  `"dataset" + "sentiment_fuzzy.csv"` with the separator missing, i.e. a bug in the
  writer. We read the properly-named one and leave both in place; the duplicate and the
  path bug should be cleaned up at source. Note also that the `.csv.xls` extension is
  wrong — the file is a plain CSV.
- **The sentiment scores cannot be re-derived in this repository**, by design — another
  subsystem owns that model. We validate what we are given; we cannot regenerate it.

