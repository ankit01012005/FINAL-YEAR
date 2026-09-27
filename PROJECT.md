3# Performance Analysis of Company Teams Using Sentiment Analysis
### Project Workbook — context, bug register, methodology, and roadmap

| | |
|---|---|
| **Title** | Performance Analysis of Company Teams and Employee Reassignment Using Sentiment Analysis |
| **Team** | Manshi Bachhan (16900123057) · Ankit Kumar (16900123017) · Nandan Kumar (16900123060) · Beauty Mondal (16900123031) · Gautam Kumar Mishra (16900123043) |
| **Guide** | Prof. / Dr. Partha Ghosh |
| **Dept / Sem** | CSE — 5th semester (synopsis stage) |
| **Doc status** | Living document. Update the Changelog (§14) on every substantive edit. |
| **Last updated** | 2026-08-31 (rev 4 — implementation; see §0.13) |

> **How to use this file.** §1–§4 are the frozen context (objective, data, what exists). §5 is the bug register — every entry has evidence and a fix. §6–§8 are the forward plan. §9–§10 are for the paper. §12 is the decision queue: things only the team can answer.

---


---

## 0. SCOPE REVISION — rev 3 *(2026-08-31)* — **read this before §4–§8**

The problem was re-scoped by the team. Sections 4, 6, 7 and 8 below were written
against the *old* scope and are partially superseded; every statement marked
**[SUPERSEDED r3]** no longer applies.

### 0.1 The new problem statement

> Two or more **already-identified low-performing teams** are handed to the system.
> Swap employees **between those teams only**. A swap is feasible only if the two
> employees hold the **same Role**. Use **performance + sentiment** to decide who moves.
> Sentiment arrives pre-computed and **fuzzified** from a separate system owned by
> another team member. No skills/domain data is available or required.

This is a *strictly smaller and strictly more defensible* problem than the old
global 782×782 reassignment. It dissolves five registered bugs outright.

### 0.2 What this dissolves (delete, do not fix)

| Artefact | Status |
|---|---|
| `codes/Embedding_emp_team_final_year.ipynb` | **Retired** — no skill text to embed |
| `dataset/employee_embeddings.npy`, `team_embeddings.npy`, `similarity_matrix.npy` | **Retired** — archive |
| `codes/Cost_metrix_estimation.ipynb` | **Retired** — global LAP replaced by scoped swap search |
| BUG-2 (row-constant cost matrix) | **Dissolved** — no global cost matrix exists any more |
| BUG-3 (similarity/cost row misalignment) | **Dissolved** — no similarity matrix |
| BUG-7 (72 distinct team profiles) | **Dissolved** — no profile text |
| BUG-8 (`job_role` contaminates profile text) | **Dissolved** — Role is now the *constraint*, not a feature |
| BUG-18 (no `skills`/`domain` column) | **Dissolved** — skills are out of scope |

### 0.3 THEOREM — the pairing is not a decision variable, so Hungarian is not needed

Let team value be additively separable over members:
`V(T) = Σ_{e ∈ T} v(e, T)` for **any** value function `v`.
Swapping `a ∈ A` with `b ∈ B` changes the total by

```
Δ(a,b) = [ v(b,A) + v(a,B) ] − [ v(a,A) + v(b,B) ]
       = [ v(a,B) − v(a,A) ]  +  [ v(b,A) − v(b,B) ]
       =        β(a)          +        α(b)
```

`Δ` is a **row constant plus a column constant** — a rank-1 additive matrix.
By the row/column-invariance of the assignment problem, *every* perfect matching
between the same two sets scores identically. **A Hungarian solve over a swap-gain
matrix is degenerate by construction — the exact BUG-2 pathology, one level up.**

Verified on `TEAM_0030` × `TEAM_0245` (3 Developers each):

```
max |rank-1 additive residual|            = 2.842e-14
Hungarian objective                       = -1.4055555556
all 6 perfect matchings: min = max        = -1.4055555556   spread = 2.8e-14
```

Stronger corollary, independent of separability: after a symmetric k-for-k swap the
two resulting **compositions depend only on the sets** `S_A` and `S_B`, never on the
bijection between them. *"Who swaps with whom" is a reporting device, not an
optimisation variable — for any objective whatsoever.*

**⇒ The decision variables are the SETS `(S_A, S_B)`, not a matching.**

### 0.4 When Hungarian IS legitimate

| Case | Correct tool |
|---|---|
| **Pairing** LAP: `employee_A × employee_B` swap-gain matrix | **Never.** Provably degenerate (§0.3) |
| K = 2 teams, separable `v` | Sort `β(e) = v(e,B) − v(e,A)` per role, take top-k. O(n log n) |
| K ≥ 3 teams, separable `v` | **Yes** — *slot* LAP / min-cost flow: `employee × team-slot` per role class, `c(e,t) = −v(e,t)`. Genuinely 2-argument, non-degenerate |
| Any K, **non-separable** objective (max-min, variance, balance) | LAP is no longer exact. Exhaustive with churn budget, or LAP warm-start + local search |

Because the objective we actually want is non-separable (§0.6), the honest answer is:
**Hungarian is optional — use exhaustive search with a churn budget for K = 2, and a
slot-LAP warm start plus local search for K ≥ 3.**

### 0.5 THE BLOCKER — a sum/mean objective makes every swap worth exactly zero

Measured on `TEAM_0030` (n=9) × `TEAM_0245` (n=8), swapping the weakest Developer of
one for the strongest of the other:

```
SUM of member Composite_Score : 1324.350000 → 1324.350000   delta = +0.0000000000
SUM of team MEANS             :  155.800417 →  155.726667   delta = −0.0737500000
```

The sum is **exactly** invariant. The mean moved only because the teams differ in size
(9 vs 8) — that is BUG-21's size-gaming artefact, not signal. With equal-size teams even
the mean is invariant.

**Therefore the system produces value only if at least one of these holds:**

1. **`v` is team-dependent** — i.e. `v(e,t)` genuinely varies with `t`. With skills gone,
   the *only* remaining channel is **sentiment**: `ŝ(e,t)`. **This is the paper's thesis:
   performance alone makes reassignment zero-sum; sentiment is what creates value.**
2. **The team aggregator is non-linear** — max-min, variance-penalised, or threshold-based.

Both must be in. Demonstrated with a max-min objective on the same pair:

```
objective = max-min team average performance
baseline  = 69.8889   →   best over 20 candidate SETS = 71.6667   (k=1 swap)   gain = +1.7778
```

### 0.6 Recommended objective

Effective contribution of employee `e` placed in team `t`:

```
v(e,t) = perf(e) · φ( ŝ(e,t) )          φ(s) = 0.5 + s,   s ∈ [0,1]
```

Team score, size-weighted; pool objective, Rawlsian (lift the worst team) with a
balance penalty:

```
J = min_{t ∈ pool} TeamScore(t)  −  λ · std_{t ∈ pool} TeamScore(t)
```

Hard constraints: role counts preserved per team (automatic under role-matched swaps);
Team Leads excluded (or swapped only with Team Leads); churn budget `B` swaps per team;
each employee moves at most once; **do-no-harm** — no individual's predicted satisfaction
may fall below a floor.

### 0.7 Counterfactual sentiment `ŝ(e,t)` — the engine

Observed sentiment exists only for an employee's *current* team, and is cohort-level
(joined on `(Title, Department)` — see §3.4). So decompose:

```
ŝ(e,t) =    s₀(e)              +        δ(e,t)
         cohort baseline            team-context adjustment
         (from the fuzzy            (fuzzy inference over RELATIONAL features)
          sentiment system)
```

`δ(e,t)` uses **relational features only** — features that change when `e` moves:
`perf(e) − Avg_Performance_Score(t)`, `PeerRating(e) − Avg_Peer_Rating(t)`,
`ManagerFeedback(e) − Avg_Manager_Feedback(t)`, `Team_Size(t)`, role scarcity of `e` in `t`,
whether `Department(e)` matches the team's modal department.

**Use a Mamdani fuzzy inference system, not a learned model.** Rationale: BUG-20 shows the
metrics are i.i.d. noise (mean |r| = 0.0109), so a learned `δ` would fit noise and inflate
results. A rule base needs no ground truth, is fully explainable in the paper, and matches
the fuzzy theme. Example rules:

```
IF perf(e) is HIGH  AND team_avg_perf is LOW   THEN satisfaction is LOW    (carrying the team)
IF peer_gap is VERY NEGATIVE                   THEN satisfaction is LOW    (relative deprivation)
IF peer_gap is NEAR ZERO AND team_size is MED  THEN satisfaction is HIGH   (person–environment fit)
IF role_scarcity is HIGH                       THEN satisfaction is MEDIUM (isolated specialist)
```

### 0.8 INTERFACE CONTRACT with the sentiment subsystem *(hand this to the owner)*

Deliver one CSV, `dataset/sentiment_fuzzy.csv`:

| column | type | meaning |
|---|---|---|
| `Employee_ID` | int | must join 1:1 against `Employee_Team_Assignment.csv` (5,000 rows) |
| `mu_neg`, `mu_neu`, `mu_pos` | float [0,1] | fuzzy membership degrees, should sum to ≈ 1 |
| `s_crisp` | float [0,1] | defuzzified (centroid) sentiment |
| `n_reviews` | int | reviews backing this employee's cohort — confidence weight |
| `join_level` | str | `title_dept` / `dept_only` / `global_prior` — provenance flag |

**Two mandatory warnings for that subsystem:**

1. **`s_crisp` must not be a deterministic function of `Overall_rating` (BUG-4 / BUG-5).**
   If it is, sentiment is a restatement of a number we already hold, it becomes another
   intrinsic per-employee constant, and by §0.5 the swap engine returns to zero-sum.
   The model must be trained on *review text*, with rating used only to build labels — and
   `rating_norm` must **not** be a fuzzy input feature.
2. **Null ratings must not fall through to POSITIVE** (BUG-5: 14,330 rows). Drop them or
   impute them explicitly before labelling.

### 0.9 Algorithm to implement

```
INPUT : pool P = {T1..TK} low-performing teams; churn budget B; role map

1  NORMALISE ROLES     role_from_title(); recompute Num_* columns          ← BLOCKER #1
2  LOAD SENTIMENT      sentiment_fuzzy.csv; validate the contract (§0.8)
3  SCREEN CANDIDATES   swap_priority(e) = w_p·(1 − perf_norm(e)) + w_s·(1 − s_crisp(e))
                       keep the bottom q per (team, role).
                       ← the "worst employee" rule; an EFFICIENCY + do-no-harm
                         device that shrinks the search space, NOT the selector
4  BUILD ŝ(e,t)        Mamdani FIS over relational features, for every screened e, every t ∈ P
5  SEARCH
     K = 2 : exhaustive over set-pairs (S_A, S_B) per role class, |S| ≤ B  → PROVEN OPTIMAL
     K ≥ 3 : slot-LAP warm start per role class, then steepest-ascent local
             search over 1-swap and 2-swap neighbourhoods
6  OUTPUT              ranked swap list with per-swap ΔJ, predicted satisfaction change
                       for both employees, and the binding constraint
```

Measured exhaustive search cost over 400 random LOW-team pairs:

```
churn budget/role   median          p95            max
None                    46      1,078,506    888,844,320    ← unbounded tail is dangerous
1                       21          2,701         95,550    ← trivially exact
2                       38         31,341     16,980,900    ← exact, still fast
3                       50         75,302     33,856,200
```

**⇒ Set B = 2 swaps per role class. Exhaustive search is then provably optimal and fast.**
This is a genuine paper claim: *"solved to proven global optimality by exhaustive
enumeration; the instance is small by construction."*

### 0.10 Bug status under the new scope

| Bug | Old | New | Why |
|---|---|---|---|
| **BUG-19** `Role` is random (χ² p = 0.51 / 0.55) | 🔴 | **🔴 BLOCKER #1** | Role is now the *only* feasibility constraint. If it is noise, every swap is meaningless |
| **BUG-21** mean objective is degenerate | 🟠 | **🔴 BLOCKER #2** | Confirmed exactly zero-sum in §0.5 |
| **BUG-20** metrics are i.i.d. (mean \|r\| = 0.0109) | 🟠 | **🔴 BLOCKER #3** | "Worst employee" is a coin flip; also drives the FIS choice in §0.7 |
| BUG-4, BUG-5 (leakage, NaN→POSITIVE) | 🔴 | 🔴 *delegated* | Owned by the sentiment subsystem — see §0.8 |
| BUG-1, BUG-15 | 🔴 | ✅ fixed | Resolved by the new backbone (§3.4) |
| BUG-2, BUG-3, BUG-7, BUG-8, BUG-18 | 🔴/🟠 | ⚪ dissolved | §0.2 |
| BUG-6, BUG-9 … BUG-14, BUG-16, BUG-17 | 🟠/🟡 | unchanged | Sentiment / hygiene side |

### 0.11 Evaluation design *(this is what makes it a paper)*

The 143 `Low` teams give **C(143,2) = 10,153** evaluable team pairs — a real experimental
population, not a case study.

Baselines to beat: (a) no swap; (b) random feasible swap; (c) performance-only swap
(sentiment term ablated); (d) sentiment-only swap (performance ablated); (e) greedy
"swap the worst for the best". Report mean / median ΔJ, win rate vs each baseline, and
the fraction of pairs where sentiment changes the chosen set relative to (c) —
**that last number is the paper's headline result.**

Ablations: λ sweep on the balance term; churn budget B ∈ {1,2,3}; φ slope; FIS rule-base
leave-one-rule-out.

### 0.12 Module layout *(as built)*

```
run.py                CLI: roles | swap | sweep | doctor
src/
  config.py           all paths and knobs; nothing else hard-codes a path (fixes BUG-16)
  roles.py            role_from_title(), measure_token_lift(), Num_* rebuild   ← BLOCKER #1
  sentiment_io.py     contract loader + validator + placeholder for the sentiment subsystem
  fis.py              Mamdani FIS, relational feature builder → ŝ(e,t)         ← the engine
  objective.py        v(e,t), TeamScore, J, do-no-harm, screening, SwapResult  ← BLOCKER #2
  search.py           exhaustive (K=2) | local search (K≥3) | slot-LAP bound
  pipeline.py         Context, pool selection, grid building, orchestration
  evaluate.py         5 baselines + 3 ablations + population sweep
output/               generated CSVs (gitignored in spirit; safe to delete)
```

Two deviations from the plan in §0.9: `pool_selection.py` folded into `pipeline.py` (it is
three functions), and `fit_fis.py` became `fis.py` since nothing is fitted — it is a rule
base, by design.

### 0.13 IMPLEMENTATION STATUS *(rev 4 — 2026-08-31)*

Built and verified end to end. `python run.py doctor` is the health check.

```
python run.py roles --tokens        # the evidence table + role-repair audit
python run.py swap --teams TEAM_0030 TEAM_0245
python run.py swap --worst 3        # three weakest low teams → local search + LAP bound
python run.py sweep --limit 300 --ablate --save
```

#### 0.13.1 BLOCKER #1 resolved — and a finding that changed the design

The first rule base keyed on the obvious tokens ("Consultant" → Business Analyst,
"Engineer" → Developer). It produced **52% Business Analysts**, which is absurd for a
software organisation. So rather than hand-tune it, `roles.measure_token_lift()` now
**measures** which tokens carry occupational signal: for each token, how concentrated is
the department mix of the titles containing it, relative to the company baseline.

```
token           n     lift   verdict
tester         34     7.40   OCCUPATION
test          112     7.21   OCCUPATION
qa             32     7.05   OCCUPATION
automation     48     7.05   OCCUPATION
network        57     4.20   OCCUPATION
administrator  62     4.19   OCCUPATION
devops         39     3.97   OCCUPATION
sap           151     3.42   OCCUPATION
developer     314     1.57   OCCUPATION
--------------------------------------- threshold lift >= 1.5
software      982     1.24   weak
manager       412     1.11   grade
engineer     1057     1.01   grade      <- appears in every department
senior       1462     1.02   grade
associate     761     0.97   grade
analyst       755     0.91   grade
consultant   2008     0.94   grade      <- 40% of all employees; a grade, not a job
lead          159     0.92   grade
```

**"Consultant", "Analyst", "Engineer", "Manager", "Senior", "Lead" and "Associate" carry
no occupational information in this organisation** — they are rungs on a promotion ladder.
Together they cover most of the workforce. Only 16 tokens clear the bar; everything else
resolves through `Department`, which is real and occupationally meaningful.

This is worth a paragraph in the paper. It is a concrete, measured caution about naive
title-based role inference in HR analytics, and it is cheap for a reviewer to re-run.

Result:

| | shipped | derived |
|---|---|---|
| χ² p (Title vs Role) | 0.5463 | **9.50e-176** |
| χ² p (Department vs Role) | 0.5112 | **0.0** |
| Developer / DevOps / BA / QA | 2432 / 667 / 422 / 909 | **2579 / 961 / 851 / 609** |
| resolved from title | — | 22.04% (19.52% measured-token) |
| resolved from department | — | 77.86% |
| resolved by default | — | **0.10%** (5 employees) |
| Σ Num_* == Team_Size | 570/570 | **570/570** |

The derived mix tracks the real department buckets (Dev 2695 / DevOps 1002 / BA 727 /
QA 576). Agreement with the shipped labels is 30.84% — close to the 25% expected by chance,
which is itself confirmation the originals were random.

#### 0.13.2 BLOCKER #2 resolved

`objective.py` implements `v(e,t) = perf_norm(e)·φ(ŝ(e,t))` with
`J = min_t TeamScore(t) − λ·std_t TeamScore(t)`. Note on BUG-21: mean-of-team-means is
normally gameable by shifting people into smaller teams, but a role-matched k-for-k swap
leaves every team size **exactly** unchanged, so that channel is closed by the constraint
rather than by the aggregator. The mean is safe here.

#### 0.13.3 Verification — the search is exact

The full system loses to the `perf_only` baseline on 11.67% of pairs, which should be
impossible for an exhaustive optimiser. It is not a defect. `perf_only` runs with a
*constant* ŝ, so the do-no-harm filter never excludes anyone and it searches a strictly
larger feasible set. Re-running with the constraint disabled:

| | mean gain | loss vs perf_only | win vs perf_only |
|---|---|---|---|
| do-no-harm **off** | 8.7938 | **0.00%** | 72.5% |
| do-no-harm **on** | 7.6851 | 11.67% | 62.5% |

**0.00% confirms the exhaustive search is exact.** The 11.67% is the measured *price of
the ethical constraint* — 12.6% of mean objective gain — and belongs in the results table
as such. Exposed as `evaluate.ablate_harm()`.

#### 0.13.4 Baseline sweep — PROVISIONAL, placeholder sentiment

300 random low-team pairs, churn budget 2, λ = 0.5:

| arm | mean gain | median | % pairs improved | full wins | full loses |
|---|---|---|---|---|---|
| **full** | **6.2221** | 3.9423 | 82.67% | — | — |
| perf_only | 5.2207 | 2.7872 | 64.67% | 67.67% | 11.67% |
| worst_for_best | 1.0983 | 0.0393 | 51.00% | 81.67% | 7.00% |
| sent_only | −1.2856 | 0.0 | 37.67% | 82.33% | 0.00% |
| random | −1.4420 | −0.5348 | 42.67% | 86.33% | 4.00% |

```
sentiment_changed_decision  79.33%     <- headline metric
proven_optimal              98.67%     (4 pairs hit the state cap -> local search)
harm violations                  0
mean moves per pair           3.61
median states evaluated         18
runtime                    298 ms/pair -> full 10,153-pair sweep ~50 min
```

> ⚠️ **These numbers are NOT a result.** `dataset/sentiment_fuzzy.csv` does not exist yet,
> so `sentiment_io` is serving a synthetic Beta(5,3) placeholder, deliberately independent
> of every performance metric. The pipeline, the search and the ablations are validated;
> the *effect sizes* are not. Every run prints an `is_placeholder` banner. Re-run every
> table in §0.13.4 once the real scores land — **no code change is required**, only the
> file.

What the placeholder run does establish: the machinery works, the search is provably
exact, the constraints bind correctly, `sent_only` correctly *underperforms* (optimising
satisfaction while ignoring performance moves the wrong people), and `random` correctly
comes out negative.

#### 0.13.5 Where the real sentiment file plugs in

Drop `dataset/sentiment_fuzzy.csv` and re-run. The loader accepts three shapes:

```
full     Employee_ID + mu_neg/mu_neu/mu_pos + s_crisp
fuzzy    Employee_ID + mu_*                    -> s_crisp by centroid defuzzification
crisp    Employee_ID + any one score column    -> mu_* by triangular fuzzification
```

It also accepts a VADER-style [−1,1] compound and rescales it. `validate()` checks the 1:1
join, duplicates, nulls and — importantly — flags a near-constant `s_crisp`, since a
constant sentiment returns the objective to exactly zero-sum. `audit_against_rating()`
tests BUG-4 leakage empirically: ship a rating column alongside and it reports the R²; above
0.95 means sentiment is a restatement of the rating rather than a reading of the text.

#### 0.13.6 Open items

- Sentiment file (external dependency, blocks §0.13.4).
- **BLOCKER #3 / BUG-20 remains open.** Metrics are still i.i.d. (mean |r| = 0.0109). The
  FIS sidesteps it — a rule base cannot overfit noise the way a learned model would — but
  `perf_norm(e)` is still drawn from a metric with no latent quality factor behind it. State
  this in Limitations; do not paper over it.
- Full 10,153-pair sweep once sentiment lands (~50 min).
- λ, budget and φ-slope ablations are implemented but only spot-checked.

---


## 1. Objective — the anchor

> **Build a data-driven decision support system that evaluates employee skill, performance, and sentiment in order to recommend the reassignment / swapping of employees between teams — replacing biased managerial judgement, so as to produce balanced teams and raise overall productivity.**

Source: `docs/HR paper (2).pdf`, `docs/synopsis (4) (1).pdf`.

**Three declared phases**

| Phase | Content |
|---|---|
| **P1** | Data collection & preprocessing — structured (skills, experience, performance) + unstructured (feedback, surveys) |
| **P2** | Sentiment analysis — BERT `[CLS]` + fuzzy logic → fuzzified positive / negative / neutral satisfaction score |
| **P3** | Performance evaluation & reassignment — employee profiles → team-level analysis → recommendations (reassign, swap, train, mentor, in rare cases terminate) |

**The operative verb is *swap / reassign*.** Every design decision in this document is judged against whether it moves us toward an actionable, defensible reassignment recommendation. Anything that does not is out of scope.

### 1.1 What "done" looks like
1. A trained sentiment model that produces an interpretable satisfaction signal from employee review text.
2. A reassignment engine that outputs a **ranked list of concrete swaps** with predicted before/after effects.
3. Quantitative evidence that the recommended swaps improve team performance and/or satisfaction versus the current allocation.
4. A conference paper reporting 1–3 with ablations.

Item 3 does not currently exist in any form. It is the difference between a project and a paper.

---

## 2. Repository map

```
D:\final_year\
├── PROJECT.md                            ← this file
├── employee_reviews.csv                  86,117 public company reviews (raw text corpus)
├── NeuroFuzzy_BERT_Sentiment.ipynb       P2 — hybrid BERT + fuzzy classifier  (NOT TRAINED)
├── codes\
│   ├── Embedding_emp_team_final_year.ipynb   P3a — MPNet profile embeddings → similarity matrix
│   ├── performance_metrix_final_year.ipynb   P3b — entropy-weighted performance score
│   └── Cost_metrix_estimation.ipynb          P3c — cost matrix + Hungarian assignment
├── dataset\
│   ├── Employee_Team_Assignment.csv      5,000 × 17 — NEW backbone: 1 row/employee  ★
│   ├── Team_Dataset.csv                    570 × 18 — NEW backbone: 1 row/team      ★
│   ├── employee_dataset_realistic.csv    782 rows × 16 — employee records  — SUPERSEDED (BUG-1)
│   ├── team_dataset_realistic.csv        782 rows × 5  — team role-slots (100 teams)  — SUPERSEDED (BUG-7)
│   ├── employee_embeddings.npy           (782, 768) MPNet, L2-normalised
│   ├── team_embeddings.npy               (782, 768) MPNet, L2-normalised
│   ├── similarity_matrix.npy             (782, 782) cosine similarity
│   ├── employee_performance_ready.csv    782 × 4 — employee_performance, employee_perf_cost
│   ├── team_metrics_ready.csv            100 × 6 — team_performance, size, variance, cost
│   └── predictions_with_sentiment.csv    782 × 8 — likes/dislikes/review + VADER scores  — SUPERSEDED
│                                          ★ = new backbone, see §3.4
├── docs\
│   ├── HR paper (2).pdf                  problem statement + methodology  ← primary spec
│   ├── synopsis (4) (1).pdf              approved synopsis                ← primary spec
│   ├── Performance Analysis Teams Sentiment Analysis.pdf   35-p internal documentation
│   ├── Intermediate Copy (IEEE Conf.)...pdf   guide's TOURISM paper — template/reference only
│   └── UARS.pdf                          not yet reviewed (see §12 Q5)
└── archive\                              orphaned — referenced by no notebook
    ├── employee_data.csv                 3,000 HR records
    ├── Employee_Performance_Dataset.csv  5,000 performance records
    ├── final_employee_dataset (1).csv    3,000 — merge of the two above
    └── WA_Fn-UseC_-HR-Employee-Attrition.csv  1,470 — IBM attrition dataset
```

### 2.1 Cleanup performed
- **Archived, not deleted.** The four CSVs in `archive\` are referenced by no notebook (verified by grepping every `.csv`/`.npy`/`.pkl`/`.pt` literal across all four notebooks). They are remnants of an earlier direction. They can be permanently deleted once the team confirms — until then they cost 2.5 MB and nothing else.
- PDFs consolidated into `docs\`. No code reads PDFs, so this is path-safe.
- `employee_reviews.csv` and `NeuroFuzzy_BERT_Sentiment.ipynb` were deliberately **left in the repository root**, because the notebook loads the CSV with a bare relative path. Moving either one breaks it. See BUG-16.

### 2.2 Note on `Intermediate Copy (IEEE Conf.).pdf`
This is a **tourism recommendation paper** (mood detection via VADER → destination recommendation → skyline hotel selection → taxicab-distance clustering → total ordering). It is **not** this project. Treat it as (a) the guide's preferred paper structure and (b) a source of reusable technique — specifically **skyline computation** for multi-criteria ranking without fixing weights, which is a genuinely good alternative to the current arbitrary weight vector (see §7, Alternative E).

---

## 3. Data assets — provenance and integrity

### 3.1 `employee_reviews.csv` — the text corpus
Public Indian company-review corpus (AmbitionBox-style; the synopsis cites *kmrmanish/Employees_Reviews_Dataset* on Hugging Face).

| Column | Notes |
|---|---|
| `Title`, `Place`, `Job_type`, `Department`, `Date` | metadata |
| `Overall_rating` | 1–5; **14,330 of 86,117 are null** |
| `work_life_balance`, `skill_development`, `salary_and_benefits`, `job_security`, `career_growth`, `work_satisfaction` | **six aspect ratings, ~75.8k non-null each — currently unused by every notebook.** This is the single most valuable untapped asset in the project (see §6, Step 2). |
| `Likes`, `Dislikes` | free text |

**True label distribution** (using the notebook's own 1–2→Neg / 3→Neu / 4–5→Pos mapping):

| Class | Share |
|---|---|
| Positive | **77.8 %** |
| Neutral | 11.7 % |
| Negative | 10.5 % |

The internal documentation states the opposite (Negative 53 % / Neutral 31 % / Positive 16 %). See BUG-6.

### 3.2 `employee_dataset_realistic.csv` / `team_dataset_realistic.csv` — the synthetic org
- 782 employee rows, 782 team role-slots, 100 teams (size 7–11).
- Employee metrics: training hours, work hours logged, peer rating, KPI score, attendance %, manager feedback, performance score, task completion %, plus `skills` (semicolon-list), `job_role`, `domain`, `project_size`, `promotion_eligibility`.
- **Provenance unknown** — no generator script in the repository. Probably derived from `archive/Employee_Performance_Dataset.csv`. See §12 Q2.
- **Integrity: broken.** See BUG-1 and BUG-7.

### 3.3 `predictions_with_sentiment.csv`
Despite the name it contains **no model predictions** — only VADER compound scores rescaled to `[0,1]`. Its `likes` field matches the public corpus **verbatim at a 100 % rate**, i.e. reviews were sampled from `employee_reviews.csv` and stapled onto synthetic employees at random. The neuro-fuzzy BERT model is connected to nothing.

### 3.4 `Team_Dataset.csv` + `Employee_Team_Assignment.csv` — the replacement org *(added 2026-08-30)*

**Verdict: adopt these as the structural backbone. They are substantially more reliable than the pair they replace — but they are not a drop-in substitute, because they carry no skills data.**

`Employee_Team_Assignment.csv` — 5,000 × 17. `Team_Dataset.csv` — 570 × 18.

#### 3.4.1 What they fix — verified

| Check | Old data | New data |
|---|---|---|
| Employee key unique | ❌ 138 people in 782 rows | ✅ **5,000 unique IDs, 1 row each** |
| One employee → one team | ❌ ~6 teams each | ✅ exactly one |
| Referential integrity | — | ✅ 570/570 teams matched; **0 orphans, 0 empty teams** |
| Nulls | — | ✅ **0** across both files |
| `Team_Size` vs actual member count | — | ✅ **0 mismatches / 570** |
| `Avg_*` vs recomputed member means | — | ✅ all 7 agree to ±0.005 (rounding) |
| `Num_Team_Lead/Developer/QA/DevOps/BA` vs actual | — | ✅ exact; **Σ Num_\* = Team_Size** for every team |
| Team sizes | 7–11, artificial | ✅ 3–20, median 8, mean 8.8 — plausible |
| Every team has a lead | — | ✅ 0 teams without a Team Lead |

This resolves **BUG-1, BUG-3 and BUG-15** at the source. Alignment stops being a hazard because the key is genuinely a key.

#### 3.4.2 The important find — reviews can now be attached *principledly*

`Title` and `Department` are drawn **entirely** from `employee_reviews.csv`:

```
Titles:       1,076 distinct in employees | 1,076 also in reviews  (100.0%)
Departments:      75 distinct in employees |     75 also in reviews  (100.0%)
employee rows whose (Title, Department) pair appears in reviews: 100.0%

reviews available per employee (by Title+Dept group):
    min 1 | 25% 3 | median 36 | 75% 303 | max 1,921
    70.6% of employees have a group with >= 5 reviews
```

This is the most valuable property of the new data. Reviews can now be joined to employees **on (Title, Department)** — an occupational match — instead of being stapled on at random. That removes the single most indefensible step in the old pipeline (§3.3) and makes the sentiment signal arguable in a paper: *"each employee is characterised by the review distribution of their occupational cohort."*

Sample within-cohort (don't take one review per employee — take the cohort distribution: mean aspect scores, dispersion, and a sampled exemplar for qualitative work). For the ~29% of employees whose cohort has < 5 reviews, back off to Department-level pooling and record the backoff level as a data-quality flag.

#### 3.4.3 Derived columns — reverse-engineered exactly

```
Peer_Rating_100      = Peer Rating × 20                                          [exact]
Composite_Score      = 0.30·Perf + 0.25·KPI + 0.15·Att + 0.10·Task + 0.20·Peer100 [exact, ±0.005]
Team_Performance     = mean(member Composite_Score)                              [exact, ±0.005]
Performance_Category = fixed bins of Team_Performance (143 High / 284 Med / 143 Low = 25/50/25)
Promotion Eligibility ≈ (Performance Score ≥ 85) AND (Manager Feedback ≥ 4.0)     [97.4% agreement,
                                                                    100% recall over the Yes class]
```

**None of these five may be used as a prediction target.** They are *definitions*, not observed outcomes — a model predicting them is recovering an algebraic identity and will score ~perfectly for no reason. This is the same class of error as BUG-4.

#### 3.4.4 New problems introduced — see BUG-18 … BUG-21

1. 🔴 **No `skills` or `domain` column** — the MPNet skill-similarity pipeline has no input at all.
2. 🔴 **`Role` is statistically independent of `Title` and `Department`** — role labels are random.
3. 🟠 **No latent employee-quality factor** — the eight performance metrics are mutually independent.
4. 🟠 **`Team_Performance` as a plain mean makes reassignment degenerate.**

#### 3.4.5 Provenance
Not derived from `archive/Employee_Performance_Dataset.csv` (only 30/5,000 ID overlap, 0 % metric agreement) — this is an independent generation. `Title`/`Department` come from `employee_reviews.csv`; the numeric metrics were drawn i.i.d.; `Team_ID`, `Role` and team composition were generated on top. **§12 Q2 is still open** — the generator script is still needed, because BUG-18 and BUG-19 are best fixed at generation time.

---

## 4. Pipeline as actually built

```
                    employee_reviews.csv (86,117)
                              │
      ┌───────────────────────┴──────────────────────┐
      │                                              │
      ▼                                              ▼
NeuroFuzzy_BERT_Sentiment.ipynb          random sample → stapled onto
  VADER + fuzzy + BERT [CLS] → MLP        782 synthetic employees
  ✗ NEVER TRAINED (KeyboardInterrupt)             │
  ✗ OUTPUT USED BY NOTHING ─────╳                 ▼
                                       predictions_with_sentiment.csv
                                          (VADER scalar only)
                                                   │
 employee_dataset_realistic ─┐                     │
 team_dataset_realistic ─────┤                     │
                             ▼                     │
            Embedding_emp_team_final_year.ipynb    │
            MPNet all-mpnet-base-v2 (768-d)        │
            → similarity_matrix.npy (782×782)      │
                             │                     │
 employee metrics ───────────┤                     │
                             ▼                     │
            performance_metrix_final_year.ipynb    │
            MinMax → entropy weights → score       │
            → employee_performance_ready.csv ──────┤
            → team_metrics_ready.csv               │
                                                   ▼
                                    Cost_metrix_estimation.ipynb
                                    cost = w₁·skill + w₂·perf + w₃·sent
                                    → scipy linear_sum_assignment
                                    → optimized_employee_assignment.csv
                                    ⚠ w₂ and w₃ are mathematically inert
```

### 4.1 Status by deliverable

| Deliverable | Status | Notes |
|---|---|---|
| Text corpus acquired | ✅ | 86,117 reviews |
| Text cleaning | ✅ | regex, lowercase, whitespace |
| VADER scoring | ✅ | |
| Fuzzy feature construction | ⚠️ | built, but leaks the label (BUG-4) |
| BERT + fuzzy hybrid model | ⚠️ | code written, **never trained**; saved bundle contains `model_state = None` |
| Sentiment integrated downstream | ❌ | downstream uses VADER, not the model |
| Skill similarity | ✅ | but see BUG-7, BUG-8 |
| Performance scoring | ✅ | entropy weighting; see BUG-9, BUG-10 |
| Cost matrix | ⚠️ | two of three criteria have no effect (BUG-2) |
| Assignment | ⚠️ | produces operationally impossible output (BUG-15) |
| **Swap / reassignment recommendations** | ❌ | **not built — this is the stated objective** |
| **Before/after evaluation** | ❌ | **not built — no evidence of improvement** |
| Employee/team profile reports | ❌ | designed in the 35-p doc, not implemented |
| Dashboard / API | ❌ | not started |

---

## 5. Bug register

Severity: 🔴 blocks a core claim · 🟠 materially distorts results · 🟡 hygiene.

---

### 🔴 BUG-1 — `employee_id` is not unique; 782 rows contain only 138 people
**Where** `dataset/employee_dataset_realistic.csv`

**Evidence**
```
rows: 782   unique employee_id: 138   max repeats: 6

employee_id 202996 "Zoya Wadhwa":
  TEAM_001  Program Manager     Data Science
  TEAM_017  Software Developer  Cybersecurity
  TEAM_035  Business Analyst    Cybersecurity
  TEAM_052  Tech Lead           Cloud
  TEAM_073  Project Manager     Robotics
  TEAM_090  Project Manager     Java
```
The performance metrics are identical across her six rows, but the **skills, job role, domain and review all differ**. Her sentiment score ranges 0.074 → 0.936 depending on which row you read (mean within-person spread across all 138 people: **0.529**, on a 0–1 scale).

**Why it matters** Every per-person claim in the project is undefined. "Zoya's satisfaction" has six contradictory values. Any employee profile, any team aggregate, any reassignment justification inherits the ambiguity.

**Fix — choose one:**
- **(a) Re-key, ½ day — recommended.** Declare each row an independent employee: `employee_uid = 0..781`. Drop the shared `employee_id` and `Name` (or regenerate names). You then have 782 employees, each in exactly one team, each with one review. This is internally consistent and dissolves BUG-3 at the same time.
- **(b) Regenerate.** If 138 people is the intent, the generator must emit one row per person with one skill set, and team membership must be a foreign key — not a duplicated row. Requires the generator script (§12 Q2).

Do **(a)** unless the team has a reason for **(b)**. Blocking: nothing downstream is trustworthy until this is settled.

---

### ⚪ DISSOLVED r3 — BUG-2 — Performance and sentiment have provably zero effect on the assignment
**Where** `codes/Cost_metrix_estimation.ipynb`, cell 25

```python
cost_matrix = (
    w_skill * skill_cost_matrix     # (782, 782)
  + w_perf  * employee_perf_cost    # (782, 1)   ← broadcasts as a ROW CONSTANT
  + w_sent  * sentiment_cost        # (782, 1)   ← broadcasts as a ROW CONSTANT
)
```

**The mathematics** In a square assignment problem every row is used exactly once. Adding a constant `rᵢ` to all of row *i* raises the objective of **every feasible assignment** by the same `Σᵢ rᵢ`. The argmin is therefore unchanged. This is the standard reduction-invariance property the Hungarian method itself is built on.

**Evidence** — three wildly different weightings, evaluated under the skill-only objective:
```
base       (w = .50/.33/.17)   144.178665
skill-only (w = .50/.00/.00)   144.178665
perf-heavy (w = .05/.90/.05)   144.178665

127 of 782 assignments differ between base and skill-only
100 % of those differing rows have IDENTICAL skill cost  → pure ties
```
Identical objective to six decimal places. Performance and sentiment do not improve the solution; they only break ties arbitrarily.

**Why it matters** This is the thesis. The claim "we combine sentiment with performance and skill" is currently false — only skill is doing work. A reviewer who checks this will reject the paper.

**Fix** Every cost term must vary with **both** the employee *i* and the slot *j*. See §6 Step 3.

---

### ⚪ DISSOLVED r3 — BUG-3 — Row misalignment between the similarity matrix and the cost inputs
**Where** `codes/Cost_metrix_estimation.ipynb`, cells 8–9

```python
employee_performance = employee_performance.sort_values("employee_id").reset_index(drop=True)
employee_sentiment   = employee_sentiment.sort_values("employee_id").reset_index(drop=True)

assert all(employee_performance["employee_id"] == employee_sentiment["employee_id"])
print("Employee alignment check PASSED")     # ← proves nothing
```

`similarity_matrix.npy` rows follow the **original, unsorted** `employee_df` order. After sorting, only **8 of 782** rows still correspond.

```
similarity_matrix row 0  →  employee 202996, TEAM_001
perf/sent      row 0     →  employee 108201, TEAM_003
```

The assert compares two *identically sorted* frames to **each other**, never to the matrix. It is a tautology. And because `employee_id` has ties (BUG-1), the sort order within a tie group is not even guaranteed stable across pandas versions.

**Fix**
```python
# Never sort. Carry the key through, and align by explicit reindex.
emp = pd.read_csv("dataset/employee_dataset_realistic.csv")     # canonical row order
emp["row"] = np.arange(len(emp))                                # matches similarity_matrix rows

perf = perf.merge(emp[["employee_uid", "row"]], on="employee_uid").sort_values("row")
sent = sent.merge(emp[["employee_uid", "row"]], on="employee_uid").sort_values("row")

assert (perf["row"].values == np.arange(len(emp))).all()        # a real check
assert (sent["row"].values == np.arange(len(emp))).all()
```
**Rule for the whole project:** the similarity matrix defines row order. Everything else reindexes onto it. Never re-sort a frame that indexes into a matrix.

---

### 🔴 BUG-4 — Label leakage in the neuro-fuzzy model
**Where** `NeuroFuzzy_BERT_Sentiment.ipynb`, cells 4 and 6

```python
df['label']       = df['Overall_rating'].apply(map_rating)          # label FROM rating
df['rating_norm'] = df['Overall_rating'].fillna(3) / 5.0            # feature FROM rating
low, mid, high = gaussmf(rating_norm, ...)                          # → r_low, r_mid, r_high
```

`r_low / r_mid / r_high` are a deterministic encoding of the target:

| `rating_norm` | → Negative | Neutral | Positive |
|---|---|---|---|
| 0.2 | 5,294 | 0 | 0 |
| 0.4 | 3,710 | 0 | 0 |
| 0.6 | 0 | 10,084 | 14,330 |
| 0.8 | 0 | 0 | 20,838 |
| 1.0 | 0 | 0 | 31,861 |

Only the 0.6 bucket is ambiguous, and that ambiguity is itself an artefact of BUG-5.

**Why it matters** Any accuracy figure this model produces is invalid — it reads the answer key. Reporting it in a paper is a research-integrity failure, not a performance bug.

**Fix** Delete `r_low`, `r_mid`, `r_high` from `fuzzy_cols`. Keep only features derivable from **text**: VADER `neg/neu/pos/compound`, word count, average word length, topic components, and (from Step 2) predicted aspect scores.

---

### 🔴 BUG-5 — 14,330 null ratings silently labelled POSITIVE
**Where** `NeuroFuzzy_BERT_Sentiment.ipynb`, cell 4

```python
def map_rating(r):
    try:    v = float(r)
    except: return 1          # ← never reached
    if   v <= 2: return 0
    elif v == 3: return 1
    else:        return 2     # ← NaN lands here
```
`float(nan)` does not raise. `nan <= 2` is `False`; `nan == 3` is `False`; control falls to `return 2`. All 14,330 null ratings become **Positive**, inflating that class from 61 % to 77.8 %.

Worse, the *same* rows are imputed as **3 (neutral)** two cells later by `fillna(3)`. Two contradictory imputations of the same missing values, in the same pipeline.

**Fix** Drop them, and make the intent explicit:
```python
df = df[df["Overall_rating"].notna()].copy()      # 71,787 rows retained
def map_rating(v):
    return 0 if v <= 2 else (1 if v == 3 else 2)
```
If the rows are kept instead, they must be imputed once, consistently, and the choice justified in the paper.

---

### 🟠 BUG-6 — Documented dataset statistics are inverted
**Where** `docs/Performance Analysis Teams Sentiment Analysis.pdf` §5.3, §9.2

Claims *"Negative 53 % / Neutral 31 % / Positive 16 % — dataset is imbalanced, more negative than positive (unhappy people are more likely to leave reviews)."*

Actual: **Positive 77.8 % / Neutral 11.7 % / Negative 10.5 %.** Imbalanced the other way.

The stated remedies (class weights favouring Positive, oversampling Positive) would therefore make the imbalance **worse**. Also §9.2 describes a 70/15/15 split; the code does 90/10 with no test set (BUG-13).

**Fix** Recompute every statistic in the documentation from the data before any submission. Class weighting must favour **Negative and Neutral**.

---

### ⚪ DISSOLVED r3 — BUG-7 — Only 72 distinct team profiles behind 782 slots
**Evidence**
```
team role-slots: 782   distinct profile texts: 72   duplicate slots: 710
most-repeated archetype appears 31 times
employee rows: 782   distinct profile texts: 782
```
Because the team profile text is `job_role + required_skills + domain`, and the generator drew from a small pool of archetypes, `similarity_matrix` has only **72 distinct columns**. This is the direct cause of the massive tie structure in BUG-2 (100 % of differing assignments were ties).

**Why it matters** The matcher has 72 distinguishable targets, not 782. Reported matching precision is illusory.

**Fix** Regenerate `team_dataset_realistic.csv` so each team's requirements vary — per-team skill emphasis, project size, seniority mix, domain-specific tooling. Target ≥ 300 distinct profiles. Alternatively add real team context (current member skill coverage, open skill gaps) into the profile text, which differentiates slots even under a shared role archetype — and is more faithful to the objective anyway.

---

### ⚪ DISSOLVED r3 — BUG-8 — `job_role` appears in both profile texts, so similarity measures title match
**Where** `codes/Embedding_emp_team_final_year.ipynb`, cells 16 and 18

```python
employee_df["profile_text"]  = job_role + " " + skills_clean + " " + domain
team_df["team_profile_text"] = job_role + " " + skills_clean + " " + domain
```
The role token is a high-weight lexical match on both sides, so cosine similarity is dominated by *"Program Manager" ≈ "Program Manager"* rather than by skill overlap. This inflates scores and defeats the point of semantic matching.

**Fix** Ablate it. Compute similarity on skills (+ domain) only, and treat role compatibility as a **separate, explicit** term — a hard eligibility mask or a learned role-transition prior. Report the ablation in the paper; it is a legitimate finding.

---

### 🟠 BUG-9 — Entropy weighting mixes a benefit criterion with two cost criteria
**Where** `codes/Cost_metrix_estimation.ipynb`, cell 17

```python
skill_score = similarity_matrix.mean(axis=1)     # higher is BETTER
X = np.vstack([skill_score,
               employee_perf_cost.flatten(),     # higher is WORSE
               sentiment_cost.flatten()]).T
```
Entropy / CRITIC weighting requires all criteria be oriented the same way before normalisation. Mixing directions makes the resulting weights uninterpretable.

Separately, `similarity_matrix.mean(axis=1)` — an employee's *average* similarity to all 782 slots — is a measure of profile genericness, not of skill quality. A generalist scores high; a deep specialist scores low.

**Fix** Convert everything to cost orientation first (`skill_cost = 1 - sim`), and use a defensible per-employee statistic (best-match cost, or mean over the top-*k* eligible slots) rather than the global mean.

---

### 🟠 BUG-10 — `alpha = 0.01` makes the entropy weights decorative
**Where** `codes/Cost_metrix_estimation.ipynb`, cell 24

```python
alpha = 0.01
final_weights = alpha * entropy_weights + (1 - alpha) * preference_weights
```

| α | final weights |
|---|---|
| 0.00 | `[0.5000, 0.3333, 0.1667]` |
| **0.01** | **`[0.4967, 0.3324, 0.1709]`** |
| 0.50 | `[0.3340, 0.2867, 0.3793]` |
| 1.00 | `[0.1679, 0.2401, 0.5919]` |

The "hybrid entropy-weighted" scheme is, to three decimals, the hardcoded preference vector `[3,2,1]`. Note also that the objective (entropy) weights would rank **sentiment highest (0.59)** — the exact opposite of the hand-set preference.

**Fix** Either justify α from data (sensitivity sweep, reported as a figure) or drop the preference vector and use pure entropy weights. Do not present an arbitrary constant as a hybrid method. A sweep of α ∈ {0, 0.25, 0.5, 0.75, 1} with the resulting objective plotted is a strong, cheap paper figure.

---

### 🟠 BUG-11 — Train/inference skew: topic features zeroed at inference
**Where** `NeuroFuzzy_BERT_Sentiment.ipynb`, cell 17

```python
topic_feats = [0.0] * TOPIC_DIM     # "if vectorizer/svd are not saved, use zeros"
```
`TfidfVectorizer` and `TruncatedSVD` are never persisted, so every inference call feeds zeros where training saw real SVD components — then passes them through a scaler fitted on non-zero values. The model sees a distribution it was never trained on.

**Fix** `joblib.dump` the vectorizer, the SVD, and the scaler into `MODEL_DIR` alongside the weights, and load them at inference. Nothing fitted on training data may be re-derived or stubbed at inference time.

---

### 🟠 BUG-12 — Preprocessing fitted on train + validation together
**Where** `NeuroFuzzy_BERT_Sentiment.ipynb`, cells 6 and 8

```python
scaler.fit_transform(fuzzy_feats)                                  # cell 6: ALL rows
tfidf = vec.fit_transform(train_df['text'] + val_df['text'])       # cell 8: train + val
svd.fit_transform(tfidf)                                           # cell 8: train + val
```
Three separate leaks. The fuzzy features are additionally standardised **twice** (cell 6, then again in cell 8), so the values fed to the model are a scaler applied to an already-scaled matrix.

**Fix** Split first, `fit` on train only, `transform` validation and test. Standardise once.

---

### 🟠 BUG-13 — No test set
Split is 90/10 train/val (`test_size=0.1`), and the same validation set drives early stopping **and** the final reported metric. There is no held-out estimate of generalisation. The documentation claims 70/15/15.

**Fix** Implement the documented 70/15/15. Early-stop on validation; touch the test set exactly once, at the end.

---

### 🟠 BUG-14 — Training configuration is not runnable
```python
EPOCHS = 100 ;  BATCH_SIZE = 16 ;  LR = 2e-5 ;  full BERT-base fine-tune
```
58,714 training samples ÷ 16 = **3,670 optimiser steps per epoch × 100 epochs**. This is why the run ended in `KeyboardInterrupt` at epoch 1, and why `model_bundle.pkl` was written with `model_state = None` — **the saved artefact is an untrained model.**

**Fix** — see §6 Step 2. Freeze BERT and cache `[CLS]` embeddings once, then train only the head (seconds per epoch on CPU); or use DistilBERT; set `EPOCHS = 4` with `patience = 2`.

---

### 🟠 BUG-15 — Assignment output is operationally impossible
`optimized_employee_assignment.csv` (not committed to the repo):
```
employee_id  team_id   assigned_role
108201       TEAM_063  Program Manager
108201       TEAM_094  Project Manager
108201       TEAM_091  Tech Lead
108201       TEAM_033  Business Analyst
108201       TEAM_094  UI/UX Designer
108201       TEAM_015  Software Developer
```
One person, six simultaneous jobs. A downstream symptom of BUG-1, compounded by there being no uniqueness constraint. There is also no check that the assignee actually satisfies the slot's required skills, and no comparison against the current allocation — so the output cannot be said to be an improvement on anything.

**Fix** BUG-1 fix (a) makes the assignment one-to-one by construction. Add role-eligibility masking (`cost = ∞` for ineligible pairs) and always report the delta against the current allocation.

---

### 🟡 BUG-16 — Hard-coded bare relative paths
Every notebook loads bare filenames (`pd.read_csv("employee_dataset_realistic.csv")`) while living in `codes\` and the data lives in `dataset\`. They only ever ran with a Colab-flat working directory. As checked out, `codes/*.ipynb` cannot find their inputs.

**Fix** Add a two-line path header to every notebook:
```python
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd().parents[0]
DATA = ROOT / "dataset"
```
then read `DATA / "employee_dataset_realistic.csv"`.

---

### ⚪ DISSOLVED r3 — BUG-18 — New datasets carry no `skills` or `domain`, so skill matching has no input
**Where** `dataset/Employee_Team_Assignment.csv`, `dataset/Team_Dataset.csv`

Neither file has a skills column. The old pair had `skills` (employee) and `required_skills` (team), which fed the MPNet encoder that produced `similarity_matrix.npy`. **Skill fit is one of the three cost criteria in the objective, and it now has nothing to compute from.**

**Fix — in preference order**
- **(a) Regenerate with skills.** Best: add a per-employee `skills` list and a per-team `required_skills` list at generation time, with enough variety to avoid BUG-7 (target ≥ 300 distinct team profiles). Requires the generator (§12 Q2).
- **(b) Derive from `Title`.** `Title` is real and highly specific ("Senior Devops And Cloud Engineer", 1,076 distinct values). Build a title → skill-set mapping (keyword rules, or embed the title string directly with MPNet). Cheap, defensible, and preserves the semantic-matching story. **Recommended if (a) is unavailable.**
- **(c) Mine skills from review text** for each (Title, Department) cohort. Most novel, most work, and noisy.

Do **not** proceed to Step 3 or Step 5 until one of these is in place — the skill term is not optional.

---

### 🔴 BLOCKER #1 (r3) — BUG-19 — `Role` is randomly assigned, unrelated to `Title` or `Department`
**Evidence**
```
chi²  p(Department vs Role) = 0.5112   → independent
chi²  p(Title      vs Role) = 0.5463   → independent

example: Title "Senior Devops And Cloud Engineer",
         Department "DevOps Department"  →  Role = "Business Analyst"
```
Role labels were drawn without reference to what the employee actually does. Any role-eligibility mask, role-criticality weight, or role-based swap constraint built on this column is built on noise — and the team-composition columns (`Num_Developer`, `Num_QA`, …), while *internally* consistent, therefore describe a fictitious structure.

**Fix** Derive `Role` from `Title` with a keyword mapping, then recompute the `Num_*` columns:
```python
RULES = [("Team Lead",        r"lead|manager|head|principal|architect"),
         ("DevOps",           r"devops|sre|cloud|infrastructure|platform"),
         ("QA",               r"\bqa\b|test|quality|automation test"),
         ("Business Analyst", r"analyst|business|consultant|functional"),
         ("Developer",        r"develop|engineer|program|software|full.?stack")]
def role_from_title(title):
    for role, pat in RULES:
        if re.search(pat, title, re.I): return role
    return "Developer"                      # documented default
```
Re-verify with the same chi² test — association must now be significant. Report the coverage of the default branch as a data-quality figure.

---

### 🔴 BLOCKER #3 (r3) — BUG-20 — No latent employee-quality factor; the metrics are i.i.d. noise
**Evidence** Across the eight performance metrics (`Performance Score`, `KPI Score`, `Attendance`, `Peer Rating`, `Task Completion`, `Work Hours`, `Training Hours`, `Manager Feedback`):
```
mean |r| = 0.0109      max |r| = 0.0272
```
Real workforce data shows correlated performance signals — a strong employee tends to score well on several axes at once. Here every metric was drawn independently, so **there is no underlying "employee quality" construct for any model to discover.** (The old dataset had the same defect: max |r| = 0.17.)

**Why it matters** Step 4's counterfactual satisfaction model will perform at chance, and the entropy weights of §5 BUG-9 are computed over pure noise. Worse, a reviewer can run this two-line check.

**Fix** Regenerate metrics from a latent factor:
```python
q = rng.normal(0, 1, n)                                  # latent employee quality
perf = 75 + 12*q + rng.normal(0, 5, n)                   # each metric loads on q
kpi  = 78 +  9*q + rng.normal(0, 7, n)                   # with its own noise
...                                                       # target pairwise r ≈ 0.3–0.6
```
Add a *team* latent factor too, so that team membership genuinely affects outcomes — without it, reassignment cannot change anything and Step 6 has nothing to measure. **This is the highest-value change available to the data generator.**

---

### 🔴 BLOCKER #2 (r3) — BUG-21 — `Team_Performance` as a plain mean makes reassignment degenerate
`Team_Performance = mean(member Composite_Score)`. Two consequences, both fatal to the optimisation:

1. **Employee-weighted total is invariant.** Σᵢ cᵢ = 396,991.3 regardless of who sits where. Optimising it is a no-op — the same structural failure as BUG-2, one level up.
2. **Mean-of-team-means is gamed by team size, not by fit.** Stacking the highest scorers into the smallest teams — with zero regard for skill, role or satisfaction — lifts it from **79.4693 → 82.6819**. The optimiser would "discover" this and call it success.

**Fix** The objective must contain terms that depend on the *composition* of a team, not just the sum of its parts:
```python
F(A) = Σ_t [ w_p · size_weighted_perf(t)          # size-weighted, so it cannot be gamed
           - λ · std(member scores in t)          # balance  → "balanced teams" (the objective)
           + μ · mean_i s_hat[i, t]               # counterfactual satisfaction (Step 4)
           - ν · skill_gap(t)                     # coverage of required skills (needs BUG-18 fixed)
           - ρ · role_imbalance(t) ]              # e.g. no team without a Developer (29 currently)
```
Only the last four terms actually vary under reassignment. **They are the optimisation.** Report each term separately in the ablation table (Step 6) so it is visible which ones do work.

---

### 🟡 BUG-17 — Miscellaneous
| Item | Detail |
|---|---|
| Dead constant | `DATA_CSV = 'employees_reviews.csv'` — typo (real file `employee_reviews.csv`), never used |
| Lost kernel state | cell 17 raises `NameError: name 'os' is not defined`; the notebook was re-run out of order |
| No-op rename | `performance_metrix` cell 8 maps `"sum_of_task_completion_(%)"`, but `clean_column_names` already rewrote `%`→`percent`; the dict matches nothing. Patched by the duplicate map in cell 10 |
| Silent signal drop | `Sum of Manager Feedback`, `Sum of Performance Score`, `Sum of Work Hours Logged` excluded from the performance score without comment. Excluding manager feedback is *defensible* (it is the bias the project exists to remove) — but that argument must be **stated**, not left implicit |
| Entropy on min-maxed data | MinMax forces an exact 0 in every column, which contributes 0 to the entropy sum and biases weights toward columns with extreme outliers |
| Fragile default | `NeuroFuzzyBert(fuzzy_dim=6)` default vs actual 11; correct only because it is passed explicitly |

---

## 6. Recommended approach **[SUPERSEDED r3 — see §0.6–§0.9]**

Ordered by impact per unit of effort. Steps 1–3 are prerequisites for everything else.

### Step 1 — Re-key the dataset *(½ day — unblocks all downstream work)*
Apply BUG-1 fix (a) and BUG-3's alignment rule. Establish `employee_uid` as the single canonical key and the similarity matrix's row order as the single canonical ordering. Regenerate `similarity_matrix.npy` afterwards.

**Deliverable** `dataset/clean/` with consistent keys + a `00_build_dataset.ipynb` that produces it deterministically from the raw inputs.

---

### Step 2 — Aspect-based neuro-fuzzy sentiment, actually trained *(3–5 days)*

This is where the project's novelty should live, and the data to support it is already sitting unused.

**2a — Fix the model.** Remove leaked features (BUG-4), drop null ratings (BUG-5), split 70/15/15 (BUG-13), fit all transforms on train only (BUG-12), persist every fitted object (BUG-11).

**2b — Make it trainable on your hardware.** Full fine-tuning of BERT-base over 65k reviews is not feasible without a GPU. Two-stage schedule:

```python
# Stage 1 — cache [CLS] embeddings ONCE (~10 min GPU / ~60 min CPU for 65k)
bert.eval()
with torch.no_grad():
    cls = torch.cat([bert(**batch).last_hidden_state[:, 0].cpu()
                     for batch in loader])          # (N, 768), write to disk

# Stage 2 — train ONLY the fuzzy + MLP head on the cached vectors
#           seconds per epoch on CPU; sweep hyperparameters freely
```
Then, if a GPU is available, unfreeze the top 2–4 BERT layers for 2 epochs at `lr = 2e-5` for a final gain. `EPOCHS = 4`, `patience = 2`. Report both stages — "frozen-encoder vs partially-unfrozen" is a clean ablation.

**2c — Upgrade from one scalar to six aspects.** `employee_reviews.csv` carries six human-assigned aspect ratings that no notebook has touched: `work_life_balance`, `skill_development`, `salary_and_benefits`, `job_security`, `career_growth`, `work_satisfaction` (~75.8k labelled each).

Train a **multi-head** model: shared BERT encoder → six fuzzy-membership heads, each producing low/medium/high membership for one aspect.

| | Now | With aspects |
|---|---|---|
| Output | one scalar in [0,1] | six interpretable aspect scores |
| Supervision | leaked from `Overall_rating` | six genuine human labels |
| Fuzzy layer | invented membership functions | fitted to real ordinal ratings |
| HR value | "morale is low" | "**career growth** is the problem in Team 12, not salary" |

This does four things at once: it kills the leakage, it gives the fuzzy component real grounding, it produces the "Contextual Analysis" capability your own 35-page document lists as future work (§11.7 item 6), and it makes the sentiment signal **actionable** — which is what the objective demands. It is also the strongest available claim to novelty (see §9).

**Deliverable** `01_absa_neurofuzzy.ipynb` + `models/absa_neurofuzzy/` (weights, tokenizer, vectorizer, SVD, scaler, metadata) + a test-set classification report per aspect.

---

### Step 3 — Make the cost matrix genuinely pairwise *(1 day — fixes the thesis-critical BUG-2)*

Every term must vary with **both** *i* and *j*:

```python
C[i, j] = w1 * (1 - sim[i, j])                    # skill misfit        — pairwise ✓
        + w2 * criticality[j] * (1 - perf[i])     # varies over j       — now pairwise ✓
        + w3 * (1 - s_hat[i, j])                  # counterfactual sat. — pairwise ✓
        + BIG_M * (~eligible[i, j])               # hard role constraint
```

- `criticality[j]` — seniority/leverage of the slot (Tech Lead, Program Manager > Developer). Placing a weak performer in a critical role must cost more than placing them in a junior one. Derive it from role seniority and team size, and state the mapping.
- `s_hat[i, j]` — Step 4.
- `eligible[i, j]` — role-family compatibility, so a UI/UX Designer is not made a DevOps Engineer.

Verify the fix with the same test that exposed the bug: re-solve at several weightings and confirm the **skill-only objective now differs**. If it does not, a term is still row-constant.

---

### Step 4 — Counterfactual satisfaction model *(3–4 days — the genuine research contribution)*

The objective requires claims of the form *"Sarah will be more satisfied in the Data team."* Nothing in the current pipeline can support that: you observe sentiment only for each employee's **current** team.

Frame it as counterfactual prediction. Train on observed `(employee i, current team t) → sentiment`, then predict `ŝ[i, t']` for teams the employee has never been in.

```python
features(i, t) = [
    skill_fit          = sim[i, slot(t)],
    skill_gap          = required_skills(t) - skills(i),      # set difference size
    domain_match       = domain(i) == domain(t),
    perf_percentile    = rank of perf[i] within team t,       # relative standing
    team_mean_perf     = team_performance[t],
    team_variance      = performance_variance[t],
    team_size          = team_size[t],
    seniority_gap      = criticality[slot] - seniority[i],
]
# Target: the six aspect scores from Step 2 (or their aggregate)
```

**Critical design rule: use only *relational* features, never employee or team identity.** With 782 observations and 100 teams, identity features would be memorised and the model would not generalise to unseen pairs. Relational features force it to learn *what kinds of fit produce satisfaction* — which is exactly what transfers.

**Be honest about the limitation.** Each employee is observed in one team, so team effects and person effects are confounded. Say so in the paper, and mitigate with:
- **Cross-fitting** — train ŝ on half the employees, evaluate improvement using a ŝ fitted on the other half. Without this you optimise against your own model's errors (Goodhart's law) and the reported gain is fictitious. This single step separates a credible result from an unfalsifiable one.
- **Sensitivity analysis** — report how the recommended swaps change as w₃ varies.

Start with gradient boosting (LightGBM / XGBoost) — strong on tabular data at this scale, and SHAP gives the per-recommendation explanation HR needs. Only try a neural model if boosting underperforms.

**Deliverable** `02_counterfactual_satisfaction.ipynb`, `s_hat` matrix (782 × 100), cross-fitted validation curve, SHAP summary.

---

### Step 5 — Reframe the optimiser as *swapping* *(2–3 days)*

Global Hungarian re-assignment of all 782 employees is not what the objective asks for, and it cannot express what actually matters. Team performance depends on the **set** of members — variance, skill coverage, seniority balance. Those are *quadratic* in the assignment; a linear assignment problem cannot represent them at all.

Use **local search over pairwise swaps**, seeded from the *current* allocation:

```python
F(A) = mean_t team_perf(A, t)                  # raise average performance
     - λ * mean_t team_std(A, t)               # penalise imbalance  → "balanced teams"
     + μ * mean_i s_hat[i, A(i)]               # raise satisfaction
     - ν * mean_t skill_gap(A, t)              # maintain skill coverage

# Steepest-descent over the swap neighbourhood
for it in range(max_iters):
    best = max(((F(swap(A,i,j)) - F(A), i, j)
                for i, j in candidate_pairs(A)), default=None)
    if best is None or best[0] <= tol: break
    A = swap(A, *best[1:])
```
Restrict `candidate_pairs` to role-compatible pairs to keep it tractable, and use random restarts or simulated annealing to escape local optima.

**Why this is the right model**
1. It matches the objective's own wording — *"exchanging experienced members between teams."*
2. It outputs **top-*k* ranked swaps** with predicted deltas — something HR can act on, unlike a full 782-person reshuffle.
3. It expresses team-level (set) objectives that Hungarian structurally cannot.
4. It starts from the status quo, so every recommendation is an incremental, auditable change.

**Deliverable** `03_swap_optimizer.ipynb` → `recommended_swaps.csv` with columns: `emp_a, team_a, emp_b, team_b, Δteam_perf_a, Δteam_perf_b, Δsatisfaction, Δvariance, rationale`.

---

### Step 6 — Evaluation *(2–3 days — without this there is no paper)*

| Element | What to report |
|---|---|
| Baselines | current allocation · random swaps · skill-only Hungarian · greedy best-fit |
| Ablations | skill → +performance → +sentiment(VADER) → +sentiment(ABSA) → +counterfactual |
| Metrics | mean team performance, performance variance (balance), mean predicted satisfaction, skill-coverage gap, % employees improved, % worsened |
| Sensitivity | sweep λ, μ, ν and α (BUG-10); plot the Pareto frontier of performance vs. satisfaction |
| Honesty | cross-fitted ŝ (Step 4); explicit statement that ground truth for realised outcomes does not exist |
| Stability | rerun with different seeds; report the variance of the recommended swap set |

The ablation table is the core result of the paper. It is the only thing that answers *"does sentiment actually help?"* — and right now, because of BUG-2, the honest answer is no.

---

## 7. Alternative routes to the objective **[SUPERSEDED r3 — see §0.4]**

Considered and ranked. Step 5 recommends **B**; the others are real options and worth a paragraph in the paper's methodology discussion.

| # | Approach | Strength | Weakness | Verdict |
|---|---|---|---|---|
| **A** | Linear assignment (Hungarian) with genuinely pairwise costs | Globally optimal, fast, exact | Cannot express team-level (set) objectives; reshuffles everyone | Keep as a **baseline** |
| **B** | **Local search over pairwise swaps** | Matches the objective's wording; handles quadratic team objectives; incremental & actionable | Local optimum only; needs restarts | ✅ **Recommended** |
| **C** | MILP / CP-SAT with explicit constraints (OR-Tools) | Exact under hard constraints (team size, role coverage, min seniority); constraints are self-documenting | Quadratic terms need linearisation; may not scale | Strong **second choice** — excellent if the team wants rigour over novelty |
| **D** | Genetic programming / RL for team formation | Highest novelty; handles arbitrary non-convex objectives | Slow, hard to tune, hard to justify at this scale | Only with spare time |
| **E** | **Skyline computation + total ordering** | Returns the Pareto-optimal set *without fixing weights* — sidesteps BUG-10 entirely; **your guide already published this technique** | Skyline set can be large; needs a tie-break | ⭐ **Strong differentiator.** Very likely to please the guide |
| **F** | Stable matching (Gale–Shapley) | Elegant HR framing: no employee–team pair both prefer to defect; explains *why* an allocation holds | Stability ≠ optimality; needs two preference orderings | Excellent **framing alternative**; cheap to add as a comparison |
| **G** | Top-*k* role recommendation (no global optimisation) | Simplest; ranking metrics (NDCG, MRR) are standard | Ignores team-level constraints entirely | Fallback if time runs out |

**Note on E.** Skyline computation over (skill fit, performance, satisfaction) returns every non-dominated candidate for each slot. It removes the need for the arbitrary `[3,2,1]` weight vector, directly addresses BUG-9/BUG-10, connects your work to your guide's prior publication, and gives HR a *set* of defensible options rather than one opaque number. If you want one differentiator beyond the ABSA work, this is the cheapest high-value addition.

---

## 8. Implementation roadmap **[SUPERSEDED r3 — see §0.9, §0.12]**

| # | Task | Effort | Depends on | Fixes |
|---|---|---|---|---|
| 0 | **Migrate to `Employee_Team_Assignment` + `Team_Dataset`** as the backbone | ½ d | — | BUG-1, 3, 15 |
| 0b | **Add skills** (derive from `Title`, or regenerate) | 1 d | 0 | **BUG-18** |
| 0c | **Derive `Role` from `Title`**; recompute `Num_*` | ½ d | 0 | **BUG-19** |
| 0d | **Regenerate metrics from a latent quality factor** | 1 d | 0 | **BUG-20** |
| 0e | Join reviews on (Title, Department); record backoff level | ½ d | 0 | replaces random stapling |
| 1 | ~~Re-key old dataset~~ — superseded by task 0 | — | — | BUG-1, 3, 15 |
| 2 | Regenerate team profiles (≥ 300 distinct); drop `job_role` from profile text | 1 d | 1 | BUG-7, 8 |
| 3 | Path header in all notebooks | 1 h | — | BUG-16 |
| 4 | Fix leakage; drop nulls; 70/15/15; persist transforms | 1 d | — | BUG-4, 5, 11, 12, 13 |
| 5 | Cache `[CLS]` embeddings; train head; then unfreeze top layers | 2 d | 4 | BUG-14 |
| 6 | Six-aspect multi-head fuzzy model | 2–3 d | 5 | — (new capability) |
| 7 | Pairwise cost matrix + eligibility mask + re-verify | 1 d | 1, 2 | **BUG-2**, 9, 10 |
| 8 | Counterfactual satisfaction model + cross-fitting | 3–4 d | 6, 7 | — (new capability) |
| 9 | Swap optimiser → `recommended_swaps.csv`; composition-aware objective | 2–3 d | 8 | — (**the objective**), **BUG-21** |
| 10 | Evaluation: baselines, ablations, sensitivity | 2–3 d | 9 | — (**the paper**) |
| 11 | Rewrite documentation statistics from data | ½ d | 4 | BUG-6 |
| 12 | *Optional* — skyline variant (Alt. E) | 2 d | 7 | BUG-10 |
| 13 | *Optional* — dashboard / FastAPI | 3–5 d | 9 | — |

**Critical path to a submittable paper: 0 → 0b → 0c → 0d → 4 → 5 → 6 → 7 → 8 → 9 → 10.** Roughly 3–4 focused weeks. Tasks 2, 3, 11 can run in parallel and are cheap.

**If time is short**, the minimum defensible paper is `1 → 4 → 5 → 7 → 9 → 10`: a trained (non-leaking) sentiment model, a cost matrix where sentiment demonstrably matters, a swap optimiser, and an ablation table. Steps 6 and 8 are what elevate it from coursework to a contribution.

---

## 9. Related work — what to reference and what to borrow

### 9.1 Directly usable

**[Rink, Meijdam & Graus (2024), "Aspect-Based Sentiment Analysis for Open-Ended HR Survey Responses", NLP4HR @ EACL 2024](https://aclanthology.org/2024.nlp4hr-1.2/)**
The closest published analogue. Randstad/UvA; identifies **six aspects** (salary, schedule, contact, communication, personal attention, agreements) by clustering, validated by domain experts; few-shot ABSA with Dutch BERT models (BERTje, RobBERT) against bag-of-words and zero-shot baselines; **1,458 annotated responses**; explicitly notes label imbalance in both aspects and sentiments.

*Why it matters to you:* it establishes ABSA-in-HR as a publishable framing, and its six aspects map almost one-to-one onto your six existing rating columns. Note the scale difference — they annotated 1,458 responses; **you already hold ~75,800 aspect-labelled reviews.** That is a defensible contribution on dataset scale alone, in a different language and market (Indian tech sector, English). Cite as your primary anchor. Their limitation — no downstream decision system — is precisely your opening.

**[Zhao et al. (2019), "The Impact of Person-Organization Fit on Talent Management", KDD '19](https://dl.acm.org/doi/10.1145/3292500.3330849)**
Person–organisation fit modelled at scale for talent management. Use for Step 4's theoretical grounding: P-O fit predicts job satisfaction, organisational commitment, and (negatively) turnover intention. This is the literature that justifies predicting `ŝ[i,t]` from fit features rather than identity.

**[Wang et al. (2023), "A Reinforcement Learning-assisted Genetic Programming Algorithm for Team Formation Problem Considering Person-Job Matching", arXiv:2304.04022](https://arxiv.org/abs/2304.04022)**
Team formation with person–job matching under uncertainty; argues precise numeric matching indices are unrealistic and proposes fuzzy/uncertain evaluation. Directly supports your fuzzy-logic choice, and is the reference for Alternative D.

**[Andrejczuk et al., "Synergistic Team Composition: A Computational Approach to Foster Diversity in Teams", arXiv:1909.11994](https://arxiv.org/abs/1909.11994)**
SynTeam — local search with multiple neighbourhoods for synergistic team composition. **The methodological template for Step 5.** Borrow the neighbourhood design and the argument for why team composition is not a linear assignment problem.

**[Quadratic assignment problem variants: a survey, and iterated local search for QAP](https://www.sciencedirect.com/science/article/abs/pii/S0377221720309814)**
Formal grounding for the swap-neighbourhood formulation and for tabu / iterated-local-search refinements. Cite when you justify abandoning Hungarian for local search.

**[Vashishtha & Susan (2023), "Sentiment analysis using fuzzy logic: a comprehensive literature review", WIREs DMKD](https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/widm.1509)**
Survey covering fuzzy-rule methods, neuro-fuzzy networks, and fuzzy emotion recognition. Your literature-review backbone for the fuzzy half.

**[A hybrid approach for sentiment analysis using fuzzy logic and DistilBERT (Discover Applied Sciences, 2025)](https://link.springer.com/article/10.1007/s42452-025-08015-9)** and **[A Three-Step Fuzzy-Based BERT Model for Sentiment Analysis (Springer, 2022)](https://link.springer.com/chapter/10.1007/978-981-19-0489-9_4)**
Two concrete fuzzy+BERT architectures. The three-step model applies fuzzy logic *after* BERT classification to resolve inconsistencies — a cheap ablation against your *concatenate-before-classifier* design. The DistilBERT variant is your evidence that a distilled encoder is acceptable, which matters for BUG-14.

**[Fuzzy ensemble of fine-tuned BERT models (PLOS ONE, 2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0300279)**
Fuzzy ensembling over several fine-tuned BERTs for domain-specific sentiment. A credible extension if you have compute to spare.

**[Zhu et al., "Natural Language Processing for Human Resources: A Survey", arXiv:2410.16498](https://arxiv.org/abs/2410.16498)**
Recent, broad, and ideal for framing your Introduction and Related Work sections.

**[Srivastava & Eachempati, "Retention Is All You Need", arXiv:2304.03103](https://arxiv.org/abs/2304.03103)**
Explainable ML for employee attrition. Relevant to the "predictive turnover" item in your own §11.7 roadmap, and a good model for how to present explainability to an HR audience.

### 9.2 Techniques to borrow
| Technique | Source | Use in this project |
|---|---|---|
| Six-aspect ABSA schema | Rink et al. 2024 | Step 2c — you already hold the labels |
| Fuzzy membership tuned by GA | fuzzy-BERT literature | Replaces the hand-set σ = 0.15 Gaussians |
| Post-hoc fuzzy correction layer | Three-Step Fuzzy BERT | Ablation vs. your concatenation design |
| Swap neighbourhoods + restarts | SynTeam / QAP survey | Step 5 |
| Skyline computation | **your guide's own tourism paper** | Alternative E — removes arbitrary weights |
| Entropy-weight ↔ TOPSIS | MCDM literature | Legitimises §5 BUG-9/10 once orientation is fixed |
| SHAP explanations | Retention Is All You Need | Per-recommendation rationale for HR |
| Cross-fitting | causal-inference practice | Step 4 — prevents optimising against your own model's error |

### 9.3 Honest positioning
Do **not** claim "first to combine BERT with fuzzy logic for sentiment" — that space is crowded (see 9.1). The defensible claims are:

1. **Scale + domain** — aspect-based employee-satisfaction analysis on ~75.8k aspect-labelled Indian tech-sector reviews, against the 1,458 Dutch responses of the closest prior work.
2. **Closing the loop** — prior work stops at the sentiment score. You carry aspect-level sentiment into a *counterfactual fit model* and then into a *combinatorial reassignment optimiser*, and you measure the end-to-end effect. **No paper in the surveyed set does this.** That is the contribution.
3. **Actionability** — output is a ranked list of concrete swaps with per-aspect rationale, not a dashboard number.

---

## 10. Mapping to an IEEE conference paper

| Section | Content | Source |
|---|---|---|
| Abstract | Problem → ABSA + counterfactual fit + swap optimisation → measured improvement | — |
| I. Introduction | Bias in HR reassignment; quantitative metrics miss morale | `HR paper (2).pdf` |
| II. Related Work | ABSA in HR · fuzzy+BERT · team formation / QAP · P-O fit | §9 |
| III. Problem Formulation | Formal statement; why linear assignment is insufficient | §7 |
| IV. Methodology A | Data & preprocessing | §3, Step 1 |
| IV. Methodology B | Aspect-based neuro-fuzzy BERT | Step 2 |
| IV. Methodology C | Counterfactual satisfaction model | Step 4 |
| IV. Methodology D | Cost formulation & swap optimiser | Steps 3, 5 |
| V. Experiments | Baselines, ablations, sensitivity, stability | Step 6 |
| VI. Results & Discussion | Ablation table; Pareto frontier; case studies | Step 6 |
| VII. Limitations | Synthetic org data; single-team observation; no realised outcomes | §11 |
| VIII. Conclusion | — | — |

**Reusable from the 35-page internal document:** the three case-study narratives (skill mismatch, team balance, at-risk employee) in §11.4 are good — regenerate them from *real model output* instead of the current hypothetical numbers, and they become the paper's qualitative results section.

---

## 11. Limitations to state explicitly

Reviewers will find these. State them first.

1. **The organisational data is synthetic.** Teams, skills, and performance metrics are generated, not observed. Reviews are real but were attached to synthetic employees at random. No claim about a real workforce is supported.
2. **Each employee is observed in exactly one team.** Person effects and team effects are confounded; counterfactual satisfaction is an extrapolation, mitigated by cross-fitting but not eliminated.
3. **No realised outcomes.** No transfer was ever executed and measured. Evaluation is necessarily a simulation study against model-predicted objectives.
4. **Reviews are cross-sectional.** The "sentiment trajectory over time" described in the internal documentation is not supported by the data.
5. **Aspect labels are self-reported** by reviewers and carry their own biases.
6. **Fairness is unaudited.** A reassignment system affects careers. Before any claim of practical use, check recommendation rates across the demographic attributes present in the archived HR datasets, and state the result.

---

## 12. Decision queue — open questions for the team

| # | Question | Blocks |
|---|---|---|
| **Q1** | **BUG-1:** is one row = one employee (→ re-key, ½ day), or are there truly 138 people (→ regenerate)? | Steps 1–5. **Highest priority.** |
| **Q2** | **Still open, now more urgent.** Where is the generator for `Employee_Team_Assignment.csv` / `Team_Dataset.csv`? BUG-18 (no skills), BUG-19 (random roles) and BUG-20 (i.i.d. metrics) are all cheapest to fix at generation time. | tasks 0b, 0c, 0d |
| **Q9** | Retire the old `employee_dataset_realistic` / `team_dataset_realistic` pair, or keep them for a comparison run? | housekeeping |
| **Q3** | Is a GPU available (Colab Pro, lab machine)? Decides DistilBERT vs BERT-base, frozen vs fine-tuned. | Step 2b |
| **Q4** | Deadline and target venue? Decides whether Steps 6 and 8 are in scope. | Scope of everything |
| **Q5** | Is `UARS.pdf` relevant? Not yet read, per instruction. If it contains a reassignment or ranking formulation it should be reviewed before Step 5 is designed. | Step 5 |
| **Q6** | Deliverable scope — notebooks + paper, or is a dashboard/API expected for the viva? | Roadmap task 13 |
| **Q7** | Does any real outcome data exist (actual transfers and what followed)? If not, Step 6 is a simulation study and will be written as such. | Step 6 |
| **Q8** | Permanently delete `archive\`, or keep it? | Housekeeping |

---

## 13. Working conventions

1. **Canonical order.** `dataset/employee_dataset_realistic.csv` row order defines matrix row order. Everything else reindexes onto it. Never `sort_values` a frame that indexes into a matrix.
2. **Fit on train only.** Every scaler, vectoriser, and decomposition is fitted on the training split and persisted with the model.
3. **No target-derived features.** If a feature can be computed from the label, it does not go in the model.
4. **Assert meaningfully.** An assertion that compares two objects derived the same way proves nothing (BUG-3). Assert against the independent source.
5. **Notebooks are runnable top-to-bottom.** Restart-and-run-all before committing. Out-of-order state caused BUG-17.
6. **Every reported number is reproducible** from a committed notebook. Nothing hand-typed into the documentation.
7. **Recompute documentation statistics from data** before any submission (BUG-6).

---

## 14. Changelog

- **rev 5 - 2026-09-23 - Sentiment unblocked; BUG-4 and BUG-5 proved and fixed.**
  BLOCKER #3 is closed. The subsystem's delivery was found in `dataset/` under a
  missing-separator filename (`datasetsentiment_fuzzy.csv`, from
  `"dataset" + "sentiment_fuzzy.csv"`), so nothing had been reading it. It audits
  clean: 5,000 rows, 1:1 against the roster, no duplicates, no unknown IDs,
  memberships summing to 1, `s_crisp` exactly their centroid, Title *and* Department
  matching the roster for all 5,000, and r2 = 0.29 against `Overall_rating` - well
  clear of the 0.95 leak threshold in 0.8. New `src/sentiment_build.py` +
  `run.py sentiment` converts it to the contract and refuses to write on a leak or a
  validation failure. The synthetic placeholder is retired.

  **BUG-4 and BUG-5 were measured, not just asserted.** Feeding the rating-derived
  memberships `r_low/r_mid/r_high` as model inputs reconstructs the 3-class label at
  **100.0% validation accuracy with the text ignored entirely** - BERT was
  contributing nothing. The only reason the leak measured 88.3% rather than 100% was
  BUG-5: `fillna(3)` gave the 14,330 null-rating rows a feature vector identical to a
  genuine rating-3 row while labelling them POSITIVE. Both are fixed in the notebook,
  along with the TF-IDF/SVD/scaler being fitted on train+val concatenated, and a
  train/inference mismatch that substituted zeros for the three topic features
  because `vec` and `svd` were never persisted in the bundle.

  **Results on real data (n = 300):** every headline claim survives. Full mean gain
  7.8865 (was 6.4246 under the placeholder), median 4.8503, damages a pool **0.00%**
  of the time, and loses to none of the four baselines. Sentiment changed the
  selected movers on **67.0%** of pairs, while `sent_only` still has a median gain of
  exactly 0.0000 and damages 42.67% - a modifier, not a driver, exactly as 0.5
  predicted. The intuitive `worst_for_best` heuristic damages the pool in **41.67%**
  of cases. The lambda ablation sharpened: lambda = 0.5 is the unique value with a
  0.0% loss rate against perf_only (0.00 loses 30.0%; 1.00 and 2.00 reintroduce 0.5%).

  **Two things this did NOT resolve.** (1) No script in the repository produces the
  delivery's `Model_Prob_*` columns, and the notebook cannot - `employee_reviews.csv`
  has no `Employee_ID`, so it can only reach cohort granularity. Whatever attached one
  review per employee lives outside the project, so working convention 6 does not hold
  for the sentiment step until it is committed. (2) `SENTIMENT_FLOOR = 0.35` was
  calibrated against the placeholder; on the delivered spread (std 0.3268) it now
  holds 791 employees rather than 288, and two low teams lose every movable member to
  it. Its price must be re-measured before the paper quotes it.

- **rev 4 - 2026-08-31 - Implementation.** Built `src/` (config, roles, sentiment_io, fis,
  objective, search, pipeline, evaluate) and `run.py`. BLOCKER #1 and #2 resolved and
  verified; BLOCKER #3 documented as an open limitation. Key finding: title tokens
  "Consultant", "Analyst", "Engineer", "Manager", "Senior", "Lead" carry no occupational
  signal in this organisation (measured lift ~1.0), so role derivation was rebuilt on the
  16 tokens that clear a lift >= 1.5 bar, with `Department` as the fallback - chi-2 p
  against Title fell from 0.5463 to 9.50e-176. Verified the exhaustive search is exact
  (0.00% loss against an unconstrained baseline) and measured the price of the do-no-harm
  constraint (-12.6% mean gain). Baseline sweep runs but is PROVISIONAL: sentiment is a
  synthetic placeholder until the real file lands. Sections 0.12 and 0.13 rewritten as-built.

- **rev 3 — 2026-08-31 — Scope revision.** Problem narrowed to role-matched swaps between a
  handed-in pool of low-performing teams, driven by performance + externally supplied
  fuzzified sentiment; skills and embeddings dropped. Added §0 with: the proof that the swap
  pairing is not a decision variable (so a pairing-Hungarian is degenerate), the exact
  zero-sum measurement that forces a team-dependent value function, the recommended objective
  and counterfactual-sentiment FIS, the interface contract for the sentiment subsystem, the
  algorithm, exhaustive-search cost measurements, re-graded bug statuses, the evaluation
  design over 10,153 low-team pairs, and the new module layout. Retired the embedding and
  global-cost-matrix notebooks. §6–§8 marked superseded.

| Date | Change |
|---|---|
| 2026-08-30 (rev 2) | Assessed `Team_Dataset.csv` + `Employee_Team_Assignment.csv`. Verdict: adopt as backbone — resolves BUG-1/3/15 and enables principled review attachment on (Title, Department). Four new issues registered (BUG-18…21). Roadmap re-sequenced with tasks 0–0e. |
| 2026-08-30 | Document created. Full audit of four notebooks and eight data files. 17 bugs registered with evidence. Roadmap, alternatives, and reference set added. Four orphaned CSVs archived; PDFs consolidated into `docs\`. |
