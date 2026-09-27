# Project status — system complete

**Project:** Team Rebalancing by Role-Matched Employee Reassignment
**Date:** 2026-09-27
**Under the guidance of:** Prof. Partha Ghosh

| | |
|---|---|
| Manshi Bachhan | 16900123057 |
| Ankit Kumar | 16900123017 |
| Nandan Kumar | 16900123060 |
| Beauty Mondal | 16900123031 |
| Gautam Kumar Mishra | 16900123043 |

---

## Summary

**The system is complete and working on real data.** All six commands run, every integrity
invariant passes, the synthetic placeholder that earlier results depended on has been
retired, and every number quoted in the paper is machine-checked against the underlying
data.

**Remaining work is the paper submission itself** — compilation, the author block, and the
venue page limit. No further engineering is required.

---

## Verify it in four commands

Every claim below can be reproduced from a clean checkout in under a minute.

```bash
pip install -r requirements.txt
python run.py doctor                      # all integrity invariants
python run.py recommend --top 10          # the actual deliverable
python paper/results/verify_numbers.py    # paper matches the data  -> exits 0
python paper/results/check_tex.py         # LaTeX structure         -> PASS
```

`doctor` is the single most useful thing to run. It loads both datasets, applies the role
repair, validates the sentiment contract, runs discovery, and solves the highest-headroom
pool as a smoke test. Every number it prints is an invariant that must hold on any correct
input.

---

## Evidence

### The pipeline runs end to end

| Component | State |
|---|---|
| Role repair | **done** — χ² p = 9.50e-176, 0.10% fallback, 570/570 teams reconcile |
| Team discovery | **done** — 143 of 570 teams flagged |
| Pool formation | **done** — 9,349 feasible pairs |
| Value function | **done** — intrinsic `v(e)`, zero-sum verified to 10 decimal places |
| Bound + pruning | **done** — 95.8% of pools skipped *with a proof* |
| Optimiser | **done** — exhaustive and proven optimal at K = 2, local search beyond |
| Protective filter | **done** — 791 employees held; measured cost 11.4% of mean gain |
| Evaluation harness | **done** — five arms, three ablations, 1,000-pair sweep |
| CLI + outputs | **done** — six commands, four CSV outputs |
| Real sentiment data | **done** — delivered, validated, placeholder retired |
| Paper draft | **done** — ~10–11 pages, all numbers verified |
| Paper submission | **open** — see below |

### Data integrity checks all pass

```
employees_loaded                  : 5000
teams_loaded                      : 570
employee_id_unique                : true
num_star_sums_to_team_size        : 570      (all 570 teams reconcile)
sentiment_is_placeholder          : false    <-- real data, not synthetic
sentiment_problems                : []       (no contract violations)
```

The sentiment dataset supplied by the sentiment subsystem was independently validated:
5,000 rows against a 5,000-employee roster at **100.0% coverage**, no duplicate IDs, no
unknown IDs, no null scores, class probabilities summing to 1 on every row, and — most
importantly — **r² = 0.29** against the star rating, confirming the model read the review
text rather than echoing the rating it was labelled from. Had that exceeded 0.95 the whole
objective would have collapsed to zero-sum, and the build step refuses to write in that case.

### Results (1,000 team pairs, real data)

| Arm | mean gain | median | % improved | % damaged |
|---|---|---|---|---|
| **proposed** | **8.0811** | **5.7184** | **82.10** | **0.00** |
| performance only | 4.8470 | 1.7425 | 59.00 | 19.80 |
| sentiment only | 0.2596 | 0.0000 | 39.40 | 41.10 |
| random | 0.1365 | 0.0000 | 45.20 | 45.30 |
| intuitive human heuristic | 2.0021 | 0.5779 | 53.10 | 37.50 |

**99.9% of pairs solved to proven global optimality.** The proposed method never loses to
any of the four baselines and never damages a pool. The intuitive human move — trading the
weak team's worst member for the strong team's best — **damages the team pool in 37.5% of
cases**, which is the practitioner-facing result of the project.

### Two corrections found during validation

Both were in the upstream sentiment model, found by our integrity checks and reported back:

1. **Target leakage.** Rating-derived features were being fed to the classifier as inputs,
   reconstructing the label at **100.0% validation accuracy with the review text ignored
   entirely** — the BERT encoder was contributing nothing.
2. **14,330 reviews with a missing rating** were being silently labelled POSITIVE.

Catching these is the reason the delivered scores can now be trusted.

---

## What remains — paper submission only

| # | Item | Blocking? |
|---|---|---|
| 1 | The paper has never been compiled — needs an Overleaf pass (no LaTeX toolchain on the dev machine) | yes |
| 2 | Institution, city and contact e-mail — `paper/main.tex:48` is still a `TODO` | yes |
| 3 | Prof. Ghosh appears as final author **and** in the Acknowledgment — one convention must be chosen | yes |
| 4 | Page limit — ~10–11 pages against a likely 6–8 cap; compressible material is already identified in priority order | depends on venue |
| 5 | Target venue and deadline not yet fixed | decision |

Full detail for whoever writes the paper is in **`HANDOVER.md`**, which carries the concept,
the flow, every verified number, and an explicit list of claims the evidence does *not*
support.

---

## Known limitations, stated deliberately

These are recorded rather than hidden, and are written into the paper:

1. **The organisational data is synthetic.** Teams and performance metrics are generated.
   The *sentiment* is modelled from 86,117 real employee reviews; the *roster* is not a real
   workforce.
2. **No ground truth.** No reassignment was ever executed and its outcome measured, so the
   system is evaluated against baselines on its own objective. This is weaker than an
   accuracy figure, and the paper says so rather than implying otherwise.
3. **The system does not predict personal fit.** An earlier design tried to predict how
   satisfied an employee would be *after* a move; it was withdrawn because nothing in the
   data can validate a counterfactual. The system recommends on role, performance and
   *observed* sentiment only, and every output states that the personal judgement stays with
   the people who know the employee.
4. **The sentiment scores cannot be regenerated from this repository** — by design, another
   subsystem owns that model.
5. **Fairness is unaudited.** A reassignment system affects careers; no check has been run on
   recommendation rates across demographic attributes.
6. **Configurable, not yet general-purpose.** The optimiser, the equalisation bound and the
   search are data-independent mathematics and would work on any organisation unchanged.
   Role derivation, however, carries a token taxonomy tuned to this organisation's job
   titles. Running on a different company's data is a configuration task, not a rewrite, and
   the design for making it fully general is recorded in `WORKFLOW.md` §13.1 as future work.

---

## Where to read further

| Document | Purpose |
|---|---|
| `README.md` | how to run it; commands, inputs, outputs, configuration |
| `WORKFLOW.md` | the concept and a stage-by-stage walkthrough of the code; §13 lists deferred work |
| `HANDOVER.md` | the paper author's pack — concept, flow, every verified number, what not to claim |
| `PROJECT.md` | the full design record: derivations, measurements, and rejected approaches |
| `paper/main.tex` | the draft |
