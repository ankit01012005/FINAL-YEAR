# IEEE conference paper

**Performance Analysis of Company Teams and Employee Reassignment Using Sentiment
Analysis: A Provably Exact Redistribution Framework**

## Compiling

The simplest route is **Overleaf**: create a blank project, upload this whole
`paper/` folder (keeping the `results/` subfolder), set the main document to
`main.tex`, and press Recompile. `IEEEtran` ships with Overleaf.

Locally, with TeX Live or MiKTeX installed:

```bash
cd paper
pdflatex main
pdflatex main        # second pass resolves table/figure/equation references
```

No `bibtex` run is needed — the bibliography is a hand-written `thebibliography`
block, so two `pdflatex` passes are sufficient.

There is **no LaTeX toolchain on this machine**, so the source has not been
compiled. It has instead been validated structurally by `results/check_tex.py`,
which checks environment balance, brace balance, math delimiters, every
`\ref`/`\eqref` against its `\label`, every `\cite` against its `\bibitem`,
tabular cell counts against each column specification, and the presence of every
included figure.

## Before submitting

1. **Fill in the institution.** `main.tex` has a `TODO` marker in the author
   block for the institution name, city and contact e-mail.
2. **Check the author list.** Prof. Partha Ghosh is currently listed as the final
   author *and* thanked in the Acknowledgment. Pick one convention.
3. **Check the page limit.** The paper is long (roughly 10–11 pages in IEEEtran
   two-column). If the venue caps at 6 or 8 pages, the compressible material is,
   in order: the Related Work subsections, Table I (token lift — the text carries
   the argument alone), and the case study table.

## Files

| Path | What it is |
|---|---|
| `main.tex` | the paper |
| `results/fig_pruning.pdf` | Fig. 1 — pools solved vs shortlist size |
| `results/fig_gains.pdf` | Fig. 2 — per-pool gain distribution by arm |
| `results/make_figures.py` | regenerates both figures from the data |
| `results/check_tex.py` | static LaTeX validation (run before compiling) |
| `results/verify_numbers.py` | re-derives **every** number in the paper from the CSVs |
| `results/ablate_*.csv` | ablation results, n = 200 pairs |
| `results/sweep1000.log`, `ablations.log` | raw run logs |
| `results/run_ablations.py` | reproduces the ablation CSVs |

## Reproducing every number

```bash
cd ..
python run.py sweep --limit 1000 --save     # Table II  (~9 min)
python paper/results/run_ablations.py       # Tables IV-VI  (~20 min)
python paper/results/make_figures.py        # Figs 1-2
python paper/results/verify_numbers.py      # asserts the paper matches the data
```

`verify_numbers.py` is the important one: it exits non-zero if any figure quoted
in `main.tex` has drifted from the underlying data, so it should be re-run after
any change to the code or the tables. It currently **passes**. Note that it holds
the paper's claims as literals in the script itself, so a table edit in
`main.tex` must be mirrored there or the check silently tests the old number.

## Status of the claims

The paper is deliberately framed as a **methodological / negative-results**
contribution, because that is what the current evidence supports.

- **Structural results** (Propositions 1–4) are data-independent and hold as
  stated.
- **Empirical comparisons** run on synthetic *organisational* data (teams,
  performance metrics), but the sentiment scores are now real: modelled from
  86,117 employee reviews and resolved to the roster on `(Title, Department)`.
  The placeholder is retired. The synthetic-org caveat is stated in the
  abstract, in Section VI-A and in Section IX — it is not buried.
- The **circularity** of comparing an optimiser against baselines on its own
  objective is stated explicitly in Section VIII-A rather than left for a
  reviewer to find.

The single result that does not depend on any of the above is that the intuitive
human heuristic (trade the weaker team's worst member for the stronger team's
best) **damages the pool in 37.5% of cases**, while the proposed method damages
it in **0.0%**.
