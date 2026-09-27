"""Ablation runs at paper scale. Writes CSVs into paper/results/."""
import warnings, pathlib
warnings.simplefilter("ignore")
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src import pipeline, evaluate

OUT = pathlib.Path(__file__).resolve().parent
ctx = pipeline.load_context()
pairs = pipeline.low_team_pairs(ctx)
N = 200

for name, fn in [("ablate_lambda", lambda: evaluate.ablate_lambda(ctx, pairs, limit=N)),
                 ("ablate_budget", lambda: evaluate.ablate_budget(ctx, pairs, limit=N)),
                 ("ablate_floor",  lambda: evaluate.ablate_floor(ctx, pairs, limit=N))]:
    df = fn()
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"=== {name} (n={N}) ===")
    print(df.to_string(index=False), flush=True)
