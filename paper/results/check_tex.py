"""Static structural check for main.tex, for use when no LaTeX toolchain is present.

Catches the errors that actually break an IEEEtran build: unbalanced environments
or braces, odd inline-math delimiters, dangling \\ref/\\cite, and - the most common
silent killer in a table-heavy paper - tabular rows whose cell count disagrees
with the column specification.
"""
import re
import pathlib

TEX = pathlib.Path(__file__).resolve().parents[1] / "main.tex"
text = TEX.read_text(encoding="utf-8")
lines = text.split("\n")
failures = []


def fail(msg):
    failures.append(msg)


# ------------------------------------------------------------------ environments
stack = []
for i, line in enumerate(lines, 1):
    if line.lstrip().startswith("%%"):
        continue
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append((env, i))
        elif not stack:
            fail(f"line {i}: \\end{{{env}}} with nothing open")
        elif stack[-1][0] != env:
            fail(f"line {i}: \\end{{{env}}} closes \\begin{{{stack[-1][0]}}} opened at line {stack[-1][1]}")
        else:
            stack.pop()
for env, i in stack:
    fail(f"line {i}: \\begin{{{env}}} never closed")

# ------------------------------------------------------------------ braces
stripped = re.sub(r"(?<!\\)%.*", "", text)
stripped = re.sub(r"\\[{}]", "", stripped)
if stripped.count("{") != stripped.count("}"):
    fail(f"brace imbalance: {stripped.count('{')} open vs {stripped.count('}')} close")

# ------------------------------------------------------------------ inline math
# Counted document-wide, not per line: an inline expression may legally straddle
# a line break, so a per-line parity test raises false alarms.
math_body = re.sub(r"(?<!\\)%.*", "", text).replace(r"\$", "")
if math_body.count("$") % 2:
    fail(f"odd number of $ in the document ({math_body.count('$')})")

# ------------------------------------------------------------------ refs / cites
labels = set(re.findall(r"\\label\{([^}]+)\}", text))
refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", text))
for r in sorted(refs - labels):
    fail(f"\\ref{{{r}}} has no matching \\label")

bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
cites = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", text):
    cites |= {c.strip() for c in m.group(1).split(",")}
for c in sorted(cites - bibitems):
    fail(f"\\cite{{{c}}} has no matching \\bibitem")

# ------------------------------------------------------------------ tabulars
for m in re.finditer(r"\\begin\{tabular\}\{([^}]+)\}(.*?)\\end\{tabular\}", text, re.S):
    spec, body = m.group(1), m.group(2)
    ncol = len(re.findall(r"[lcr]|p\{[^}]*\}", spec))
    lineno = text[: m.start()].count("\n") + 1
    for row in body.split(r"\\"):
        clean = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", row)
        clean = re.sub(r"\\cmidrule\{[^}]*\}", "", clean).strip()
        if not clean or clean.startswith("%"):
            continue
        cells = re.split(r"(?<!\\)&", clean)
        n = 0
        for cell in cells:
            mc = re.search(r"\\multicolumn\{(\d+)\}", cell)
            n += int(mc.group(1)) if mc else 1
        if n != ncol:
            fail(f"tabular at line {lineno} (spec '{spec}' = {ncol} cols): row has {n} -> {clean[:58]}")

# ------------------------------------------------------------------ figures
for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
    if not (TEX.parent / "results" / fig).exists():
        fail(f"figure not found: results/{fig}")

# ------------------------------------------------------------------ report
print(f"environments : {'balanced' if not stack else 'UNBALANCED'}")
print(f"labels/refs  : {len(labels)} labels, {len(refs)} refs")
print(f"bibliography : {len(bibitems)} entries, {len(cites)} cited")
uncited = bibitems - cites
if uncited:
    print(f"  note: uncited entries -> {sorted(uncited)}")
unref = labels - refs
if unref:
    print(f"  note: unreferenced labels -> {sorted(unref)}")
print()
if failures:
    print(f"FAIL ({len(failures)} issues)")
    for f in failures:
        print("  -", f)
else:
    print("PASS - no structural errors found")
