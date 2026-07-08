"""
Paper-material export: results tables as CSV and LaTeX, written to paper/.

Every experiment calls export_table at the end so the paper directory
always holds the current numbers; figures are generated separately from
the saved .npz arrays (see experiments/make_figures.py).
"""

import csv
from pathlib import Path

from . import config as cfg

PAPER_DIR = cfg.PROJECT_ROOT / "paper"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"
for _d in (PAPER_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(exist_ok=True)

_COLUMNS = [
    ('method', 'Method', 's'),
    ('cov95', 'Cov. 95', '.3f'),
    ('cov90', 'Cov. 90', '.3f'),
    ('len95', 'Len. 95', '.2f'),
    ('w1_truth', 'W1 truth', '.3f'),
    ('w1_ref', 'W1 ref', '.3f'),
]


def _fmt(row, key, spec):
    v = row.get(key)
    if v is None:
        return ''   # blank cell; '--' would typeset as an en-dash
    if spec == 's':
        # '|' typesets as an em-dash in text mode; '~' is a nbsp
        return str(v).replace('|', ' / ')
    if v != v:  # NaN
        return ''
    return format(v, spec)


def export_table(rows: list, name: str, caption: str = ''):
    """Write paper/tables/{name}.csv and .tex from evaluate_method rows."""
    csv_path = TABLES_DIR / f"{name}.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in _COLUMNS])
        for r in rows:
            w.writerow([_fmt(r, k, spec) for k, _, spec in _COLUMNS])

    tex_path = TABLES_DIR / f"{name}.tex"
    cap = (caption or name).replace(' ~ ', ' $\\sim$ ')
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{cap}}}",
        f"\\label{{tab:{name}}}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        " & ".join(h for _, h, _ in _COLUMNS) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        cells = [_fmt(r, k, spec) for k, _, spec in _COLUMNS]
        cells[0] = cells[0].replace('_', '\\_')
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"    exported {csv_path.name}, {tex_path.name} -> paper/tables/")
