"""Print the null-calibration / power tables for the primary test (D-026). CPU only.

Usage: python scripts/null_grid.py [--n-grids 200000] [--n-power 20000]
Main table: D-026 balanced designs (result-eligible). Second table: old unbalanced designs, old D-016 test
(informational, not result-eligible).
"""

from __future__ import annotations

import argparse

from robocurate.report.simulate import grid, markdown

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grids", type=int, default=200_000, help="grids per cell")
    ap.add_argument("--n-power", type=int, default=20_000)
    a = ap.parse_args()
    print("Result-eligible (D-026 test):\n")
    print(markdown(grid(a.n_grids, a.n_power)))
    print("\nInformational, not result-eligible (old D-016/D-019 test, unbalanced):\n")
    print(markdown(grid(a.n_grids, a.n_power, informational=True)))
