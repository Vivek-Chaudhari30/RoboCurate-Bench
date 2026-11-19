"""Print the null-calibration / power table for the primary test (D-016 + D-019). CPU only.

Usage: python scripts/d016_null_grid.py [--n-small 40000] [--n-large 200000]
"""

from __future__ import annotations

import argparse

from robocurate.report.simulate import grid, markdown

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-small", type=int, default=40_000, help="grids per K<8 cell")
    ap.add_argument("--n-large", type=int, default=200_000, help="grids per K>=8 cell")
    ap.add_argument("--n-power", type=int, default=20_000)
    a = ap.parse_args()
    print(markdown(grid(a.n_small, a.n_large, a.n_power)))
