# ruff: noqa
# Diagnostic for the CUPID sanity investigation (PROGRESS.md). Usage: python scripts/diagnostics/cupid/premise.py <repo root>. CPU only.
"""Does removing the planted defects improve BC closed-loop success on the toybed? (CUPID's premise.)"""
import json
import sys

sys.path.insert(0, sys.argv[1] + "/tests")
from toybed.demos import flatten, make_dataset, success_rate

from robocurate.curation.policy import MLPPolicy, train_bc

view, labels = make_dataset()
seeds_eval = range(400000, 400300)  # toy env seeds; not protocol eval/calibration/curation seeds
def sr(ids, s):
    return success_rate(train_bc(MLPPolicy(4, 2), *flatten(view, ids), steps=400, seed=s), seeds_eval)
allid = view.episode_indices
sets = {"all": allid, "no_wrong": [i for i in allid if labels[i] != "wrong"],
        "no_pause": [i for i in allid if labels[i] != "pause"], "no_jitter": [i for i in allid if labels[i] != "jitter"],
        "clean_only": [i for i in allid if labels[i] == "clean"]}
out = {k: [round(sr(v, s), 3) for s in range(3)] for k, v in sets.items()}
print(json.dumps(out))
