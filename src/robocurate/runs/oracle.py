"""Label-oracle reference condition (D-012). NOT an entrant and NOT in the cleaner registry.

Selects the `budget` best episodes by hidden ground-truth labels. Shows whether curation can help on a task
at all. Besides robocurate.report, this is the only code allowed to read label sidecars.

Ranking (deterministic): quality tier (clean < degraded < bad), then number of defects, then episode_index.
"""

from __future__ import annotations

from robocurate.data.labels import TIERS, LabelsSidecar


def oracle_select(labels: LabelsSidecar, budget: int, episode_indices: list[int] | None = None) -> list[int]:
    by = labels.by_index()
    ids = sorted(by) if episode_indices is None else sorted(episode_indices)
    missing = set(ids) - set(by)
    if missing:
        raise ValueError(f"labels missing for episodes {sorted(missing)[:10]}")
    if not 0 < budget <= len(ids):
        raise ValueError(f"budget {budget} out of range for {len(ids)} episodes")
    rank = sorted(ids, key=lambda i: (TIERS.index(by[i].quality_tier), len(by[i].defects), i))
    return sorted(rank[:budget])
