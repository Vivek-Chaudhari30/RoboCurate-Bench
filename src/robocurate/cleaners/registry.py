"""Cleaner registry: built-ins plus third-party plugins from the entry-point group "robocurate.cleaners".

A plugin package declares, in its own pyproject.toml:

    [project.entry-points."robocurate.cleaners"]
    my_cleaner = "my_pkg.module:MyCleaner"

The entry point may name a Cleaner subclass or a zero-argument factory returning a Cleaner instance.
The oracle (D-012) is deliberately NOT registered here; it lives in robocurate.runs.oracle.
"""

from __future__ import annotations

from importlib import metadata

from robocurate.cleaners.base import Cleaner
from robocurate.cleaners.builtin import LengthHeuristic, RandomSubset

ENTRY_POINT_GROUP = "robocurate.cleaners"
BUILTINS: dict[str, type[Cleaner]] = {c.name: c for c in (RandomSubset, LengthHeuristic)}

# Launch entrants (m1-core item 5), imported lazily so heavy/optional dependencies load only when used.
_E = "robocurate.cleaners.entrants"
ENTRANTS: dict[str, str] = {
    "lerobot_doctor": f"{_E}.checkers:LerobotDoctor",
    "kinema": f"{_E}.checkers:Kinema",
    "roboaudit": f"{_E}.checkers:RoboAudit",
    "rda": f"{_E}.checkers:RDA",
    "robometer_filter": f"{_E}.robometer:RobometerFilter",
    "rabc": f"{_E}.robometer:RABC",
    "rcb_score": f"{_E}.rcb_score:RCBScore",
    "demo_score": f"{_E}.demo_score:DemoScore",
    "deminf": f"{_E}.deminf:DemInf",
    "cupid": f"{_E}.cupid:Cupid",
}


class RegistryError(LookupError):
    pass


def instantiate(obj) -> Cleaner:
    inst = obj() if callable(obj) else obj
    if not isinstance(inst, Cleaner):
        raise RegistryError(f"{obj!r} did not produce a Cleaner")
    if not inst.name or not inst.version:
        raise RegistryError(f"{type(inst).__name__} must set non-empty name and version")
    return inst


def plugin_entry_points() -> dict[str, metadata.EntryPoint]:
    return {ep.name: ep for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}


def available() -> list[str]:
    return sorted(set(BUILTINS) | set(ENTRANTS) | set(plugin_entry_points()))


def disabled_reason(name: str) -> str | None:
    """The status string of a disabled registered entrant (e.g. failed its toybed sanity check), else None."""
    try:
        st = getattr(get(name), "status", "active")
    except Exception:  # unknown names are handled by the caller's own lookup
        return None
    return None if st == "active" else st


def _load_entrant(target: str):
    import importlib

    mod, attr = target.split(":")
    return getattr(importlib.import_module(mod), attr)


def get(name: str, version: str | None = None) -> Cleaner:
    if name in BUILTINS:
        inst = instantiate(BUILTINS[name])
    elif name in ENTRANTS:
        inst = instantiate(_load_entrant(ENTRANTS[name]))
    else:
        eps = plugin_entry_points()
        if name not in eps:
            raise RegistryError(f"unknown cleaner {name!r}; available: {available()}")
        inst = instantiate(eps[name].load())
        if inst.name != name:
            raise RegistryError(f"entry point {name!r} produced cleaner named {inst.name!r}")
    if version is not None and str(inst.version) != str(version):
        raise RegistryError(f"cleaner {name!r} is version {inst.version!r}, grid asks for {version!r}")
    return inst
