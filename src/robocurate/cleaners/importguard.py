"""Static import checks: cleaners must not reach labels, eval code or the oracle (CLAUDE.md rule 7).

`forbidden_reach(source)` parses the source with `ast`, collects its imports, then follows imports of
robocurate's own modules transitively (statically, by reading their source files) and reports any path that
reaches a forbidden module. Limitation: static analysis cannot see `importlib.import_module("...")` with
computed strings or `__import__`; the submit check therefore also flags any use of importlib/__import__/
exec/eval, and checks sys.modules after running the cleaner.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

FORBIDDEN = ("robocurate.data.labels", "robocurate.eval", "robocurate.runs.oracle")
SUSPICIOUS_CALLS = ("__import__", "exec", "eval", "compile")
SUSPICIOUS_MODULES = ("importlib",)


def _is_forbidden(mod: str) -> bool:
    return any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN)


def _resolve_relative(module: str | None, level: int, current: str, is_pkg: bool) -> str:
    if level == 0:
        return module or ""
    parts = current.split(".")
    base = parts if is_pkg else parts[:-1]
    base = base[: len(base) - (level - 1)] if level > 1 else base
    return ".".join(base + ([module] if module else []))


def imports_of(source: str, current: str = "__submitted__", is_pkg: bool = False) -> set[str]:
    """Fully-qualified module names imported by `source` (including `from a import b` -> a and a.b)."""
    tree = ast.parse(source)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = _resolve_relative(node.module, node.level, current, is_pkg)
            if mod:
                out.add(mod)
                out.update(f"{mod}.{a.name}" for a in node.names if a.name != "*")
    return out


def suspicious_uses(source: str) -> list[str]:
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SUSPICIOUS_CALLS:
            found.append(f"call to {node.func.id}() at line {node.lineno}")
    for m in imports_of(source):
        if m.split(".")[0] in SUSPICIOUS_MODULES:
            found.append(f"import of {m}")
    return found


def _module_source(mod: str) -> tuple[str, bool] | None:
    """(source, is_package) for a robocurate module, without importing it."""
    try:
        spec = importlib.util.find_spec(mod)
    except (ModuleNotFoundError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin).read_text(), spec.submodule_search_locations is not None


def forbidden_reach(source: str, current: str = "__submitted__", is_pkg: bool = False) -> list[list[str]]:
    """Import paths from `source` to any forbidden module. Empty list = clean."""
    paths: list[list[str]] = []
    seen: set[str] = set()
    stack: list[tuple[str, list[str]]] = [
        (m, [m]) for m in sorted(imports_of(source, current, is_pkg))
    ]
    while stack:
        mod, path = stack.pop()
        if _is_forbidden(mod):
            paths.append(path)
            continue
        if mod in seen or not mod.startswith("robocurate"):
            continue
        seen.add(mod)
        # importing a.b.c also executes a and a.b
        parents = [".".join(mod.split(".")[:i]) for i in range(1, len(mod.split(".")))]
        for parent in parents:
            if parent not in seen:
                stack.append((parent, [*path[:-1], parent]))
        src = _module_source(mod)
        if src is None:
            continue
        for sub in sorted(imports_of(src[0], mod, src[1])):
            stack.append((sub, [*path, sub]))
    return paths


def module_forbidden_reach(mod: str) -> list[list[str]]:
    src = _module_source(mod)
    if src is None:
        raise ModuleNotFoundError(mod)
    return forbidden_reach(src[0], mod, src[1])
