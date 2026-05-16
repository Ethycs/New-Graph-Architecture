"""Atom census test: every doc-listed atom has an implementation and
at least one caller.

This is the executable form of "the architecture is built exactly to spec".
It walks docs/arch/ and docs/drivers/, derives the canonical atom list,
asserts each name maps to a Python module under src/nga/arch/ or
src/nga/drivers/, and asserts each module is imported by at least one
runner under src/nga/exp/ or by src/nga/cli.py (transitive use through
another arch/driver atom counts).
"""
from __future__ import annotations
import ast
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARCH_DOCS = REPO_ROOT / "docs/arch"
DRIVER_DOCS = REPO_ROOT / "docs/drivers"
ARCH_SRC = REPO_ROOT / "src/nga/arch"
DRIVER_SRC = REPO_ROOT / "src/nga/drivers"
EXP_SRC = REPO_ROOT / "src/nga/exp"
CLI = REPO_ROOT / "src/nga/cli.py"

OPTIONAL = {
    "_index", "README",
    "weighted-plumbing-graph", "dynkin-ade-classification",
    "graph-realization-functor", "singularity-extraction-functor",
    "cli-runner",
    # Driver docs cover wire formats; the implementation file may be the
    # same name but the canonical atom is the doc itself.
}

# (doc-stem, src-dir-name) -> module-stem aliases for cases where the doc
# filename and the Python module filename diverge for documented reasons.
# Keep short and explicit; each entry is a load-bearing decision the user
# should be able to read.
DOC_MODULE_ALIASES: dict[tuple[str, str], str] = {
    # The doc was originally named with the trailing -E to disambiguate from
    # the partition function in math notes; the Python module dropped the
    # suffix. Same atom.
    ("energy-function-E", "arch"): "energy_function",
    # The arch-side typed-score-record doc describes the BUILDER; the
    # wire format with the same name lives under drivers/.
    ("typed-score-record", "arch"): "typed_score_record_builder",
}

def _doc_atoms(d: Path) -> set[str]:
    if not d.exists(): return set()
    return {p.stem for p in d.glob("*.md")
            if p.stem not in OPTIONAL}

def _module_imports(p: Path) -> set[str]:
    """Return {module_path} like 'nga.arch.foo' for every import in p.

    Catches BOTH `from nga.arch.foo import X` (where node.module is
    'nga.arch.foo') AND `from nga.arch import foo` (where node.module is
    'nga.arch' and 'foo' is one of node.names). Without the second case
    the test under-counts callers and produces false negatives.
    """
    try: tree = ast.parse(p.read_text())
    except SyntaxError: return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module)
                # `from nga.arch import foo` -> nga.arch.foo
                if node.module in {"nga.arch", "nga.drivers", "nga.exp"}:
                    for a in node.names:
                        found.add(f"{node.module}.{a.name}")
    return found

def _slug(doc_stem: str, src_dir: Path) -> str:
    key = (doc_stem, src_dir.name)
    return DOC_MODULE_ALIASES.get(key, doc_stem.replace("-", "_"))

@pytest.mark.parametrize("doc_stem,src_dir", [
    *((s, ARCH_SRC) for s in _doc_atoms(ARCH_DOCS)),
    *((s, DRIVER_SRC) for s in _doc_atoms(DRIVER_DOCS)),
])
def test_atom_has_implementation(doc_stem, src_dir):
    slug = _slug(doc_stem, src_dir)
    expected = src_dir / f"{slug}.py"
    assert expected.exists(), (
        f"Doc {doc_stem}.md (in {src_dir.parent.name}/) has no implementation "
        f"at {expected.relative_to(REPO_ROOT)}"
    )

def test_every_implemented_atom_has_a_caller():
    """Walk arch/, drivers/, exp/, cli.py; for each implemented atom,
    confirm at least one .py file under exp/ or cli.py imports it
    directly OR transitively via another consumed atom.
    """
    all_modules = {}
    for d in (ARCH_SRC, DRIVER_SRC):
        for p in d.glob("*.py"):
            if p.name.startswith("_") or p.name == "__init__.py":
                continue
            all_modules[f"nga.{d.name}.{p.stem}"] = p

    # Direct imports from runners + cli
    direct_imports: set[str] = set()
    for p in list(EXP_SRC.glob("*.py")) + [CLI]:
        if p.name == "__init__.py": continue
        direct_imports |= _module_imports(p)

    # Transitive closure: an atom is "consumed" if it's directly imported,
    # OR imported by another consumed atom.
    consumed = {m for m in all_modules if m in direct_imports}
    changed = True
    while changed:
        changed = False
        for m, path in all_modules.items():
            if m in consumed: continue
            its_imports = _module_imports(path)
            if its_imports & consumed:
                consumed.add(m); changed = True

    # Torch atoms are now consumed by E12 (substrate-independence runner)
    # and torch_energy_trainer by E14 (torch-native end-to-end training);
    # the sklearn-substrate siblings remain consumed by E0..E11.
    #
    # Phase 26 atoms (labelled hypergraph + KL signature + SAE adapter) are
    # scaffolding shipped per `docs/proposals/labelled-hypergraph.md`. The
    # proposal explicitly defers wiring into PCG-X runners (E28/E30) to a
    # follow-up; until then these atoms exist + are unit-tested but have no
    # runner caller. Remove from this list when the integration proposal lands.
    KNOWN_UNCONSUMED: set[str] = {
        "nga.arch.labelled_hypergraph",
        "nga.arch.kl_regime_signature",
        "nga.arch.sae_adapter",
    }

    unused = sorted(set(all_modules) - consumed - KNOWN_UNCONSUMED)
    # A handful of modules may legitimately have no caller during the
    # build-out (this is what the test catches). Failure message lists them.
    assert not unused, f"Atoms with no caller: {unused}"
