"""Phase 24 -- visualise extracted regime graphs vs. gold FSMs.

For each grammar, render side-by-side:
  - LEFT: the hand-authored FSM (from tests/fixtures/graphs/<g>.fsm.yaml).
  - RIGHT: the regime graph extracted by E30 (Phase 24, frozen pretrained GPT-2).

Node colour encodes dominant-current-state purity on the extracted side
(green = high, red = low); node size encodes support; edge thickness
encodes transition probability * count. Saves one PNG per grammar to
``runs/phase24_regime_graphs/`` plus a combined grid.

Reads from ``runs/E30_phase24_<grammar>/control_graph.json``. Re-run
``scripts/phase24_gpt2_substrate_sweep.py`` first if those don't exist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from nga.arch.graph_fsm import GraphFSM  # noqa: E402
from nga.drivers import graph_fsm_spec as graph_fsm_spec_mod  # noqa: E402
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH  # noqa: E402

GRAMMARS = ["listops", "python_expr", "python_big", "json", "python_control"]
OUT_DIR = REPO_ROOT / "runs" / "phase24_regime_graphs"


def _load_gold_fsm(grammar: str) -> GraphFSM:
    dispatch = GRAMMAR_DISPATCH[grammar]
    return GraphFSM(graph_fsm_spec_mod.load(REPO_ROOT / dispatch["fsm_yaml_path"]))


def _build_gold_graph(fsm: GraphFSM) -> nx.DiGraph:
    g = nx.DiGraph()
    for v_id in fsm.vertex_ids:
        g.add_node(v_id)
    L = fsm.legality_matrix
    for i, src in enumerate(fsm.vertex_ids):
        for j, dst in enumerate(fsm.vertex_ids):
            if bool(L[i, j]):
                g.add_edge(src, dst)
    return g


def _build_regime_graph(control_graph_path: Path) -> tuple[nx.DiGraph, dict]:
    data = json.loads(control_graph_path.read_text())
    g = nx.DiGraph()
    for r in data["regimes"]:
        node_id = r["regime_id"]
        g.add_node(
            node_id,
            support=int(r["support"]),
            failure_rate=float(r["failure_rate"]) if r["failure_rate"] is not None else 0.0,
            purity=float(r["purity_against_current_state"]),
            label_state=str(r["dominant_current_state"]),
        )
    for e in data["edges"]:
        g.add_edge(
            int(e["src"]),
            int(e["dst"]),
            probability=float(e["probability"]),
            count=int(e["count"]),
        )
    return g, data


def _draw_gold(ax, fsm: GraphFSM) -> None:
    g = _build_gold_graph(fsm)
    pos = nx.kamada_kawai_layout(g) if len(g) <= 30 else nx.spring_layout(g, seed=0, k=1.2)
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color="#cfd8dc", node_size=420, edgecolors="#37474f"
    )
    nx.draw_networkx_edges(
        g, pos, ax=ax, arrows=True, edge_color="#90a4ae", width=0.8, arrowsize=8
    )
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=5)
    ax.set_title(f"gold FSM (V={fsm.vertex_count})", fontsize=10)
    ax.set_axis_off()


def _draw_regime(ax, g: nx.DiGraph, data: dict, grammar: str) -> None:
    if len(g) == 0:
        ax.set_title("(empty)", fontsize=10)
        ax.set_axis_off()
        return
    pos = nx.kamada_kawai_layout(g) if len(g) <= 30 else nx.spring_layout(g, seed=0, k=1.2)

    supports = np.array([g.nodes[n]["support"] for n in g.nodes])
    purities = np.array([g.nodes[n]["purity"] for n in g.nodes])
    sup_norm = supports / supports.max() if supports.max() > 0 else supports

    cmap = cm.get_cmap("RdYlGn")
    node_colors = [cmap(p) for p in purities]
    node_sizes = (200 + 600 * sup_norm).tolist()

    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
        edgecolors="#263238", linewidths=0.6,
    )
    edge_weights = np.array(
        [g.edges[u, v]["probability"] for u, v in g.edges]
    )
    widths = 0.4 + 2.2 * edge_weights
    nx.draw_networkx_edges(
        g, pos, ax=ax, arrows=True, edge_color="#455a64", width=widths.tolist(),
        arrowsize=8, connectionstyle="arc3,rad=0.05",
    )
    labels = {
        n: f"r{n}\n{g.nodes[n]['label_state']}\n{g.nodes[n]['purity']:.2f}"
        for n in g.nodes
    }
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=4)

    mean_purity = float(purities.mean())
    n_regimes = data["n_regimes_after_merge"]
    ax.set_title(
        f"extracted regimes (frozen GPT-2)  K={n_regimes}, mean purity {mean_purity:.3f}",
        fontsize=10,
    )
    ax.set_axis_off()


def render_one(grammar: str) -> Path | None:
    cg_path = REPO_ROOT / "runs" / f"E30_phase24_{grammar}" / "control_graph.json"
    if not cg_path.exists():
        print(f"  skip {grammar}: missing {cg_path}")
        return None
    fsm = _load_gold_fsm(grammar)
    g_extracted, data = _build_regime_graph(cg_path)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    _draw_gold(ax_l, fsm)
    _draw_regime(ax_r, g_extracted, data, grammar)
    fig.suptitle(f"{grammar}: gold FSM vs PCG-X regimes on frozen GPT-2", fontsize=11)
    fig.tight_layout()

    out_path = OUT_DIR / f"{grammar}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return out_path


def render_grid(per_grammar_paths: list[Path]) -> None:
    """Combined 5x2 grid: one row per grammar, gold on left, extracted on right."""
    fig, axes = plt.subplots(len(GRAMMARS), 2, figsize=(14, 4 * len(GRAMMARS)))
    if len(GRAMMARS) == 1:
        axes = [axes]
    for i, grammar in enumerate(GRAMMARS):
        cg_path = REPO_ROOT / "runs" / f"E30_phase24_{grammar}" / "control_graph.json"
        if not cg_path.exists():
            axes[i][0].set_title(f"{grammar}: (missing)")
            axes[i][0].set_axis_off()
            axes[i][1].set_axis_off()
            continue
        fsm = _load_gold_fsm(grammar)
        g_extracted, data = _build_regime_graph(cg_path)
        _draw_gold(axes[i][0], fsm)
        axes[i][0].set_title(f"{grammar}: gold FSM (V={fsm.vertex_count})", fontsize=10)
        _draw_regime(axes[i][1], g_extracted, data, grammar)
    fig.suptitle("Phase 24 — PCG-X regimes on frozen GPT-2 vs. gold FSMs", fontsize=12)
    fig.tight_layout()
    out_path = OUT_DIR / "all_grammars_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {OUT_DIR}")
    per_grammar_paths: list[Path] = []
    for grammar in GRAMMARS:
        path = render_one(grammar)
        if path is not None:
            per_grammar_paths.append(path)
    render_grid(per_grammar_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
