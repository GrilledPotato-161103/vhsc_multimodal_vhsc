# jit_graph_viz.py
# Visualize a TorchScript JIT graph using matplotlib (no graphviz required).
#
# Usage:
#   python jit_graph_viz.py --model tiny_vlm_traced.pt --out jit_graph.png
#   python jit_graph_viz.py --model tiny_vlm_traced.pt --show
#
# Notes:
# - Works best on traced/frozen models (inlined_graph).
# - For big graphs, use --max-nodes and --font.

import argparse
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


@dataclass
class VizNode:
    nid: int
    label: str
    kind: str
    inputs: List[int]
    outputs: List[int]


def _sanitize(s: str, max_len: int = 70) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _value_id(v) -> Optional[int]:
    # TorchScript Value has unique() (int). Sometimes returns None.
    try:
        return int(v.unique())
    except Exception:
        return None


def _node_kind(n) -> str:
    try:
        return str(n.kind())
    except Exception:
        return "unknown"


def _node_outputs(n) -> List[int]:
    outs = []
    for v in n.outputs():
        vid = _value_id(v)
        if vid is not None:
            outs.append(vid)
    return outs


def _node_inputs(n) -> List[int]:
    ins = []
    for v in n.inputs():
        vid = _value_id(v)
        if vid is not None:
            ins.append(vid)
    return ins


def _make_node_label(n) -> Tuple[str, str]:
    kind = _node_kind(n)
    # Attempt to include some useful info (dtype/shape if present) from outputs
    out_parts = []
    for v in n.outputs():
        try:
            t = v.type()
            out_parts.append(str(t))
        except Exception:
            pass
        if len(out_parts) >= 1:
            break
    extra = _sanitize(out_parts[0]) if out_parts else ""
    label = kind if not extra else f"{kind}\n{extra}"
    return _sanitize(label, 80), kind


def extract_viz_graph(
    module: torch.jit.ScriptModule,
    use_inlined: bool = True,
    max_nodes: int = 300,
) -> Tuple[List[VizNode], List[Tuple[int, int]]]:
    """
    Returns (nodes, edges) where edges are between node indices (nid -> nid),
    based on Value unique IDs.
    """
    g = module.inlined_graph if use_inlined else module.graph

    # Create pseudo nodes for graph inputs so edges have a source
    nodes: List[VizNode] = []
    value_producer: Dict[int, int] = {}  # value_id -> node_nid

    def add_node(label: str, kind: str, in_vals: List[int], out_vals: List[int]) -> int:
        nid = len(nodes)
        nodes.append(VizNode(nid=nid, label=label, kind=kind, inputs=in_vals, outputs=out_vals))
        for ov in out_vals:
            value_producer[ov] = nid
        return nid

    # Graph inputs
    for inp in g.inputs():
        vid = _value_id(inp)
        if vid is None:
            continue
        try:
            t = str(inp.type())
        except Exception:
            t = ""
        add_node(label=_sanitize(f"prim::Param\n{t}", 80), kind="prim::Param", in_vals=[], out_vals=[vid])

    # Real nodes
    for n in g.nodes():
        if len(nodes) >= max_nodes:
            break
        label, kind = _make_node_label(n)
        in_vals = _node_inputs(n)
        out_vals = _node_outputs(n)
        add_node(label=label, kind=kind, in_vals=in_vals, out_vals=out_vals)

    # Graph outputs -> pseudo sink node
    out_value_ids = []
    for out in g.outputs():
        vid = _value_id(out)
        if vid is not None:
            out_value_ids.append(vid)
    sink_nid = add_node(label="prim::Return", kind="prim::Return", in_vals=out_value_ids, out_vals=[])

    # Build edges based on value flow: producer(value) -> consumer(node)
    edges_set = set()
    for nd in nodes:
        if nd.nid == sink_nid:
            # sink consumes output values
            for vid in nd.inputs:
                src = value_producer.get(vid)
                if src is not None:
                    edges_set.add((src, nd.nid))
            continue

        for vid in nd.inputs:
            src = value_producer.get(vid)
            if src is not None:
                edges_set.add((src, nd.nid))

    edges = sorted(edges_set)
    return nodes, edges


def _topo_layers(nodes: List[VizNode], edges: List[Tuple[int, int]]) -> List[List[int]]:
    """
    Simple topological layering:
    - assign level = 0 for nodes with indegree 0
    - level(v) = 1 + max(level(u)) for all u->v
    """
    n = len(nodes)
    indeg = [0] * n
    preds: List[List[int]] = [[] for _ in range(n)]
    succs: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        succs[u].append(v)
        preds[v].append(u)
        indeg[v] += 1

    # Kahn + compute levels
    level = [0] * n
    q = [i for i in range(n) if indeg[i] == 0]
    head = 0
    order = []
    while head < len(q):
        u = q[head]
        head += 1
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # If cycles (rare in JIT graphs), fall back to node index ordering
    if len(order) != n:
        order = list(range(n))

    for v in order:
        if preds[v]:
            level[v] = 1 + max(level[u] for u in preds[v])
        else:
            level[v] = 0

    max_level = max(level) if level else 0
    layers: List[List[int]] = [[] for _ in range(max_level + 1)]
    for i, lv in enumerate(level):
        layers[lv].append(i)
    return layers


def draw_jit_graph_matplotlib(
    nodes: List[VizNode],
    edges: List[Tuple[int, int]],
    out_path: Optional[str] = None,
    show: bool = False,
    font_size: int = 7,
    box_w: float = 3.6,
    box_h: float = 1.0,
    x_gap: float = 1.2,
    y_gap: float = 0.6,
    max_label_lines: int = 4,
):
    layers = _topo_layers(nodes, edges)

    # Position nodes by layer (x) and within-layer index (y)
    pos: Dict[int, Tuple[float, float]] = {}
    max_rows = max((len(layer) for layer in layers), default=1)
    width = len(layers) * (box_w + x_gap) + x_gap
    height = max_rows * (box_h + y_gap) + y_gap

    for lx, layer in enumerate(layers):
        for i, nid in enumerate(layer):
            x = x_gap + lx * (box_w + x_gap)
            # spread vertically, top-to-bottom
            y = height - (y_gap + i * (box_h + y_gap) + box_h)
            pos[nid] = (x, y)

    fig_w = max(10, int(math.ceil(width * 1.1)))
    fig_h = max(6, int(math.ceil(height * 0.9)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, width + x_gap)
    ax.set_ylim(0, height + y_gap)
    ax.axis("off")

    # Draw edges first (so boxes are on top)
    for u, v in edges:
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        start = (x1 + box_w, y1 + box_h / 2)
        end = (x2, y2 + box_h / 2)
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.8,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    # Draw nodes
    for nd in nodes:
        if nd.nid not in pos:
            continue
        x, y = pos[nd.nid]
        rect = Rectangle((x, y), box_w, box_h, fill=False, linewidth=1.0)
        ax.add_patch(rect)

        # limit label lines to keep readable
        lines = nd.label.split("\n")
        if len(lines) > max_label_lines:
            lines = lines[: max_label_lines - 1] + ["…"]
        text = "\n".join(lines)
        ax.text(
            x + 0.05, y + box_h - 0.08,
            text,
            fontsize=font_size,
            va="top",
            ha="left",
        )

    title = "TorchScript JIT Graph (matplotlib)"
    ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top", ha="left", fontsize=11)

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to TorchScript .pt (traced/scripted).")
    ap.add_argument("--out", type=str, default="jit_graph.png", help="Output image path.")
    ap.add_argument("--show", action="store_true", help="Show interactive window.")
    ap.add_argument("--no-inlined", action="store_true", help="Use module.graph instead of module.inlined_graph.")
    ap.add_argument("--max-nodes", type=int, default=300, help="Cap nodes for readability.")
    ap.add_argument("--font", type=int, default=7, help="Font size for node labels.")
    args = ap.parse_args()

    m = torch.jit.load(args.model, map_location="cpu").eval()

    nodes, edges = extract_viz_graph(
        m,
        use_inlined=not args.no_inlined,
        max_nodes=args.max_nodes,
    )

    print(f"Nodes: {len(nodes)} | Edges: {len(edges)}")
    draw_jit_graph_matplotlib(
        nodes,
        edges,
        out_path=args.out if args.out else None,
        show=args.show,
        font_size=args.font,
    )


if __name__ == "__main__":
    main()
