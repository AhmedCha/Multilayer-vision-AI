# Builds the taxonomy tree exclusively from saved model checkpoints under models/.
# Converts the tree and highlighted prediction paths into a Graphviz DOT string
# with color-blind-safe styling, shared-path stripes, and a status legend.

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from hiervision import model, pipeline

# Color-blind-safe 6-color palette (red, blue, green, purple, orange, brown)
PALETTE: List[str] = [
    "#D55E00",  # red-vermilion
    "#0072B2",  # blue
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#8C564B",  # brown
]

DARK_FILLS: Set[str] = {"#D55E00", "#0072B2", "#009E73", "#8C564B"}


@dataclass
class TreeNode:
    path: str
    name: str
    children: List["TreeNode"]


def _build_node(node_path: str, name: str) -> TreeNode:
    # Recursively inspect saved model checkpoints to discover child nodes and leaves.
    loaded = model.load_head(node_path)
    if loaded is None:
        return TreeNode(path=node_path, name=name, children=[])

    _, class_names = loaded
    children: List[TreeNode] = []
    for cls in class_names:
        child_path = cls if node_path == "" else f"{node_path}/{cls}"
        if model.model_path(child_path).exists():
            children.append(_build_node(child_path, cls))
        else:
            children.append(TreeNode(path=child_path, name=cls, children=[]))

    return TreeNode(path=node_path, name=name, children=children)


def load_taxonomy() -> Optional[TreeNode]:
    # Starts at the root model and builds the full tree; returns None if untrained.
    if model.load_head("") is None:
        return None
    return _build_node("", "root")


def step_paths(result: pipeline.Result) -> List[Tuple[str, float, bool]]:
    # Maps each prediction step to (class_path, confidence, accepted).
    items: List[Tuple[str, float, bool]] = []
    for step in result.steps:
        class_path = step.label if step.node == "" else f"{step.node}/{step.label}"
        items.append((class_path, step.confidence, step.accepted))
    return items


def _escape(text: str) -> str:
    # Escapes quotes and backslashes for valid DOT labels.
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _collect_highlights(
    entries: List[Tuple[str, str, pipeline.Result]],
) -> Tuple[Dict[str, List[Tuple[str, str, float, bool]]], Dict[Tuple[str, str], List[Tuple[str, str, bool]]], Set[str]]:
    # Aggregates per-node and per-edge highlights across all selected entries.
    node_hits: Dict[str, List[Tuple[str, str, float, bool]]] = {}
    edge_hits: Dict[Tuple[str, str], List[Tuple[str, str, bool]]] = {}
    used_nodes: Set[str] = {""}

    for entry_name, color, res in entries:
        steps = step_paths(res)
        if any(acc for _, _, acc in steps):
            # Mark root as active when at least one step succeeded.
            node_hits.setdefault("", []).append((entry_name, color, 1.0, True))

        parent = ""
        for class_path, conf, accepted in steps:
            used_nodes.add(class_path)
            node_hits.setdefault(class_path, []).append((entry_name, color, conf, accepted))
            edge_hits.setdefault((parent, class_path), []).append((entry_name, color, accepted))
            parent = class_path

    return node_hits, edge_hits, used_nodes


def _prune_tree(node: TreeNode, used_nodes: Set[str]) -> TreeNode:
    # Keeps only branches touched by at least one entry while preserving the root.
    kept = [_prune_tree(c, used_nodes) for c in node.children if c.path in used_nodes]
    return TreeNode(path=node.path, name=node.name, children=kept)


def _format_node(
    node: TreeNode,
    node_id: str,
    hits: List[Tuple[str, str, float, bool]],
) -> str:
    # Formats a single DOT node line based on accepted/rejected/unvisited status.
    # Note: striped fillcolor ("c1:c2") is used for multi-entry nodes; Graphviz falls back cleanly if unsupported.
    if not hits:
        label = _escape(node.name)
        return f'  {node_id} [label="{label}", style="rounded,filled", fillcolor="#F5F5F5", color="#CCCCCC", fontcolor="#777777"];'

    accepted_hits = [h for h in hits if h[3]]
    if accepted_hits:
        colors = [h[1] for h in accepted_hits]
        unique_colors = list(dict.fromkeys(colors))
        if node.path == "":
            label = _escape(node.name)
        elif len(accepted_hits) == 1:
            label = f"{_escape(node.name)}\\n{accepted_hits[0][2]:.0%}"
        else:
            confs = " / ".join(f"{h[2]:.0%}" for h in accepted_hits)
            label = f"{_escape(node.name)}\\n{confs}"

        if len(unique_colors) > 1:
            fillcolor = ":".join(unique_colors)
            style = "filled,striped"
        else:
            fillcolor = unique_colors[0]
            style = "rounded,filled"

        fontcolor = "#FFFFFF" if all(c in DARK_FILLS for c in unique_colors) else "#111111"
        return (
            f'  {node_id} [label="{label}", style="{style}", fillcolor="{fillcolor}", '
            f'color="{unique_colors[0]}", penwidth=2.5, fontcolor="{fontcolor}"];'
        )

    # Rejected guess node: dashed outline in the entry's color.
    entry_name, color, conf, _ = hits[0]
    guess_text = f"guess {conf:.0%}" if len(hits) == 1 else " / ".join(f"guess {h[2]:.0%}" for h in hits)
    label = f"{_escape(node.name)}\\n{guess_text}"
    return (
        f'  {node_id} [label="{label}", style="rounded,dashed,filled", fillcolor="#FFFFFF", '
        f'color="{color}", penwidth=2.0, fontcolor="{color}"];'
    )


def _format_edge(
    src_id: str,
    dst_id: str,
    hits: List[Tuple[str, str, bool]],
) -> str:
    # Formats a DOT edge line; shared edges use colon-separated colors, rejected edges use dashed lines.
    if not hits:
        return f'  {src_id} -> {dst_id} [color="#CCCCCC", fontcolor="#777777"];'

    accepted_hits = [h for h in hits if h[2]]
    if accepted_hits:
        unique_colors = list(dict.fromkeys(h[1] for h in accepted_hits))
        color_attr = ":".join(unique_colors)
        return f'  {src_id} -> {dst_id} [color="{color_attr}", penwidth=2.5];'

    color = hits[0][1]
    return f'  {src_id} -> {dst_id} [color="{color}", style="dashed", penwidth=2.0];'


def to_dot(
    root: TreeNode,
    entries: List[Tuple[str, str, pipeline.Result]],
    direction: str = "LR",
    only_used: bool = False,
) -> str:
    # Converts the taxonomy tree and prediction entries into a complete Graphviz DOT string.
    node_hits, edge_hits, used_nodes = _collect_highlights(entries)
    tree = _prune_tree(root, used_nodes) if only_used else root

    lines: List[str] = [
        "digraph Taxonomy {",
        f'  rankdir="{direction}";',
        '  node [shape=box, fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]

    id_map: Dict[str, str] = {}

    def walk(node: TreeNode) -> None:
        nid = f"n{len(id_map)}"
        id_map[node.path] = nid
        lines.append(_format_node(node, nid, node_hits.get(node.path, [])))
        for child in node.children:
            walk(child)
            cid = id_map[child.path]
            lines.append(_format_edge(nid, cid, edge_hits.get((node.path, child.path), [])))

    walk(tree)

    # Legend cluster showing each selected entry's color, name, final label, and status.
    if entries:
        lines.append("  subgraph cluster_legend {")
        lines.append('    label="Predictions";')
        lines.append('    fontname="Helvetica";')
        lines.append('    fontsize=11;')
        lines.append('    color="#CCCCCC";')
        lines.append('    style="rounded";')
        for idx, (entry_name, color, res) in enumerate(entries):
            fontcolor = "#FFFFFF" if color in DARK_FILLS else "#111111"
            text = _escape(f"{entry_name}: {res.final_label} ({res.status})")
            lines.append(
                f'    leg_{idx} [label="{text}", style="rounded,filled", fillcolor="{color}", '
                f'color="{color}", fontcolor="{fontcolor}", fontsize=10];'
            )
        lines.append("  }")

    lines.append("}")
    return "\n".join(lines)