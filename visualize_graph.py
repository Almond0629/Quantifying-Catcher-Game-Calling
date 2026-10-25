"""Visualize the full MLB at-bat graph with all nodes and edges."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import matplotlib.pyplot as plt

from baseball_graph.core import AtBatGraph, PITCH_TYPES, TWO_STRIKE_COUNTS


def build_positions(graph: AtBatGraph) -> Dict[str, Tuple[float, float]]:
    """Create deterministic node positions so count clusters appear in order."""

    count_columns = {
        "0-0": 1,
        "0-1": 2,
        "1-0": 2,
        "0-2": 3,
        "1-1": 3,
        "2-0": 3,
        "1-2": 4,
        "2-1": 4,
        "3-0": 4,
        "2-2": 5,
        "3-1": 5,
        "3-2": 6,
    }
    count_group_offsets = {
        "0-0": 0.0,
        "0-1": 2.2,
        "1-0": -2.2,
        "0-2": 4.4,
        "1-1": 0.0,
        "2-0": -4.4,
        "1-2": 4.4,
        "2-1": 0.0,
        "3-0": -4.4,
        "2-2": 2.2,
        "3-1": -2.2,
        "3-2": 0.0,
    }
    pitch_offsets = {
        "FF": 1.2,
        "SL": 0.4,
        "CH": -0.4,
        "CB": -1.2,
    }

    positions: Dict[str, Tuple[float, float]] = {"source": (0.0, 0.0)}

    for count in graph.all_counts():
        x = float(count_columns[count])
        base_y = count_group_offsets[count]
        for pitch in PITCH_TYPES:
            positions[graph.pitch_node(count, pitch)] = (x, base_y + pitch_offsets[pitch])
        if count in TWO_STRIKE_COUNTS:
            positions[graph.foul_node(count)] = (x + 0.25, base_y + 2.0)

    positions["terminal"] = (7.2, 0.0)

    return positions


def draw_graph(graph: AtBatGraph, output_path: str | None, show: bool) -> None:
    positions = build_positions(graph)
    fig, ax = plt.subplots(figsize=(20, 12))

    for start, neighbors in graph.adjacency.items():
        sx, sy = positions[start]
        for end, weight in neighbors.items():
            ex, ey = positions[end]
            ax.annotate(
                "",
                xy=(ex, ey),
                xytext=(sx, sy),
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 0.6,
                    "alpha": 0.18,
                    "color": "tab:gray",
                    "shrinkA": 8,
                    "shrinkB": 8,
                },
                zorder=1,
            )
            mx = (sx + ex) / 2
            my = (sy + ey) / 2
            ax.text(mx, my, f"{weight:.2f}", fontsize=5, alpha=0.45, color="dimgray")

    for node, (x, y) in positions.items():
        if node == "source":
            color = "#f4a261"
            size = 420
        elif node == "terminal":
            color = "#e63946"
            size = 380
        elif node.endswith("_Foul"):
            color = "#457b9d"
            size = 320
        else:
            color = "#2a9d8f"
            size = 260

        ax.scatter([x], [y], s=size, c=color, edgecolors="black", linewidths=0.6, zorder=2)
        ax.text(x, y, node, fontsize=7, ha="center", va="center", color="black", zorder=3)

    ax.set_title("MLB At-Bat Directed Weighted Graph (All Nodes and Edges)", fontsize=14)
    ax.set_xlabel("Count Progression Layers")
    ax.set_ylabel("Cluster/Pitch Position")
    ax.grid(alpha=0.15, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=220)
        print(f"Saved graph visualization to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the MLB at-bat graph.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used to build edge weights.")
    parser.add_argument(
        "--output",
        type=str,
        default="at_bat_graph.png",
        help="Output image file path. Use empty string to skip saving.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = AtBatGraph(seed=args.seed)
    output_path = args.output if args.output else None
    draw_graph(graph, output_path=output_path, show=args.show)


if __name__ == "__main__":
    main()
