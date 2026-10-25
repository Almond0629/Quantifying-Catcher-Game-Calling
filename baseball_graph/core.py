"""Core graph, shortest path, and catcher evaluation logic."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import random
from typing import Dict, Iterable, List, Optional, Tuple


PITCH_TYPES = ("FF", "SL", "CH", "CB")
TWO_STRIKE_COUNTS = {"0-2", "1-2", "2-2", "3-2"}
TERMINAL_NODES = {
    "terminal",
}
INFINITY = float("inf")


@dataclass(frozen=True)
class PitchEvent:
    """A single pitch in a real or mock at-bat."""

    count_before: str
    actual_pitch: str
    result: str


class AtBatGraph:
    """Directed weighted graph for MLB at-bat pitch and count states."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.random = random.Random(seed)
        self.adjacency: Dict[str, Dict[str, float]] = {}
        self.count_progression = {
            "0-0": ["0-1", "1-0"],
            "0-1": ["0-2", "1-1"],
            "1-0": ["1-1", "2-0"],
            "0-2": ["1-2"],
            "1-1": ["1-2", "2-1"],
            "2-0": ["2-1", "3-0"],
            "1-2": ["2-2"],
            "2-1": ["2-2", "3-1"],
            "3-0": ["3-1"],
            "2-2": ["3-2"],
            "3-1": ["3-2"],
            "3-2": [],
        }
        self.terminal_edges = {
            "0-2": ["terminal"],
            "1-2": ["terminal"],
            "3-0": ["terminal"],
            "2-2": ["terminal"],
            "3-1": ["terminal"],
            "3-2": ["terminal"],
        }
        self._build_graph()

    def _build_graph(self) -> None:
        self._add_node("source")
        for terminal in TERMINAL_NODES:
            self._add_node(terminal)

        for count in self.all_counts():
            for pitch_type in PITCH_TYPES:
                self._add_node(self.pitch_node(count, pitch_type))
            if count in TWO_STRIKE_COUNTS:
                self._add_node(self.foul_node(count))

        for pitch_type in PITCH_TYPES:
            self._add_edge("source", self.pitch_node("0-0", pitch_type))

        for count, next_counts in self.count_progression.items():
            for pitch_type in PITCH_TYPES:
                node = self.pitch_node(count, pitch_type)
                for next_count in next_counts:
                    for next_pitch in PITCH_TYPES:
                        self._add_edge(node, self.pitch_node(next_count, next_pitch))

                for terminal in self.terminal_edges.get(count, []):
                    self._add_edge(node, terminal)

                if count in TWO_STRIKE_COUNTS:
                    self._add_edge(node, self.foul_node(count))

            if count in TWO_STRIKE_COUNTS:
                foul = self.foul_node(count)
                for pitch_type in PITCH_TYPES:
                    self._add_edge(foul, self.pitch_node(count, pitch_type))

    def _add_node(self, node: str) -> None:
        self.adjacency.setdefault(node, {})

    def _add_edge(self, start: str, end: str) -> None:
        self.adjacency[start][end] = self.random.uniform(0.1, 1.0)

    @staticmethod
    def pitch_node(count: str, pitch_type: str) -> str:
        return f"{count}_{pitch_type}"

    @staticmethod
    def foul_node(count: str) -> str:
        return f"{count}_Foul"

    @staticmethod
    def all_counts() -> List[str]:
        return [
            "0-0",
            "0-1",
            "1-0",
            "0-2",
            "1-1",
            "2-0",
            "1-2",
            "2-1",
            "3-0",
            "2-2",
            "3-1",
            "3-2",
        ]

    def edge_weight(self, start: str, end: str) -> float:
        return self.adjacency[start][end]

    def node_count(self) -> int:
        return len(self.adjacency)

    def edge_count(self) -> int:
        return sum(len(neighbors) for neighbors in self.adjacency.values())


def dijkstra_shortest_paths(
    adjacency: Dict[str, Dict[str, float]], source: str
) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """Compute shortest paths from source using Dijkstra's algorithm."""

    distances = {node: INFINITY for node in adjacency}
    predecessors: Dict[str, Optional[str]] = {node: None for node in adjacency}
    distances[source] = 0.0

    queue: List[Tuple[float, str]] = [(0.0, source)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance > distances[node]:
            continue

        for neighbor, weight in adjacency[node].items():
            new_distance = current_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessors[neighbor] = node
                heapq.heappush(queue, (new_distance, neighbor))

    return distances, predecessors


def reconstruct_path(
    predecessors: Dict[str, Optional[str]], start: str, end: str
) -> List[str]:
    """Reconstruct a path from start to end using a predecessor map."""

    if start == end:
        return [start]

    path: List[str] = []
    node: Optional[str] = end
    while node is not None:
        path.append(node)
        if node == start:
            break
        node = predecessors[node]

    if not path or path[-1] != start:
        return []
    path.reverse()
    return path


class CatcherGameCallingEvaluator:
    """Produces recommendations and compares them to real at-bat pitch calls."""

    def __init__(self, graph: AtBatGraph) -> None:
        self.graph = graph
        self.recommended_terminal_order = ["terminal"]

    def choose_terminal_target(self, start_node: str) -> Optional[str]:
        distances, _ = dijkstra_shortest_paths(self.graph.adjacency, start_node)
        reachable = [
            (distances[target], target)
            for target in self.recommended_terminal_order
            if distances[target] < INFINITY
        ]
        if not reachable:
            return None
        return min(reachable, key=lambda item: (item[0], item[1]))[1]

    def _best_terminal_cost_from_node(self, start_node: str) -> float:
        distances, _ = dijkstra_shortest_paths(self.graph.adjacency, start_node)
        reachable_costs = [
            distances[target]
            for target in self.recommended_terminal_order
            if distances[target] < INFINITY
        ]
        if not reachable_costs:
            return INFINITY
        return min(reachable_costs)

    def recommend_next_pitch(self, start_node: str) -> Optional[str]:
        candidates: List[Tuple[float, str]] = []
        for neighbor, weight in self.graph.adjacency[start_node].items():
            if neighbor in TERMINAL_NODES or neighbor.endswith("_Foul"):
                continue
            downstream_cost = self._best_terminal_cost_from_node(neighbor)
            if downstream_cost == INFINITY:
                continue
            total_cost = weight + downstream_cost
            candidates.append((total_cost, neighbor.rsplit("_", 1)[1]))

        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], item[1]))[1]

    def recommended_pitch_flow(self, start_node: str = "source") -> List[str]:
        flow: List[str] = []
        current_node = start_node
        visited = 0

        while visited < 20:
            recommendation = self.recommend_next_pitch(current_node)
            if recommendation is None:
                break
            flow.append(recommendation)

            neighbors = self.graph.adjacency[current_node]
            next_pitch_nodes = [
                neighbor
                for neighbor in neighbors
                if neighbor.endswith(f"_{recommendation}")
            ]
            if not next_pitch_nodes:
                break

            next_node = min(
                next_pitch_nodes,
                key=lambda neighbor: self.graph.edge_weight(current_node, neighbor),
            )
            if next_node in TERMINAL_NODES:
                break
            current_node = next_node
            visited += 1

        return flow

    def evaluate_at_bat(self, at_bat: Iterable[PitchEvent]) -> Dict[str, object]:
        recommended_flow: List[str] = []
        actual_flow: List[str] = []
        pitch_results: List[Tuple[str, str, bool]] = []
        correct = 0
        incorrect = 0
        current_node = "source"

        for event in at_bat:
            recommendation = self.recommend_next_pitch(current_node)
            if recommendation is None:
                break

            recommended_flow.append(recommendation)
            actual_flow.append(event.actual_pitch)
            is_correct = event.actual_pitch == recommendation
            pitch_results.append((event.count_before, event.actual_pitch, is_correct))
            if is_correct:
                correct += 1
            else:
                incorrect += 1

            if event.result == "terminal":
                current_node = event.result
                break

            current_node = self.graph.pitch_node(event.result, event.actual_pitch)

        attempts = correct + incorrect
        at_bat_correct_pct = correct / attempts if attempts else 0.0
        return {
            "recommended_flow": recommended_flow,
            "actual_flow": actual_flow,
            "pitch_results": pitch_results,
            "correct": correct,
            "incorrect": incorrect,
            "correct_pct": at_bat_correct_pct,
        }

    def evaluate_season(self, at_bats: Iterable[Iterable[PitchEvent]]) -> Dict[str, object]:
        details = []
        total_correct = 0
        total_incorrect = 0

        for index, at_bat in enumerate(at_bats, start=1):
            result = self.evaluate_at_bat(at_bat)
            result["at_bat_number"] = index
            details.append(result)
            total_correct += int(result["correct"])
            total_incorrect += int(result["incorrect"])

        attempts = total_correct + total_incorrect
        season_pct = total_correct / attempts if attempts else 0.0
        return {
            "at_bats": details,
            "total_correct": total_correct,
            "total_incorrect": total_incorrect,
            "correct_pct": season_pct,
        }


def load_mock_at_bats() -> List[List[PitchEvent]]:
    """Small, replaceable mock dataset of pitch-by-pitch at-bats."""

    return [
        [
            PitchEvent(count_before="0-0", actual_pitch="FF", result="0-1"),
            PitchEvent(count_before="0-1", actual_pitch="SL", result="1-1"),
            PitchEvent(count_before="1-1", actual_pitch="CH", result="1-2"),
            PitchEvent(count_before="1-2", actual_pitch="SL", result="terminal"),
        ],
        [
            PitchEvent(count_before="0-0", actual_pitch="CB", result="1-0"),
            PitchEvent(count_before="1-0", actual_pitch="FF", result="2-0"),
            PitchEvent(count_before="2-0", actual_pitch="CH", result="3-0"),
            PitchEvent(count_before="3-0", actual_pitch="FF", result="terminal"),
        ],
        [
            PitchEvent(count_before="0-0", actual_pitch="SL", result="0-1"),
            PitchEvent(count_before="0-1", actual_pitch="SL", result="0-2"),
            PitchEvent(count_before="0-2", actual_pitch="CB", result="0-2"),
            PitchEvent(count_before="0-2", actual_pitch="FF", result="1-2"),
            PitchEvent(count_before="1-2", actual_pitch="CH", result="2-2"),
            PitchEvent(count_before="2-2", actual_pitch="FF", result="terminal"),
        ],
    ]
