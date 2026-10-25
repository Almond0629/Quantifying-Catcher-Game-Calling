"""Demo entrypoint for the MLB at-bat graph project."""

from baseball_graph.core import (
    AtBatGraph,
    CatcherGameCallingEvaluator,
    dijkstra_shortest_paths,
    load_mock_at_bats,
)


def print_demo() -> None:
    graph = AtBatGraph(seed=7)
    evaluator = CatcherGameCallingEvaluator(graph)

    print("MLB At-Bat Graph Demo")
    print(f"Nodes: {graph.node_count()}")
    print(f"Edges: {graph.edge_count()}")

    distances, predecessors = dijkstra_shortest_paths(graph.adjacency, "source")
    print("\nShortest distances from source to terminal node:")
    for target in sorted(node for node in graph.adjacency if node == "terminal"):
        print(f"  {target}: {distances[target]:.3f} via {predecessors[target]}")

    print("\nRecommended pitch flow from source:")
    print(f"  {evaluator.recommended_pitch_flow()}")

    season_results = evaluator.evaluate_season(load_mock_at_bats())
    print("\nMock at-bat evaluation:")
    for at_bat in season_results["at_bats"]:
        print(f"  At-bat {at_bat['at_bat_number']}:")
        print(f"    Recommended: {at_bat['recommended_flow']}")
        print(f"    Actual:      {at_bat['actual_flow']}")
        print(f"    Per pitch:   {at_bat['pitch_results']}")
        print(f"    Correct %:   {at_bat['correct_pct']:.2%}")

    print("\nSeason summary:")
    print(f"  Correct:   {season_results['total_correct']}")
    print(f"  Incorrect: {season_results['total_incorrect']}")
    print(f"  Correct %: {season_results['correct_pct']:.2%}")


if __name__ == "__main__":
    print_demo()
