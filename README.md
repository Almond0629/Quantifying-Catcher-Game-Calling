# Quantifying Catcher Game Calling

This project models a full MLB at-bat as a directed weighted graph. Each count is
represented as a cluster of four pitch-type nodes (`FF`, `SL`, `CH`, `CB`), and
two-strike counts add a dedicated foul node that cycles back into the same count.

The implementation makes one explicit interpretation where the next recommended
pitch is the pitch type on the lowest-cost outgoing path toward the closest
reachable terminal state under Dijkstra's algorithm. Random positive edge weights
are used now, but the graph stores them cleanly enough to swap in real pitch,
count, or outcome data later.

Run the demo with:

```bash
python3 main.py
```
