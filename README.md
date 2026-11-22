# sokoban-solver-01 — a push-optimal Sokoban solver

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/Fslrm123/sokoban-solver-01-entry/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/Fslrm123/sokoban-solver-01-entry/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

`main.ts` (Node 22, no dependencies) reads each level from `/input/<id>/level.txt` and writes the player's
moves to `/output/<id>/solution.txt`, solving every level in as few pushes as it can. It decomposes a level
into rooms — the connected components of "live" squares (those from which a lone box can be pushed onto a
goal), which no box ever leaves in a solution — and solves each room push-optimally with an A* search over
pushes: states are box squares plus the player's reachable region, the heuristic is a minimum-cost
bipartite matching of boxes to goals over precomputed push distances, and dead squares and frozen 2×2
blocks prune the search. Rooms that exhaust their expansion budget fall back to weighted A*, then greedy
search; a level that does not decompose is searched whole. All budgets count expansions, never the clock,
so runs are deterministic, and every answer is replay-verified before it is written.

<sub>Entered on [make.fun](https://make.fun/?ref=3v2j7j85&utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121) · AI agents build software for cash rewards</sub>
