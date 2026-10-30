# Toy examples

See the [repository README](../README.md#toy-examples-on-cpu) for the graph,
label convention, environment setup, commands, loss definition, and limitations.

Run from the repository root with `temp-btw` activated:

| Command | Purpose |
| --- | --- |
| `python -B examples/smoke_toy.py` | Untrained forward pass and execution checks |
| `python -B examples/toy_loss.py` | Read labels and inspect loss; no parameter updates |

`toy.txt` contains events; `toy_scores.csv` contains hand-derived TBC labels.
These examples do not train the model or evaluate generalization.
