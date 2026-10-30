# CLGNN
This repository contains implementation for paper "CLGNN: A Contrastive Learning-based GNN for  Temporal Betweenness  Prediction under Extreme Value Imbalance"

Code is written in Python and the proposed model is implemented using Pytorch.

The test datasets are available on http://snap.stanford.edu/data and https://graphdata.net.

## Toy examples on CPU

This branch adds small examples around the original CLGNN model: data loading,
forward inference and loss calculation.

### Graph and labels

Each arrow is a directed temporal event:

```text
        t=1          t=2          t=3
   (1) -------> (2) -------> (3) -------> (4)
TBC: 0           2           2           0
```

[`examples/toy.txt`](examples/toy.txt) stores `source destination timestamp`.
[`examples/toy_scores.csv`](examples/toy_scores.csv) stores `node_id,TBC`
without a header. Both files are read using the repository's original loaders.

The hand-derived labels use directed, strictly increasing temporal paths,
ordered source-target pairs, excluded endpoints, and no normalization.
TBC sums the fraction of paths through an internal vertex over reachable pairs.
Every reachable pair in this chain has a unique path: node 2 contributes to
1→3 and 1→4, while node 3 contributes to 1→4 and 2→4. These are teaching
conventions, not a claim about the published datasets' label-generation protocol.

### Setup

Use Python 3.12. If needed, create the environment with
`conda create -n temp-btw python=3.12`. From the repository root:

```bash
conda activate temp-btw
python -m pip install -r requirements.txt
```

The examples use CPU, seed 42, feature dimension 16, one sampled neighbor,
cutoff 4, and zero dropout. They leave the original model source unchanged.
Numba caches use the system temporary directory. Requirements are unpinned;
this is not a specification of the paper's original software environment.

### 1. Forward pass

```bash
python -B examples/smoke_toy.py
```

Loads the graph and calls an untrained model without labels. Prints neighbors,
four scores, and embeddings of shape `(4, 16)`. Checks finite values, shapes,
and upstream error logs. Random scores are not meaningful TBC estimates.

### 2. Labels and loss

```bash
python -B examples/toy_loss.py
```

Loads events and labels, then computes the original objective without updating weights:

```text
regression_loss = mean(abs(prediction - log1p(TBC)))
total_loss = 0.8 * regression_loss + 0.2 * contrastive_loss
```

Contrastive loss is zero on this toy: positive pairs require
`0 < label_difference <= 0.1 * median(positive labels)`. Differences here
are 0 or 2, while the threshold is 0.2. Neither example updates model weights.

### VS Code

Open the `TemporalBetweenness` folder and trust the workspace. Install Microsoft's
Python and Python Debugger extensions. Use `Python: Select Interpreter` to
select `temp-btw`. In Run and Debug, select one of these configurations and press F5:

- `CLGNN: toy CPU forward`
- `CLGNN: toy CPU loss`

Useful breakpoints: `model(...)`, `CLGNN.forward`, and `compute_total_loss`.

### Code map

| File | Responsibility |
| --- | --- |
| `graph.py` | Event loading, incoming adjacency, initial node features |
| `neighbor_finder.py` | Temporal neighbor lookup and path-count inputs |
| `module.py` | CLGNN architecture, encoders, aggregation, prediction and losses |
| `nx2graphs.py` | Dataset discovery and label loading |
| `main.py` | Full experiment training, evaluation and checkpoints |
| `examples/` | Standalone CPU walkthroughs |

The default `main.py` run is a larger experiment (30 seeds, three learning rates,
15 epochs). Full reproduction also requires training data and an agreed
evaluation protocol; the toy examples do not reproduce the paper's results.
