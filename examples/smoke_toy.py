"""CPU forward pass of the original CLGNN. Requires torch/numpy/numba/scikit-learn.

No training: randomly initialized scores do not measure TBC prediction quality.
"""
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Resolve paths relative to this script, regardless of the terminal's directory.
HERE = Path(__file__).resolve().parent

# Keep generated Python bytecode and Numba compilation caches out of the repo.
# NUMBA_CACHE_DIR must be set before graph imports neighbor_finder and Numba.
sys.dont_write_bytecode = True
os.environ['NUMBA_CACHE_DIR'] = str(Path(tempfile.gettempdir()) / 'clgnn_numba_cache')

# Upstream uses flat imports (e.g. "from neighbor_finder import ...") and is not
# an installable package. Add the repository root so this standalone example
# can import its modules. These local imports intentionally follow the setup;
# standard-library and third-party imports are grouped at the top as usual.
sys.path.insert(0, str(HERE.parent))

from graph import build_ngh, compute_node_features, load_graph
from module import CLGNN


class CaptureErrors(logging.Handler):
    """Collect upstream error logs so fallback outputs cannot hide a failure."""

    def __init__(self):
        super().__init__(logging.ERROR)
        self.errors = []

    def emit(self, record):
        self.errors.append(record.getMessage())


# Upstream catches exceptions internally: explicitly fail on logged errors.
errors = CaptureErrors()
logging.getLogger().addHandler(errors)

# Fix initialization and use one CPU thread for this tiny demonstration.
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)

# Each row is a directed event (source, destination, timestamp):
# 1 -> 2 at t=1, 2 -> 3 at t=2, and 3 -> 4 at t=3.
# build_ngh stores incoming events for each destination node.
src, dst, ts, idx = load_graph(HERE / 'toy.txt')
ngh = build_ngh(src, dst, ts, idx)
nodes = np.array([1, 2, 3, 4], dtype=np.int32)

# Neighbor lookup uses event_time < cutoff; t=4 includes the final event at t=3.
cutoffs = np.full(4, 4.0, dtype=np.float64)

# Allocate rows 0..4: row 0 is padding, rows 1..4 are real nodes.
# The first three columns contain log1p(in-degree), log1p(out-degree), and
# their sum; the remaining columns start at zero. Degrees count events.
features = compute_node_features(src, dst, feat_size=5, feat_dim=16)

# Use the original architecture with a small feature dimension and random
# weights. eval() disables training-time behavior such as dropout; it does
# not train the model or load a checkpoint.
model = CLGNN(ngh, features, num_layers=3, n_head=2, drop_out=0.0).cpu().eval()
# One neighbor per node avoids random repetition on this simple chain.
print('Neighbors and timestamps:', ngh.get_temporal_neighbor(nodes, cutoffs, 1))

# No labels are passed, so forward returns scores and embeddings without
# computing contrastive loss. no_grad() disables gradient tracking.
with torch.no_grad():
    scores, embeddings = model(nodes, cutoffs, num_neighbors=1)

# Verify execution and tensor shapes, not prediction quality. Random scores
# (including negative values) are not meaningful TBC estimates.
if errors.errors:
    raise RuntimeError('\n'.join(errors.errors))
assert scores.numel() == 4 and embeddings.shape == (4, 16)
assert torch.isfinite(scores).all() and torch.isfinite(embeddings).all()
print('Embedding shape:', tuple(embeddings.shape))
print('Random, untrained scores:', scores.flatten().tolist())
print('CPU forward passed. No training, contrastive loss or quality evaluation performed.')
