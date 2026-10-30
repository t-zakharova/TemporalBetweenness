"""Load a toy temporal graph and TBC labels, then compute CLGNN losses on CPU.

This is a loss calculation with random weights, not a training experiment.
See README.md for the exact convention used to derive the toy labels.
"""
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
# Configure Numba before importing the repository's neighbor finder.
os.environ['NUMBA_CACHE_DIR'] = str(Path(tempfile.gettempdir()) / 'clgnn_numba_cache')
# Upstream uses flat imports rather than an installable Python package.
sys.path.insert(0, str(HERE.parent))

from graph import build_ngh, compute_node_features, load_graph
from module import CLGNN
from nx2graphs import load_scores


class CaptureErrors(logging.Handler):
    """Prevent upstream's logged exceptions from silently passing this example."""

    def __init__(self):
        super().__init__(logging.ERROR)
        self.errors = []

    def emit(self, record):
        self.errors.append(record.getMessage())


errors = CaptureErrors()
logging.getLogger().addHandler(errors)
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)

# Read events and labels separately, using the same loaders as main.py.
# The CSV has no header: each row is node_id,raw_tbc.
src, dst, ts, idx = load_graph(HERE / 'toy.txt')
node_ids, raw_tbc = load_scores(HERE / 'toy_scores.csv')
nodes = np.asarray(node_ids, dtype=np.int32)
labels = torch.tensor(raw_tbc, dtype=torch.float32)
assert len(set(node_ids)) == len(node_ids), 'Duplicate node labels'
assert set(node_ids) == set(src) | set(dst), 'Labels must cover the toy nodes'
assert torch.isfinite(labels).all() and (labels >= 0).all()

ngh = build_ngh(src, dst, ts, idx)
features = compute_node_features(src, dst, feat_size=int(nodes.max()) + 1, feat_dim=16)
cutoffs = np.full(len(nodes), ts.max() + 1.0, dtype=np.float64)
model = CLGNN(ngh, features, num_layers=3, n_head=2, drop_out=0.0, alpha=0.2).cpu().eval()

# Passing labels changes the return value: forward also computes contrastive
# loss. eval() does not skip that loss; the presence of tbc_labels controls it.
# No optimizer or backward call is needed to inspect the loss values.
with torch.no_grad():
    predictions, contrastive_loss, embeddings = model(
        nodes, cutoffs, tbc_labels=labels, num_neighbors=1
    )
    total_loss, regression_loss = model.compute_total_loss(
        predictions, labels, contrastive_loss
    )

if errors.errors:
    raise RuntimeError('\n'.join(errors.errors))
assert predictions.numel() == len(nodes)
assert embeddings.shape == (len(nodes), 16)
assert torch.isfinite(predictions).all() and torch.isfinite(embeddings).all()
assert all(torch.isfinite(loss).item() for loss in (total_loss, regression_loss, contrastive_loss))

# Check the original objective explicitly: MAE in log1p(TBC) space, mixed
# with contrastive loss. Raw predictions are compared directly to log targets.
log_targets = torch.log1p(labels)
manual_mae = (predictions.flatten() - log_targets).abs().mean()
torch.testing.assert_close(regression_loss, manual_mae)
torch.testing.assert_close(total_loss, 0.2 * contrastive_loss + 0.8 * manual_mae)

print('node | raw TBC | log1p(TBC) | untrained prediction')
for node, label, target, pred in zip(nodes, labels, log_targets, predictions.flatten()):
    print(f'{node:4d} | {label.item():7.1f} | {target.item():10.6f} | {pred.item():20.6f}')
print(f'Regression loss (log-space MAE): {regression_loss.item():.6f}')
print(f'Contrastive loss:               {contrastive_loss.item():.6f}')
print(f'Total (0.8 * MAE + 0.2 * CL):    {total_loss.item():.6f}')
print('For labels [0, 2, 2, 0], contrastive loss is zero: no positive pairs satisfy')
print('the original rule 0 < label difference <= 0.1 * positive-label median.')
print('Loss calculation passed. No parameter updates or quality evaluation performed.')
