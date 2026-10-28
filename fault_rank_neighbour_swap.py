"""
QGIS Python console script: refine fault sort_order using nearest neighbours.

Inputs: the active polyline layer (with "sort_order" from fault_sort_order.py)
and the "Fault node distance matrix" layer from fault_node_distance_matrix.py.

Each pass walks the faults from the lowest priority (largest sort_order number)
to the highest. For each fault A, take its nearest neighbour B in the distance
matrix; if B has a higher rank (smaller number) than A, swap their ranks.
Repeats until a pass makes no swaps or MAX_PASSES is reached, then writes the
new ranks back to the polyline layer.
"""
import time
import numpy as np
from qgis.core import QgsProject
from qgis.utils import iface

ID_FIELD = "fid"   # matrix row-header field: QGIS feature id (OBJECTID is not unique)
RANK_FIELD = "sort_order"
MATRIX_NAME = "Fault node distance matrix"
MAX_PASSES = 10

layer = iface.activeLayer()
if layer is None or layer.geometryType() != 1:
    raise Exception("Select the polyline layer in the Layers panel first.")
if RANK_FIELD not in layer.fields().names():
    raise Exception(f"Field '{RANK_FIELD}' missing - run fault_sort_order.py first.")
mats = QgsProject.instance().mapLayersByName(MATRIX_NAME)
if not mats:
    raise Exception(f"Layer '{MATRIX_NAME}' not found - run fault_node_distance_matrix.py first.")
matrix = mats[-1]

# --- read matrix, make it symmetric, find each fault's nearest neighbour once
rows = list(matrix.getFeatures())
names = [str(r[ID_FIELD]) for r in rows]
n = len(names)
col_order = [matrix.fields().indexFromName(nm) for nm in names]


def to_float(v):
    """NULL cells arrive as QVariant(NULL), which float() rejects."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


D = np.empty((n, n))
for k, r in enumerate(rows):
    attrs = r.attributes()          # fetch once per row, not once per cell
    D[k] = [to_float(attrs[c]) for c in col_order]
D = np.where(np.isnan(D), D.T, D)          # mirror upper triangle to lower
np.fill_diagonal(D, np.inf)
D = np.where(np.isnan(D), np.inf, D)
nn = D.argmin(axis=1)                       # nearest neighbour index per fault

# --- current ranks for faults present in the matrix
fid_of, rank = {}, np.full(n, -1, dtype=int)
pos = {nm: i for i, nm in enumerate(names)}
for f in (layer.selectedFeatures() or layer.getFeatures()):
    key = str(f.id())
    i = pos.get(key)
    if i is not None and f[RANK_FIELD] is not None:
        rank[i] = int(f[RANK_FIELD])
        fid_of[i] = f.id()
valid = np.array(sorted(fid_of))
if len(valid) == 0:
    raise Exception("No faults matched between layer and matrix, or sort_order is empty.")

# --- swap passes
t0 = time.time()
for p in range(1, MAX_PASSES + 1):
    swaps = 0
    # lowest priority first = largest rank number first (snapshot at pass start)
    for a in valid[np.argsort(-rank[valid], kind="stable")]:
        b = nn[a]
        if b in fid_of and np.isfinite(D[a, b]) and rank[b] < rank[a]:
            rank[a], rank[b] = rank[b], rank[a]
            swaps += 1
    print(f"Pass {p}/{MAX_PASSES}: {swaps} swaps, {time.time() - t0:.1f} s elapsed")
    if swaps == 0:
        break

# --- write back
layer.startEditing()
idx = layer.fields().indexFromName(RANK_FIELD)
for i, fid in fid_of.items():
    layer.changeAttributeValue(fid, idx, int(rank[i]))
layer.commitChanges()
print(f"Done. Updated '{RANK_FIELD}' for {len(fid_of)} faults.")
