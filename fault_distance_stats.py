"""
QGIS Python console script: diagnostics on the "Fault node distance matrix".

Reports how many fault pairs / faults are within MAX_DIST of each other, and how
the survive rule (nearest neighbour only, same cluster, lower rank) whittles
that down. Run with the polyline layer active.
"""
import numpy as np
from qgis.core import QgsProject
from qgis.utils import iface

ID_FIELD = "fid"   # matrix row-header field: QGIS feature id (OBJECTID is not unique)
RANK_FIELD = "sort_order"
ORIENT_FIELD = "cluster"
MATRIX_NAME = "Fault node distance matrix"
MAX_DIST = 5000.0

layer = iface.activeLayer()
matrix = QgsProject.instance().mapLayersByName(MATRIX_NAME)[-1]


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


rows = list(matrix.getFeatures())
names = [str(r[ID_FIELD]) for r in rows]
n = len(names)
col_order = [matrix.fields().indexFromName(nm) for nm in names]
D = np.empty((n, n))
for k, r in enumerate(rows):
    attrs = r.attributes()
    D[k] = [to_float(attrs[c]) for c in col_order]
D = np.where(np.isnan(D), D.T, D)
np.fill_diagonal(D, np.inf)
D = np.where(np.isnan(D), np.inf, D)

close = D <= MAX_DIST
pairs = int(np.triu(close, 1).sum())
has_nb = close.any(axis=1)
print(f"Faults in matrix:                      {n}")
print(f"Pairs within {MAX_DIST:.0f}:                    {pairs}  (of {n * (n - 1) // 2})")
print(f"Faults with >=1 neighbour in range:    {int(has_nb.sum())}")
print(f"Neighbours in range per fault: mean {close.sum(1).mean():.1f}, max {int(close.sum(1).max())}")

# attributes aligned with the matrix
pos = {nm: i for i, nm in enumerate(names)}
rank = np.full(n, -1)
orient = np.full(n, None, dtype=object)
for f in layer.getFeatures():
    i = pos.get(str(f.id()))
    if i is not None and f[RANK_FIELD] is not None:
        rank[i], orient[i] = int(f[RANK_FIELD]), f[ORIENT_FIELD]

same = np.array([[orient[i] == orient[j] for j in range(n)] for i in range(n)])
print(f"Pairs in range AND same {ORIENT_FIELD}:      {int(np.triu(close & same, 1).sum())}")
print(f"Faults with an in-range same-{ORIENT_FIELD} neighbour (any, not just nearest): "
      f"{int((close & same).any(axis=1).sum())}")

nn = D.argmin(axis=1)
idx = np.arange(n)
nn_ok = close[idx, nn]
nn_same = nn_ok & same[idx, nn]
nn_lower = nn_same & (rank > rank[nn])
print(f"Nearest neighbour in range:            {int(nn_ok.sum())}")
print(f"  ... and same {ORIENT_FIELD}:                  {int(nn_same.sum())}")
print(f"  ... and fault is lower ranked:       {int(nn_lower.sum())}  <- survive = 0 under current rule")
