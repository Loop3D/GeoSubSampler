"""
QGIS Python console script: flag redundant faults in the active polyline layer.

Adds an integer field "survive" (default 1). Faults are visited from best rank
(smallest sort_order) to worst; a fault gets survive = 0 if an already-kept fault
  - is within MAX_DIST (per the "Fault node distance matrix" layer),
  - is in the same orientation class (ORIENT_FIELD), and
  - has a better rank.
Uses the selected features, or all features if none are selected.
"""
import time
import numpy as np
from qgis.core import QgsFeatureRequest, QgsField, QgsProject
from qgis.PyQt.QtCore import QVariant
from qgis.utils import iface

ID_FIELD = "fid"   # matrix row-header field: QGIS feature id (OBJECTID is not unique)
RANK_FIELD = "sort_order"
ORIENT_FIELD = "cluster"           # orientation class field
OUT_FIELD = "survive"
MATRIX_NAME = "Fault node distance matrix"
MAX_DIST = 5000.0                  # 5 km, in layer CRS units (metres for a projected CRS)

layer = iface.activeLayer()
if layer is None or layer.geometryType() != 1:
    raise Exception("Select the polyline layer in the Layers panel first.")
for fld in (RANK_FIELD, ORIENT_FIELD):
    if fld not in layer.fields().names():
        raise Exception(f"Field '{fld}' not found. Available fields: {layer.fields().names()}")
mats = QgsProject.instance().mapLayersByName(MATRIX_NAME)
if not mats:
    raise Exception(f"Layer '{MATRIX_NAME}' not found - run fault_node_distance_matrix.py first.")
matrix = mats[-1]


def to_float(v):
    """NULL cells arrive as QVariant(NULL), which float() rejects."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


t0 = time.time()
# --- matrix -> symmetric array -> nearest neighbour per fault
rows = list(matrix.getFeatures())
names = [str(r[ID_FIELD]) for r in rows]
n = len(names)
col_order = [matrix.fields().indexFromName(nm) for nm in names]
D = np.empty((n, n))
for k, r in enumerate(rows):
    attrs = r.attributes()          # fetch once per row, not once per cell
    D[k] = [to_float(attrs[c]) for c in col_order]
D = np.where(np.isnan(D), D.T, D)
np.fill_diagonal(D, np.inf)
D = np.where(np.isnan(D), np.inf, D)

# --- fault attributes, indexed like the matrix
pos = {nm: i for i, nm in enumerate(names)}
fid_of, rank, orient = {}, {}, {}
for f in (layer.selectedFeatures() or layer.getFeatures()):
    i = pos.get(str(f.id()))
    if i is not None and f[RANK_FIELD] is not None:
        fid_of[i], rank[i], orient[i] = f.id(), int(f[RANK_FIELD]), f[ORIENT_FIELD]

# --- decide survivors
# Greedy pass, best rank first: a fault is removed if an already-kept fault of the
# same cluster, with a better rank, lies within MAX_DIST.
codes = {}
cluster = np.full(n, -1)
for i in fid_of:
    cluster[i] = codes.setdefault(orient[i], len(codes))
kept = np.zeros(n, dtype=bool)
survive = {}
for a in sorted(fid_of, key=lambda i: rank[i]):
    redundant = (kept & (cluster == cluster[a]) & (D[a] <= MAX_DIST)).any()
    survive[a] = 0 if redundant else 1
    kept[a] = not redundant

# --- write back
print(f"Analysis done in {time.time() - t0:.1f} s, writing...")
# One batched provider write (bypasses the per-edit signals/undo stack of the
# edit buffer, which is very slow with thousands of changes).
if layer.isEditable():
    raise Exception("Layer is in edit mode - save/stop editing and rerun.")
prov = layer.dataProvider()
if OUT_FIELD not in layer.fields().names():
    prov.addAttributes([QgsField(OUT_FIELD, QVariant.Int)])
    layer.updateFields()
idx = layer.fields().indexFromName(OUT_FIELD)
changes = {f.id(): {idx: 1} for f in layer.getFeatures(QgsFeatureRequest().setFlags(QgsFeatureRequest.NoGeometry))}
for i, s in survive.items():
    changes[fid_of[i]][idx] = s
prov.changeAttributeValues(changes)
layer.reload()
layer.triggerRepaint()
print(f"Write done in {time.time() - t0:.1f} s")
print(f"Done: {sum(1 for s in survive.values() if s == 0)} of {len(survive)} faults set to survive = 0.")
