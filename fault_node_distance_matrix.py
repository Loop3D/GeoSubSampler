"""
QGIS Python console script: closest-node distance between every pair of
polylines in the active layer (selected features only, or all if none selected).

Output: a no-geometry memory layer "Fault node distance matrix" with a "fid"
row header column and one column per fault, both holding the QGIS feature id
(OBJECTID is not unique in this data). Only the upper
triangle is filled (row index < column index); everything else is NULL.
"""
import numpy as np
from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsField
from qgis.PyQt.QtCore import QVariant
from qgis.utils import iface

MAX_DIST = 5000.0  # 5 km in layer CRS units; pairs further apart are stored as 2 * MAX_DIST

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def vertices(geom):
    """All vertices of a (multi)polyline as an (n, 2) float array."""
    lines = geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
    return np.array([(p.x(), p.y()) for line in lines for p in line], dtype=float)


def min_dist_numpy(a, b, chunk=2000):
    """Brute-force fallback if scipy is unavailable."""
    best = np.inf
    for i in range(0, len(a), chunk):
        d = a[i:i + chunk, None, :] - b[None, :, :]
        best = min(best, np.einsum("ijk,ijk->ij", d, d).min())
    return np.sqrt(best)


layer = iface.activeLayer()
if layer is None or layer.geometryType() != 1:  # 1 = line
    raise Exception("Select a polyline layer in the Layers panel first.")

feats = layer.selectedFeatures() or list(layer.getFeatures())
ids, pts = [], []
for f in feats:
    g = f.geometry()
    if g is None or g.isEmpty():
        continue
    v = vertices(g)
    if len(v) == 0:
        continue
    ids.append(f.id())   # OBJECTID is not unique, so key on the QGIS feature id
    pts.append(v)

n = len(ids)
mat = np.full((n, n), np.nan)
FAR = 2.0 * MAX_DIST

trees = [cKDTree(v) for v in pts] if cKDTree else None
bb = np.array([[v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()] for v in pts])

for i in range(n - 1):
    # vectorised bounding-box gap to every later fault; > MAX_DIST means too far
    gx = np.maximum(0, np.maximum(bb[i + 1:, 0] - bb[i, 2], bb[i, 0] - bb[i + 1:, 2]))
    gy = np.maximum(0, np.maximum(bb[i + 1:, 1] - bb[i, 3], bb[i, 1] - bb[i + 1:, 3]))
    near = np.hypot(gx, gy) <= MAX_DIST
    mat[i, i + 1:] = FAR
    for j in (np.nonzero(near)[0] + i + 1):
        if trees:
            # query the smaller vertex set against the larger set's tree
            a, t = (pts[i], trees[j]) if len(pts[i]) <= len(pts[j]) else (pts[j], trees[i])
            d = t.query(a, k=1, distance_upper_bound=MAX_DIST)[0].min()
        else:
            d = min_dist_numpy(pts[i], pts[j])
        mat[i, j] = d if d <= MAX_DIST else FAR

# Build output table
out = QgsVectorLayer("NoGeometry", "Fault node distance matrix", "memory")
pr = out.dataProvider()
names = [str(x) for x in ids]
pr.addAttributes([QgsField("fid", QVariant.String)] +
                 [QgsField(nm, QVariant.Double) for nm in names])
out.updateFields()

rows = []
for i in range(n):
    f = QgsFeature(out.fields())
    f.setAttributes([names[i]] + [None if np.isnan(x) else float(x) for x in mat[i]])
    rows.append(f)
pr.addFeatures(rows)
QgsProject.instance().addMapLayer(out)
print(f"Done: {n} faults, {n * (n - 1) // 2} pairs ({'scipy' if cKDTree else 'numpy'}).")
