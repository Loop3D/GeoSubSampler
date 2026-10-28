"""
QGIS Python console script: rank faults and flag redundant ones, in one pass.

Run with the polyline layer active. Uses the selected features, or all features
if none are selected (unselected faults keep survive = 1 and sort_order untouched).

Steps
  1. Rank faults -> "sort_order" (1 = best):
       max_off_lv (largest offset first), then edge_type (X-X, X-Z, Z-Z),
       then length (longest first).
  2. Closest-node distances between faults, kept only where <= DIST
     (vertex-to-vertex, via KD-trees; bounding-box pre-filter for speed).
  3. Greedy thinning -> "survive": visiting faults from best rank to worst, a fault
     gets survive = 0 if an already-kept fault of the same ORIENT_FIELD class lies
     within DIST.
  4. Write both fields in one batched provider write and print diagnostics.
  5. Where two faults continue end-to-end (see TOUCH_TOL) but their touching
     endpoints are not at exactly the same location, add a short straight
     segment joining the two endpoints, with attributes copied from the longer
     of the two faults (its survive is 1 only if both faults survive).

The script first copies the active layer to a new timestamped GeoPackage next to
the source file and works only on that copy, so the original is never modified.
Faults are keyed on the QGIS feature id because OBJECTID is not unique.
"""
import datetime
import os
import re
import time
import numpy as np
from qgis.core import (QgsFeature, QgsFeatureRequest, QgsField, QgsGeometry, QgsPointXY,
                       QgsProject, QgsVectorFileWriter, QgsVectorLayer, QgsWkbTypes)
from qgis.PyQt.QtCore import QVariant
from qgis.utils import iface

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

# ---------------------------------------------------------------- settings
DIST = 1000.0                 # the single distance buffer, in layer CRS units (m)
OFF_FIELD = "max_off_lv"
EDGE_FIELD = "edge_type"
ORIENT_FIELD = "cluster"      # orientation class
RANK_FIELD = "sort_order"
SURVIVE_FIELD = "survive"
# If max_off_lv is text, list its categories from LARGEST to SMALLEST offset,
# e.g. ["High", "Medium", "Low"]. Leave empty if numeric (bigger = larger offset).
OFF_ORDER = []
EDGE_ORDER = {"X-X": 0, "X-Z": 1, "Z-Z": 2}
# Also create the old "Fault node distance matrix" table layer (upper triangle).
WRITE_MATRIX_LAYER = False
# Continuation test: two faults are parts of one longer fault (and so are NOT
# thinned against each other) if an endpoint of one lies within TOUCH_TOL of an
# endpoint of the other AND the path through that shared point bends by no more
# than MAX_BEND_DEG. The bend is measured using the point TANGENT_LEN back along
# each fault from the shared endpoint.
TOUCH_TOL = 200
MAX_BEND_DEG = 30.0
TANGENT_LEN = DIST
# Add a bridging segment only if the two touching endpoints are further apart than this.
BRIDGE_MIN_GAP = 1e-6
# ------------------------------------------------------------------------

t0 = time.time()
src = iface.activeLayer()
if src is None or src.geometryType() != 1:
    raise Exception("Select a polyline layer in the Layers panel first.")
if src.isEditable():
    raise Exception("Layer is in edit mode - save/stop editing and rerun.")
for fld in (OFF_FIELD, EDGE_FIELD, ORIENT_FIELD):
    if fld not in src.fields().names():
        raise Exception(f"Field '{fld}' not found. Available fields: {src.fields().names()}")

# ------------------------------- 0. duplicate the layer on disk, work on the copy
src_path = src.source().split("|")[0]
out_dir = os.path.dirname(src_path) if os.path.isfile(src_path) else os.path.expanduser("~")
safe = re.sub(r"\W+", "_", src.name()).strip("_") or "faults"
copy_name = f"{safe}_thinned_{datetime.datetime.now():%Y%m%d_%H%M%S}"
copy_path = os.path.join(out_dir, copy_name + ".gpkg")
opts = QgsVectorFileWriter.SaveVectorOptions()
opts.driverName = "GPKG"
opts.layerName = copy_name
res = QgsVectorFileWriter.writeAsVectorFormatV3(src, copy_path, QgsProject.instance().transformContext(), opts)
if res[0] != QgsVectorFileWriter.NoError:
    raise Exception(f"Could not duplicate layer: {res[1]}")
layer = QgsVectorLayer(f"{copy_path}|layername={copy_name}", copy_name, "ogr")
if not layer.isValid():
    raise Exception(f"Copy at {copy_path} could not be loaded.")
QgsProject.instance().addMapLayer(layer)
iface.setActiveLayer(layer)

# carry the source selection across (feature ids differ between source and copy)
no_geom = QgsFeatureRequest().setFlags(QgsFeatureRequest.NoGeometry)
src_ids = sorted(f.id() for f in src.getFeatures(no_geom))
copy_ids = sorted(f.id() for f in layer.getFeatures(no_geom))
if len(src_ids) != len(copy_ids):
    raise Exception("Copy has a different feature count from the source.")
to_copy = dict(zip(src_ids, copy_ids))
sel_ids = [to_copy[i] for i in src.selectedFeatureIds()]
print(f"[{time.time() - t0:5.1f} s] working on copy: {copy_path}")


def vertices(geom):
    """All vertices of a (multi)polyline as an (n, 2) float array."""
    lines = geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
    return np.array([(p.x(), p.y()) for line in lines for p in line], dtype=float)


def off_key(v):
    """Smaller key = larger offset (sorts first)."""
    if OFF_ORDER:
        s = str(v).strip()
        if s not in OFF_ORDER:
            raise Exception(f"{OFF_FIELD} value '{s}' not in OFF_ORDER {OFF_ORDER}")
        return OFF_ORDER.index(s)
    try:
        return -float(v)
    except (TypeError, ValueError):
        raise Exception(f"{OFF_FIELD} value '{v}' is not numeric; set OFF_ORDER.")


def edge_key(v):
    return EDGE_ORDER.get(str(v).strip().upper(), len(EDGE_ORDER))  # unknown last


def end_info(v):
    """[(endpoint, point TANGENT_LEN back along the fault)] for both ends of v."""
    out = []
    for p, seq in ((v[0], v), (v[-1], v[::-1])):
        d = np.hypot(*(seq - p).T)
        k = np.nonzero(d >= TANGENT_LEN)[0]
        out.append((p, seq[k[0]] if len(k) else seq[-1]))
    return out


COS_BEND = np.cos(np.radians(MAX_BEND_DEG))


def continues(ends_a, ends_b):
    """(endpoint_a, endpoint_b) if the faults meet end-to-end and run on roughly
    straight through the join, else None."""
    for pa, ia in ends_a:
        for pb, ib in ends_b:
            if np.hypot(*(pa - pb)) > TOUCH_TOL:
                continue
            u, w = pa - ia, ib - pb          # heading into the join, heading out of it
            nu, nw = np.hypot(*u), np.hypot(*w)
            if nu > 0 and nw > 0 and (u @ w) / (nu * nw) >= COS_BEND:
                return pa, pb
    return None


# ------------------------------------------------ 1. collect faults and rank
fids, pts, sort_keys, cluster_vals, lengths = [], [], [], [], []
feats = (layer.getFeatures(QgsFeatureRequest().setFilterFids(sel_ids)) if sel_ids
         else layer.getFeatures())
for f in feats:
    g = f.geometry()
    if g is None or g.isEmpty():
        continue
    v = vertices(g)
    if len(v) == 0:
        continue
    fids.append(f.id())
    pts.append(v)
    lengths.append(g.length())
    sort_keys.append((off_key(f[OFF_FIELD]), edge_key(f[EDGE_FIELD]), -g.length()))
    cluster_vals.append(f[ORIENT_FIELD])
n = len(fids)
if n == 0:
    raise Exception("No faults to process.")

order = sorted(range(n), key=lambda i: sort_keys[i])     # best first
rank = np.empty(n, dtype=int)
rank[order] = np.arange(1, n + 1)
print(f"[{time.time() - t0:5.1f} s] ranked {n} faults")

# ------------------------------------- 2. distances, kept only where <= DIST
D = np.full((n, n), np.inf)
cont = np.zeros((n, n), dtype=bool)
joins = {}                     # (i, j) with i < j -> (touching endpoint of i, of j)
ends = [end_info(v) for v in pts]
bb = np.array([[v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()] for v in pts])
trees = [cKDTree(v) for v in pts] if cKDTree else None
for i in range(n - 1):
    # vectorised bounding-box gap to every later fault; > DIST means too far
    gx = np.maximum(0, np.maximum(bb[i + 1:, 0] - bb[i, 2], bb[i, 0] - bb[i + 1:, 2]))
    gy = np.maximum(0, np.maximum(bb[i + 1:, 1] - bb[i, 3], bb[i, 1] - bb[i + 1:, 3]))
    for j in np.nonzero(np.hypot(gx, gy) <= DIST)[0] + i + 1:
        if trees:
            # query the smaller vertex set against the larger set's tree
            a, t = (pts[i], trees[j]) if len(pts[i]) <= len(pts[j]) else (pts[j], trees[i])
            d = t.query(a, k=1, distance_upper_bound=DIST)[0].min()
        else:
            dd = pts[i][:, None, :] - pts[j][None, :, :]
            d = np.sqrt(np.einsum("ijk,ijk->ij", dd, dd).min())
        if d <= DIST:
            D[i, j] = D[j, i] = d
            if d <= TOUCH_TOL:
                m = continues(ends[i], ends[j])
                if m is not None:
                    cont[i, j] = cont[j, i] = True
                    joins[(i, j)] = m
close = D <= DIST
link = close & ~cont          # in range and NOT the same fault continuing on
print(f"[{time.time() - t0:5.1f} s] distances done "
      f"({'scipy' if cKDTree else 'numpy'}), {int(np.triu(close, 1).sum())} pairs within {DIST:.0f}, "
      f"{int(np.triu(cont, 1).sum())} of them end-to-end continuations")

# ------------------------------------------------------- 3. greedy thinning
codes = {}
cluster = np.array([codes.setdefault(c, len(codes)) for c in cluster_vals])
kept = np.zeros(n, dtype=bool)
survive = np.ones(n, dtype=int)
for a in order:                                           # best rank first
    if (kept & (cluster == cluster[a]) & link[a]).any():
        survive[a] = 0
    else:
        kept[a] = True

# ------------------------------------------------------------ diagnostics
same = cluster[:, None] == cluster[None, :]
print(f"[{time.time() - t0:5.1f} s] thinning done")
print(f"  faults processed:                     {n}")
print(f"  pairs within {DIST:.0f}:                    {int(np.triu(close, 1).sum())} (of {n * (n - 1) // 2})")
print(f"  faults with >=1 neighbour in range:   {int(close.any(axis=1).sum())}")
print(f"  neighbours in range: mean {close.sum(1).mean():.1f}, max {int(close.sum(1).max())}")
print(f"  pairs in range and same {ORIENT_FIELD}:      {int(np.triu(close & same, 1).sum())}")
print(f"  ... of which end-to-end continuations (exempt): {int(np.triu(close & same & cont, 1).sum())}")
print(f"  survive = 1: {int(survive.sum())}   survive = 0: {int((survive == 0).sum())}")

# ----------------------------------------------------------- 4. write back
# One batched provider write (the edit buffer is very slow for thousands of changes).
prov = layer.dataProvider()
new = [QgsField(nm, QVariant.Int) for nm in (RANK_FIELD, SURVIVE_FIELD)
       if nm not in layer.fields().names()]
if new:
    prov.addAttributes(new)
    layer.updateFields()
r_idx = layer.fields().indexFromName(RANK_FIELD)
s_idx = layer.fields().indexFromName(SURVIVE_FIELD)
changes = {f.id(): {s_idx: 1} for f in layer.getFeatures(no_geom)}
for k, fid in enumerate(fids):
    changes[fid][r_idx] = int(rank[k])
    changes[fid][s_idx] = int(survive[k])
prov.changeAttributeValues(changes)

# ------------------------------------------------- 5. bridge end-to-end gaps
pk = set(prov.pkAttributeIndexes())
is_multi = QgsWkbTypes.isMultiType(layer.wkbType())
bridges = []
for (i, j), (pa, pb) in joins.items():
    if np.hypot(*(pa - pb)) <= BRIDGE_MIN_GAP:
        continue                                          # endpoints already coincide
    big = i if lengths[i] >= lengths[j] else j            # copy properties from the longer fault
    donor = layer.getFeature(fids[big])
    attrs = list(donor.attributes())
    for p in pk:
        attrs[p] = None                                   # let the provider assign a new key
    attrs[r_idx] = int(rank[big])
    attrs[s_idx] = int(survive[i] and survive[j])
    geom = QgsGeometry.fromPolylineXY([QgsPointXY(*pa), QgsPointXY(*pb)])
    if is_multi:
        geom.convertToMultiType()
    nf = QgsFeature(layer.fields())
    nf.setAttributes(attrs)
    nf.setGeometry(geom)
    bridges.append(nf)
if bridges:
    prov.addFeatures(bridges)
print(f"  bridging segments added: {len(bridges)} "
      f"(of {len(joins)} end-to-end continuations; the rest already share an endpoint)")
layer.reload()
layer.setRenderer(layer.renderer().clone())   # force Layers-panel feature counts to refresh
layer.dataChanged.emit()
layer.triggerRepaint()

# ------------------------------- optional: old-style upper-triangle table
if WRITE_MATRIX_LAYER:
    out = QgsVectorLayer("NoGeometry", "Fault node distance matrix", "memory")
    pr = out.dataProvider()
    names = [str(x) for x in fids]
    pr.addAttributes([QgsField("fid", QVariant.String)] +
                     [QgsField(nm, QVariant.Double) for nm in names])
    out.updateFields()
    rows = []
    for i in range(n):
        row = QgsFeature(out.fields())
        row.setAttributes([names[i]] + [float(D[i, j]) if j > i and np.isfinite(D[i, j]) else None
                                        for j in range(n)])
        rows.append(row)
    pr.addFeatures(rows)
    QgsProject.instance().addMapLayer(out)

print(f"[{time.time() - t0:5.1f} s] done: wrote '{RANK_FIELD}' and '{SURVIVE_FIELD}' for {n} faults")
