"""
QGIS Python console script: classify the faults of the active polyline layer by
end-to-end orientation (non-directional, 0 = 180 degrees) and centroid position,
choosing the number of classes with the best silhouette score.

Run with the polyline layer active (selected features only, or all if none are
selected). The source layer is not modified: the result is a copy written to a new
timestamped GeoPackage next to the source file, with these fields added:
    cluster     class number (0 = largest class)
    cluster_az  azimuth (0-180 deg) of the class mean orientation

Each fault becomes a point in a 4-D feature space
    ( a * x/E , a * y/E , (1 - a) * cos 2t , (1 - a) * sin 2t )
where t = azimuth of the line joining its end points, (x, y) = its centroid, E = the
larger side of the layer's extent (so position runs 0-1) and a = DISTANCE_WEIGHT.
Doubling the angle makes 0 and 180 degrees the same direction.
    DISTANCE_WEIGHT = 0   orientation only
    DISTANCE_WEIGHT = 1   position only
"""
import datetime
import os
import re
import time
import numpy as np
from qgis.core import QgsFeature, QgsFeatureRequest, QgsField, QgsProject, QgsVectorFileWriter, QgsVectorLayer
from qgis.PyQt.QtCore import QVariant
from qgis.utils import iface
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------- settings
DISTANCE_WEIGHT = 0.8     # 0-1: 0 = orientation only, 1 = centroid position only
N_CLUSTERS = None         # None = pick the best by silhouette score between K_MIN and K_MAX
K_MIN, K_MAX = 2, 20
WEIGHT_BY_LENGTH = True   # longer faults count more in class means and the silhouette score
N_INIT = 5                # k-means restarts per number of classes
MAX_SILHOUETTE_SAMPLE = 3000   # score a random sample if there are more faults than this
# ------------------------------------------------------------------------


def kmeans(X, w, k, rng, iters=100):
    """One weighted k-means run with k-means++ start. Returns (labels, centres, inertia)."""
    cen = [X[rng.choice(len(X), p=w / w.sum())]]
    for _ in range(k - 1):
        d2 = np.min([((X - c) ** 2).sum(axis=1) for c in cen], axis=0) * w
        cen.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
    cen = np.array(cen)
    for _ in range(iters):
        lab = cKDTree(cen).query(X)[1]
        new = cen.copy()
        for j in range(k):
            m = lab == j
            if m.any():
                new[j] = (X[m] * w[m, None]).sum(axis=0) / w[m].sum()
        done = np.allclose(new, cen)
        cen = new
        if done:
            break
    d, lab = cKDTree(cen).query(X)
    return lab, cen, float((w * d ** 2).sum())


def silhouette(X, lab, k, w):
    """Weighted mean silhouette score (labels 0..k-1)."""
    onehot = np.eye(k)[lab]
    cnt = onehot.sum(axis=0)
    if (cnt > 0).sum() < 2:
        return -1.0
    sums = cdist(X, X) @ onehot
    idx = np.arange(len(X))
    a = sums[idx, lab] / np.maximum(cnt[lab] - 1, 1)
    other = sums / np.maximum(cnt, 1)
    other[idx, lab] = np.inf
    other[:, cnt == 0] = np.inf
    b = other.min(axis=1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    s[cnt[lab] <= 1] = 0.0
    return float((s * w).sum() / w.sum())


def end_vector(geom):
    """Vector from the first to the last vertex of a (multi)polyline."""
    lines = geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
    lines = [l for l in lines if len(l) > 1]
    if not lines:
        return None
    a, b = lines[0][0], lines[-1][-1]
    return b.x() - a.x(), b.y() - a.y()


src = iface.activeLayer()
if src is None or src.geometryType() != 1:
    raise Exception("Select a polyline layer in the Layers panel first.")
t0 = time.time()

# ------------------------------------------------ 1. fault features
fids, cx, cy, ang, length = [], [], [], [], []
for f in (src.selectedFeatures() or src.getFeatures()):
    g = f.geometry()
    if g is None or g.isEmpty():
        continue
    v = end_vector(g)
    if v is None or (v[0] == 0 and v[1] == 0):
        continue                                    # closed or degenerate: no end-to-end direction
    c = g.centroid().asPoint()
    fids.append(f.id())
    cx.append(c.x())
    cy.append(c.y())
    ang.append(np.arctan2(v[1], v[0]))
    length.append(g.length())
n = len(fids)
if n < 3:
    raise Exception(f"Need at least 3 faults with a usable orientation (found {n}).")
cx, cy, ang, length = map(np.array, (cx, cy, ang, length))
w = length if WEIGHT_BY_LENGTH else np.ones(n)

extent = max(np.ptp(cx), np.ptp(cy)) or 1.0
a = float(np.clip(DISTANCE_WEIGHT, 0.0, 1.0))
X = np.column_stack([a * (cx - cx.min()) / extent, a * (cy - cy.min()) / extent,
                     (1 - a) * np.cos(2 * ang), (1 - a) * np.sin(2 * ang)])

# ------------------------------------------------ 2. choose the number of classes
rng = np.random.default_rng(0)
ks = [min(N_CLUSTERS, n)] if N_CLUSTERS else list(range(K_MIN, min(K_MAX, n - 1) + 1))
sub = rng.choice(n, MAX_SILHOUETTE_SAMPLE, replace=False) if n > MAX_SILHOUETTE_SAMPLE else np.arange(n)
best, scores = None, {}
for k in ks:
    lab, cen, _ = min((kmeans(X, w, k, rng) for _ in range(N_INIT)), key=lambda r: r[2])
    sc = silhouette(X[sub], lab[sub], k, w[sub]) if len(ks) > 1 else float("nan")
    scores[k] = sc
    if best is None or sc > best[0]:
        best = (sc, k, lab, cen)
_, k, lab, cen = best
if len(ks) > 1:
    print("silhouette scores: " + ", ".join(f"{kk}:{v:.3f}" for kk, v in scores.items()))

# number classes largest first; class azimuth from the orientation part of the centre
order = np.argsort(-np.bincount(lab, weights=w, minlength=k), kind="stable")
lab = np.argsort(order)[lab]
cen = cen[order]
if a < 1:
    az = (90 - np.degrees(np.arctan2(cen[:, 3], cen[:, 2])) / 2) % 180
else:                                               # position only: use the weighted mean of members
    az = np.array([(90 - np.degrees(np.arctan2(np.average(np.sin(2 * ang[lab == j]), weights=w[lab == j]),
                                               np.average(np.cos(2 * ang[lab == j]), weights=w[lab == j]))) / 2) % 180
                   for j in range(k)])
print(f"{n} faults -> {k} classes (distance weight {a:g}"
      f"{', best silhouette' if len(ks) > 1 else ''})")
for j in range(k):
    print(f"  class {j}: {int((lab == j).sum()):5d} faults, azimuth {az[j]:5.1f} deg")

# ------------------------------------------------ 3. write a copy with the new fields
src_path = src.source().split("|")[0]
out_dir = os.path.dirname(src_path) if os.path.isfile(src_path) else os.path.expanduser("~")
safe = re.sub(r"\W+", "_", src.name()).strip("_") or "faults"
name = f"{safe}_classified_{datetime.datetime.now():%Y%m%d_%H%M%S}"
path = os.path.join(out_dir, name + ".gpkg")
opts = QgsVectorFileWriter.SaveVectorOptions()
opts.driverName, opts.layerName = "GPKG", name
res = QgsVectorFileWriter.writeAsVectorFormatV3(src, path, QgsProject.instance().transformContext(), opts)
if res[0] != QgsVectorFileWriter.NoError:
    raise Exception(f"Could not write {path}: {res[1]}")
out = QgsVectorLayer(f"{path}|layername={name}", name, "ogr")

# copy and source have different feature ids, so match them by position
no_geom = QgsFeatureRequest().setFlags(QgsFeatureRequest.NoGeometry)
src_ids = sorted(f.id() for f in src.getFeatures(no_geom))
out_ids = sorted(f.id() for f in out.getFeatures(no_geom))
if len(src_ids) != len(out_ids):
    raise Exception("Copy has a different feature count from the source.")
to_out = dict(zip(src_ids, out_ids))

prov = out.dataProvider()
prov.addAttributes([QgsField("cluster", QVariant.Int), QgsField("cluster_az", QVariant.Double)])
out.updateFields()
ci, ai = out.fields().indexFromName("cluster"), out.fields().indexFromName("cluster_az")
prov.changeAttributeValues({to_out[fid]: {ci: int(lab[i]), ai: float(az[lab[i]])}
                            for i, fid in enumerate(fids)})
out.reload()
QgsProject.instance().addMapLayer(out)
print(f"[{time.time() - t0:.1f} s] wrote {path}")
