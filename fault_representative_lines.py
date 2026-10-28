"""
QGIS Python console script: collapse series of similar faults into long
"representative" faults.

Run with the polyline layer active (selected features only, or all if none are
selected). The source layer is never modified; the result is written to a new
timestamped GeoPackage next to the source file and added to the project.

Strategy
  1. LINK  - two faults are linked if they are within LINK_DIST of each other
             (closest-node distance, KD-tree), their orientations differ by at
             most MAX_ANGLE_DEG and (optionally) they share the same ORIENT_FIELD.
  2. GROUP - connected components of that link graph are the "series".
  3. FIT   - for each series with >= 2 faults, densify the faults to evenly spaced
             points, take the principal axis of all points (the series'
             orientation), cut the series into BIN_WIDTH slices along that axis
             and take the mean sideways offset in each slice, smooth the offsets,
             and stretch the line over the full along-axis extent of the series.
             Gaps between faults of one series are bridged by interpolation.
  4. WRITE - one line per series, with n_src, source length, azimuth, cluster,
             best (smallest) sort_order and the source feature ids.
             Single faults are copied through unchanged if KEEP_SINGLES.
"""
import datetime
import os
import re
import time
import numpy as np
from qgis.core import (QgsFeature, QgsField, QgsFeatureRequest, QgsGeometry, QgsPointXY,
                       QgsProject, QgsVectorFileWriter, QgsVectorLayer)
from qgis.PyQt.QtCore import QVariant
from qgis.utils import iface

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

# ---------------------------------------------------------------- settings
LINK_DIST = 3000.0        # max closest-node distance for two faults to be in one series (m)
MAX_ANGLE_DEG = 25.0      # max orientation difference between linked faults
USE_CLUSTER = True        # only link faults with the same ORIENT_FIELD value
ORIENT_FIELD = "cluster"
RANK_FIELD = "sort_order"
SURVIVE_FIELD = "survive"
ONLY_SURVIVORS = False    # True: build series from survive == 1 faults only
STEP = 250.0              # point spacing when densifying faults (m)
BIN_WIDTH = 1000.0        # along-axis slice width for the representative line (m)
SMOOTH_BINS = 5           # moving-average window (slices) for the sideways offset
SIMPLIFY_TOL = 100.0      # Douglas-Peucker tolerance on the output lines (m); 0 = off
KEEP_SINGLES = True       # copy faults with no linked neighbour through unchanged
# ------------------------------------------------------------------------


# --- pure geometry helpers (numpy only) ---------------------------------
def densify(v, step):
    """Resample an (n, 2) polyline at ~even spacing, keeping both end points."""
    seg = np.hypot(*np.diff(v, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] == 0:
        return v[:1]
    s = np.unique(np.append(np.arange(0.0, cum[-1], step), cum[-1]))
    return np.column_stack([np.interp(s, cum, v[:, 0]), np.interp(s, cum, v[:, 1])])


def principal_axis(P):
    """(centre, along-axis unit vector, sideways unit vector) of a point cloud."""
    c = P.mean(axis=0)
    a = np.linalg.svd(P - c, full_matrices=False)[2][0]
    return c, a, np.array([-a[1], a[0]])


def axial_angle(a, b):
    """Angle in degrees (0-90) between two undirected unit vectors."""
    return np.degrees(np.arccos(min(1.0, abs(float(a @ b)))))


def centreline(P, bin_w, smooth):
    """Representative polyline through point cloud P (n, 2), spanning its full extent."""
    c, a, nrm = principal_axis(P)
    t, s = (P - c) @ a, (P - c) @ nrm
    t0, t1 = t.min(), t.max()
    nb = max(int(np.ceil((t1 - t0) / bin_w)), 1)
    idx = np.minimum(((t - t0) / bin_w).astype(int), nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    ok = cnt > 0
    tc = (np.bincount(idx, weights=t, minlength=nb)[ok]) / cnt[ok]
    sc = (np.bincount(idx, weights=s, minlength=nb)[ok]) / cnt[ok]
    if smooth > 1 and len(sc) > 1:
        k = min(smooth, len(sc)) | 1                       # odd window
        pad = np.pad(sc, k // 2, mode="edge")
        sc = np.convolve(pad, np.ones(k) / k, mode="valid")
    if len(tc) == 1:
        tc, sc = np.array([t0, t1]), np.array([sc[0], sc[0]])
    else:
        tc[0], tc[-1] = t0, t1                             # stretch to the full extent
    return c + np.outer(tc, a) + np.outer(sc, nrm)
# --- end helpers --------------------------------------------------------


def parts(geom):
    return [np.array([(p.x(), p.y()) for p in line], dtype=float)
            for line in (geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()])
            if len(line) > 1]


if iface is not None:       # i.e. running inside QGIS
    t0 = time.time()
    src = iface.activeLayer()
    if src is None or src.geometryType() != 1:
        raise Exception("Select a polyline layer in the Layers panel first.")
    names = src.fields().names()
    use_cluster = USE_CLUSTER and ORIENT_FIELD in names
    use_surv = ONLY_SURVIVORS and SURVIVE_FIELD in names

    # ------------------------------------------------ collect faults
    fids, pts, geoms, cl, rk = [], [], [], [], []
    for f in (src.selectedFeatures() or src.getFeatures()):
        g = f.geometry()
        if g is None or g.isEmpty() or (use_surv and f[SURVIVE_FIELD] == 0):
            continue
        ps = parts(g)
        if not ps:
            continue
        fids.append(f.id())
        pts.append(np.vstack([densify(p, STEP) for p in ps]))
        geoms.append(QgsGeometry(g))
        cl.append(f[ORIENT_FIELD] if use_cluster else None)
        rk.append(f[RANK_FIELD] if RANK_FIELD in names and f[RANK_FIELD] is not None else None)
    n = len(fids)
    if n == 0:
        raise Exception("No faults to process.")
    axes = [principal_axis(p)[1] if len(p) > 1 else np.array([1.0, 0.0]) for p in pts]

    # ------------------------------------------------ 1-2. link and group
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    bb = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in pts])
    trees = [cKDTree(p) for p in pts] if cKDTree else None
    n_links = 0
    for i in range(n - 1):
        gx = np.maximum(0, np.maximum(bb[i + 1:, 0] - bb[i, 2], bb[i, 0] - bb[i + 1:, 2]))
        gy = np.maximum(0, np.maximum(bb[i + 1:, 1] - bb[i, 3], bb[i, 1] - bb[i + 1:, 3]))
        for j in np.nonzero(np.hypot(gx, gy) <= LINK_DIST)[0] + i + 1:
            if use_cluster and cl[i] != cl[j]:
                continue
            if axial_angle(axes[i], axes[j]) > MAX_ANGLE_DEG:
                continue
            if find(i) == find(j):
                continue                                   # already in one series
            if trees:
                a, t = (pts[i], trees[j]) if len(pts[i]) <= len(pts[j]) else (pts[j], trees[i])
                d = t.query(a, k=1, distance_upper_bound=LINK_DIST)[0].min()
            else:
                dd = pts[i][:, None, :] - pts[j][None, :, :]
                d = np.sqrt(np.einsum("ijk,ijk->ij", dd, dd).min())
            if d <= LINK_DIST:
                parent[find(i)] = find(j)
                n_links += 1
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    sizes = sorted((len(m) for m in groups.values()), reverse=True)
    print(f"[{time.time() - t0:5.1f} s] {n} faults -> {len(groups)} series "
          f"(largest {sizes[:5]}, singles {sum(1 for s in sizes if s == 1)})")

    # ------------------------------------------------ 3. representative lines
    out = QgsVectorLayer(f"LineString?crs={src.crs().authid()}", "representative_faults", "memory")
    pr = out.dataProvider()
    pr.addAttributes([QgsField("series_id", QVariant.Int), QgsField("n_src", QVariant.Int),
                      QgsField("src_len", QVariant.Double), QgsField("azimuth", QVariant.Double),
                      QgsField("cluster", QVariant.String), QgsField("best_rank", QVariant.Int),
                      QgsField("src_fids", QVariant.String, len=254)])
    out.updateFields()
    feats = []
    for sid, members in enumerate(groups.values(), start=1):
        if len(members) == 1 and not KEEP_SINGLES:
            continue
        P = np.vstack([pts[i] for i in members])
        if len(members) == 1:
            geom = geoms[members[0]]
            a = axes[members[0]]
        else:
            line = centreline(P, BIN_WIDTH, SMOOTH_BINS)
            geom = QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in line])
            if SIMPLIFY_TOL > 0:
                geom = geom.simplify(SIMPLIFY_TOL)
            a = principal_axis(P)[1]
        ranks = [rk[i] for i in members if rk[i] is not None]
        f = QgsFeature(out.fields())
        f.setGeometry(geom)
        f.setAttributes([sid, len(members), float(sum(g.length() for g in (geoms[i] for i in members))),
                         float(np.degrees(np.arctan2(a[0], a[1])) % 180),
                         str(cl[members[0]]) if use_cluster else None,
                         int(min(ranks)) if ranks else None,
                         ",".join(str(fids[i]) for i in members)[:254]])
        feats.append(f)
    pr.addFeatures(feats)

    # ------------------------------------------------ 4. write to a new file
    src_path = src.source().split("|")[0]
    out_dir = os.path.dirname(src_path) if os.path.isfile(src_path) else os.path.expanduser("~")
    safe = re.sub(r"\W+", "_", src.name()).strip("_") or "faults"
    name = f"{safe}_representative_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    path = os.path.join(out_dir, name + ".gpkg")
    opts = QgsVectorFileWriter.SaveVectorOptions()
    opts.driverName, opts.layerName = "GPKG", name
    res = QgsVectorFileWriter.writeAsVectorFormatV3(out, path, QgsProject.instance().transformContext(), opts)
    if res[0] != QgsVectorFileWriter.NoError:
        raise Exception(f"Could not write {path}: {res[1]}")
    QgsProject.instance().addMapLayer(QgsVectorLayer(f"{path}|layername={name}", name, "ogr"))
    print(f"[{time.time() - t0:5.1f} s] wrote {len(feats)} representative faults to {path}")
