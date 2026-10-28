"""
QGIS Python console script: build long representative faults by growing them
through the straight segments of the faults in each orientation cluster.

Run with the polyline layer active (selected features only, or all if none are
selected). The source layer is never modified; the result goes to a new
timestamped GeoPackage next to the source file and is added to the project.

Strategy (per cluster; clusters come from a joint orientation + position k-means)
  1. BREAK  - every fault is split into its straight segments, each sampled into
              points (spacing PT_SPACING) that carry the segment's direction.
  2. SEED   - take the best-ranked unused fault point; gather unused points within
              SEED_RADIUS whose direction is within MAX_ANGLE_DEG; the seed is their
              centroid and the starting heading is their mean direction.
  3. GROW   - from the seed, in both directions, step forward by STEP. At each step
              look at unused points ahead (within LOOKAHEAD, +/- WIDTH sideways,
              direction within MAX_ANGLE_DEG of the heading):
                * heading <- blend of current heading and their mean direction
                              (MOMENTUM = weight of the current heading)
                * position is nudged sideways towards their centroid (LAT_GAIN)
              If nothing is ahead, coast along the heading; stop once the gap
              exceeds GAP_MAX and trim the coasted tail (end of available faults).
  4. USE UP - points within WIDTH of the finished line, with a similar direction,
              are consumed. Repeat from the next unused seed until none remain.
  5. WRITE  - lines shorter than MIN_LINE_LEN are dropped.
"""
import datetime
import os
import re
import time
import numpy as np
from qgis.core import (QgsFeature, QgsField, QgsGeometry, QgsPointXY, QgsProject,
                       QgsVectorFileWriter, QgsVectorLayer)
from qgis.PyQt.QtCore import QVariant
from qgis.utils import iface

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

# ---------------------------------------------------------------- settings
ORIENT_FIELD = "cluster"          # existing cluster field, used if the layer has it
USE_EXISTING_CLUSTER = False      # False: always cluster by orientation AND position (below)
# Clustering of faults by orientation (optionally also position) with k-means:
N_CLUSTERS = None                 # None: choose the number automatically (best silhouette score)
K_MIN, K_MAX = 1, 3              # range tried when N_CLUSTERS is None
ORIENT_FROM_ENDPOINTS = True      # orientation = azimuth of the line joining each fault's endpoints
                                  # (as in FaultClusterOrientation.py); False = mean segment direction
LENGTH_POWER = 2.0                # growth heading weights each point by (its fault's length)**this;
                                  # 0 = every point equal, 1 = proportional to length, 2 = even more
WEIGHT_BY_LENGTH = True           # longer faults count more in each cluster's mean orientation
USE_POSITION = False              # False: cluster on orientation only (0 deg = 180 deg);
                                  # True: also on fault centroid position
ORIENT_SCALE = 50000.0            # (USE_POSITION only) two perpendicular faults count as 2 * this many metres apart;
                                  # larger = orientation matters more, smaller = position does
RANK_FIELD = "sort_order"         # best-ranked faults seed first; falls back to longest
SURVIVE_FIELD = "survive"
ONLY_SURVIVORS = False            # True: use survive == 1 faults only
PT_SPACING = 250.0                # sampling spacing along each straight segment (m)
STEP = 500.0                      # growth step (m)
LOOKAHEAD = 1000.0                # how far ahead of the tip to look for faults (m)
WIDTH = 2500.0                    # sideways half-window, and use-up distance (m)
MAX_ANGLE_DEG = 25.0              # max direction difference to the heading
MOMENTUM = 0.7                    # 0-1; higher = straighter, slower to turn
LAT_GAIN = 0.3                    # 0-1; how strongly the line is pulled to the fault centroid
GAP_MAX = 3000.0                  # longest gap with no faults the line may cross (m)
SEED_RADIUS = 2000.0              # neighbourhood used to place the seed centroid (m)
MIN_LINE_LEN = 3000.0             # drop lines shorter than this (m)
SIMPLIFY_TOL = 50.0               # Douglas-Peucker tolerance on output (m); 0 = off
USE_RBF = True                    # elliptical Gaussian weighting of the local heading / sideways pull
RBF_LONG = 0.5 * LOOKAHEAD        # kernel std dev ALONG the line heading (m)
RBF_SHORT = 0.3 * WIDTH           # kernel std dev ACROSS the line (m); keep below WIDTH
RBF_AHEAD = 0.5 * LOOKAHEAD       # kernel centre in front of the line tip (m)
MAX_STEPS = 20000                 # safety stop per arm
# ------------------------------------------------------------------------


# --- pure geometry helpers (numpy + scipy) ------------------------------
def segment_points(v, spacing):
    """Points (k, 2) and unit directions (k, 2) sampled along each straight segment of v."""
    a, d = v[:-1], np.diff(v, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    keep = L > 0
    a, d, L = a[keep], d[keep], L[keep]
    if not len(L):
        return np.empty((0, 2)), np.empty((0, 2))
    u = d / L[:, None]
    k = np.maximum(1, np.ceil(L / spacing)).astype(int)
    seg = np.repeat(np.arange(len(L)), k)
    j = np.arange(len(seg)) - np.repeat(np.cumsum(k) - k, k)
    frac = (j + 0.5) / np.repeat(k, k)
    return a[seg] + d[seg] * frac[:, None], u[seg]


def _unit(v):
    return v / np.hypot(v[0], v[1])


def _mean_dir(U, ref, w=None):
    """(Weighted) mean of undirected unit vectors U, signs aligned to ref."""
    v = U * np.sign(U @ ref)[:, None]
    return _unit((v if w is None else v * w[:, None]).sum(axis=0))


def trace_lines(P, U, seed_order, step, look, width, max_ang, momentum, lat_gain,
                gap_max, seed_r, min_len, max_steps=20000, wts=None,
                rbf_long=None, rbf_short=None, rbf_ahead=0.0):
    """Grow representative lines through points P (m, 2) with directions U (m, 2).
    wts: per-point weights for the heading average (e.g. parent fault length).
    rbf_long / rbf_short: if set, the local heading and sideways offset use an elliptical
    Gaussian kernel (std dev rbf_long along the current heading, rbf_short across it,
    centred rbf_ahead in front of the tip) instead of a plain average over the search
    window. The first step of each arm uses a circular kernel of std dev
    sqrt(rbf_long * rbf_short), since there is no heading to follow yet.
    Returns a list of (line_xy (k, 2), consumed point indices)."""
    cos_a = np.cos(np.radians(max_ang))
    tree = cKDTree(P)
    used = np.zeros(len(P), dtype=bool)
    reach = np.hypot(look, width)
    out = []

    for s in seed_order:
        if used[s]:
            continue
        # ---- seed: centroid and mean direction of the nearby unused, aligned points
        idx = np.array(tree.query_ball_point(P[s], seed_r), dtype=int)
        idx = idx[~used[idx]]
        h = U[s].copy()
        sel = np.array([s])
        for _ in range(2):
            sel = idx[np.abs(U[idx] @ h) >= cos_a]
            gs = np.ones(len(sel)) if rbf_long is None else                 np.exp(-0.5 * (((P[sel] - P[s]) ** 2).sum(axis=1) / (0.5 * seed_r) ** 2))
            h = _mean_dir(U[sel], h, gs if wts is None else gs * wts[sel])
        c = np.average(P[sel], axis=0, weights=gs)

        # ---- grow the two arms
        arms = []
        for sgn in (1.0, -1.0):
            pos, hd = c.copy(), h * sgn
            path, heads, last_ok, gap = [pos.copy()], [hd.copy()], 0, 0.0
            for _ in range(max_steps):
                nrm = np.array([-hd[1], hd[0]])
                ci = np.array(tree.query_ball_point(pos, reach), dtype=int)
                ci = ci[~used[ci]]
                rel = P[ci] - pos
                al, lat = rel @ hd, rel @ nrm
                ok = (al >= -0.5 * step) & (al <= look) & (np.abs(lat) <= width) \
                    & (np.abs(U[ci] @ hd) >= cos_a)
                if ok.any():
                    if rbf_long is None:
                        g = np.ones(int(ok.sum()))
                    else:                           # elliptical Gaussian, long axis along the heading
                        sa = sc = np.sqrt(rbf_long * rbf_short) if len(path) == 1 else None
                        if sa is None:
                            sa, sc = rbf_long, rbf_short
                        g = np.exp(-0.5 * (((al[ok] - rbf_ahead) / sa) ** 2 + (lat[ok] / sc) ** 2))
                        if g.sum() < 1e-12:
                            g = np.ones_like(g)
                    hd = _unit(momentum * hd + (1 - momentum) * _mean_dir(
                        U[ci[ok]], hd, g if wts is None else g * wts[ci[ok]]))
                    nrm = np.array([-hd[1], hd[0]])
                    pos = pos + step * hd + lat_gain * np.average(lat[ok], weights=g) * nrm
                    gap = 0.0
                    supported = True
                else:
                    pos = pos + step * hd
                    gap += step
                    supported = False
                    if gap > gap_max:
                        break
                path.append(pos.copy())
                heads.append(hd.copy())
                if supported:
                    last_ok = len(path) - 1
            arms.append((np.array(path[:last_ok + 1]), np.array(heads[:last_ok + 1])))

        (pa, ha), (pb, hb) = arms
        line = np.vstack([pb[:0:-1], pa])                  # arm B reversed, then arm A
        heads = np.vstack([-hb[:0:-1], ha])
        used[sel] = True                                    # guarantees progress
        length = float(np.hypot(*np.diff(line, axis=0).T).sum()) if len(line) > 1 else 0.0
        if length < min_len:
            continue

        # ---- use up the faults the line represents
        un = np.nonzero(~used)[0]
        eaten = sel
        if len(un):
            d, j = cKDTree(line).query(P[un], distance_upper_bound=width)
            fin = np.isfinite(d)
            j = np.where(fin, j, 0)
            good = fin & (np.abs((U[un] * heads[j]).sum(axis=1)) >= cos_a)
            eaten = np.concatenate([sel, un[good]])
            used[un[good]] = True
        out.append((line, eaten))
    return out


def _kmeans(X, w, k, rng, iters):
    """One weighted k-means run (k-means++ start). Returns (labels, centres, inertia)."""
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


def silhouette(X, lab, k, w=None):
    """Mean (optionally weighted) silhouette score of a clustering (labels 0..k-1)."""
    from scipy.spatial.distance import cdist
    n = len(X)
    onehot = np.eye(k)[lab]
    cnt = onehot.sum(axis=0)
    if (cnt > 0).sum() < 2:
        return -1.0
    sums = cdist(X, X) @ onehot                                     # (n, k) summed distances
    idx = np.arange(n)
    a = sums[idx, lab] / np.maximum(cnt[lab] - 1, 1)
    other = sums / np.maximum(cnt, 1)
    other[idx, lab] = np.inf
    other[:, cnt == 0] = np.inf
    b = other.min(axis=1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    s[cnt[lab] <= 1] = 0.0
    return float(s.mean() if w is None else (s * w).sum() / w.sum())


def cluster_faults(cx, cy, C, S, k, orient_scale, kmin=2, kmax=30, n_init=5, iters=100,
                   seed=0, max_sil=3000, use_position=True, weights=None):
    """Cluster faults by orientation (and, if use_position, centroid position) with k-means.
    cx, cy: fault centroids; C, S: cos(2t), sin(2t) of each fault's orientation t
    (doubling makes 0 and 180 degrees the same direction).
    Each fault becomes a point (x, y, s*cos2t, s*sin2t) with s = orient_scale metres, so
    two perpendicular faults are as 'far apart' as 2 * orient_scale metres of position.
    weights: per-fault weights (e.g. length) for the cluster means and silhouette score.
    k=None: try kmin..kmax and keep the number with the best silhouette score.
    Returns (labels 0..k-1 largest cluster first, centroid x, centroid y and azimuth of each
    cluster, {k: silhouette score})."""
    nrm = np.hypot(C, S)
    pos = np.column_stack([cx, cy]) if use_position else np.zeros((len(cx), 2))
    X = np.column_stack([pos, orient_scale * C / np.where(nrm > 0, nrm, 1),
                         orient_scale * S / np.where(nrm > 0, nrm, 1)])
    n = len(X)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    ks = [min(k, n)] if k else list(range(kmin, min(kmax, n - 1) + 1)) or [min(2, n)]
    sub = rng.choice(n, max_sil, replace=False) if n > max_sil else np.arange(n)
    scores, best = {}, None
    for kk in ks:
        lab, cen, _ = min((_kmeans(X, w, kk, rng, iters) for _ in range(n_init)), key=lambda r: r[2])
        sc = silhouette(X[sub], lab[sub], kk, w[sub]) if len(ks) > 1 else float("nan")
        scores[kk] = sc
        if best is None or sc > best[0]:
            best = (sc, kk, lab, cen)
    _, kk, lab, cen = best
    order = np.argsort(-np.bincount(lab, minlength=kk), kind="stable")
    rank = np.argsort(order)
    theta = np.degrees(np.arctan2(cen[:, 3], cen[:, 2])) / 2          # direction from x axis
    lab = rank[lab]
    kx = np.array([np.average(cx[lab == j], weights=w[lab == j]) if (lab == j).any() else np.nan
                   for j in range(kk)])
    ky = np.array([np.average(cy[lab == j], weights=w[lab == j]) if (lab == j).any() else np.nan
                   for j in range(kk)])
    return lab, kx, ky, (90 - theta[order]) % 180, scores
# --- end helpers --------------------------------------------------------


def parts(geom):
    return [np.array([(p.x(), p.y()) for p in line], dtype=float)
            for line in (geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()])
            if len(line) > 1]


if iface is not None:       # i.e. running inside QGIS
    if cKDTree is None:
        raise Exception("scipy is required for this script.")
    t0 = time.time()
    src = iface.activeLayer()
    if src is None or src.geometryType() != 1:
        raise Exception("Select a polyline layer in the Layers panel first.")
    names = src.fields().names()
    use_cluster = USE_EXISTING_CLUSTER and ORIENT_FIELD in names
    use_surv = ONLY_SURVIVORS and SURVIVE_FIELD in names
    has_rank = RANK_FIELD in names

    # --------------------------------------------- 1. break faults into segments
    fault_fid, fault_rank, fault_len, fault_cluster = [], [], [], []
    fault_ep = []                                  # endpoint-to-endpoint vector of each fault
    P_all, U_all, owner = [], [], []
    for f in (src.selectedFeatures() or src.getFeatures()):
        g = f.geometry()
        if g is None or g.isEmpty() or (use_surv and f[SURVIVE_FIELD] == 0):
            continue
        k = len(fault_fid)
        got = False
        ps = parts(g)
        for v in ps:
            p, u = segment_points(v, PT_SPACING)
            if len(p):
                P_all.append(p)
                U_all.append(u)
                owner.append(np.full(len(p), k))
                got = True
        if got:
            fault_fid.append(f.id())
            fault_len.append(g.length())
            fault_ep.append(ps[-1][-1] - ps[0][0])
            r = f[RANK_FIELD] if has_rank else None
            fault_rank.append(float(r) if r is not None else np.inf)
            fault_cluster.append(f[ORIENT_FIELD] if use_cluster else None)
    if not fault_fid:
        raise Exception("No faults to process.")
    P_all, U_all, owner = np.vstack(P_all), np.vstack(U_all), np.concatenate(owner)
    fault_rank, fault_len = np.array(fault_rank), np.array(fault_len)
    nf = len(fault_fid)
    if use_cluster:
        print(f"Using existing '{ORIENT_FIELD}' field for clusters.")
    else:
        # joint clustering on orientation and centroid position of each fault
        cnt = np.bincount(owner, minlength=nf)
        cx = np.bincount(owner, weights=P_all[:, 0], minlength=nf) / cnt
        cy = np.bincount(owner, weights=P_all[:, 1], minlength=nf) / cnt
        C = np.bincount(owner, weights=U_all[:, 0] ** 2 - U_all[:, 1] ** 2, minlength=nf)
        S = np.bincount(owner, weights=2 * U_all[:, 0] * U_all[:, 1], minlength=nf)
        if ORIENT_FROM_ENDPOINTS:
            ep = np.array(fault_ep)
            ok = np.hypot(ep[:, 0], ep[:, 1]) > 0                 # else keep the segment-based value
            th = np.arctan2(ep[ok, 1], ep[ok, 0])
            C, S = C.copy(), S.copy()
            C[ok], S[ok] = np.cos(2 * th), np.sin(2 * th)
        labels, kx, ky, az, scores = cluster_faults(cx, cy, C, S, N_CLUSTERS, ORIENT_SCALE, K_MIN, K_MAX,
                                                      use_position=USE_POSITION,
                                                      weights=fault_len if WEIGHT_BY_LENGTH else None)
        fault_cluster = [int(x) for x in labels]
        if len(scores) > 1:
            print("  silhouette scores: " + ", ".join(f"{k}:{v:.3f}" for k, v in scores.items()))
        print(f"Clustered {nf} faults by orientation{' + position' if USE_POSITION else ''} into {len(kx)} clusters"
              f"{' (best silhouette)' if N_CLUSTERS is None else ''}:")
        for j in range(len(kx)):
            print(f"  cluster {j}: {int((labels == j).sum()):5d} faults, azimuth {az[j]:5.1f} deg, "
                  f"centre ({kx[j]:.0f}, {ky[j]:.0f})")
    # seed priority: best rank first, then longest
    fault_prio = np.lexsort((-fault_len, fault_rank))
    prio_of = np.empty(len(fault_prio), dtype=int)
    prio_of[fault_prio] = np.arange(len(fault_prio))
    print(f"[{time.time() - t0:5.1f} s] {len(fault_fid)} faults -> {len(P_all)} segment points")

    # --------------------------------------------- 2-4. seed, grow, use up
    out = QgsVectorLayer(f"LineString?crs={src.crs().authid()}", "representative_faults", "memory")
    pr = out.dataProvider()
    pr.addAttributes([QgsField("line_id", QVariant.Int), QgsField("cluster", QVariant.String),
                      QgsField("length", QVariant.Double), QgsField("azimuth", QVariant.Double),
                      QgsField("n_src", QVariant.Int), QgsField("best_rank", QVariant.Int),
                      QgsField("src_fids", QVariant.String, len=254)])
    out.updateFields()
    feats, leftover = [], 0
    pt_cluster = np.array([str(fault_cluster[o]) for o in owner])
    # growth heading: each point is weighted by (length of its fault) ** LENGTH_POWER
    pt_w = fault_len[owner] ** LENGTH_POWER
    for cl in sorted(set(pt_cluster)):
        ci = np.nonzero(pt_cluster == cl)[0]
        P, U, ow = P_all[ci], U_all[ci], owner[ci]
        seed_order = np.argsort(prio_of[ow], kind="stable")
        lines = trace_lines(P, U, seed_order, STEP, LOOKAHEAD, WIDTH, MAX_ANGLE_DEG, MOMENTUM,
                            LAT_GAIN, GAP_MAX, SEED_RADIUS, MIN_LINE_LEN, MAX_STEPS,
                            wts=pt_w[ci],
                            rbf_long=RBF_LONG if USE_RBF else None, rbf_short=RBF_SHORT,
                            rbf_ahead=RBF_AHEAD)
        covered = np.zeros(len(P), dtype=bool)
        for line, eaten in lines:
            covered[eaten] = True
            geom = QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in line])
            if SIMPLIFY_TOL > 0:
                geom = geom.simplify(SIMPLIFY_TOL)
            srcs = np.unique(ow[eaten])
            ranks = [fault_rank[k] for k in srcs if np.isfinite(fault_rank[k])]
            d = line[-1] - line[0]
            f = QgsFeature(out.fields())
            f.setGeometry(geom)
            f.setAttributes([len(feats) + 1, cl, float(geom.length()),
                             float(np.degrees(np.arctan2(d[0], d[1])) % 180), len(srcs),
                             int(min(ranks)) if ranks else None,
                             ",".join(str(fault_fid[k]) for k in srcs)[:254]])
            feats.append(f)
        leftover += int((~covered).sum())
        print(f"  cluster {cl}: {len(ci)} points -> {len(lines)} lines")
    pr.addFeatures(feats)
    print(f"[{time.time() - t0:5.1f} s] {len(feats)} lines; {leftover} of {len(P_all)} points "
          f"({100 * leftover / len(P_all):.0f}%) not represented (too short / isolated)")

    # --------------------------------------------- 5. write to a new file
    src_path = src.source().split("|")[0]
    out_dir = os.path.dirname(src_path) if os.path.isfile(src_path) else os.path.expanduser("~")
    safe = re.sub(r"\W+", "_", src.name()).strip("_") or "faults"
    name = f"{safe}_grown_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    path = os.path.join(out_dir, name + ".gpkg")
    opts = QgsVectorFileWriter.SaveVectorOptions()
    opts.driverName, opts.layerName = "GPKG", name
    res = QgsVectorFileWriter.writeAsVectorFormatV3(out, path, QgsProject.instance().transformContext(), opts)
    if res[0] != QgsVectorFileWriter.NoError:
        raise Exception(f"Could not write {path}: {res[1]}")
    QgsProject.instance().addMapLayer(QgsVectorLayer(f"{path}|layername={name}", name, "ogr"))
    print(f"[{time.time() - t0:5.1f} s] wrote {path}")
