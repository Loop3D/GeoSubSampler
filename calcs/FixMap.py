# -*- coding: utf-8 -*-
"""
Fix Map: make polygons and faults topologically consistent.

Digitised geological maps usually contain small errors: gaps and overlaps between polygons,
shared boundaries whose nodes do not match, triple junctions made of two or three nodes that
almost coincide, faults that run almost (but not exactly) along a contact, doubled features and
polygons lying inside another polygon without a hole.  ``fix_map`` removes them within a
tolerance given in metres.  The whole process is described step by step in docs/FixMap.md; in
short (geometry is handled as a flat stream of coordinates, all vectorised):

1. Repair invalid polygons (buffer(0), make_valid as a fallback).
2. Remove exact duplicate polygons / faults.
3. A polygon at least 90% inside a bigger polygon gets cut out of it as a hole; then every enclosed gap is
   filled with the neighbour that shares the longest boundary (gaps first, so the snapping has less to do).
4. Conform (``_conform``).  Every path (polygon ring or fault) has a rank: polygons before faults, bigger
   before smaller, longer faults before shorter.  A vertex within the tolerance of a higher-ranked line moves
   onto its nearest node, or onto the nearest point of its segment (which becomes a node); a higher-ranked
   vertex within the tolerance of a lower-ranked segment is inserted into it.  Segments are handled once, not
   once per polygon, so shared boundaries stay shared.  No vertex ends up more than the tolerance from where
   it started.  Passes repeat until nothing changes (after the first, only around what changed, with a final
   full check).
5. Rounds: the geometries are rebuilt, the polygons that changed are checked (repair, holes for contained
   polygons, thin overlaps cut from the smaller polygon) and the conform repeats.
6. Once settled (or out of rounds): the remaining overlaps are cut out of one polygon, and any new gaps are
   filled; another conform follows.
7. Nodes are added where lines cross (polygon boundaries and faults), then a local conform; anything still
   self-intersecting is repaired.
8. A text report lists what was found before and what remains.
"""

import time
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

from .Densify import _layer_paths, densify_paths, check_same_crs, M_PER_DEG_LAT, M_PER_DEG_LON_AT_EQUATOR

BIG = np.iinfo(np.int64).max // 4


# =============================================================================
# SCALING TO METRES
# =============================================================================

def _metric_scale(layers, geographic):
    """Factors converting CRS units to metres (approximate for degrees)."""
    if not geographic:
        return np.array([1.0, 1.0])
    b = np.array([g.total_bounds for g in layers if len(g)])
    lat = 0.5 * (b[:, 1].min() + b[:, 3].max())
    return np.array([M_PER_DEG_LON_AT_EQUATOR * np.cos(np.radians(lat)), M_PER_DEG_LAT])


def _kind(gdf):
    t = shapely.get_type_id(gdf.geometry.values)
    t = t[~shapely.is_missing(gdf.geometry.values)]
    if len(t) and np.isin(t, (3, 6)).any():
        return "polygon"
    if len(t) and np.isin(t, (1, 2, 5)).any():
        return "line"
    return "none"


# =============================================================================
# STEPS 1-3: REPAIR, DUPLICATES, POLYGONS INSIDE POLYGONS
# =============================================================================

_OVERLAY_FAILS = []          # overlay operations GEOS could not do (reset at the start of every fix_map)


def _grid(gdf=None, bounds=None):
    """Snap-rounding grid for the fallback overlay: 1e-9 degrees, or 1e-6 in projected units."""
    if gdf is not None and gdf.crs is not None:
        return 1e-9 if gdf.crs.is_geographic else 1e-6
    b = np.asarray(bounds if bounds is not None else gdf.total_bounds, dtype=float)
    return 1e-9 if np.nanmax(np.abs(b)) <= 360 else 1e-6


def _robust(op, a, b, grid, what=""):
    """
    ``op(a, b)`` (shapely.intersection / difference / union) that survives GEOS TopologyExceptions on nearly degenerate
    input: it is retried on repaired geometry, then with snap-rounding.  Returns None (and records ``what``) if it still fails.
    """
    for attempt in range(3):
        try:
            if attempt == 0:
                return op(a, b)
            if attempt == 1:
                return op(shapely.make_valid(a), shapely.make_valid(b))
            return op(a, b, grid_size=grid)
        except shapely.errors.GEOSException:
            continue
    _OVERLAY_FAILS.append(str(what))
    return None


def _robust_union_all(arr, grid, what=""):
    for attempt in range(3):
        try:
            if attempt == 0:
                return shapely.union_all(arr)
            if attempt == 1:
                return shapely.union_all(shapely.make_valid(arr))
            return shapely.union_all(arr, grid_size=grid)
        except shapely.errors.GEOSException:
            continue
    _OVERLAY_FAILS.append(str(what))
    return None


def repair_polygons(gdf):
    """buffer(0) on the invalid polygons (make_valid if that empties one)."""
    geoms = gdf.geometry.values.copy()
    bad = np.flatnonzero(~shapely.is_valid(geoms) & ~shapely.is_missing(geoms))
    info = {"invalid": len(bad), "repaired": 0, "failed": [], "indices": list(gdf.index[bad][:20])}
    for k in bad:
        new = shapely.buffer(geoms[k], 0)
        if shapely.is_empty(new) or not shapely.is_valid(new):
            new = shapely.make_valid(geoms[k])
            polys = [p for p in shapely.get_parts(new) if shapely.get_type_id(p) == 3]
            new = shapely.multipolygons(polys) if polys else None
        if new is None or shapely.is_empty(new):
            info["failed"].append(gdf.index[k])
            continue
        if shapely.get_type_id(new) == 6 and shapely.get_num_geometries(new) == 1:
            new = shapely.get_geometry(new, 0)
        geoms[k] = new
        info["repaired"] += 1
    res = gdf.copy()
    res["geometry"] = gpd.GeoSeries(geoms, index=gdf.index, crs=gdf.crs)
    return res, info


def _plain_index(x):
    """A feature index as a plain Python value, so the report shows 909 rather than np.int64(909)."""
    return x.item() if hasattr(x, "item") else x


def _on_segment(a, b, pts, K, near_m):
    """Of the points (m, 2), those within ``near_m`` metres of the inside of segment a-b, ordered from a to b."""
    if len(pts) == 0:
        return []
    A, B, Q = np.asarray(a) * K, np.asarray(b) * K, np.asarray(pts) * K
    d = B - A
    L2 = float(d @ d)
    if L2 == 0:
        return []
    t = ((Q - A) @ d) / L2
    dist = np.hypot(*(Q - (A + t[:, None] * d)).T)
    keep = (t > 0) & (t < 1) & (dist <= near_m)
    return [tuple(pts[k]) for k in np.flatnonzero(keep)[np.argsort(t[keep], kind="stable")]]


def _cut_loops(c, max_m, K, near_m=0.0):
    """
    Cut the loops out of one line's vertices (n, 2): wherever the line passes through the same point twice,
    the vertices between the two visits are dropped, if all of them lie within ``max_m`` metres of that point
    (K: metres per layer unit in x and y). Dropped vertices lying along one of the two segments now meeting at
    the point (within ``near_m`` metres of it) are put back on it, in order, so that a fault that ran out along
    a contact and back keeps the contact's nodes once. Repeated neighbouring vertices are merged. A closed line
    keeps its closing vertex. Returns (vertices, loops cut, a larger loop was left).
    """
    closed = len(c) > 3 and (c[0] == c[-1]).all()
    body = [tuple(p) for p in (c[:-1] if closed else c)]
    cut = 0
    big = False
    for _ in range(4 * len(body) + 4):              # every step shortens the line or fixes a loop; bounded anyway
        seen, done = {}, True
        for pos, xy in enumerate(body):
            i = seen.get(xy)
            if i is not None:
                loop = np.asarray(body[i:pos + 1])
                extent = np.hypot(*((loop - loop[0]) * K).T).max()
                if pos - i == 1 or extent <= max_m:
                    dropped = body[i + 1:pos]
                    body = body[:i + 1] + body[pos + 1:]
                    if dropped and near_m > 0:
                        present = set(body)
                        dropped = list(dict.fromkeys(p for p in dropped if p not in present))
                        after = _on_segment(xy, body[i + 1], np.asarray(dropped), K, near_m) if i + 1 < len(body) else []
                        rest = [p for p in dropped if p not in set(after)]
                        before = _on_segment(body[i - 1], xy, np.asarray(rest), K, near_m) if i > 0 and rest else []
                        body = body[:i] + before + [xy] + after + body[i + 1:]
                    cut += pos - i > 1
                    done = False
                    break
                big = True
            seen[xy] = pos
        if done:
            break
    out = np.asarray(body + ([body[0]] if closed else []))
    return out, cut, big


def remove_line_loops(gdf, max_m, K, near_m=0.0):
    """
    Cut short loops out of lines.  Snapping a vertex onto a node, and adding that node to a nearby segment of the
    same line, can make a line pass through the node twice with a loop of a few metres in between (often where the
    line was digitised with a hairpin or spike).  Loops whose vertices all lie within ``max_m`` metres of the
    repeated point are cut out; nothing is moved.  Returns (gdf, info): info has "loops" (number cut), "indices"
    (features changed) and "left" (features that still cross or touch themselves).
    """
    geoms = gdf.geometry.values.copy()
    info = {"loops": 0, "indices": [], "left": []}
    ok = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
    for k in np.flatnonzero(ok & ~shapely.is_simple(np.where(ok, geoms, None))):
        parts = shapely.get_parts(geoms[k])
        new_parts, n_cut = [], 0
        for part in parts:
            c = shapely.get_coordinates(part)
            c2, cut, _ = _cut_loops(c, max_m, K, near_m)
            n_cut += cut
            new_parts.append(shapely.linestrings(c2) if len(c2) >= 2 and cut else part)
        if n_cut:
            geoms[k] = new_parts[0] if shapely.get_type_id(geoms[k]) == 1 else shapely.multilinestrings(new_parts)
            info["loops"] += n_cut
            info["indices"].append(_plain_index(gdf.index[k]))
        if not shapely.is_simple(geoms[k]):
            info["left"].append(_plain_index(gdf.index[k]))
    res = gdf.copy()
    res["geometry"] = gpd.GeoSeries(geoms, index=gdf.index, crs=gdf.crs)
    return res, info


def remove_duplicates(gdf):
    """Remove features whose geometry is identical (any start point / direction) to an earlier one."""
    geoms = gdf.geometry.values
    ok = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
    wkb = np.full(len(geoms), None, dtype=object)
    wkb[ok] = shapely.to_wkb(shapely.normalize(geoms[ok]))
    dup = pd.Series(wkb).duplicated().values & ok
    info = {"removed": int(dup.sum()), "indices": list(gdf.index[dup][:20])}
    return gdf.loc[~dup].copy(), info


def cut_contained_polygons(gdf, pad, min_inside=0.9, only=None):
    """
    A polygon that lies (for at least ``min_inside`` of its area) inside a larger polygon, without
    a hole in it, gets cut out of the larger polygon as a hole.  ``pad`` is a bounding-box margin
    in CRS units.
    """
    geoms = gdf.geometry.values.copy()
    n = len(geoms)
    grid = _grid(gdf)
    info = {"holes_cut": 0, "containers": 0, "examples": []}
    if n < 2:
        return gdf, info
    ok = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
    area = np.where(ok, shapely.area(geoms), 0.0)
    tree = shapely.STRtree(geoms)
    if only is None:
        qi, ti = tree.query(geoms, predicate="contains")
        qo, to = tree.query(geoms, predicate="overlaps")
    else:                                    # only pairs that involve a polygon that changed
        only = np.asarray(only, dtype=np.int64)
        if len(only) == 0:
            return gdf, info
        a_, b_ = tree.query(geoms[only], predicate="contains")           # a changed polygon contains b
        c_, d_ = tree.query(geoms[only], predicate="within")             # a changed polygon lies inside d
        qi = np.concatenate([only[a_], d_])
        ti = np.concatenate([b_, only[c_]])
        e_, f_ = tree.query(geoms[only], predicate="overlaps")
        qo = np.concatenate([only[e_], f_])                              # both orientations: the
        to = np.concatenate([f_, only[e_]])                              # bigger one is picked below
    m = (qi != ti) & (area[qi] > area[ti])
    ci, cj = qi[m], ti[m]
    m = area[qo] > area[to]
    qo, to = qo[m], to[m]
    if len(qo):
        b = shapely.bounds(geoms)
        # at least ``min_inside`` of the smaller polygon's area must fall in the overlap of the two bounding boxes
        # (a multi-part polygon with a small remote part has a bounding box far larger than its area suggests)
        ix = np.clip(np.minimum(b[qo, 2], b[to, 2]) - np.maximum(b[qo, 0], b[to, 0]), 0, None)
        iy = np.clip(np.minimum(b[qo, 3], b[to, 3]) - np.maximum(b[qo, 1], b[to, 1]), 0, None)
        near = ix * iy >= min_inside * area[to]
        qo, to = qo[near], to[near]
        if len(qo):
            try:
                inter = shapely.area(shapely.intersection(geoms[qo], geoms[to]))
            except shapely.errors.GEOSException:                          # one pair is too degenerate: do them one by one
                inter = np.zeros(len(qo))
                for k_, (a_, b_) in enumerate(zip(qo, to)):
                    r_ = _robust(shapely.intersection, geoms[a_], geoms[b_], grid, f"{gdf.index[a_]}/{gdf.index[b_]}")
                    inter[k_] = shapely.area(r_) if r_ is not None else 0.0
            m = inter >= min_inside * area[to]
            qo, to = qo[m], to[m]
    ci = np.concatenate([ci, qo])
    cj = np.concatenate([cj, to])
    if len(ci) == 0:
        return gdf, info
    order = np.argsort(ci, kind="stable")
    ci, cj = ci[order], cj[order]
    for i in np.unique(ci):
        js = np.unique(cj[ci == i])
        un = _robust_union_all(geoms[js], grid, f"{gdf.index[i]}")
        new = _robust(shapely.difference, geoms[i], un, grid, f"{gdf.index[i]}") if un is not None else None
        if new is None:
            continue
        polys = [p for p in shapely.get_parts(new) if shapely.get_type_id(p) == 3]
        if not polys:
            continue
        geoms[i] = polys[0] if len(polys) == 1 and shapely.get_type_id(geoms[i]) == 3 else shapely.multipolygons(polys)
        info["containers"] += 1
        info["holes_cut"] += len(js)
        if len(info["examples"]) < 10:
            info["examples"].append((gdf.index[i], [gdf.index[j] for j in js][:5]))
    res = gdf.copy()
    res["geometry"] = gpd.GeoSeries(geoms, index=gdf.index, crs=gdf.crs)
    return res, info


def _gap_polygons(u, grid=None):
    """
    The gaps in a union of polygons: every hole in it, minus any polygon that lies inside that hole
    (the outline of a hole also encloses such islands).
    """
    parts = shapely.get_parts(u)
    grid = grid if grid is not None else _grid(bounds=shapely.bounds(u))
    tree = shapely.STRtree(parts)
    holes = []
    for pi, part in enumerate(parts):
        for ring in shapely.get_rings(part)[1:]:
            h = shapely.polygons(ring)
            inside = [k for k in tree.query(h, predicate="contains") if k != pi]
            if inside:
                un = _robust_union_all(parts[inside], grid, "gap")
                h2 = _robust(shapely.difference, h, un, grid, "gap") if un is not None else None
                h = h2 if h2 is not None else h
            if not shapely.is_empty(h):
                holes.append(h)
    return holes


def remove_overlaps(gdf, keep="smaller"):
    """
    Every remaining overlap between two polygons (partial overlaps, or one inside the other) is cut out of one of
    them, so that no two polygons overlap any more.  ``keep="smaller"``: the smaller polygon keeps the overlap and the
    bigger one loses it (the same convention as a hole cut for a polygon inside another); ``keep="larger"``: the other
    way round.  A polygon that is entirely overlapped disappears.  Returns (gdf, info).
    """
    geoms = gdf.geometry.values.copy()
    grid = _grid(gdf)
    info = {"pairs": 0, "area": 0.0, "removed": [], "examples": []}
    n = len(geoms)
    if n < 2:
        return gdf, info
    ok = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
    area0 = np.where(ok, shapely.area(geoms), 0.0)
    tree = shapely.STRtree(geoms)
    a1, b1 = tree.query(geoms, predicate="overlaps")
    a2, b2 = tree.query(geoms, predicate="contains")
    qi = np.concatenate([a1, a2])
    ti = np.concatenate([b1, b2])
    m = qi != ti
    qi, ti = qi[m], ti[m]
    if len(qi) == 0:
        return gdf, info
    lo_, hi_ = np.minimum(qi, ti).astype(np.int64), np.maximum(qi, ti).astype(np.int64)
    _, first = np.unique(lo_ * n + hi_, return_index=True)              # each pair once
    lo_, hi_ = lo_[first], hi_[first]
    for i, j in zip(lo_, hi_):
        bigger, smaller = (i, j) if (area0[i], -i) > (area0[j], -j) else (j, i)
        loser, winner = (bigger, smaller) if keep == "smaller" else (smaller, bigger)
        if shapely.is_empty(geoms[loser]) or shapely.is_empty(geoms[winner]):
            continue
        what = f"{gdf.index[loser]}/{gdf.index[winner]}"
        ov_ = _robust(shapely.intersection, geoms[loser], geoms[winner], grid, what)
        if ov_ is None:
            continue
        inter = shapely.area(ov_)
        if inter <= 1e-9 * max(min(area0[i], area0[j]), 1e-30):
            continue                                                    # only touching, or numerical noise
        new = _robust(shapely.difference, geoms[loser], geoms[winner], grid, what)
        if new is None:
            continue
        parts = [x for x in shapely.get_parts(new) if shapely.get_type_id(x) == 3 and x.area > 0]
        if not parts:
            geoms[loser] = shapely.Polygon()
            info["removed"].append(gdf.index[loser])
        else:
            geoms[loser] = parts[0] if (len(parts) == 1 and shapely.get_type_id(geoms[loser]) == 3) else shapely.multipolygons(parts)
        info["pairs"] += 1
        info["area"] += float(inter)
        if len(info["examples"]) < 10:
            info["examples"].append((gdf.index[loser], gdf.index[winner]))
    res = gdf.copy()
    res["geometry"] = gpd.GeoSeries(geoms, index=gdf.index, crs=gdf.crs)
    if info["removed"]:
        res = res.loc[~res.index.isin(info["removed"])].copy()
    return res, info


def fill_thin_gaps(gdf, width):
    """
    Gaps between polygons that are narrower than ``width`` (CRS units) everywhere (all of them if ``width`` is
    None) are added to the
    neighbouring polygon with which they share the longest boundary.  Only holes completely enclosed
    by polygons are gaps; open slivers along the edge of the map and wide holes are left alone.
    Returns (gdf, info).
    """
    geoms = gdf.geometry.values.copy()
    grid = _grid(gdf)
    info = {"gaps": 0, "area": 0.0, "examples": []}
    ok = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
    if ok.sum() < 2:
        return gdf, info
    valid = np.where(shapely.is_valid(geoms), geoms, shapely.make_valid(geoms))
    u = _robust_union_all(valid[ok], grid, "union of all polygons")
    if u is None:
        return gdf, info
    holes = _gap_polygons(u, grid)
    if not holes:
        return gdf, info
    holes = np.array(holes, dtype=object)
    if width is not None:                                  # None: every gap completely surrounded by polygons
        thin = shapely.is_empty(shapely.buffer(holes, -0.5 * width))
        holes = holes[thin]
    if len(holes) == 0:
        return gdf, info
    tree = shapely.STRtree(geoms)
    for h in holes:
        cand = tree.query(h, predicate="intersects")
        cand = [k for k in cand if ok[k]]
        if not cand:
            continue
        hb = shapely.boundary(h)
        shared = []
        for k in cand:
            r_ = _robust(shapely.intersection, hb, geoms[k], grid, f"gap/{gdf.index[k]}")
            shared.append(shapely.length(r_) if r_ is not None else 0.0)
        best = cand[int(np.argmax(shared))]
        if max(shared) <= 0:
            continue
        new = _robust(shapely.union, geoms[best], h, grid, f"gap/{gdf.index[best]}")
        if new is None:
            continue
        try:                      # the union copies a corner from two neighbours with float noise between: one vertex is enough
            new = shapely.remove_repeated_points(new, tolerance=10 * grid)
        except (AttributeError, shapely.errors.GEOSException):
            pass
        parts = [x for x in shapely.get_parts(new) if shapely.get_type_id(x) == 3]
        if not parts:
            continue
        geoms[best] = parts[0] if (len(parts) == 1 and shapely.get_type_id(geoms[best]) == 3) else shapely.multipolygons(parts)
        info["gaps"] += 1
        info["area"] += float(shapely.area(h))
        if len(info["examples"]) < 10:
            info["examples"].append(gdf.index[best])
    res = gdf.copy()
    res["geometry"] = gpd.GeoSeries(geoms, index=gdf.index, crs=gdf.crs)
    return res, info


def resolve_thin_overlaps(gdf, width, only=None):
    """
    Where two polygons overlap in a strip narrower than ``width`` (CRS units) everywhere, the overlap is
    taken out of the smaller polygon.  Wider overlaps are left alone.  Returns (gdf, info).
    """
    geoms = gdf.geometry.values.copy()
    grid = _grid(gdf)
    info = {"pairs": 0, "area": 0.0, "examples": []}
    n = len(geoms)
    if n < 2:
        return gdf, info
    ok = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
    area = np.where(ok, shapely.area(geoms), 0.0)
    tree = shapely.STRtree(geoms)
    if only is None:
        qi, ti = tree.query(geoms, predicate="overlaps")
    else:
        only = np.asarray(only, dtype=np.int64)
        if len(only) == 0:
            return gdf, info
        a_, b_ = tree.query(geoms[only], predicate="overlaps")
        qi = np.concatenate([only[a_], b_])                              # both orientations; the next lines
        ti = np.concatenate([b_, only[a_]])                              # keep the one with qi bigger
    m = (area[qi] > area[ti]) | ((area[qi] == area[ti]) & (qi < ti))      # qi is the bigger one
    qi, ti = qi[m], ti[m]
    if len(qi) == 0:
        return gdf, info
    for i, j in zip(qi, ti):
        big, small = geoms[i], geoms[j]
        what = f"{gdf.index[j]}/{gdf.index[i]}"
        inter = _robust(shapely.intersection, big, small, grid, what)
        if inter is None:
            continue
        polys = [x for x in shapely.get_parts(inter) if shapely.get_type_id(x) == 3]
        if not polys:
            continue
        ov = shapely.multipolygons(polys)
        if not shapely.is_empty(shapely.buffer(ov, -0.5 * width)):
            continue                                                   # wide: a real overlap, leave it
        new = _robust(shapely.difference, small, big, grid, what)
        if new is None:
            continue
        parts = [x for x in shapely.get_parts(new) if shapely.get_type_id(x) == 3]
        if not parts:
            continue
        geoms[j] = parts[0] if (len(parts) == 1 and shapely.get_type_id(small) == 3) else shapely.multipolygons(parts)
        info["pairs"] += 1
        info["area"] += float(shapely.area(ov))
        if len(info["examples"]) < 10:
            info["examples"].append((gdf.index[j], gdf.index[i]))
    res = gdf.copy()
    res["geometry"] = gpd.GeoSeries(geoms, index=gdf.index, crs=gdf.crs)
    return res, info


# =============================================================================
# STEP 4: CONFORM
# =============================================================================

def _candidates(coords, pid, rank, tol, dirty=None, orig=None):
    """
    Everything one pass needs: unique vertices (sites) and unique segments, and every pair of a site
    with a segment of another path that lies within ``tol`` of it, without being one of its nodes.

    With ``dirty`` (an array of points where something changed in the previous pass) only the
    neighbourhood of those points is examined: sites within 2 tol of one, and the segments that pass
    within 3 tol of one.  Everything else was examined before and has not changed.
    """
    from scipy.spatial import cKDTree
    P = len(rank)
    n = len(coords)
    has_next = np.r_[pid[:-1] == pid[1:], False]
    vpos = np.flatnonzero(has_next)
    queried = None
    if dirty is None:
        rows = None
        sub = coords
        sub_pid = pid
    else:
        Dt = cKDTree(dirty)
        d_row, _ = Dt.query(coords, k=1, distance_upper_bound=3.0 * tol, workers=-1)
        # A segment with a changed end point has moved as a whole: a long one can now pass close to
        # vertices far from its ends.  Treat points along such segments as changed too.
        seg_len = np.hypot(*(coords[vpos + 1] - coords[vpos]).T)
        changed_long = ((d_row[vpos] == 0) | (d_row[vpos + 1] == 0)) & (seg_len > 2.0 * tol)
        if changed_long.any():
            va = vpos[changed_long]
            ns = np.ceil(seg_len[changed_long] / tol).astype(np.int64)
            if ns.sum() > 0.25 * n:
                return _candidates(coords, pid, rank, tol, None, orig)  # too much changed: look at everything
            rep_s = np.repeat(np.arange(len(va)), ns)
            frac = (np.arange(len(rep_s)) - np.repeat(np.cumsum(ns) - ns, ns) + 0.5) / ns[rep_s]
            extra = coords[va][rep_s] + frac[:, None] * (coords[va + 1][rep_s] - coords[va][rep_s])
            dirty = np.vstack([dirty, extra])
            Dt = cKDTree(dirty)
            d_row, _ = Dt.query(coords, k=1, distance_upper_bound=3.0 * tol, workers=-1)
        near3 = np.isfinite(d_row)
        mid = 0.5 * (coords[vpos] + coords[vpos + 1])
        half = 0.5 * np.hypot(*(coords[vpos + 1] - coords[vpos]).T)
        d_mid, _ = Dt.query(mid, k=1, workers=-1)
        vpos = vpos[d_mid <= half + 3.0 * tol]
        sel = near3.copy()
        sel[vpos] = True
        sel[vpos + 1] = True
        rows = np.flatnonzero(sel)
        sub = coords[rows]
        sub_pid = pid[rows]
        near2_rows = d_row[rows] <= 2.0 * tol
    uv, inv = np.unique(sub, axis=0, return_inverse=True)
    inv = inv.ravel()
    prank = rank[sub_pid]
    site_rank = np.full(len(uv), BIG, dtype=np.int64)
    np.minimum.at(site_rank, inv, prank)
    if dirty is not None:
        queried = np.zeros(len(uv), dtype=bool)
        queried[inv[near2_rows]] = True
        inv_full = np.full(n, -1, dtype=np.int64)
        inv_full[rows] = inv
    else:
        inv_full = inv
    site_lo = site_hi = None
    if orig is not None:                     # the box around where the rows of each site started from
        osub = orig if rows is None else orig[rows]
        site_lo = np.full((len(uv), 2), np.inf)
        site_hi = np.full((len(uv), 2), -np.inf)
        for d_ in (0, 1):
            np.minimum.at(site_lo[:, d_], inv, osub[:, d_])
            np.maximum.at(site_hi[:, d_], inv, osub[:, d_])

    A, B = coords[vpos], coords[vpos + 1]
    flip = (A[:, 0] > B[:, 0]) | ((A[:, 0] == B[:, 0]) & (A[:, 1] > B[:, 1]))
    lo = np.where(flip[:, None], B, A)
    hi = np.where(flip[:, None], A, B)
    uniq, sinv = np.unique(np.concatenate([lo, hi], axis=1), axis=0, return_inverse=True)
    sinv = sinv.ravel()
    ulo, uhi = uniq[:, :2], uniq[:, 2:]
    U = len(uniq)
    seg_path = pid[vpos]
    seg_rank = np.full(U, BIG, dtype=np.int64)
    np.minimum.at(seg_rank, sinv, rank[seg_path])
    path_of_rank = np.empty(P, dtype=np.int64)
    path_of_rank[rank] = np.arange(P)
    seg_owner = path_of_rank[np.minimum(seg_rank, P - 1)]

    seg_lo_site = np.empty(U, dtype=np.int64)
    seg_hi_site = np.empty(U, dtype=np.int64)
    s0, s1 = inv_full[vpos], inv_full[vpos + 1]
    seg_lo_site[sinv] = np.where(flip, s1, s0)
    seg_hi_site[sinv] = np.where(flip, s0, s1)

    segs = shapely.linestrings(np.stack([ulo, uhi], axis=1))
    tree = shapely.STRtree(segs)
    if queried is None:
        si, gi = tree.query(shapely.points(uv), predicate="dwithin", distance=tol)
    else:
        qsites = np.flatnonzero(queried)
        si, gi = tree.query(shapely.points(uv[qsites]), predicate="dwithin", distance=tol)
        si = qsites[si]
    Q = uv[si]
    a, b = ulo[gi], uhi[gi]
    keep = ~(np.all(Q == a, axis=1) | np.all(Q == b, axis=1))
    si, gi, Q, a, b = si[keep], gi[keep], Q[keep], a[keep], b[keep]

    # drop pairs where the site and the segment belong to a common path
    sp = np.unique(inv.astype(np.int64) * P + sub_pid)
    gp = np.unique(sinv.astype(np.int64) * P + seg_path)
    lo_i = np.searchsorted(sp, si.astype(np.int64) * P)
    hi_i = np.searchsorted(sp, (si.astype(np.int64) + 1) * P)
    cnt = hi_i - lo_i
    rep = np.repeat(np.arange(len(si)), cnt)
    off = np.arange(len(rep)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    pth = sp[lo_i[rep] + off] - si[rep].astype(np.int64) * P
    key = gi[rep].astype(np.int64) * P + pth
    pos = np.minimum(np.searchsorted(gp, key), len(gp) - 1)
    shared = np.bincount(rep, weights=(gp[pos] == key), minlength=len(si)) > 0
    si, gi, Q, a, b = si[~shared], gi[~shared], Q[~shared], a[~shared], b[~shared]

    d = b - a
    L2 = (d * d).sum(axis=1)
    t = ((Q - a) * d).sum(axis=1) / L2
    q = a + np.clip(t, 0, 1)[:, None] * d
    dist = np.hypot(Q[:, 0] - q[:, 0], Q[:, 1] - q[:, 1])
    ok = dist <= tol
    # a vertex whose nearest point is a node belonging to its own path is just a short segment
    clamped = np.flatnonzero(ok & ((t <= 0) | (t >= 1)))
    if len(clamped):
        e_site = np.where(t[clamped] <= 0, seg_lo_site[gi[clamped]], seg_hi_site[gi[clamped]])
        s_c = si[clamped].astype(np.int64)
        lo_i = np.searchsorted(sp, s_c * P)
        hi_i = np.searchsorted(sp, (s_c + 1) * P)
        cnt = hi_i - lo_i
        rep = np.repeat(np.arange(len(clamped)), cnt)
        off = np.arange(len(rep)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        pth = sp[lo_i[rep] + off] - s_c[rep] * P
        key = e_site[rep].astype(np.int64) * P + pth
        pos = np.minimum(np.searchsorted(sp, key), len(sp) - 1)
        own = np.bincount(rep, weights=(sp[pos] == key), minlength=len(clamped)) > 0
        ok[clamped[own]] = False
    return {"uv": uv, "inv": inv, "rows": rows, "vpos": vpos, "flip": flip, "sinv": sinv, "U": U,
            "site_lo": site_lo, "site_hi": site_hi,
            "site_rank": site_rank, "seg_rank": seg_rank, "seg_owner": seg_owner, "P": P,
            "si": si[ok], "gi": gi[ok], "t": t[ok], "q": q[ok], "dist": dist[ok],
            "a": a[ok], "b": b[ok]}


def _count_defects(coords, pid, rank, tol):
    """(unmatched nodes, near-duplicate nodes): what a conform pass would act on."""
    if len(coords) == 0:
        return 0, 0
    c = _candidates(coords, pid, rank, tol)
    P = c["P"]
    interior = (c["t"] > 0) & (c["t"] < 1)
    unmatched = len(np.unique(c["si"][interior].astype(np.int64) * P + c["seg_owner"][c["gi"][interior]]))
    near = ~interior
    pa = np.where((c["t"][near] <= 0)[:, None], c["a"][near], c["b"][near])
    pair = np.unique(np.concatenate([c["si"][near, None].astype(float), pa], axis=1), axis=0)
    return int(unmatched), int(len(pair))


def _insert_nodes(coords, pid, c, ins_seg, ins_t, ins_xy, orig=None):
    """Put the new nodes into the paths (once per unique segment, read in each path's direction)."""
    n = len(coords)
    U = c["U"]
    order = np.lexsort((ins_t, ins_seg))
    ins_seg, ins_xy = ins_seg[order], ins_xy[order]
    m_u = np.bincount(ins_seg, minlength=U)
    off = np.cumsum(m_u) - m_u
    cnt = np.zeros(n, dtype=np.int64)
    cnt[c["vpos"]] = m_u[c["sinv"]]
    ub = np.zeros(n, dtype=np.int64)
    ub[c["vpos"]] = c["sinv"]
    fl = np.zeros(n, dtype=bool)
    fl[c["vpos"]] = c["flip"]
    place = np.cumsum(1 + cnt) - (1 + cnt)
    out = np.empty((int((1 + cnt).sum()), 2))
    out[place] = coords
    v = np.repeat(np.arange(n), cnt)
    m = np.arange(len(v)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    take = off[ub[v]] + np.where(fl[v], cnt[v] - 1 - m, m)
    out[place[v] + 1 + m] = ins_xy[take]
    orig_out = None
    if orig is not None:                     # a new node starts where it is
        orig_out = np.empty_like(out)
        orig_out[place] = orig
        orig_out[place[v] + 1 + m] = ins_xy[take]
    return out, np.repeat(pid, 1 + cnt), orig_out


def _cleanup(coords, pid, path_is_poly, orig=None):
    """Drop repeated vertices and spikes (A-B-A); drop paths that collapsed."""
    for _ in range(8):
        if len(coords) < 2:
            break
        same_prev = np.r_[False, (pid[1:] == pid[:-1]) & np.all(coords[1:] == coords[:-1], axis=1)]
        spike = np.zeros(len(coords), dtype=bool)
        if len(coords) > 2:
            spike[1:-1] = (pid[1:-1] == pid[:-2]) & (pid[1:-1] == pid[2:]) & np.all(coords[:-2] == coords[2:], axis=1)
        drop = same_prev | spike
        if not drop.any():
            break
        coords, pid = coords[~drop], pid[~drop]
        if orig is not None:
            orig = orig[~drop]
    need = np.where(path_is_poly, 4, 2)
    rows = np.bincount(pid, minlength=len(path_is_poly))
    dead = rows < need
    if path_is_poly.any() and len(coords) > 2:
        first = np.r_[True, pid[1:] != pid[:-1]]
        f0 = np.maximum.accumulate(np.where(first, np.arange(len(pid)), 0))
        xs, ys = coords[:, 0] - coords[f0, 0], coords[:, 1] - coords[f0, 1]
        cross = np.zeros(len(pid))
        same = pid[:-1] == pid[1:]
        cross[:-1] = np.where(same, xs[:-1] * ys[1:] - xs[1:] * ys[:-1], 0.0)
        ring_area = np.abs(np.bincount(pid, weights=cross, minlength=len(path_is_poly))) / 2
        dead |= path_is_poly & (ring_area < 1e-3)           # rings that collapsed to a line
    if dead.any():
        k = ~dead[pid]
        coords, pid = coords[k], pid[k]
        if orig is not None:
            orig = orig[k]
    return coords, pid, orig


def _conform_pass(coords, pid, rank, path_is_poly, tol, dirty=None, orig=None, cap=None):
    """One conform pass.  With ``cap`` (and ``orig``, where every row started) no vertex ends up more than
    ``cap`` from where it started."""
    c = _candidates(coords, pid, rank, tol, dirty, orig if cap is not None else None)
    lo_, hi_ = c["site_lo"], c["site_hi"]

    def allowed(sites, targets):
        if cap is None or lo_ is None:
            return np.ones(len(sites), dtype=bool)
        far = np.maximum(np.abs(targets - lo_[sites]), np.abs(targets - hi_[sites]))
        return np.hypot(far[:, 0], far[:, 1]) <= cap

    P = c["P"]
    si, gi, t, q, dist, a, b = c["si"], c["gi"], c["t"], c["q"], c["dist"], c["a"], c["b"]
    uv = c["uv"]
    slave = c["site_rank"][si] > c["seg_rank"][gi]

    # slave vertices: move onto the nearest node of a higher-ranked line within the tolerance
    # (a node is on that line anyway, and this merges the near-coincident nodes of a junction);
    # if there is none, onto the nearest point of the nearest segment, which becomes a node
    idx = np.flatnonzero(slave)
    new_uv = uv.copy()
    ins_seg = np.zeros(0, dtype=np.int64); ins_t = np.zeros(0); ins_xy = np.zeros((0, 2))
    n_move = 0
    parts_seg, parts_t, parts_xy = [], [], []
    if len(idx):
        Qs = uv[si[idx]]
        nd_a = np.hypot(*(Qs - a[idx]).T)
        nd_b = np.hypot(*(Qs - b[idx]).T)
        use_a = nd_a <= nd_b
        nd = np.where(use_a, nd_a, nd_b)
        nxy = np.where(use_a[:, None], a[idx], b[idx])
        has = (nd <= tol) & allowed(si[idx], nxy)
        done = np.zeros(len(uv), dtype=bool)
        if has.any():
            h = np.flatnonzero(has)
            order = h[np.lexsort((nd[h], si[idx][h]))]
            sites = si[idx][order]
            first = np.r_[True, sites[1:] != sites[:-1]]
            site = sites[first]
            new_uv[site] = nxy[order][first]
            done[site] = True
        rest = idx[~done[si[idx]]]
        if len(rest):
            order = rest[np.lexsort((gi[rest], dist[rest], si[rest]))]
            first = np.r_[True, si[order][1:] != si[order][:-1]]
            sel = order[first]
            sel = sel[allowed(si[sel], q[sel])]
            site = si[sel]
            new_uv[site] = q[sel]
            parts_seg.append(gi[sel]); parts_t.append(t[sel]); parts_xy.append(q[sel])
        moved_sites = np.flatnonzero(np.any(new_uv != uv, axis=1))
        n_move = int(len(moved_sites))

    # higher-ranked vertices: become nodes of the lower-ranked line passing near them
    m_idx = np.flatnonzero(~slave & (t > 0) & (t < 1))
    if len(m_idx):
        key = si[m_idx].astype(np.int64) * P + c["seg_owner"][gi[m_idx]]
        order = m_idx[np.lexsort((dist[m_idx], key))]
        ks = key[np.lexsort((dist[m_idx], key))]
        first = np.r_[True, ks[1:] != ks[:-1]]
        sel = order[first]
        parts_seg.append(gi[sel]); parts_t.append(t[sel]); parts_xy.append(uv[si[sel]])

    if parts_seg:
        ins_seg = np.concatenate(parts_seg); ins_t = np.concatenate(parts_t); ins_xy = np.concatenate(parts_xy)
        # the same node can arrive from several vertices
        k = np.round(ins_xy / 1e-6).astype(np.int64)
        _, u = np.unique(np.column_stack([ins_seg, k]), axis=0, return_index=True)
        ins_seg, ins_t, ins_xy = ins_seg[u], ins_t[u], ins_xy[u]

    changed_sites = np.flatnonzero(np.any(new_uv != uv, axis=1))
    if c["rows"] is None:
        coords = new_uv[c["inv"]]
    else:
        coords = coords.copy()
        coords[c["rows"]] = new_uv[c["inv"]]
    if len(ins_seg):
        coords, pid, orig_new = _insert_nodes(coords, pid, c, ins_seg, ins_t, ins_xy, orig if cap is not None else None)
        if cap is not None:
            orig = orig_new
    coords, pid, orig = _cleanup(coords, pid, path_is_poly, orig if cap is not None else None)
    # where something changed: the next pass only needs to look around these points
    dirty_out = np.concatenate([new_uv[changed_sites], uv[changed_sites], ins_xy]) if (len(changed_sites) or len(ins_seg)) \
        else np.zeros((0, 2))
    return coords, pid, n_move, int(len(ins_seg)), dirty_out, orig


def _conform(coords, pid, rank, path_is_poly, tol, max_pass=40, log=None, dirty=None, cap=None):
    """
    Conform passes until nothing changes.  The first pass looks at everything (or, given ``dirty``,
    around those points); later passes only look around what the previous pass changed.  When the local
    passes find nothing a last full pass checks that nothing was missed.
    """
    stats = {"passes": 0, "vertices_moved": 0, "nodes_inserted": 0}
    orig = coords.copy() if cap is not None else None
    last_ni, same_ni = -1, 0
    for _ in range(max_pass):
        t0 = time.time()
        local = dirty is not None
        coords, pid, nm, ni, d_out, orig = _conform_pass(coords, pid, rank, path_is_poly, tol, dirty, orig, cap)
        stats["passes"] += 1
        stats["vertices_moved"] += nm
        stats["nodes_inserted"] += ni
        if log:
            log(f"  pass {stats['passes']}{' (local)' if local else ' (full)'}: {nm} vertices moved, "
                f"{ni} nodes inserted ({time.time() - t0:.1f} s)")
        if nm == 0 and ni == 0:
            if not local:
                break                                    # a full pass found nothing: finished
            dirty = None                                 # verify with one full pass
            continue
        # the same nodes inserted again and again, with nothing moving, means they are not sticking: stop
        same_ni = same_ni + 1 if (nm == 0 and ni == last_ni) else 0
        last_ni = ni
        if same_ni >= 12:
            if log:
                log(f"  stopped: the same {ni} nodes were inserted in 12 passes in a row without anything moving")
            stats["stalled"] = ni
            break
        dirty = d_out
    stats["max_move"] = float(np.hypot(*(coords - orig).T).max()) if (orig is not None and len(coords)) else 0.0
    return coords, pid, stats


# =============================================================================
# PUTTING LAYERS BACK TOGETHER
# =============================================================================

def _assemble(gdf, info, coords, pid, base, alive):
    """New geometries of a layer from its (possibly changed) paths; features that vanished are dropped."""
    n_paths = len(info["paths"])
    n_rings = info["n_rings"]
    alive = alive[base:base + n_paths]
    sel = (pid >= base) & (pid < base + n_paths)
    cc, pp = coords[sel], pid[sel] - base

    ring_geoms = np.full(n_rings, None, dtype=object)
    ridx = np.flatnonzero(alive[:n_rings])
    if len(ridx):
        m = pp < n_rings
        _, rel = np.unique(pp[m], return_inverse=True)
        ring_geoms[ridx] = shapely.linearrings(cc[m], indices=rel)
    parts = info["parts"]
    new_parts = np.full(len(parts), None, dtype=object)
    if n_rings:
        rp = info["ring_poly"]
        first = np.r_[True, rp[1:] != rp[:-1]]
        shell_pos = np.flatnonzero(first)
        ends = np.r_[shell_pos[1:], n_rings]
        polys = np.full(len(shell_pos), None, dtype=object)
        shell_alive = alive[shell_pos]
        simple = shell_alive & (ends - shell_pos == 1)
        if simple.any():
            polys[simple] = shapely.polygons(ring_geoms[shell_pos[simple]])
        for q in np.flatnonzero(shell_alive & ~simple):
            holes = [ring_geoms[k] for k in range(shell_pos[q] + 1, ends[q]) if alive[k]]
            polys[q] = shapely.Polygon(ring_geoms[shell_pos[q]], holes)
        # an empty polygon has no rings: ring_poly says which polygon each group of rings belongs to
        all_polys = np.full(len(info["poly_pos"]), None, dtype=object)
        all_polys[rp[shell_pos]] = polys
        new_parts[info["poly_pos"]] = all_polys
    if len(info["lines"]):
        line_alive = alive[n_rings:]
        lines = np.full(len(info["lines"]), None, dtype=object)
        lidx = np.flatnonzero(line_alive)
        if len(lidx):
            m = pp >= n_rings
            _, rel = np.unique(pp[m], return_inverse=True)
            lines[lidx] = shapely.linestrings(cc[m], indices=rel)
        new_parts[info["line_pos"]] = lines

    geoms, feat = info["geoms"], info["feat"]
    out = geoms.copy()
    otype = shapely.get_type_id(geoms)
    alive_part = ~shapely.is_missing(new_parts)
    cnt = np.bincount(feat[alive_part], minlength=len(geoms))
    had = np.bincount(feat, minlength=len(geoms)) > 0
    dropped = had & (cnt == 0)
    ap = np.flatnonzero(alive_part)
    single = np.isin(otype, (1, 2, 3)) & (cnt >= 1)
    if single.any():
        uf, fi = np.unique(feat[ap], return_index=True)
        pick = dict(zip(uf, ap[fi]))
        for f in np.flatnonzero(single):
            out[f] = new_parts[pick[f]]
    for type_id, make in ((6, shapely.multipolygons), (5, shapely.multilinestrings)):
        feats = np.flatnonzero((otype == type_id) & (cnt >= 1))
        if len(feats):
            keep = alive_part & np.isin(feat, feats)
            uf, inv = np.unique(feat[keep], return_inverse=True)
            out[uf] = make(new_parts[keep], indices=inv)
    res = gdf.copy()
    res["geometry"] = gpd.GeoSeries(out, index=gdf.index, crs=gdf.crs)
    return res.loc[~dropped].copy(), list(gdf.index[dropped])


# =============================================================================
# MEASURING
# =============================================================================

def _stream(layers, infos):
    paths = [i["paths"] for i in infos if len(i["paths"])]
    if not paths:
        return None
    all_paths = np.concatenate(paths)
    coords, pid = shapely.get_coordinates(all_paths, return_index=True)
    is_poly = np.concatenate([np.r_[np.ones(i["n_rings"], bool), np.zeros(len(i["paths"]) - i["n_rings"], bool)] for i in infos])
    size = np.concatenate([
        np.r_[shapely.area(i["parts"][i["poly_pos"]])[i["ring_poly"]] if i["n_rings"] else np.zeros(0),
              shapely.length(i["lines"]) if len(i["lines"]) else np.zeros(0)] for i in infos])
    # rank: polygons before lines, larger before smaller (stable on ties)
    order = np.lexsort((-size, ~is_poly))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return all_paths, coords, pid, is_poly, rank


def topology_report(layers, tol_m, geographic, kinds=None):
    """Counts of the defects Fix Map looks for (used before and after)."""
    kinds = kinds or [_kind(g) for g in layers]
    K = _metric_scale([g for g in layers], geographic)
    r = {}
    infos = [_layer_paths(g) for g in layers]
    st = _stream(layers, infos)
    if st is not None:
        _, coords, pid, is_poly, rank = st
        r["unmatched_nodes"], r["near_duplicate_nodes"] = _count_defects(coords * K, pid, rank, tol_m)
    poly_layers = [g for g, k in zip(layers, kinds) if k == "polygon"]
    for g, k in zip(layers, kinds):
        geoms = g.geometry.values
        ok = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
        if k == "polygon":
            r["polygons"] = int(ok.sum())
            r["invalid_polygons"] = int((~shapely.is_valid(geoms[ok])).sum())
            r["duplicate_polygons"] = int(pd.Series(shapely.to_wkb(shapely.normalize(geoms[ok]))).duplicated().sum())
            tree = shapely.STRtree(geoms[ok])
            area = shapely.area(geoms[ok])
            qi, ti = tree.query(geoms[ok], predicate="contains")
            r["polygons_inside_polygons"] = int(((qi != ti) & (area[qi] > area[ti])).sum())
            if ok.sum():
                gv = geoms[ok]
                gv = np.where(shapely.is_valid(gv), gv, shapely.make_valid(gv))
                u = _robust_union_all(gv, _grid(g), "union for the report")
                if u is None:
                    continue
                area = shapely.area(gv)
                scale = K[0] * K[1]
                r["_union"] = u
                r["overlap_area_m2"] = max(0.0, float((area.sum() - shapely.area(u)) * scale))
                holes = _gap_polygons(u)
                r["gaps"] = len(holes)
                r["gaps_area_m2"] = float(sum(shapely.area(h) for h in holes) * scale)
        elif k == "line":
            r["faults"] = int(ok.sum())
            r["duplicate_faults"] = int(pd.Series(shapely.to_wkb(shapely.normalize(geoms[ok]))).duplicated().sum())
    return r


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def fix_map(layers, tol_m, geographic=None, repair=True, node_crossings=True, min_inside=0.9, progress=None,
            sources=None, sliver_m=None, fill_gaps=True, overlap_rule="smaller", max_rounds=5,
            gaps_first=True):
    """
    Make polygon and fault layers topologically consistent within ``tol_m`` metres.

    Parameters
    ----------
    layers : list of GeoDataFrames (polygons and/or lines), same CRS.
    tol_m  : tolerance in metres (degrees are converted approximately).
    sources : optional list of the input file paths (listed in the report).
    overlap_rule : what to do with the overlaps that remain at the end (wider than the sliver width):
                "smaller" cuts them out of the bigger polygon (the smaller one stays whole), "larger" cuts them out
                of the smaller polygon, None leaves them.
    fill_gaps : at the end, fill every gap that is completely surrounded by polygons (whatever its size) with the
                neighbouring polygon that shares the longest boundary with it.
    sliver_m : width in metres below which a gap or overlap between polygons is a sliver and is closed
               (default: the tolerance).  Slivers wider than the tolerance are closed by trimming the smaller
               polygon (overlaps) or adding the gap to the neighbour with the longest shared boundary (gaps);
               they do not move any vertex.
    Returns (list of fixed GeoDataFrames in the same order, report text, statistics dict).
    """
    t_start = time.time()
    lines_out = []
    del _OVERLAY_FAILS[:]

    def log(s=""):
        lines_out.append(s)
        if progress:
            progress(s)

    layers = [g for g in layers if g is not None]
    check_same_crs(layers)
    if tol_m is None or tol_m <= 0:
        raise ValueError("The tolerance must be a positive number of metres.")
    if geographic is None:
        geographic = any(g.crs is not None and g.crs.is_geographic for g in layers)
    kinds = [_kind(g) for g in layers]
    K = _metric_scale(layers, geographic)
    pad = tol_m / K.min()
    sliver_m = max(float(sliver_m), float(tol_m)) if sliver_m else float(tol_m)
    stats = {}
    log("FIX MAP REPORT")
    log("=" * 60)
    log(f"Tolerance: {tol_m:g} m" + (" (degrees converted approximately: %.0f m/deg E-W, %.0f m/deg N-S)" % tuple(K) if geographic else ""))
    if sliver_m > tol_m:
        log(f"Sliver width: {sliver_m:g} m (gaps and overlaps narrower than this are closed)")
    for i, (g, k) in enumerate(zip(layers, kinds)):
        src = f"   [{sources[i]}]" if sources and i < len(sources) and sources[i] else ""
        log(f"Layer {i + 1}: {len(g)} {k} features, {shapely.get_num_coordinates(g.geometry.values).sum()} vertices{src}")
    log("")

    t0 = time.time()
    before = topology_report(layers, tol_m, geographic, kinds)
    log(f"Measured the starting state ({time.time() - t0:.1f} s)")

    work = [g.copy() for g in layers]
    lost = {}
    for i, (g, k) in enumerate(zip(work, kinds)):
        t0 = time.time()
        if k == "polygon":
            if repair:
                work[i], rinfo = repair_polygons(g)
                stats["repair"] = rinfo
                log(f"Repaired invalid polygons of layer {i + 1} ({time.time() - t0:.1f} s)")
                t0 = time.time()
            work[i], dinfo = remove_duplicates(work[i])
            stats["duplicate_polygons_removed"] = dinfo["removed"]
            log(f"Looked for duplicate polygons ({time.time() - t0:.1f} s)")
            t0 = time.time()
            work[i], cinfo = cut_contained_polygons(work[i], pad, min_inside)
            stats["containment"] = cinfo
            stats["dup_poly"] = dinfo
            log(f"Looked for polygons inside polygons ({time.time() - t0:.1f} s)")
            if fill_gaps and gaps_first:
                # every gap completely surrounded by polygons goes to the neighbour that shares the longest boundary,
                # before anything is snapped: the filled polygon takes over its neighbours' own vertices along the gap
                t0 = time.time()
                work[i], ginfo = fill_thin_gaps(work[i], None)
                if ginfo["gaps"]:
                    sg = stats.setdefault("start_gaps", {"gaps": 0, "area": 0.0, "examples": []})
                    sg["gaps"] += ginfo["gaps"]
                    sg["area"] += ginfo["area"] * K[0] * K[1]
                    sg["examples"] = (sg["examples"] + ginfo["examples"])[:10]
                log(f"Filled {ginfo['gaps']} gaps completely surrounded by polygons ({time.time() - t0:.1f} s)")
        elif k == "line":
            work[i], dinfo = remove_duplicates(g)
            stats["duplicate_faults_removed"] = dinfo["removed"]
            stats["dup_fault"] = dinfo
            log(f"Looked for duplicate faults ({time.time() - t0:.1f} s)")

    conform = {"passes": 0, "vertices_moved": 0, "nodes_inserted": 0}
    node_stats = {}
    max_shift = 0.0
    max_move = 0.0
    rounds = 0
    node_now = False                      # set once the geometry has settled: the next round adds the crossing nodes
    gaps_done = False
    prev_changed = {}                     # polygons changed in the previous round, to spot a repair/snap cycle
    node_rounds = 0
    for rounds in range(1, max_rounds + 5):     # max_rounds of conform + repair, then the settle and crossing-node rounds
        infos = [_layer_paths(g) for g in work]
        st = _stream(work, infos)
        if st is None:
            break
        _, coords, pid, is_poly, rank = st
        if node_now:
            # the geometry is already conformed: add the nodes where lines cross, then conform around them
            log("Adding nodes where lines cross:")
            t0 = time.time()
            c, p, ns = densify_paths(coords, pid, 1e12, is_poly, geographic=geographic, node_crossings=True,
                                     line_crossings=True)
            node_stats["nodes_added_at_crossings"] = (node_stats.get("nodes_added_at_crossings", 0)
                                                      + ns.get("nodes_added_at_crossings", 0))
            log(f"  {ns.get('nodes_added_at_crossings', 0)} nodes added ({time.time() - t0:.1f} s)")
            new_pts = c[~np.isin(c[:, 0] + 1j * c[:, 1], coords[:, 0] + 1j * coords[:, 1])]
            if len(new_pts):
                c, p, cs = _conform(c * K, p, rank, is_poly, float(tol_m), log=log, dirty=new_pts * K, cap=float(tol_m))
                c = c / K
                for k_ in conform:
                    conform[k_] += cs[k_]
        else:
            log(f"Conforming{' (round %d, after repairing polygons that became invalid)' % rounds if rounds > 1 else ''}:")
            c, p, cs = _conform(coords * K, pid, rank, is_poly, float(tol_m), log=log, cap=float(tol_m))
            for k_ in conform:
                conform[k_] += cs[k_]
            if rounds == 1:
                max_move = cs.get("max_move", 0.0)
            c = c / K
        alive = np.bincount(p, minlength=len(is_poly)) > 0
        if rounds == 1:                                  # largest shift of a vertex away from its own original path
            changed = np.flatnonzero(~np.isin(c[:, 0] + 1j * c[:, 1], coords[:, 0] + 1j * coords[:, 1]))
            if len(changed):
                changed = changed[:: max(1, len(changed) // 50000)]
                dd = shapely.distance(shapely.points(c[changed]), st[0][p[changed]])
                max_shift = float(dd.max() * K.mean())
        base = 0
        t0 = time.time()
        before_geoms = [g.geometry for g in work]
        for i, (g, info) in enumerate(zip(work, infos)):
            work[i], nd = _assemble(g, info, c, p, base, alive)
            base += len(info["paths"])
            if nd:
                lost.setdefault(i, []).extend(nd)
        log(f"  rebuilt the geometries ({time.time() - t0:.1f} s)")
        t0 = time.time()
        for i, (g, k) in enumerate(zip(work, kinds)):
            if k == "polygon":
                work[i], d2 = remove_duplicates(g)
                stats["duplicate_polygons_removed"] = stats.get("duplicate_polygons_removed", 0) + d2["removed"]
            elif k == "line":
                work[i], d2 = remove_duplicates(g)
                stats["duplicate_faults_removed"] = stats.get("duplicate_faults_removed", 0) + d2["removed"]
        log(f"  looked for duplicates ({time.time() - t0:.1f} s)")
        t0 = time.time()
        # snapping can leave a polygon self-intersecting: repair it and conform the neighbours to the repair
        again = False
        n_changed = 0
        for i, (g, k) in enumerate(zip(work, kinds)):
            if k == "polygon" and rounds < max_rounds + 1:
                # only polygons that differ from the previous round need checking
                old = before_geoms[i]
                new_g = g.geometry
                same_idx = new_g.index.isin(old.index)
                changed = np.ones(len(g), dtype=bool)
                if same_idx.any():
                    changed[same_idx] = ~shapely.equals_exact(old.loc[new_g.index[same_idx]].values,
                                                              new_g.values[same_idx], 0.0)
                only = np.flatnonzero(changed)
                n_changed += len(only)
                if len(only) == 0:
                    continue
                # the same polygons changing again and again: repairing them and snapping them undo each other
                labels = frozenset(g.index[only])
                if rounds > 1 and prev_changed.get(i) == labels:
                    stats.setdefault("cycling", set()).update(labels)
                    continue
                prev_changed[i] = labels
                if repair:
                    bad = only[~shapely.is_valid(g.geometry.values[only])]
                    if len(bad):
                        work[i], rinfo = repair_polygons(g)
                        stats.setdefault("repair_after", []).append(rinfo)
                        again = True
                        only = np.union1d(only, bad)
                # polygons that now lie inside another one (their boundaries have just been matched)
                work[i], cinfo = cut_contained_polygons(work[i], pad, min_inside, only=only)
                work[i], oinfo = resolve_thin_overlaps(work[i], sliver_m / K.max(), only=only)
                if oinfo["pairs"]:
                    so = stats.setdefault("thin_overlaps", {"pairs": 0, "area": 0.0, "examples": []})
                    so["pairs"] += oinfo["pairs"]
                    so["area"] += oinfo["area"] * K[0] * K[1]
                    so["examples"] = (so["examples"] + oinfo["examples"])[:10]
                    again = True
                if cinfo["holes_cut"]:
                    ci = stats["containment"]
                    ci["holes_cut"] += cinfo["holes_cut"]
                    ci["containers"] += cinfo["containers"]
                    ci["examples"] = (ci["examples"] + cinfo["examples"])[:10]
                    again = True
        log(f"  checked validity, polygons inside polygons and thin overlaps ({time.time() - t0:.1f} s)")
        # a late round that only touches a handful of polygons is the repair/snap cycle again: stop here
        if again and not node_now and rounds >= 3 and n_changed <= 25:
            again = False
            stats["stopped_early"] = (rounds, n_changed)
        out_of_rounds = rounds >= max_rounds
        if node_now:
            node_rounds += 1
            if again and not out_of_rounds and node_rounds < 2:
                node_now = False                         # something changed after the crossing nodes: settle, then node again
                continue
            break
        if not again or out_of_rounds:
            # settled (or out of rounds): remove the remaining overlaps and close the gaps, once, on the whole map
            if not gaps_done and (sliver_m > tol_m or fill_gaps or overlap_rule):
                gaps_done = True
                changed_settle = False
                t0 = time.time()
                for i, k in enumerate(kinds):
                    if k == "polygon" and overlap_rule:
                        work[i], winfo = remove_overlaps(work[i], overlap_rule)
                        if winfo["pairs"]:
                            sw_ = stats.setdefault("wide_overlaps", {"pairs": 0, "area": 0.0, "removed": [], "examples": []})
                            sw_["pairs"] += winfo["pairs"]
                            sw_["area"] += winfo["area"] * K[0] * K[1]
                            sw_["removed"] += winfo["removed"]
                            sw_["examples"] = (sw_["examples"] + winfo["examples"])[:10]
                            changed_settle = True
                log(f"  removed the remaining overlaps ({time.time() - t0:.1f} s)")
                t0 = time.time()
                for i, k in enumerate(kinds):
                    if k == "polygon":
                        for label, width in (("thin_gaps", sliver_m / K.max() if sliver_m > tol_m else -1),
                                             ("all_gaps", None if fill_gaps else -1)):
                            if width == -1:
                                continue
                            work[i], ginfo = fill_thin_gaps(work[i], width)
                            if ginfo["gaps"]:
                                sg = stats.setdefault(label, {"gaps": 0, "area": 0.0, "examples": []})
                                sg["gaps"] += ginfo["gaps"]
                                sg["area"] += ginfo["area"] * K[0] * K[1]
                                sg["examples"] = (sg["examples"] + ginfo["examples"])[:10]
                                changed_settle = True
                log(f"  closed gaps ({time.time() - t0:.1f} s)")
                if changed_settle:
                    continue                             # one more conform round gives the new edges shared nodes
            if node_crossings:
                node_now = True                          # settled: now add the crossing nodes
                continue
            break
        # otherwise something changed and rounds remain: conform again
    stats["rounds"] = rounds

    # whatever is still self-intersecting after the last round (e.g. polygons left alone because their repair and
    # the snapping kept undoing each other) is repaired now, so that no invalid polygon is delivered
    if repair:
        for i, (g, k) in enumerate(zip(work, kinds)):
            if k == "polygon" and (~shapely.is_valid(g.geometry.values)).any():
                work[i], rinfo = repair_polygons(g)
                stats["repair_final"] = rinfo
                log(f"Repaired {rinfo['repaired']} polygons that were still self-intersecting at the end")

    # the same for faults: a vertex snapped onto a node and that node added to a nearby segment of the same fault
    # make the fault pass through the node twice, with a small loop in between; cut such loops out
    for i, (g, k) in enumerate(zip(work, kinds)):
        if k == "line":
            work[i], linfo = remove_line_loops(g, 2.0 * float(tol_m), K, near_m=float(tol_m))
            if linfo["loops"] or linfo["left"]:
                sl = stats.setdefault("line_loops", {"loops": 0, "indices": [], "left": []})
                for key in sl:
                    sl[key] += linfo[key]
            if linfo["loops"]:
                log(f"Cut {linfo['loops']} small loops out of {len(linfo['indices'])} faults that passed through a node twice")

    # the last conform rounds, the crossing nodes and the final repair can move an edge by up to the tolerance (or add
    # a loop back), which can open a small new overlap after the settle step: one last sweep with the same rule
    if overlap_rule:
        t0 = time.time()
        for i, k in enumerate(kinds):
            if k == "polygon":
                work[i], winfo = remove_overlaps(work[i], overlap_rule)
                if winfo["pairs"]:
                    sw_ = stats.setdefault("wide_overlaps", {"pairs": 0, "area": 0.0, "removed": [], "examples": []})
                    sw_["pairs"] += winfo["pairs"]
                    sw_["area"] += winfo["area"] * K[0] * K[1]
                    sw_["removed"] += winfo["removed"]
                    sw_["examples"] = (sw_["examples"] + winfo["examples"])[:10]
                    log(f"Final sweep: cut {winfo['pairs']} small overlaps that the last rounds had opened ({time.time() - t0:.1f} s)")

    t0 = time.time()
    after = topology_report(work, tol_m, geographic, kinds)
    log(f"Measured the final state ({time.time() - t0:.1f} s)")
    log("")

    # ---- report ------------------------------------------------------------
    log("WHAT WAS DONE")
    log("-" * 60)
    if "repair" in stats:
        r = stats["repair"]
        log(f"Invalid polygons repaired (buffer 0): {r['repaired']} of {r['invalid']}" +
            (f"; could not repair: {r['failed']}" if r["failed"] else ""))
        if r["indices"]:
            log(f"  feature index: {r['indices']}")
    if "dup_poly" in stats or "duplicate_polygons_removed" in stats:
        log(f"Duplicate polygons removed: {stats.get('duplicate_polygons_removed', 0)}" +
            (f"  (index {stats['dup_poly']['indices']})" if stats.get("dup_poly", {}).get("indices") else ""))
    if "dup_fault" in stats or "duplicate_faults_removed" in stats:
        log(f"Duplicate faults removed: {stats.get('duplicate_faults_removed', 0)}" +
            (f"  (index {stats['dup_fault']['indices']})" if stats.get("dup_fault", {}).get("indices") else ""))
    if "containment" in stats:
        ci = stats["containment"]
        log(f"Polygons inside other polygons: {ci['holes_cut']} holes cut in {ci['containers']} polygons")
        for cont, inner in ci["examples"]:
            log(f"  polygon {cont} now has a hole for polygon(s) {inner}")
    log(f"Conform passes: {conform['passes']};  vertices moved onto other lines/nodes: {conform['vertices_moved']};  "
        f"nodes added to lines: {conform['nodes_inserted']}")
    if node_stats:
        log(f"Nodes added where lines cross / touch: {node_stats.get('nodes_added_at_crossings', 0)}")
    if max_move:
        log(f"Largest distance a vertex was moved in the first round (limit {tol_m:g} m): {max_move:.2f} m")
    if max_shift > tol_m:
        log(f"Largest distance of a new line from the original line: {max_shift:.1f} m (a few places where vertices that "
            f"snapped onto the same nodes left a straight chord instead of a short detour)")
    if "thin_gaps" in stats:
        sg = stats["thin_gaps"]
        log(f"Gaps narrower than {sliver_m:g} m added to the neighbour with the longest shared boundary: "
            f"{sg['gaps']} ({sg['area']:,.0f} m2)")
    if "wide_overlaps" in stats:
        sw_ = stats["wide_overlaps"]
        side = "bigger" if overlap_rule == "smaller" else "smaller"
        log(f"Remaining overlaps cut out of the {side} polygon (the {overlap_rule} one stays whole): "
            f"{sw_['pairs']} pairs, {sw_['area']:,.0f} m2")
        for loser, winner in sw_["examples"]:
            log(f"  polygon {loser} lost the part it shared with polygon {winner}")
        if sw_["removed"]:
            log(f"  {len(sw_['removed'])} polygons were entirely overlapped and removed; index {sw_['removed'][:20]}")
    if "start_gaps" in stats:
        sg = stats["start_gaps"]
        log(f"Gaps completely surrounded by polygons filled at the start with the neighbour that shares the longest "
            f"boundary: {sg['gaps']} ({sg['area']:,.0f} m2)")
    if "all_gaps" in stats:
        sg = stats["all_gaps"]
        log(f"Other gaps completely surrounded by polygons filled with the neighbour that shares the longest boundary: "
            f"{sg['gaps']} ({sg['area']:,.0f} m2)")
    if "stopped_early" in stats:
        r_, n_ = stats["stopped_early"]
        log(f"Stopped repairing after round {r_}: only {n_} polygons were still changing (the repair/snap cycle); "
            f"anything still invalid was repaired at the end.")
    if "thin_overlaps" in stats:
        so = stats["thin_overlaps"]
        log(f"Overlaps narrower than {sliver_m:g} m removed from the smaller polygon: "
            f"{so['pairs']} ({so['area']:,.1f} m2)")
        for small, big in so["examples"]:
            log(f"  polygon {small} trimmed where it overlapped polygon {big}")
    if stats.get("cycling"):
        cyc = sorted(str(x) for x in stats["cycling"])
        log(f"Polygons whose repair and snapping kept undoing each other were left as they are: {cyc[:20]}"
            + (f" and {len(cyc) - 20} more" if len(cyc) > 20 else ""))
    if "repair_final" in stats:
        r = stats["repair_final"]
        log(f"Polygons still self-intersecting after the last round, repaired at the end (a loop removed or a node "
            f"added where the ring crosses itself): {r['repaired']} of {r['invalid']}; feature index {r['indices']}")
    for r in stats.get("repair_after", []):
        log(f"Polygons that became invalid while snapping and were repaired again: {r['repaired']} of {r['invalid']}")
    if stats.get("line_loops", {}).get("loops"):
        sl = stats["line_loops"]
        log(f"Small loops cut out of faults that passed through the same node twice (all within {2 * tol_m:g} m of "
            f"that node; nothing moved): {sl['loops']} in {len(sl['indices'])} faults; feature index {sl['indices'][:20]}")
    if _OVERLAY_FAILS:
        uniq_f = sorted(set(_OVERLAY_FAILS))
        log(f"{len(_OVERLAY_FAILS)} geometry operations were skipped because GEOS could not do them even after repair and "
            f"snap-rounding (features involved, as index/index): {uniq_f[:20]}" + (f" and {len(uniq_f) - 20} more" if len(uniq_f) > 20 else ""))
    union_after = after.pop("_union", None)
    before.pop("_union", None)
    for i, nd in lost.items():
        nd = [x for x in nd if x in layers[i].index]
        covered, narrow = [], []
        if kinds[i] == "polygon" and union_after is not None:
            gl = layers[i].loc[nd].geometry.values
            ar = shapely.area(gl)
            try:
                cov = np.where(ar > 0, shapely.area(shapely.intersection(gl, union_after)) / np.where(ar > 0, ar, 1), 1.0)
            except shapely.errors.GEOSException:
                cov = np.ones(len(gl))
            for x, cv, a_ in zip(nd, cov, ar):
                (covered if cv >= 0.99 else narrow).append((x, a_ * K[0] * K[1]))
        else:
            narrow = [(x, 0.0) for x in nd]
        if covered:
            log(f"Layer {i + 1}: {len(covered)} polygons were completely covered by other polygons once those were cut out of them "
                f"(or were slivers inside them) and were removed; the area is still mapped by the other polygons "
                f"(together {sum(a for _, a in covered):,.0f} m2 in the input). Index {[x for x, _ in covered][:20]}")
        if narrow:
            log(f"Layer {i + 1}: {len(narrow)} features collapsed to nothing (narrower than the tolerance) and were removed; "
                f"index {[x for x, _ in narrow][:20]}")
    still = []
    for g, k in zip(work, kinds):
        if k == "polygon":
            geoms = g.geometry.values
            bad = np.flatnonzero(~shapely.is_valid(geoms))
            still += [(g.index[j], shapely.is_valid_reason(geoms[j])) for j in bad]
    if still:
        log("Polygons still invalid (edit by hand): " + "; ".join(f"{i}: {r}" for i, r in still[:10]))
    left_faults = stats.get("line_loops", {}).get("left", [])
    if left_faults:
        log(f"Faults that still cross or touch themselves (edit by hand): {len(left_faults)}; "
            f"feature index {left_faults[:20]}")
    log("")
    log("BEFORE -> AFTER")
    log("-" * 60)
    rows = [("unmatched_nodes", "Nodes lying on another line but not shared with it"),
            ("near_duplicate_nodes", "Pairs of nodes closer than the tolerance, not shared"),
            ("invalid_polygons", "Invalid polygons"),
            ("duplicate_polygons", "Duplicate polygons"),
            ("duplicate_faults", "Duplicate faults"),
            ("polygons_inside_polygons", "Polygons inside a polygon without a hole"),
            ("gaps", "Gaps between polygons (holes in the map)"),
            ("gaps_area_m2", "  total gap area (m2)"),
            ("overlap_area_m2", "Polygon overlap area (m2)"),
            ("polygons", "Polygons"), ("faults", "Faults")]
    for key, label in rows:
        if key in before or key in after:
            b, a = before.get(key, "-"), after.get(key, "-")
            fmt = (lambda v: f"{v:,.1f}") if "area" in key else (lambda v: f"{v:,}" if isinstance(v, int) else str(v))
            log(f"{label:58s} {fmt(b):>14s} -> {fmt(a):>14s}")
    log("")
    left = [lbl for k, lbl in rows[:6] if after.get(k, 0)]
    ga, gb = after.get("gaps_area_m2", 0), before.get("gaps_area_m2", 0)
    oa = after.get("overlap_area_m2", 0)
    if oa > 1 or ga - gb > 1:
        log(f"Note: overlap area is now {oa:,.1f} m2 and gap area changed by {ga - gb:+,.0f} m2. Overlaps wider than the tolerance are not")
        log("changed; gaps can also appear where a polygon narrower than the tolerance collapsed (listed above).")
    if left:
        log("Not fully resolved: " + "; ".join(left) + ". Features that differ by more than the tolerance are never changed;")
        log("raise the tolerance or edit those by hand.")
    else:
        log("No remaining nodes, duplicates or containments were found at this tolerance.")
    log(f"Total time {time.time() - t_start:.1f} s")
    stats.update({"before": before, "after": after, "conform": conform, "max_shift_m": max_shift, "max_move_m": max_move})
    return work, "\n".join(lines_out), stats
