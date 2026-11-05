# -*- coding: utf-8 -*-
"""
Conforming densification of polygons and polylines.

Inserts nodes so that no segment is longer than a chosen spacing, in a way that keeps the
data topologically consistent: every new node is created once per *segment* (not once per
polygon), so the polygons on both sides of a shared boundary - and any fault that runs along
it - get exactly the same new nodes.  Densifying each polygon separately does not do this:
floating-point differences between the two directions in which a shared edge is traversed
leave the two copies with slightly different nodes.

Optionally (``node_crossings``) nodes are also added where lines cross polygon boundaries, and
where a vertex lies on the inside of another feature's segment (a T-junction), so that the
features share a node there as well.

The shapes are not changed: new nodes lie on the existing straight segments.

The algorithm (everything is vectorised numpy, O(V log V)):

1. Flatten all polygon rings and lines of all layers into one coordinate stream, and weld
   vertices that coincide to within rounding noise.
2. Take the segments, put the end points of each in a fixed order (lexicographic) and keep the
   unique ones.  A boundary shared by two polygons, or run along by a fault, is the same row.
3. (optional) Find nodes to add to unique segments: vertices lying inside another segment
   (T-junctions) and proper crossings between a polygon boundary and any other segment.  Each
   such node is computed once and inserted in both segments.
4. Split every unique segment at those nodes and subdivide each piece into
   ``ceil(length / spacing)`` equal parts.  The new nodes are computed once per piece, always
   in the fixed (lo -> hi) direction.
5. Rebuild every ring and line by reading each segment's nodes in the direction in which that
   ring or line runs.
"""

import numpy as np
import geopandas as gpd
import shapely

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_AT_EQUATOR = 111320.0


# =============================================================================
# SPLITTING LAYERS INTO PATHS AND PUTTING THEM BACK TOGETHER
# =============================================================================

def _layer_paths(gdf):
    """The rings (of polygons) and lines of one layer, with what is needed to rebuild it."""
    geoms = gdf.geometry.values
    parts, feat = shapely.get_parts(geoms, return_index=True)
    tid = shapely.get_type_id(parts)
    is_poly = tid == 3
    is_line = (tid == 1) | (tid == 2)
    pos = np.arange(len(parts))
    rings, ring_poly = shapely.get_rings(parts[is_poly], return_index=True)
    lines = parts[is_line]
    return {
        "geoms": geoms, "parts": parts, "feat": feat,
        "poly_pos": pos[is_poly], "line_pos": pos[is_line],
        "rings": rings, "ring_poly": ring_poly, "lines": lines,
        "paths": np.concatenate([rings, lines]) if (len(rings) or len(lines)) else rings,
        "n_rings": len(rings),
    }


def _rebuild_layer(gdf, info, coords, path_id, base):
    """New geometries of a layer from the densified coordinates of its paths."""
    n_paths = len(info["paths"])
    sel = (path_id >= base) & (path_id < base + n_paths)
    c, p = coords[sel], path_id[sel] - base
    geoms = info["geoms"]
    parts = info["parts"]
    new_parts = parts.copy()

    n_rings = info["n_rings"]
    if n_rings:
        ring_sel = p < n_rings
        new_rings = shapely.linearrings(c[ring_sel], indices=p[ring_sel])
        rp = info["ring_poly"]
        first = np.r_[True, rp[1:] != rp[:-1]]
        shell_pos = np.flatnonzero(first)
        polys = shapely.polygons(new_rings[shell_pos])            # shells, no holes yet
        ends = np.r_[shell_pos[1:], len(new_rings)]
        for q in np.flatnonzero(ends - shell_pos > 1):            # polygons with holes
            polys[q] = shapely.Polygon(new_rings[shell_pos[q]],
                                       list(new_rings[shell_pos[q] + 1:ends[q]]))
        # an empty polygon has no rings, so keep it (empty) in its own place
        keep = info["poly_pos"].copy()
        ordered = parts[keep].copy()
        ordered[rp[shell_pos]] = polys
        new_parts[info["poly_pos"]] = ordered
    if len(info["lines"]):
        line_sel = p >= n_rings
        new_parts[info["line_pos"]] = shapely.linestrings(c[line_sel], indices=p[line_sel] - n_rings)

    # parts -> features
    out = geoms.copy()
    feat = info["feat"]
    otype = shapely.get_type_id(geoms)
    first_part = np.searchsorted(feat, np.arange(len(geoms)))
    single = ((otype == 3) | (otype == 1) | (otype == 2)) & ~shapely.is_empty(geoms)
    if single.any():
        out[single] = new_parts[first_part[single]]
    for type_id, make in ((6, shapely.multipolygons), (5, shapely.multilinestrings)):
        feats = np.flatnonzero(otype == type_id)
        if len(feats):
            keep = np.isin(feat, feats)
            uf, inv = np.unique(feat[keep], return_inverse=True)
            out[uf] = make(new_parts[keep], indices=inv)
    res = gdf.copy()
    res["geometry"] = gpd.GeoSeries(out, index=gdf.index, crs=gdf.crs)
    return res


# =============================================================================
# THE CORE: NODES ON UNIQUE SEGMENTS
# =============================================================================

def _weld(coords, tol):
    """
    Merge vertices closer together than ``tol`` into one exact coordinate.

    Nodes that are shared between features are often shared only to within floating-point
    noise (the same corner written 2e-8 m apart by different polygons).  Such nodes are not
    really shared and no densification can make their neighbours agree, so they are welded
    first: every cluster of vertices within ``tol`` of each other gets the coordinates of its
    lexicographically first member.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    uv, inv = np.unique(coords, axis=0, return_inverse=True)
    inv = inv.ravel()
    if len(uv) < 2:
        return coords, 0
    pairs = cKDTree(uv).query_pairs(tol, output_type="ndarray")
    if len(pairs) == 0:
        return coords, 0
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(uv), len(uv)))
    ncomp, labels = connected_components(graph, directed=False)
    _, first = np.unique(labels, return_index=True)
    rep = uv[first[labels]]
    moved = int(np.any(rep != uv, axis=1).sum())
    return rep[inv], moved


def _segment_length_m(lo, hi, geographic):
    """Approximate length in metres of segments given in CRS units (degrees if geographic)."""
    dx = hi[:, 0] - lo[:, 0]
    dy = hi[:, 1] - lo[:, 1]
    if geographic:
        mid_lat = np.radians(0.5 * (lo[:, 1] + hi[:, 1]))
        return np.hypot(dx * M_PER_DEG_LON_AT_EQUATOR * np.cos(mid_lat), dy * M_PER_DEG_LAT)
    return np.hypot(dx, dy)


def _insertions(ulo, uhi, u_is_poly, snap_tol, line_crossings=False):
    """
    Nodes to add inside unique segments: T-junctions and crossings with polygon boundaries.
    Returns (segment, t along lo->hi, x, y) arrays, sorted by segment and t, de-duplicated.
    """
    n = len(ulo)
    segs = shapely.linestrings(np.stack([ulo, uhi], axis=1))
    tree = shapely.STRtree(segs)
    d = uhi - ulo
    seg_len2 = (d ** 2).sum(axis=1)

    seg_l, t_l, x_l, y_l = [], [], [], []

    # (a) vertices lying inside another segment (T-junctions)
    allv = np.concatenate([ulo, uhi])
    uv, inv_v = np.unique(allv, axis=0, return_inverse=True)
    inv_v = inv_v.ravel()
    v_is_poly = np.zeros(len(uv), dtype=bool)
    np.logical_or.at(v_is_poly, inv_v, np.concatenate([u_is_poly, u_is_poly]))
    vi, si = tree.query(shapely.points(uv), predicate="dwithin", distance=snap_tol)
    if len(vi):
        v = uv[vi]
        ok = (u_is_poly[si] | v_is_poly[vi]) if not line_crossings else np.ones(len(si), dtype=bool)  # not line-against-line only
        ok &= (np.abs(v - ulo[si]).max(axis=1) > snap_tol) & (np.abs(v - uhi[si]).max(axis=1) > snap_tol)
        vi, si, v = vi[ok], si[ok], v[ok]
        t = ((v - ulo[si]) * d[si]).sum(axis=1) / seg_len2[si]
        inside = (t > 0) & (t < 1)
        seg_l.append(si[inside]); t_l.append(t[inside]); x_l.append(v[inside, 0]); y_l.append(v[inside, 1])

    # (b) proper crossings (interior point of both segments), at least one a polygon boundary
    pairs = tree.query(segs, predicate="crosses")
    i, j = pairs[0], pairs[1]
    keep = (i < j) & ((u_is_poly[i] | u_is_poly[j]) if not line_crossings else True)
    i, j = i[keep], j[keep]
    if len(i):
        p, r = ulo[i], d[i]
        q, s = ulo[j], d[j]
        denom = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
        good = np.abs(denom) > 0
        i, j, p, r, q, s, denom = i[good], j[good], p[good], r[good], q[good], s[good], denom[good]
        qp = q - p
        ti = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / denom
        tj = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / denom
        pt = p + ti[:, None] * r                                  # computed once, inserted in both
        # a crossing on top of an existing vertex of either segment is a T-junction, handled in (a)
        away = np.ones(len(i), dtype=bool)
        for seg in (i, j):
            away &= (np.abs(pt - ulo[seg]).max(axis=1) > snap_tol) & (np.abs(pt - uhi[seg]).max(axis=1) > snap_tol)
        i, j, ti, tj, pt = i[away], j[away], ti[away], tj[away], pt[away]
        for seg, tt in ((i, ti), (j, tj)):
            m = (tt > 0) & (tt < 1)
            seg_l.append(seg[m]); t_l.append(tt[m]); x_l.append(pt[m, 0]); y_l.append(pt[m, 1])

    if not seg_l or sum(len(a) for a in seg_l) == 0:
        z = np.zeros(0)
        return np.zeros(0, dtype=np.int64), z, z, z
    seg = np.concatenate(seg_l); t = np.concatenate(t_l); x = np.concatenate(x_l); y = np.concatenate(y_l)
    # de-duplicate (the same node arrives from several pairs), then sort along each segment
    key = np.stack([seg.astype(float), np.round(x / snap_tol), np.round(y / snap_tol)], axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    seg, t, x, y = seg[first], t[first], x[first], y[first]
    order = np.lexsort((t, seg))
    return seg[order], t[order], x[order], y[order]


def densify_paths(coords, path_id, spacing_m, path_is_poly, geographic=False,
                  node_crossings=True, snap_tol=None, line_crossings=False):
    """
    Densify a stream of paths.  ``coords`` (n, 2), ``path_id`` (n,) non-decreasing,
    ``path_is_poly`` (number of paths,) bool.  Returns (new coords, new path ids, stats).
    """
    if snap_tol is None:
        snap_tol = 1e-10 if geographic else 1e-6
    coords, n_welded = _weld(coords, snap_tol)
    n = len(coords)
    has_next = np.r_[path_id[:-1] == path_id[1:], False]
    vpos = np.flatnonzero(has_next)                           # vertices that start a segment
    A = coords[vpos]
    B = coords[vpos + 1]
    flip = (A[:, 0] > B[:, 0]) | ((A[:, 0] == B[:, 0]) & (A[:, 1] > B[:, 1]))
    lo = np.where(flip[:, None], B, A)
    hi = np.where(flip[:, None], A, B)
    uniq, inv = np.unique(np.concatenate([lo, hi], axis=1), axis=0, return_inverse=True)
    inv = inv.ravel()
    ulo, uhi = uniq[:, :2], uniq[:, 2:]
    U = len(uniq)
    u_is_poly = np.zeros(U, dtype=bool)
    np.logical_or.at(u_is_poly, inv, path_is_poly[path_id[vpos]])

    # nodes to add inside unique segments
    if node_crossings:
        ins_seg, ins_t, ins_x, ins_y = _insertions(ulo, uhi, u_is_poly, snap_tol, line_crossings)
    else:
        ins_seg = np.zeros(0, dtype=np.int64); ins_x = ins_y = np.zeros(0)
    m_u = np.bincount(ins_seg, minlength=U)                   # added nodes per segment
    off_ins = np.cumsum(m_u) - m_u

    # atoms: the pieces between consecutive nodes of a segment
    n_atoms_u = m_u + 1
    atom_u = np.repeat(np.arange(U), n_atoms_u)
    first_atom = np.cumsum(n_atoms_u) - n_atoms_u
    atom_k = np.arange(len(atom_u)) - first_atom[atom_u]
    ins_xy = np.stack([ins_x, ins_y], axis=1)
    alo = np.where((atom_k == 0)[:, None], ulo[atom_u], ins_xy[np.minimum(off_ins[atom_u] + atom_k - 1, max(len(ins_xy) - 1, 0))] if len(ins_xy) else ulo[atom_u])
    ahi = np.where((atom_k == m_u[atom_u])[:, None], uhi[atom_u], ins_xy[np.minimum(off_ins[atom_u] + atom_k, max(len(ins_xy) - 1, 0))] if len(ins_xy) else uhi[atom_u])
    notlast = (atom_k < m_u[atom_u]).astype(np.int64)

    # subdivide every atom, once, in its fixed lo -> hi direction
    L = _segment_length_m(alo, ahi, geographic)
    kk = np.maximum(1, np.ceil(L / spacing_m)).astype(np.int64)
    inner_n = kk - 1
    count = inner_n + notlast
    start = np.cumsum(count) - count
    EP = np.empty((int(count.sum()), 2))
    rep = np.repeat(np.arange(len(atom_u)), inner_n)
    j = np.arange(len(rep)) - np.repeat(np.cumsum(inner_n) - inner_n, inner_n) + 1
    EP[start[rep] + j - 1] = alo[rep] + (j / kk[rep])[:, None] * (ahi[rep] - alo[rep])
    has_end = notlast == 1
    EP[(start + inner_n)[has_end]] = ahi[has_end]
    e_u = np.bincount(atom_u, weights=count, minlength=U).astype(np.int64)   # nodes emitted per segment
    off_u = start[first_atom]

    # rebuild every path: each vertex is followed by the nodes of the segment that starts at it
    c = np.zeros(n, dtype=np.int64)
    c[vpos] = e_u[inv]
    fl = np.zeros(n, dtype=bool)
    fl[vpos] = flip
    ub = np.zeros(n, dtype=np.int64)
    ub[vpos] = inv
    place = np.cumsum(1 + c) - (1 + c)
    out = np.empty((int((1 + c).sum()), 2))
    out[place] = coords
    v = np.repeat(np.arange(n), c)
    m = np.arange(len(v)) - np.repeat(np.cumsum(c) - c, c)
    take = off_u[ub[v]] + np.where(fl[v], c[v] - 1 - m, m)    # reversed where the path runs against lo -> hi
    out[place[v] + 1 + m] = EP[take]
    stats = {"unique_segments": int(U), "nodes_added_at_crossings": int(len(ins_seg)),
             "vertices_welded": n_welded,
             "vertices_in": int(n), "vertices_out": int(len(out))}
    return out, np.repeat(path_id, 1 + c), stats


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def check_same_crs(layers):
    """Fix Map and Densify compare coordinates between layers, so the layers must share a coordinate system."""
    crs = [g.crs for g in layers if g is not None and g.crs is not None]
    if len(crs) > 1 and any(not c.equals(crs[0]) for c in crs[1:]):
        names = ", ".join(c.to_string() for c in crs)
        raise ValueError("The polygon and fault layers are in different coordinate systems (" + names + "). "
                         "Reproject one of them so that both use the same coordinate system, then run again.")


def densify_layers(layers, spacing_m, geographic=None, node_crossings=True, snap_tol=None):
    """
    Densify polygon and line layers together so that the new nodes are shared.

    Parameters
    ----------
    layers         : list of GeoDataFrames (polygons and/or lines, in the same CRS).
    spacing_m      : the largest segment length wanted, in metres.  Each segment is divided
                     into ``ceil(length / spacing_m)`` equal parts, so new nodes are between
                     ``spacing_m / 2`` and ``spacing_m`` apart.  In a geographic CRS lengths
                     are converted from degrees to metres approximately (111 km per degree of
                     latitude, scaled by cos(latitude) for longitude).
    geographic     : treat coordinates as degrees (default: taken from the layers' CRS).
    node_crossings : also add shared nodes where a line or polygon boundary crosses a polygon
                     boundary, and where a vertex lies inside another feature's segment.
    snap_tol       : tolerance (CRS units) used to weld vertices that are the same corner to within
                     rounding noise, and to decide that a vertex lies on a segment.  Default
                     1e-6 (1e-10 degrees for a geographic CRS).

    Returns
    -------
    (list of densified GeoDataFrames in the same order, dict of statistics)
    """
    if spacing_m is None or spacing_m <= 0:
        raise ValueError("The node spacing must be a positive number of metres.")
    layers = [g for g in layers if g is not None]
    check_same_crs(layers)
    if geographic is None:
        geographic = any(g.crs is not None and g.crs.is_geographic for g in layers)
    infos = [_layer_paths(g) for g in layers]
    all_paths = np.concatenate([i["paths"] for i in infos if len(i["paths"])]) if any(len(i["paths"]) for i in infos) else None
    if all_paths is None:
        return [g.copy() for g in layers], {"vertices_in": 0, "vertices_out": 0}
    path_is_poly = np.concatenate([
        np.r_[np.ones(i["n_rings"], dtype=bool), np.zeros(len(i["paths"]) - i["n_rings"], dtype=bool)]
        for i in infos])
    coords, path_id = shapely.get_coordinates(all_paths, return_index=True)
    new_coords, new_pid, stats = densify_paths(coords, path_id, float(spacing_m), path_is_poly,
                                               geographic=geographic, node_crossings=node_crossings,
                                               snap_tol=snap_tol)
    out, base = [], 0
    for g, info in zip(layers, infos):
        out.append(_rebuild_layer(g, info, new_coords, new_pid, base))
        base += len(info["paths"])
    return out, stats
