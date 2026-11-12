# -*- coding: utf-8 -*-
"""
Conforming node thinning of polygons and polylines (the opposite of Densify).

Removes nodes that are closer together than a chosen distance, in a way that keeps the data
topologically consistent: a node is removed from *every* feature that uses it at once, so the
polygons on both sides of a shared boundary - and any fault running along it - lose exactly the
same nodes and stay coincident.

A node is removed only if all of these hold:

* it is an ordinary node along a boundary or line: exactly two distinct neighbours in the
  network of all polygon boundaries and lines.  Triple junctions (three or more polygons),
  places where a fault meets, leaves or crosses a boundary or another fault, the end points of
  lines, and nodes where a ring touches itself are never removed;
* it is closer than the threshold to one of its two neighbours;
* removing it does not create a self-intersection: the new segment joining its two
  neighbours does not cross or overlap any other segment of any polygon or line, and the
  triangle cut off (neighbour - node - neighbour) contains no other node, so no feature ends up
  on the wrong side of a boundary.  A ring is never reduced below three nodes.

The algorithm works in rounds (everything vectorised with numpy/shapely):

1. Flatten all polygon rings and lines of all layers into one coordinate stream, weld nodes
   that coincide to within rounding noise, and build the network of unique segments.
2. Fix the nodes that must stay (above), and one node of every closed ring that touches
   nothing else (so that each ring keeps an anchor).
3. Each round, the candidates are the removable nodes with a neighbour closer than the
   threshold.  Of these, a node whose shorter segment is shorter than that of both its
   neighbours is taken (so no two neighbouring nodes go in the same round), checked for
   intersections against the current network, and removed if it passes.  A node that fails is
   set aside until one of its neighbours is removed.
4. Rebuild every ring and line, keeping the nodes that are left.
"""

import numpy as np
import shapely

from .Densify import _layer_paths, _rebuild_layer, _weld, _segment_length_m, check_same_crs


def _neighbours(E, n):
    """For each node, its first two neighbours in the edge list E (-1 where there is none), and its degree."""
    ends = np.concatenate([E[:, 0], E[:, 1]])
    others = np.concatenate([E[:, 1], E[:, 0]])
    order = np.argsort(ends, kind="stable")
    ends, others = ends[order], others[order]
    deg = np.bincount(ends, minlength=n)
    first = np.cumsum(deg) - deg
    nb = np.full((n, 2), -1, dtype=np.int64)
    has1 = deg >= 1
    nb[has1, 0] = others[first[has1]]
    has2 = deg >= 2
    nb[has2, 1] = others[first[has2] + 1]
    return nb, deg


def _unique_edges(a, b):
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keep = lo != hi
    E = np.stack([lo[keep], hi[keep]], axis=1)
    return np.unique(E, axis=0) if len(E) else E.reshape(0, 2)


def _query(small, big, predicate):
    """Pairs (i in small, j in big) with predicate(small[i], big[j]); the tree is built on the smaller set."""
    if len(small) <= len(big):
        j, i = shapely.STRtree(small).query(big, predicate=predicate)
    else:
        i, j = shapely.STRtree(big).query(small, predicate=predicate)
    return i, j


def _valid_removals(V, pts, E, segs, alive, v, a, b):
    """
    Which of the removals (a - v - b  ->  a - b) leave the network free of self-intersections.
    Checked against the current network (edges E with geometries ``segs``; node points ``pts``):
    the new segment must not share interior points with any other segment, and the triangle
    a, v, b must contain no other live node (also on its edges).
    """
    k = len(v)
    ok = np.ones(k, dtype=bool)
    new_segs = shapely.linestrings(np.stack([V[a], V[b]], axis=1))

    # (1) the new segment against every current segment except the two it replaces
    qi, ei = _query(new_segs, segs, "intersects")
    if len(qi):
        e0, e1 = E[ei, 0], E[ei, 1]
        own = (e0 == v[qi]) | (e1 == v[qi])                    # a-v and v-b
        qi, ei = qi[~own], ei[~own]
        if len(qi):
            bad = shapely.relate_pattern(new_segs[qi], segs[ei], "T********")   # interiors meet
            ok[np.unique(qi[bad])] = False

    # (2) no live node inside or on the triangle a, v, b (other than a, v, b)
    #     (a flat triangle - a, v, b in line - is tested as the line a-v-b)
    live = np.flatnonzero(alive)
    tris = shapely.polygons(np.stack([V[a], V[v], V[b], V[a]], axis=1))
    flat = ~(shapely.area(tris) > 0)
    if flat.any():
        tris[flat] = shapely.linestrings(np.stack([V[a[flat]], V[v[flat]], V[b[flat]]], axis=1))
    ti, pi = _query(tris, pts[live], "intersects")
    if len(ti):
        p = live[pi]
        other = (p != a[ti]) & (p != v[ti]) & (p != b[ti])
        ok[np.unique(ti[other])] = False

    # (3) two removals in the same round must not create the same segment (a ring collapsing to a-b-a)
    idx = np.flatnonzero(ok)
    if len(idx) > 1:
        key = np.stack([np.minimum(a[idx], b[idx]), np.maximum(a[idx], b[idx])], axis=1)
        _, first = np.unique(key, axis=0, return_index=True)
        dup = np.ones(len(idx), dtype=bool)
        dup[first] = False
        ok[idx[dup]] = False
    return ok


def thin_paths(coords, path_id, min_dist_m, path_is_poly, path_poly_id, geographic=False,
               snap_tol=None, progress=None):
    """
    Thin a stream of paths.  ``coords`` (n, 2), ``path_id`` (n,) non-decreasing, ``path_is_poly``
    and ``path_poly_id`` (number of paths,): whether each path is a polygon ring, and a polygon
    number shared by all rings of one polygon (-1 for lines).
    Returns (new coords, new path ids, stats).
    """
    if snap_tol is None:
        snap_tol = 1e-10 if geographic else 1e-6
    # Welding is used only to decide which nodes are the same node; the surviving nodes are written
    # at their original coordinates, so nothing moves (welding can join two visits of a ring to a
    # point a few nanometres apart and make a valid polygon touch itself).
    orig = coords
    coords, n_welded = _weld(coords, snap_tol)
    n_in = len(coords)
    V, uid = np.unique(coords, axis=0, return_inverse=True)
    uid = uid.ravel()
    nV = len(V)

    same_next = np.r_[path_id[:-1] == path_id[1:], False]
    first_of_path = np.r_[True, path_id[1:] != path_id[:-1]]
    last_of_path = np.r_[path_id[:-1] != path_id[1:], True]
    E = _unique_edges(uid[:-1][same_next[:-1]], uid[1:][same_next[:-1]])
    nb, deg = _neighbours(E, nV)

    # --- nodes that must stay ------------------------------------------------
    fixed = deg != 2
    is_line_pt = ~path_is_poly[path_id]
    fixed[uid[is_line_pt & (first_of_path | last_of_path)]] = True     # line end points
    # nodes shared by three or more polygons (normally already degree > 2)
    on_ring = path_is_poly[path_id]
    pairs = np.unique(np.stack([uid[on_ring], path_poly_id[path_id[on_ring]]], axis=1), axis=0)
    n_polys = np.bincount(pairs[:, 0], minlength=nV) if len(pairs) else np.zeros(nV, dtype=np.int64)
    n_triple = int(((n_polys >= 3) & (deg == 2)).sum())
    fixed |= n_polys >= 3
    # closed loops made only of removable nodes (e.g. an island polygon) get one fixed node
    free_edge = ~fixed[E[:, 0]] & ~fixed[E[:, 1]]
    if free_edge.any():
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        Ef = E[free_edge]
        g = coo_matrix((np.ones(len(Ef)), (Ef[:, 0], Ef[:, 1])), shape=(nV, nV))
        _, lab = connected_components(g, directed=False)
        free = ~fixed
        touches_fixed = np.zeros(nV, dtype=bool)
        mixed = fixed[E[:, 0]] != fixed[E[:, 1]]
        for col in (0, 1):
            m = mixed & ~fixed[E[:, col]]
            touches_fixed[lab[E[m, col]]] = True
        comps = np.unique(lab[free])
        loops = comps[~touches_fixed[comps]]
        if len(loops):
            fv = np.flatnonzero(free)
            _, first = np.unique(lab[fv], return_index=True)
            anchor = fv[first]
            fixed[anchor[np.isin(lab[anchor], loops)]] = True
    n_fixed = int(fixed.sum())

    # --- rounds ----------------------------------------------------------------
    alive = np.ones(nV, dtype=bool)
    blocked = np.zeros(nV, dtype=bool)
    rounds = 0
    n_rejected_total = 0
    BIG = np.iinfo(np.int64).max
    tiebreak = np.random.default_rng(0).permutation(nV)
    pts = shapely.points(V)
    segs = shapely.linestrings(np.stack([V[E[:, 0]], V[E[:, 1]]], axis=1))   # kept in step with E
    while True:
        nb, deg = _neighbours(E, nV)
        movable = alive & ~fixed & (deg == 2)
        mv = np.flatnonzero(movable)
        if len(mv) == 0:
            break
        l0 = _segment_length_m(V[mv], V[nb[mv, 0]], geographic)
        l1 = _segment_length_m(V[mv], V[nb[mv, 1]], geographic)
        short = np.minimum(l0, l1)
        cmask = (short < min_dist_m) & ~blocked[mv]
        cand, cshort = mv[cmask], short[cmask]
        if len(cand) == 0:
            break
        # ties (e.g. evenly spaced nodes) are broken by a fixed pseudo-random order, so that about
        # one node in three of an evenly spaced chain goes each round, not one per chain
        # Take nodes shortest-first, never two neighbours: repeatedly pick the candidates ranked before
        # both neighbours still in the pool, then drop them and their neighbours from the pool.
        rank = np.full(nV, BIG, dtype=np.int64)
        rank[cand[np.lexsort((tiebreak[cand], cshort))]] = np.arange(len(cand))
        pool = cand
        picked = []
        for _ in range(50):
            if len(pool) == 0:
                break
            r = np.full(nV, BIG, dtype=np.int64)
            r[pool] = rank[pool]
            pick = pool[(r[pool] < r[nb[pool, 0]]) & (r[pool] < r[nb[pool, 1]])]
            picked.append(pick)
            drop = np.zeros(nV, dtype=bool)
            drop[pick] = True
            drop[nb[pick, 0]] = True
            drop[nb[pick, 1]] = True
            pool = pool[~drop[pool]]
        sel = np.concatenate(picked)
        a, b = nb[sel, 0], nb[sel, 1]
        ok = _valid_removals(V, pts, E, segs, alive, sel, a, b)
        rounds += 1
        rem, ra, rb = sel[ok], a[ok], b[ok]
        rejected = sel[~ok]
        n_rejected_total += len(rejected)
        blocked[rejected] = True
        if len(rem):
            alive[rem] = False
            blocked[ra] = False
            blocked[rb] = False
            gone = ~alive[E[:, 0]] | ~alive[E[:, 1]]
            E_new = np.stack([np.minimum(ra, rb), np.maximum(ra, rb)], axis=1)
            E = np.concatenate([E[~gone], E_new])
            segs = np.concatenate([segs[~gone],
                                   shapely.linestrings(np.stack([V[E_new[:, 0]], V[E_new[:, 1]]], axis=1))])
        if progress is not None:
            progress(f"Thin nodes: round {rounds}: {len(rem)} nodes removed, {len(rejected)} kept to "
                     f"avoid self-intersections ({int((~alive).sum())} removed so far)")

    # --- rebuild paths: keep live nodes; re-close rings ----------------------
    is_ring_pt = path_is_poly[path_id]
    keep = alive[uid] & ~(is_ring_pt & last_of_path)
    c, p = orig[keep], path_id[keep]
    ring_paths = np.flatnonzero(path_is_poly)
    if len(ring_paths):
        # each ring's first kept node, appended after its last kept node
        fk = np.r_[True, p[1:] != p[:-1]]
        start_idx = np.flatnonzero(fk)
        start_path = p[start_idx]
        is_r = path_is_poly[start_path]
        end_idx = np.r_[start_idx[1:], len(p)]
        ins_at = end_idx[is_r]
        c = np.insert(c, ins_at, c[start_idx[is_r]], axis=0)
        p = np.insert(p, ins_at, start_path[is_r])

    stuck = alive & blocked & ~fixed
    stats = {"vertices_in": int(n_in), "vertices_out": int(len(c)),
             "nodes_removed": int((~alive).sum()), "fixed_nodes": n_fixed,
             "triple_nodes_deg2": n_triple,
             "kept_to_avoid_intersection": int(stuck.sum()),
             "vertices_welded": int(n_welded), "rounds": rounds}
    return c, p, stats


def thin_layers(layers, min_dist_m, geographic=None, snap_tol=None, progress=None):
    """
    Remove nodes closer together than ``min_dist_m`` from polygon and line layers together, so that
    shared boundaries and faults along them lose the same nodes and stay coincident.

    Parameters
    ----------
    layers     : list of GeoDataFrames (polygons and/or lines, in the same CRS).
    min_dist_m : nodes closer than this (in metres) to a neighbouring node are removed, where this is
                 allowed (see the module description).  In a geographic CRS distances are converted
                 from degrees to metres approximately.
    geographic : treat coordinates as degrees (default: taken from the layers' CRS).
    snap_tol   : tolerance (CRS units) used to weld nodes that are the same corner to within rounding
                 noise.  Default 1e-6 (1e-10 degrees for a geographic CRS).
    progress   : optional callable receiving one line of text per round.

    Returns
    -------
    (list of thinned GeoDataFrames in the same order, dict of statistics)
    """
    if min_dist_m is None or min_dist_m <= 0:
        raise ValueError("The minimum node spacing must be a positive number of metres.")
    layers = [g for g in layers if g is not None]
    check_same_crs(layers)
    if geographic is None:
        geographic = any(g.crs is not None and g.crs.is_geographic for g in layers)
    infos = [_layer_paths(g) for g in layers]
    if not any(len(i["paths"]) for i in infos):
        return [g.copy() for g in layers], {"vertices_in": 0, "vertices_out": 0, "nodes_removed": 0}
    all_paths = np.concatenate([i["paths"] for i in infos if len(i["paths"])])
    path_is_poly, path_poly_id = [], []
    poly_base = 0
    for i in infos:
        nr, nl = i["n_rings"], len(i["paths"]) - i["n_rings"]
        path_is_poly.append(np.r_[np.ones(nr, dtype=bool), np.zeros(nl, dtype=bool)])
        poly_feat = i["feat"][i["poly_pos"]][i["ring_poly"]] if nr else np.zeros(0, dtype=np.int64)
        path_poly_id.append(np.r_[poly_feat + poly_base, -np.ones(nl, dtype=np.int64)])
        poly_base += len(i["geoms"])
    path_is_poly = np.concatenate(path_is_poly)
    path_poly_id = np.concatenate(path_poly_id).astype(np.int64)
    coords, path_id = shapely.get_coordinates(all_paths, return_index=True)
    new_coords, new_pid, stats = thin_paths(coords, path_id, float(min_dist_m), path_is_poly, path_poly_id,
                                            geographic=geographic, snap_tol=snap_tol, progress=progress)
    out, base = [], 0
    for g, info in zip(layers, infos):
        out.append(_rebuild_layer(g, info, new_coords, new_pid, base))
        base += len(info["paths"])
    return out, stats
