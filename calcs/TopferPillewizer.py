"""
Töpfer & Pillewizer (1966) map generalisation scaling.

TN = ON * (OS/TS) ^ (x/2)

  ON  — original feature count
  OS  — original map scale denominator  (e.g. 50 000)
  TS  — target  map scale denominator  (e.g. 500 000)
  x   — 1 (points), 2 (lines), 3 (polygons)

The user supplies OS/TS as a ratio in the range (0, 1] for reduction to a
smaller scale (OS/TS < 1 means the target denominator is larger, i.e. coarser).

Iteration schedule
------------------
If the supplied increment is smaller than the target ratio the module builds a
sequence of intermediate ratios starting at `increment`, with each successive
step 1.5× the previous step size, until the target ratio is reached.
"""

import math
import os
import tempfile
import shutil

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Core T&P calculation
# ---------------------------------------------------------------------------

def topfer_count(on, ratio, x):
    """Return integer target count TN = ON * (OS/TS)^(x/2).  Minimum 1."""
    if on <= 0 or ratio <= 0:
        return 0
    return max(1, round(on * (ratio ** (x / 2.0))))


def scale_iterations(target_ratio, increment):
    """
    Build the sequence of OS/TS ratios for iterative T&P scaling.

    Steps are additive: ratio_n = 1 - n * increment.
    The sequence runs from (1 - increment) down to target_ratio.

    Example: target_ratio=0.1, increment=0.1
      0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1  (9 steps)

    If increment <= 0 or the first step already reaches the target,
    a single-element list [target_ratio] is returned.
    """
    if increment <= 0:
        return [target_ratio]

    ratios = []
    current = round(1.0 - increment, 12)
    while current > target_ratio:
        ratios.append(current)
        current = round(current - increment, 12)

    ratios.append(target_ratio)
    return ratios


# ---------------------------------------------------------------------------
# Point scaling  (x = 1)
# ---------------------------------------------------------------------------

def _count_grid(engine, method, gdf, min_x, max_x, min_y, max_y, gs):
    """Run grid engine method at cell size gs, return number of valid output cells."""
    tmpdir = tempfile.mkdtemp()
    try:
        if method == 'gridcell_average':
            stem = engine.gridcell_average(gdf, min_x, max_x, min_y, max_y, n=gs, path_out=tmpdir)
        elif method == 'spherical_kent':
            stem = engine.spherical_kent(gdf, min_x, max_x, min_y, max_y, n=gs, path_out=tmpdir)
        else:
            stem = engine.outlier_removal(gdf, min_x, max_x, min_y, max_y, n=gs, path_out=tmpdir)
        df = pd.read_csv(os.path.join(tmpdir, stem + '.csv'))
        return int((df['DIP'] != -999).sum()) if not df.empty else 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def scale_points_tp(gdf, ratio, engine, method, dip_col, dipdir_col,
                    grid_size=5000):
    """
    Reduce a point GeoDataFrame to TN = ON*(OS/TS)^0.5 using the chosen method.

    ON is always the raw point count (len(gdf)).  For stochastic, the fraction
    is drawn directly.  For grid methods, the grid cell size is found iteratively
    so that the number of occupied cells matches TN — the analytical estimate
    sqrt(total_area/TN) fails for non-uniform / sparse data, so a proportional
    seed + binary search is used instead.
    """
    from .FirstOrderOrientation import save_grid_to_shapefile
    import time

    on = len(gdf)
    tn = topfer_count(on, ratio, x=1)
    if tn >= on:
        return gdf.copy()

    # --- Stochastic ---
    if method == 'stochastic':
        tmpdir = tempfile.mkdtemp()
        try:
            result = engine.stochastic(gdf, frac=tn / on,
                                       random_state=int(time.time()),
                                       path_out=tmpdir)
            return result.set_crs(gdf.crs, allow_override=True)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # --- Grid methods ---
    bounds = gdf.total_bounds
    min_x, max_x = int(bounds[0]), int(bounds[2]) + 1
    min_y, max_y = int(bounds[1]), int(bounds[3]) + 1

    is_geo = gdf.crs is not None and gdf.crs.is_geographic
    gs_ref = grid_size / 110000.0 if is_geo else float(grid_size)

    def count(gs):
        return _count_grid(engine, method, gdf, min_x, max_x, min_y, max_y, gs)

    # Seed with reference count to drive proportional estimate
    n_ref = count(gs_ref)
    if n_ref == 0:
        # No data at all — fall back to stochastic
        tmpdir = tempfile.mkdtemp()
        try:
            result = engine.stochastic(gdf, frac=max(0.001, tn / on),
                                       random_state=int(time.time()),
                                       path_out=tmpdir)
            return result.set_crs(gdf.crs, allow_override=True)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Proportional first estimate: N ∝ 1/gs²  →  gs ∝ sqrt(1/N)
    gs1 = gs_ref * math.sqrt(max(1, n_ref) / max(1, tn))
    n1 = count(gs1)

    tol = 0.15
    if abs(n1 - tn) <= tol * tn:
        gs_best = gs1
    else:
        # Build binary-search brackets:
        #   fine_gs (small) → n >= tn;  coarse_gs (large) → n <= tn
        if n1 >= tn:
            fine_gs, fine_n = gs1, n1
            coarse_gs = gs1 * 2
            for _ in range(10):
                coarse_n = count(coarse_gs)
                if coarse_n <= tn:
                    break
                coarse_gs *= 2
        else:
            coarse_gs, coarse_n = gs1, n1
            fine_gs = gs1 / 2
            for _ in range(10):
                fine_n = count(fine_gs)
                if fine_n >= tn:
                    break
                fine_gs /= 2

        gs_best, best_n = gs1, n1
        for _ in range(12):
            if coarse_gs - fine_gs < 0.1:
                break
            mid = (fine_gs + coarse_gs) / 2
            mid_n = count(mid)
            if abs(mid_n - tn) < abs(best_n - tn):
                gs_best, best_n = mid, mid_n
            if abs(mid_n - tn) <= tol * tn:
                break
            if mid_n > tn:
                fine_gs = mid
            else:
                coarse_gs = mid

    # Final run at best grid size
    tmpdir = tempfile.mkdtemp()
    try:
        if method == 'gridcell_average':
            stem = engine.gridcell_average(gdf, min_x, max_x, min_y, max_y,
                                           n=gs_best, path_out=tmpdir)
        elif method == 'spherical_kent':
            stem = engine.spherical_kent(gdf, min_x, max_x, min_y, max_y,
                                         n=gs_best, path_out=tmpdir)
        else:
            stem = engine.outlier_removal(gdf, min_x, max_x, min_y, max_y,
                                          n=gs_best, path_out=tmpdir)
        result = save_grid_to_shapefile(tmpdir, stem)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if result is not None and not result.empty:
        result = result.rename(columns={'DIP': dip_col, 'DIP_DIR': dipdir_col}, errors='ignore')
        return result.set_crs(gdf.crs, allow_override=True)

    # Fallback to stochastic if grid method yielded nothing
    tmpdir = tempfile.mkdtemp()
    try:
        result = engine.stochastic(gdf, frac=max(0.001, tn / on),
                                   random_state=int(time.time()),
                                   path_out=tmpdir)
        return result.set_crs(gdf.crs, allow_override=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Line (fault) scaling  (x = 2)
# ---------------------------------------------------------------------------

def _edge_type_rank(edge_type):
    """
    Importance rank for graph edge_type (lower rank = higher importance = kept longer).

    x-x  → 0  (most important, crosses two X-nodes)
    z-z  → 3  (least important, isolated at both ends — removed first)
    else → 1  (x-z, y-*, etc.)
    """
    if pd.isna(edge_type):
        return 2
    et = str(edge_type).lower().strip()
    if et == 'x-x':
        return 0
    if et == 'z-z':
        return 3
    return 1


def scale_lines_tp(gdf, ratio, method):
    """
    Reduce a line GeoDataFrame to TN = ON*(OS/TS)^1 features.

    A ``tp_sel_val`` field is always appended to record the criterion used:
      length        — geometry length
      graph         — edge_type value  (falls back to length if field absent)
      strat_offset  — stratigraphic-offset value  (falls back to length)
      clusters      — cluster label  (falls back to length)

    Sorting rules by method:
      length        — keep longest features
      graph         — sort by edge_type rank (x-x > x-z/y-* > z-z),
                      secondary sort by length
      strat_offset  — keep highest absolute stratigraphic-offset value;
                      falls back to length if no offset field found
      clusters      — keep members of the largest orientation clusters;
                      falls back to length if no cluster field found
    """
    on = len(gdf)
    tn = topfer_count(on, ratio, x=2)

    def _to_str(series):
        """Convert a Series to string, replacing NaN/None with ''."""
        return ['' if pd.isna(v) else str(v) for v in series]

    # Fast path: all features kept — add fields without sorting
    if tn >= on:
        result = gdf.copy()
        if 'line_len' not in result.columns:
            result['line_len'] = result.geometry.length
        if method == 'graph' and 'edge_type' in result.columns:
            result['tp_sel_val'] = _to_str(result['edge_type'])
        elif method == 'strat_offset':
            strat_col = next((c for c in result.columns
                              if 'strat' in c.lower() or 'offset' in c.lower()), None)
            result['tp_sel_val'] = _to_str(
                pd.to_numeric(result[strat_col], errors='coerce') if strat_col
                else pd.Series([''] * len(result), index=result.index))
        elif method == 'clusters':
            cluster_col = next((c for c in result.columns
                                if 'cluster' in c.lower()), None)
            result['tp_sel_val'] = _to_str(result[cluster_col]) if cluster_col else ''
        else:  # length or graph-without-edge_type
            result['tp_sel_val'] = ('' if method == 'graph'
                                    else _to_str(result['line_len']))
        return result

    n_keep = min(tn, on)
    work = gdf.copy()
    # Reuse existing line_len to avoid recomputing geometry lengths on every step
    if 'line_len' in work.columns:
        work['_len'] = work['line_len']
    else:
        work['_len'] = work.geometry.length

    if method == 'graph' and 'edge_type' in work.columns:
        work['_rank'] = work['edge_type'].apply(_edge_type_rank)
        work = work.sort_values(['_rank', '_len'], ascending=[True, False])
        work['tp_sel_val'] = _to_str(work['edge_type'])

    elif method == 'graph':  # graph selected but edge_type column absent
        work = work.sort_values('_len', ascending=False)
        work['tp_sel_val'] = ''

    elif method == 'strat_offset':
        strat_col = next(
            (c for c in work.columns
             if 'strat' in c.lower() or 'offset' in c.lower()),
            None
        )
        if strat_col:
            work['_sort'] = pd.to_numeric(work[strat_col], errors='coerce').abs()
            work = work.sort_values(['_sort', '_len'], ascending=False)
            work['tp_sel_val'] = _to_str(pd.to_numeric(work[strat_col], errors='coerce'))
        else:
            work = work.sort_values('_len', ascending=False)
            work['tp_sel_val'] = ''

    elif method == 'clusters':
        cluster_col = next(
            (c for c in work.columns if 'cluster' in c.lower()), None
        )
        if cluster_col:
            sizes = work[cluster_col].value_counts()
            work['_csz'] = work[cluster_col].map(sizes)
            work = work.sort_values(['_csz', '_len'], ascending=[False, False])
            work['tp_sel_val'] = _to_str(work[cluster_col])
        else:
            work = work.sort_values('_len', ascending=False)
            work['tp_sel_val'] = ''

    else:   # length
        work = work.sort_values('_len', ascending=False)
        work['tp_sel_val'] = _to_str(work['_len'])

    if 'line_len' not in work.columns:
        work['line_len'] = work['_len']

    drop_cols = [c for c in ['_len', '_rank', '_sort', '_csz'] if c in work.columns]
    return work.head(n_keep).drop(columns=drop_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Polygon scaling helpers
# ---------------------------------------------------------------------------

def _vals_match(a, b):
    """True if both values are non-null and equal."""
    try:
        if pd.isna(a) or pd.isna(b):
            return False
    except (TypeError, ValueError):
        pass
    return a == b


def _best_neighbor_idx(gdf, si, neighbors, priority_cols):
    """
    Return the positional index of the best merge target for the polygon at
    position *si*, honouring the attribute priority hierarchy.

    Parameters
    ----------
    gdf          : current working GeoDataFrame (reset_index applied)
    si           : positional index of the polygon to absorb
    neighbors    : list of (positional_index, shared_boundary_length) tuples
    priority_cols: column names tried in order (lithoname first, then strat1..4)

    Within each priority tier the candidate with the *longest* shared boundary
    is preferred.  Falls back to longest boundary when no attribute match exists.
    """
    small_row = gdf.iloc[si]

    for col in priority_cols:
        matching = [
            (ci, sl) for ci, sl in neighbors
            if _vals_match(small_row[col], gdf.iloc[ci][col])
        ]
        if matching:
            return max(matching, key=lambda x: x[1])[0]

    return max(neighbors, key=lambda x: x[1])[0]


def _merge_to_count(gdf, target_count, priority_cols=None):
    """
    Count-controlled merge: repeatedly dissolve the smallest polygon into its
    best stratigraphically-matching neighbour until len(gdf) == target_count.

    Every polygon is *merged* (dissolved via union), never deleted, so no
    holes are created in the map.  The priority hierarchy is:

        lithoname match > strat1 match > strat2 match > strat3 match
        > strat4 match > longest shared boundary (fallback)

    Parameters
    ----------
    gdf           : input GeoDataFrame
    target_count  : desired number of polygons
    priority_cols : list of column names to check in priority order
    """
    if priority_cols is None:
        priority_cols = []

    result = gdf.copy().reset_index(drop=True)
    geom_col = result.geometry.name
    crs = result.crs

    while len(result) > target_count:
        result = result.reset_index(drop=True)
        areas = result.geometry.area.values
        si = int(np.argmin(areas))
        small_geom = result.geometry.iloc[si]

        # Bounding-box candidates via spatial index
        try:
            candidates = list(result.sindex.intersection(small_geom.bounds))
        except Exception:
            candidates = list(range(len(result)))
        candidates = [c for c in candidates if c != si]

        # Measure actual shared boundary length for touching candidates
        neighbors = []
        for ci in candidates:
            try:
                ni_geom = result.geometry.iloc[ci]
                if ni_geom.touches(small_geom) or ni_geom.intersects(small_geom):
                    shared = small_geom.intersection(ni_geom)
                    sl = shared.length if not shared.is_empty else 0.0
                    if sl > 0:
                        neighbors.append((ci, sl))
            except Exception:
                pass

        if not neighbors:
            # Isolated polygon — nearest centroid as emergency fallback
            if candidates:
                best_ci = min(
                    candidates,
                    key=lambda c: small_geom.centroid.distance(
                        result.geometry.iloc[c].centroid)
                )
                neighbors = [(best_ci, 0.0)]
            else:
                # No candidates at all — remove to avoid infinite loop
                result = result.drop(index=si).reset_index(drop=True)
                continue

        # Select best neighbour honouring the attribute priority hierarchy
        best_ci = _best_neighbor_idx(result, si, neighbors, priority_cols)

        # Union geometries, drop the absorbed polygon
        geoms = result.geometry.tolist()
        geoms[best_ci] = unary_union([geoms[si], geoms[best_ci]])
        geoms.pop(si)   # si removed; if si < best_ci the updated geom shifts left

        result = (result
                  .drop(columns=[geom_col])
                  .drop(index=si)
                  .reset_index(drop=True))
        result = gpd.GeoDataFrame(result, geometry=geoms, crs=crs)

    return result


# ---------------------------------------------------------------------------
# Polygon scaling  (x = 3)
# ---------------------------------------------------------------------------

def scale_polygons_tp(gdf, ratio, polygon_subsampler_class,
                      distance_threshold=1e-6,
                      lithoname=None, strat1=None, strat2=None,
                      strat3=None, strat4=None,
                      dyke_field=None, dyke_codes=None,
                      triangulator_class=None):
    """
    Reduce a polygon GeoDataFrame to TN = ON*(OS/TS)^1.5 features.

    Strategy
    --------
    1. Run ``clean_small_polygons_and_holes_new`` once with a threshold sized
       to cover all the intended-to-remove polygons.  This efficiently handles
       the bulk of merges with full stratigraphic hierarchy.
    2. If the count is still above TN (the threshold-based algorithm stalls when
       small polygons are only adjacent to other small polygons), fall through to
       ``_merge_to_count``, which iteratively absorbs the smallest remaining
       polygon into its best stratigraphically-matching neighbour until the T&P
       target is reached.  Every polygon is merged (dissolved), never deleted,
       so no holes appear in the output map.
    """
    def _or_none(v):
        return v if v else None

    on = len(gdf)
    tn = topfer_count(on, ratio, x=3)
    if tn >= on:
        return gdf.copy()

    work = gdf.copy()

    # Optional dyke pre-processing (once, before merging)
    if (dyke_field and dyke_codes and triangulator_class is not None
            and dyke_field in work.columns):
        trig = triangulator_class(
            gdf=work,
            id_column=dyke_field,
            min_area_threshold=0.0,
            distance_threshold=distance_threshold,
            strat1=strat1, strat2=strat2, strat3=strat3, strat4=strat4,
            lithoname=lithoname,
        )
        work = trig.triangulate_polygons(target_ids=dyke_codes)

    merge_kwargs = dict(
        distance_threshold=distance_threshold,
        lithoname=_or_none(lithoname),
        strat1=_or_none(strat1),
        strat2=_or_none(strat2),
        strat3=_or_none(strat3),
        strat4=_or_none(strat4),
    )

    # Priority column list for count-controlled merge (same order as hierarchy)
    priority_cols = [
        c for c in [lithoname, strat1, strat2, strat3, strat4]
        if c and c in work.columns
    ]

    # Pass 1: threshold-based bulk merge (efficient)
    n_current = len(work)
    if n_current > tn:
        n_remove = n_current - tn
        areas = work.geometry.area.sort_values().values
        idx = min(n_remove, len(areas) - 1)
        threshold = float(areas[idx]) * (1.0 + 1e-9) + 1e-6
        try:
            sub = polygon_subsampler_class(work)
            merged = sub.clean_small_polygons_and_holes_new(
                work, min_area_threshold=threshold, **merge_kwargs)
            if len(merged) < n_current:
                work = merged
        except Exception as exc:
            print(f"T&P polygon bulk-merge error: {exc}")

    # Pass 2: count-controlled merge for any remainder — no holes, priorities respected
    if len(work) > tn:
        work = _merge_to_count(work, tn, priority_cols=priority_cols)

    return work