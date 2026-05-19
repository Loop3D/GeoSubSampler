"""
Polygon Simplification Package — Geological Map Vector Simplification

Overview
--------
Simplifies polygon and line shapefiles for geological maps while preserving
topological integrity.  Designed for GSWA-style datasets where polygon features
represent geological units and a separate line layer represents the fault network.

The main entry point is  vector_simplify_file_two_stage().  It runs a
three-stage pipeline:

  Stage 0 — Topology pre-processing
      1. Snaps all polygon vertex coordinates to a decimal grid to eliminate
         floating-point mismatches between adjacent features.
      2. Detects and corrects hairline overlaps and gaps between polygon rings
         so that polygon positions are finalised before any snapping occurs.
      3. Snaps fault↔polygon vertex pairs to their MIDPOINT so that fault and
         polygon boundaries that should coincide share exactly the same coordinate.
         Using the midpoint (rather than snapping one side to the other) avoids
         the position-swap bug: one-sided bidirectional snapping causes fault and
         polygon vertices that are millimetres apart to trade places — fault ends
         up where the polygon was and vice versa.  Moving both to the midpoint
         eliminates the swap.  This snap runs AFTER the overlap/gap fix so it
         targets the final corrected polygon positions.

  Stage 1 — Fault network simplification
      Simplifies the fault line network using Modified Visvalingam-Whyatt.
      Every vertex that lies on a polygon boundary is pinned (given infinite
      weight) so it is never removed.  This preserves the exact position of
      every fault-polygon crossing point through Stage 1.

  Stage 2 — Polygon simplification
      Simplifies each polygon using the chosen algorithm.  Before simplification:
        • Junction detection: every point where ≥ 3 features meet gets infinite
          weight and cannot be removed.
        • All-vertex pinning: every fault vertex that coincides with a polygon
          boundary vertex is pinned so fault–polygon alignment is maintained.
        • Boundary preservation: exterior arc vertices are either hard-pinned
          (cannot be removed) or soft-scaled (resist removal) depending on
          the boundary_preserve setting.
        • Unique geological contact pinning: for each unique CODE-pair that
          shares a polygon boundary, one representative vertex is pinned to
          ensure the contact relationship is always detectable in the output.
        • Thin-body arc pinning: for every narrow polygon at risk of collapsing
          at the requested threshold, its boundary arcs are pre-simplified at a
          conservative safe threshold and the surviving intermediate vertices are
          pinned.  This guarantees that both the thin body and its neighbours
          simplify the shared arc to the same vertex set, preventing gaps and
          overlaps on thin-body boundaries.
      Thin-body features are processed last so that all neighbour geometries are
      available if the collapse fallback is needed.
      After simplification a contact-health report is printed showing how many
      shared boundaries retain ≥ 3 vertices (meaningful shape), exactly 2
      (barely a line), 1 (point only — relationship preserved, shape lost),
      or 0 (boundaries diverged — reduce tolerance).

Geological contact preservation
---------------------------------
A "unique geological contact" is a shared boundary between two polygon
features with DIFFERENT values in the unit identifier field (default: CODE).
For each such pair the algorithm:
  1. Computes the shared boundary (boundary_A ∩ boundary_B via Shapely).
  2. Selects the vertex closest to the centroid of the shared boundary as the
     representative — a stable, central anchor point.
  3. Adds that vertex to dict_junctions with infinite area weight so it is
     never removed by Visvalingam-Whyatt regardless of the threshold.
After simplification, the shared boundaries of every contact pair are
re-examined and classified as preserved / minimal / shape-lost / diverged.

Self-intersection prevention
------------------------------
Thin polygons (fold stripes, dyke outlines) can produce self-intersecting
("bowtie") output after simplification.  Two mechanisms prevent this:

  1. Ring-mode guard (whole-ring simplification path):
     Before committing to any vertex removal, the proposed replacement edge
     (prev → next) is tested against every non-adjacent ring edge using a
     fast pure-Python cross-product test (_segments_cross).  If it would
     cross, the vertex is kept (treated as constrained for the rest of the
     run) and simplification continues with the next candidate.

  2. Arc-assembly guard (arc-cache path for junction-constrained polygons):
     Each arc is simplified independently and the results are reassembled
     into a ring.  If the assembled ring is not geometrically simple (two
     arcs cross at the tip of a thin polygon), the code falls back to
     whole-ring simplification, which has the ring-mode guard above.

  Both paths also apply a post-simplification make_valid() repair as an
  ultimate safety net: if the finished ring is still not simple (a sequence
  of individually-safe removals can cumulatively produce a crossing), the
  ring is repaired by splitting at the crossing and keeping the largest
  valid polygon part.

  The arc cache only stores geometrically simple (non-self-intersecting)
  arcs.  A self-intersecting arc is not cached so adjacent polygons that
  share the same boundary arc are not affected.

Thin body detection and topology-safe simplification
------------------------------------------------------
Before simplification, every polygon body is checked for approximate minimum
width (≈ 2 × area / perimeter).  If min_width < √(threshold) / 2, the body
is flagged as at-risk of collapsing.

Thin bodies (fold stripes, dyke outlines) are narrow polygons that share
long boundary arcs with wider neighbours.  If a shared arc is simplified too
aggressively, it collapses to just its two endpoint junctions — which
eliminates the thin body's interior.  To prevent this, every thin-body arc
is pre-simplified at a conservative safe threshold (= smallest thin-body area
/ 4) before the main simplification run.  Every intermediate vertex that
survives the pre-simplification is pinned into dict_junctions (given infinite
area weight).  During the main run every polygon sharing those arcs — the
thin body and all its neighbours — treats those vertices as immovable hard
constraints.  This guarantees that the shared boundary simplifies to the same
vertex set for every touching polygon regardless of processing order, so no
gaps or overlaps can arise on thin-body boundaries.

Thin bodies are also processed last in the feature loop, after all their
neighbours have already been simplified.  This ensures that the collapse
fallback (below) has access to the fully-simplified neighbour geometries.

Collapse fallback
-----------------
If arc-mode MVW still cannot produce a valid simplified ring (the arc
subdivision can leave very short arcs that individually cannot be simplified),
three successive fallbacks are tried:

  1. Ring-mode retry: simplify the whole exterior as a single ring — no arc
     splitting — at the original threshold, stepping down 10 % per attempt
     until a result is found that does not overlap any already-simplified
     neighbour.

  2. Neighbour-clip: if every ring-mode threshold overlaps a neighbour,
     compute  this_polygon = original_geometry − union(overlapping_neighbours).
     The shared boundary then exactly matches the neighbour's simplified edge.

  3. Write original: if the clipped remainder is empty (the polygon's entire
     territory has been absorbed by its simplified neighbours at this
     tolerance), the original unsimplified geometry is written.  The output
     always contains the same number of features as the input; reduce the
     threshold to obtain a simplified result for these features.

Six simplification algorithms
-------------------------------
1. Decimation              — nth-point removal (Tobler 1966, Miller 2004)
2. Douglas-Peucker         — perpendicular distance (Douglas & Peucker 1973)
3. Douglas-Peucker TP      — topology-preserving GEOS variant (Saalfeld 1999)
4. Bend Simplification     — compactness-based bend removal (Visvalingam & Whyatt 1990)
5. Visvalingam-Whyatt      — area-based vertex removal (Visvalingam & Whyatt 1992)
6. Modified Visvalingam-Whyatt — topology-preserving two-stage pipeline with
                               fault network consistency (arc caching, junction
                               pinning, boundary preservation, contact pinning)

Key parameters
--------------
input_file        : polygon shapefile path
output_file       : output shapefile path
method            : algorithm name (see list above)
threshold         : Visvalingam-Whyatt area threshold in map units squared (m²)
fault_file        : fault network shapefile (enables two-stage MVW pipeline)
boundary_preserve : "hard" — exterior arc vertices pinned (cannot be removed)
                    "soft" — exterior arc vertices scaled to resist removal
unit_field        : attribute field that identifies the geological unit code
                    (default "CODE" for GSWA datasets)
preprocess        : if True (default), run Stage 0 topology pre-processing
snap_decimals     : decimal places for coordinate grid in Stage 0 (default 7)
"""

import heapq
import copy
import os
import time
import tempfile
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional, Set
import fiona
import math
try:
    from fiona import mapping
except ImportError:
    from shapely.geometry import mapping
from shapely.geometry import (
    Point, LineString, MultiLineString, Polygon, MultiPolygon, LinearRing, box
)
from shapely.ops import unary_union
from shapely.validation import make_valid as _make_valid
from shapely.strtree import STRtree
try:
    from shapely import make_valid          # Shapely ≥ 2.0
except ImportError:
    from shapely.validation import make_valid
import geopandas as gpd
import numpy as np
from collections import defaultdict

# Global validation flag for debugging
VALIDATE_GEOMETRY = True


class TriangleCalculator:
    """
    Represents a vertex in the Visvalingam-Whyatt algorithm as part of a triangular
    area calculation.  Used in min-heap operations.

    area_scale
        Multiplier applied to the raw triangle area in calcArea().  Defaults to
        1.0 (standard behaviour).  Set to (threshold / soft_tolerance) for
        soft-exterior vertices so they behave as if their effective area is larger
        — they are removed only when their triangle area falls below soft_tolerance,
        not the main threshold.  This lets exterior boundary vertices resist
        simplification without being permanently pinned.
    """

    def __init__(self, point: Tuple[float, float], index: int,
                 is_constrained: bool = False, area_scale: float = 1.0):
        self.point           = point
        self.ringIndex       = index
        self.is_constrained  = is_constrained
        self.area_scale      = area_scale        # multiplier for soft-exterior resistance
        self.prevTriangle    = None
        self.nextTriangle    = None

    def __lt__(self, other) -> bool:
        return self.calcArea() < other.calcArea()

    def calcArea(self) -> float:
        """
        Effective area of the triangle formed by this vertex and its neighbours.
        Constrained points return infinity.  Soft-exterior points return
        actual_area × area_scale, making them resistant to removal.
        """
        if self.is_constrained:
            return float('inf')
        if not self.prevTriangle or not self.nextTriangle:
            return float('inf')

        p1 = self.point
        p2 = self.prevTriangle.point
        p3 = self.nextTriangle.point

        area = abs(p1[0] * (p2[1] - p3[1]) +
                   p2[0] * (p3[1] - p1[1]) +
                   p3[0] * (p1[1] - p2[1])) / 2.0

        return area * self.area_scale    # soft-exterior scaling applied here


class SimplificationEngine:
    """
    Main engine for vector geometry simplification with comprehensive topology
    preservation.

    Handles junction detection, arc caching, constraint management, boundary
    preservation, unique geological contact pinning, and all six simplification
    algorithms.  Instantiate one engine per simplification run; the engine
    accumulates topology state (junctions, shared boundaries, arc cache) as
    features are processed.
    """

    def __init__(self, dict_junctions: Dict = None):
        self.dict_junctions    = dict_junctions if dict_junctions is not None else {}
        self.dict_simple_arcs  = {}
        self.quantitization_factor = (0.1, 0.1)

        self.boundary_segments      = {}
        self.shared_boundaries      = defaultdict(list)
        self.junction_neighbors     = defaultdict(set)
        self.geometry_registry      = {}
        self.feature_types          = {}
        self.fault_polygon_boundaries      = defaultdict(list)
        self.polygon_polygon_boundaries    = defaultdict(list)
        self.topologically_significant_junctions = set()

        # ── Boundary-preservation state ──────────────────────────────────────
        # exterior_vertices: coordinates that lie only on exterior arcs (face void)
        # soft_scale: area multiplier applied to soft-exterior vertices so they
        #             resist removal without being permanently pinned
        self.exterior_vertices: Set[Tuple[float, float]] = set()
        self.soft_scale: float = 1.0

    # =========================================================================
    # UTILITY
    # =========================================================================

    def set_quantitization_factor(self, quant_value: float):
        self.quantitization_factor = (quant_value, quant_value)

    def quantitize(self, point: Tuple[float, float]) -> Tuple[float, float]:
        qx = int(round(point[0] / self.quantitization_factor[0])) * self.quantitization_factor[0]
        qy = int(round(point[1] / self.quantitization_factor[1])) * self.quantitization_factor[1]
        return (qx, qy)

    def create_segment_hash(self, point1, point2) -> str:
        p1 = self.quantitize(point1)
        p2 = self.quantitize(point2)
        if (p1[0], p1[1]) <= (p2[0], p2[1]):
            return f"{p1[0]},{p1[1]}|{p2[0]},{p2[1]}"
        return f"{p2[0]},{p2[1]}|{p1[0]},{p1[1]}"

    def _iter_all_coords(self, geom_dict):
        """Yield every coordinate from a fiona geometry dict."""
        gtype = geom_dict['type']
        if gtype == 'Polygon':
            for ring in geom_dict['coordinates']:
                for pt in ring:
                    yield pt
        elif gtype == 'MultiPolygon':
            for poly in geom_dict['coordinates']:
                for ring in poly:
                    for pt in ring:
                        yield pt

    # =========================================================================
    # BOUNDARY SEGMENT REGISTRATION
    # =========================================================================

    def _register_boundary_segments_typed(self, feature_id, points_list, feature_type):
        self.feature_types[feature_id] = feature_type
        for i in range(len(points_list) - 1):
            seg_hash = self.create_segment_hash(points_list[i], points_list[i + 1])
            self.shared_boundaries[seg_hash].append(
                {'id': feature_id, 'type': feature_type, 'segment_index': i}
            )
            if seg_hash not in self.boundary_segments:
                self.boundary_segments[seg_hash] = (points_list[i], points_list[i + 1])

    def _register_boundary_segments(self, feature_id, points_list):
        for i in range(len(points_list) - 1):
            seg_hash = self.create_segment_hash(points_list[i], points_list[i + 1])
            self.shared_boundaries[seg_hash].append(feature_id)
            if seg_hash not in self.boundary_segments:
                self.boundary_segments[seg_hash] = (points_list[i], points_list[i + 1])

    # =========================================================================
    # JUNCTION DETECTION
    # Identifies vertices where local neighbour connectivity changes across
    # ring encounters — i.e. points where three or more features meet.
    # These points are marked in dict_junctions with infinite area weight.
    # =========================================================================

    def _append_junctions_typed(self, junctions, neighbors, points_list,
                                 feature_id=None, feature_type='unknown'):
        cleaned = []
        for i, pt in enumerate(points_list):
            if i == 0 or pt != points_list[i - 1]:
                cleaned.append(pt)
        if feature_id:
            self._register_boundary_segments_typed(feature_id, cleaned, feature_type)
        if VALIDATE_GEOMETRY:
            dict_check = {}
        for index, point in enumerate(cleaned):
            qpt = self.quantitize(point)
            if VALIDATE_GEOMETRY:
                if qpt in dict_check:
                    continue
                dict_check[qpt] = point
            qnbrs = []
            if index > 0:
                qnbrs.append(self.quantitize(cleaned[index - 1]))
            if index + 1 < len(cleaned):
                qnbrs.append(self.quantitize(cleaned[index + 1]))
            if qpt in neighbors:
                if set(neighbors[qpt]) != set(qnbrs):
                    junctions[qpt] = 1
                    self.junction_neighbors[qpt].update(neighbors[qpt])
                    self.junction_neighbors[qpt].update(qnbrs)
            else:
                neighbors[qpt] = qnbrs
                self.junction_neighbors[qpt].update(qnbrs)

    def _append_junctions(self, junctions, neighbors, points_list, feature_id=None):
        cleaned = []
        for i, pt in enumerate(points_list):
            if i == 0 or pt != points_list[i - 1]:
                cleaned.append(pt)
        if feature_id:
            self._register_boundary_segments(feature_id, cleaned)
        if VALIDATE_GEOMETRY:
            dict_check = {}
        for index, point in enumerate(cleaned):
            qpt = self.quantitize(point)
            if VALIDATE_GEOMETRY:
                if qpt in dict_check:
                    continue
                dict_check[qpt] = point
            qnbrs = []
            if index > 0:
                qnbrs.append(self.quantitize(cleaned[index - 1]))
            if index + 1 < len(cleaned):
                qnbrs.append(self.quantitize(cleaned[index + 1]))
            if qpt in neighbors:
                if set(neighbors[qpt]) != set(qnbrs):
                    junctions[qpt] = 1
                    self.junction_neighbors[qpt].update(neighbors[qpt])
                    self.junction_neighbors[qpt].update(qnbrs)
            else:
                neighbors[qpt] = qnbrs
                self.junction_neighbors[qpt].update(qnbrs)

    # =========================================================================
    # POLYGON TRIPLE JUNCTION DETECTION
    # Scans the polygon dataset and returns every coordinate where three or
    # more polygon rings meet.  Used to identify which fault endpoints must
    # be pinned so they are never moved by fault simplification.
    # =========================================================================

    def _collect_polygon_triple_junctions(self, polygon_file: str) -> Set[Tuple[float, float]]:
        """Scan the polygon dataset and return every polygon triple-junction coordinate."""
        poly_junctions: Dict = {}
        poly_neighbors: Dict = {}
        with fiona.open(polygon_file, 'r') as layer:
            for record in layer:
                geom = record['geometry']
                if geom['type'] == 'Polygon':
                    self._append_junctions(poly_junctions, poly_neighbors,
                                           geom['coordinates'][0])
                    for ring in geom['coordinates'][1:]:
                        self._append_junctions(poly_junctions, poly_neighbors, ring)
                elif geom['type'] == 'MultiPolygon':
                    for poly_coords in geom['coordinates']:
                        self._append_junctions(poly_junctions, poly_neighbors,
                                               poly_coords[0])
                        for ring in poly_coords[1:]:
                            self._append_junctions(poly_junctions, poly_neighbors, ring)
        return set(poly_junctions.keys())

    # =========================================================================
    # ALL-VERTEX POLYGON PINNING
    # Returns every polygon vertex coordinate so that any fault vertex
    # coinciding with the polygon boundary — whether at a triple junction
    # or an ordinary mid-arc point — is given infinite area weight and
    # cannot be removed during fault simplification (Stage 1).
    # This is a superset of triple-junction pinning and closes the gap
    # where fault–polygon coincident mid-arc vertices would otherwise be
    # removed by Stage 1, breaking fault–polygon alignment in Stage 2.
    # =========================================================================

    def _collect_all_polygon_vertices(self, polygon_file: str) -> Set[Tuple[float, float]]:
        """
        Return the quantized coordinates of EVERY vertex in the polygon dataset.

        Any fault vertex that lies on the polygon boundary — regardless of
        whether it is at a triple junction or a plain mid-arc point — is
        given infinite area weight and kept through Stage 1 simplification,
        ensuring fault–polygon alignment is preserved into Stage 2.
        """
        all_verts: Set = set()
        with fiona.open(polygon_file, 'r') as layer:
            for record in layer:
                geom = record['geometry']
                if geom['type'] == 'Polygon':
                    for ring in geom['coordinates']:
                        for pt in ring:
                            all_verts.add(self.quantitize(pt))
                elif geom['type'] == 'MultiPolygon':
                    for poly_coords in geom['coordinates']:
                        for ring in poly_coords:
                            for pt in ring:
                                all_verts.add(self.quantitize(pt))
        return all_verts

    # =========================================================================
    # UNIQUE GEOLOGICAL CONTACT REPRESENTATIVE PINNING
    # For every pair of adjacent polygon features with different unit codes,
    # selects one representative vertex on the shared boundary and adds it
    # to dict_junctions so it can never be removed.  This guarantees that
    # the contact relationship between any two geological units always has
    # at least one surviving shared vertex in the simplified output.
    # =========================================================================

    def _collect_unique_contact_representatives(
        self,
        polygon_file: str,
        unit_field:   str = 'CODE',
    ) -> Tuple[Dict, Set]:
        """
        Identify every unique geological unit contact and select one
        representative vertex per contact pair to pin in dict_junctions.

        A "unique contact" is a shared polygon boundary between two features
        with DIFFERENT values in `unit_field`.  Self-contacts (same unit
        touching itself across multiple polygon bodies) are excluded.

        Algorithm
        ---------
        1. Load all polygon features with their unit code and Shapely geometry.
        2. Build an STRtree for fast candidate-pair lookup.
        3. For each candidate pair (i, j) where code_i ≠ code_j:
             shared = boundary_i ∩ boundary_j
             If non-empty: collect the vertex in `shared` that is closest
             to the centroid of `shared` as the representative.
        4. Add the quantized representative to dict_junctions (infinite
           area weight → cannot be removed by VW regardless of threshold).

        Parameters
        ----------
        polygon_file : path to polygon shapefile
        unit_field   : attribute field that identifies the geological unit
                       (default 'CODE' for GSWA datasets)

        Returns
        -------
        (contacts : dict, representative_vertices : set)

        contacts maps  frozenset({code_a, code_b})  →
            {
              'code_a': str, 'code_b': str,
              'shared_length': float,        # total shared boundary length (m)
              'n_shared_verts': int,          # vertices on shared boundary
              'representative': (x, y),       # chosen pin coordinate
            }

        representative_vertices is the set of quantized (x, y) coordinates
        added to dict_junctions (one per unique contact pair, unless a
        representative was already in dict_junctions from a prior step).
        """
        from shapely.geometry import shape as _shape, MultiLineString, LineString

        # ── Validate unit_field ────────────────────────────────────────────────
        with fiona.open(polygon_file, 'r') as _lyr:
            _schema_fields = list(_lyr.schema.get('properties', {}).keys())
        if unit_field not in _schema_fields:
            raise ValueError(
                f"unit_field='{unit_field}' not found in polygon schema.\n"
                f"Available fields: {_schema_fields}\n"
                f"Call inspect_unit_field(polygon_file) to see field details."
            )

        # ── Load features ──────────────────────────────────────────────────────
        records = []
        with fiona.open(polygon_file, 'r') as layer:
            for rec in layer:
                try:
                    code = rec['properties'].get(unit_field) or ''
                    geom = _shape(rec['geometry'])
                    if geom.is_valid and not geom.is_empty:
                        records.append({'code': str(code).strip(), 'geom': geom})
                except Exception:
                    pass

        if not records:
            return {}, set()

        # ── Spatial index ──────────────────────────────────────────────────────
        geom_list = [r['geom'] for r in records]
        tree      = STRtree(geom_list)

        contacts:  Dict = {}   # frozenset({a, b}) → info dict
        rep_verts: Set  = set()

        def _extract_coords(geom) -> List[Tuple[float, float]]:
            """All (x, y) vertices from a LineString / MultiLineString / Point."""
            pts: List = []
            if geom.geom_type == 'LineString':
                pts = [(c[0], c[1]) for c in geom.coords]
            elif geom.geom_type == 'MultiLineString':
                for part in geom.geoms:
                    pts.extend([(c[0], c[1]) for c in part.coords])
            elif geom.geom_type == 'GeometryCollection':
                for g in geom.geoms:
                    pts.extend(_extract_coords(g))
            elif geom.geom_type == 'Point':
                pts = [(geom.x, geom.y)]
            elif geom.geom_type == 'MultiPoint':
                pts = [(p.x, p.y) for p in geom.geoms]
            return pts

        for i in range(len(records)):
            ri   = records[i]
            bi   = ri['geom'].boundary
            cands = tree.query(ri['geom'], predicate='intersects')

            for j in cands:
                if j <= i:
                    continue
                rj = records[j]

                # Skip self-contacts
                if ri['code'] == rj['code']:
                    continue

                key = frozenset({ri['code'], rj['code']})
                if key in contacts:
                    continue   # already found this pair (may occur with multi-body units)

                try:
                    shared = bi.intersection(rj['geom'].boundary)
                except Exception:
                    continue

                if shared is None or shared.is_empty:
                    continue

                coords = _extract_coords(shared)
                if not coords:
                    continue

                # Representative = vertex closest to centroid of shared geometry
                cx, cy   = shared.centroid.x, shared.centroid.y
                rep_xy   = min(coords,
                               key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)
                rep_q    = self.quantitize(rep_xy)

                shared_length = shared.length if hasattr(shared, 'length') else 0.0

                contacts[key] = {
                    'code_a':        ri['code'],
                    'code_b':        rj['code'],
                    'shared_length': shared_length,
                    'n_shared_verts': len(coords),
                    'representative': rep_xy,
                }

                # Pin the representative in dict_junctions
                if rep_q not in self.dict_junctions:
                    self.dict_junctions[rep_q] = 1
                    rep_verts.add(rep_q)

        return contacts, rep_verts

    # =========================================================================
    # BOUNDARY PRESERVATION
    # Detects whether the dataset boundary is rectangular or irregular and
    # applies the appropriate preservation strategy so the outer edge of the
    # map is never altered by simplification:
    #   "hard" mode — exterior arc vertices are added to dict_junctions
    #                 (infinite area weight, permanently pinned)
    #   "soft" mode — exterior arc vertices get area_scale = threshold / soft_tolerance
    #                 so they are only removed when their triangle area is below
    #                 soft_tolerance, making them strongly resistant but not immovable
    # Rectangular datasets always use hard-pin on all four bbox edges.
    # =========================================================================

    def _is_rectangular_dataset(self, polygon_file: str) -> Tuple[bool, Optional[Tuple]]:
        """
        Determine whether the outer boundary of the polygon dataset is
        (approximately) rectangular.

        Method: compute the unary union of all polygon geometries, then compare
        the convex hull of that union against the bounding box.  If the
        symmetric difference between the two is less than 0.01 % of the
        bounding-box area, the dataset is considered rectangular.

        Returns
        -------
        (is_rectangular: bool, bounds: (xmin, ymin, xmax, ymax) or None)
        """
        geoms = []
        with fiona.open(polygon_file, 'r') as layer:
            for rec in layer:
                try:
                    from shapely.geometry import shape as _shape
                    g = _shape(rec['geometry'])
                    if g.is_valid and not g.is_empty:
                        geoms.append(g)
                except Exception:
                    pass

        if not geoms:
            return False, None

        union  = unary_union(geoms)
        bounds = union.bounds                    # (xmin, ymin, xmax, ymax)
        bbox   = box(*bounds)

        # Convex-hull of union vs exact bounding box
        sym_diff   = union.convex_hull.symmetric_difference(bbox)
        rel_diff   = sym_diff.area / bbox.area if bbox.area > 0 else 1.0
        is_rect    = rel_diff < 1e-4            # < 0.01 % area difference

        return is_rect, bounds

    def _collect_bbox_boundary_vertices(self, polygon_file: str,
                                         bounds: Tuple) -> Set[Tuple[float, float]]:
        """
        Return the set of quantized polygon vertices that lie on any of the
        four edges of the bounding box defined by *bounds*.

        A vertex is considered "on the edge" if its X coordinate is within one
        quantization unit of xmin or xmax, OR its Y is within one quantization
        unit of ymin or ymax.
        """
        xmin, ymin, xmax, ymax = bounds
        eps = self.quantitization_factor[0]   # 0.1 m for this dataset

        bbox_verts: Set = set()
        with fiona.open(polygon_file, 'r') as layer:
            for rec in layer:
                for coord in self._iter_all_coords(rec['geometry']):
                    x, y = coord[0], coord[1]
                    if (abs(x - xmin) <= eps or abs(x - xmax) <= eps or
                            abs(y - ymin) <= eps or abs(y - ymax) <= eps):
                        bbox_verts.add(self.quantitize(coord))

        return bbox_verts

    def _collect_exterior_vertices_from_shared_boundaries(
            self) -> Set[Tuple[float, float]]:
        """
        Identify vertices that lie exclusively on exterior polygon arcs — arcs
        that belong to only ONE polygon feature (i.e. they face the void outside
        the dataset, not another polygon).

        This method MUST be called after find_all_junctions_with_faults() (or
        find_all_junctions()) so that self.shared_boundaries is populated.

        Algorithm
        ---------
        For every segment tracked in shared_boundaries:
          • Count how many DISTINCT polygon features reference it.
          • Segment shared by ≥ 2 polygon features → interior segment.
          • Segment referenced by exactly 1 polygon feature → exterior segment.
        A vertex is "exterior-only" if it appears on at least one exterior
        segment and NEVER on an interior segment.  Vertices already in
        dict_junctions are excluded (they are already fully constrained).

        Fault features are ignored so that fault-polygon co-location does
        not incorrectly classify polygon exterior arcs as interior.

        Returns
        -------
        Set of quantized (x, y) coordinates.
        """
        on_interior: Set = set()   # ever seen on a polygon-polygon shared segment
        on_exterior: Set = set()   # ever seen on a polygon-only exterior segment

        for seg_hash, features in self.shared_boundaries.items():
            coords = self.boundary_segments.get(seg_hash)
            if coords is None:
                continue

            p1 = self.quantitize(coords[0])
            p2 = self.quantitize(coords[1])

            # Count polygon-only features for this segment
            poly_count = 0
            for f in features:
                if isinstance(f, dict):
                    if f.get('type') == 'polygon':
                        poly_count += 1
                elif isinstance(f, str):
                    # Plain-string feature IDs use the naming convention
                    # 'polygon_N_exterior' or 'polygon_N_interior_M'
                    if 'polygon' in f and 'fault' not in f:
                        poly_count += 1

            if poly_count >= 2:
                on_interior.add(p1)
                on_interior.add(p2)
            elif poly_count == 1:
                on_exterior.add(p1)
                on_exterior.add(p2)

        # Pure exterior = on exterior segment but NEVER on interior segment
        pure_exterior = on_exterior - on_interior

        # Exclude vertices already fully constrained as junctions
        pure_exterior -= set(self.dict_junctions.keys())

        return pure_exterior

    def detect_and_apply_boundary_preservation(
            self,
            polygon_file:      str,
            boundary_preserve: str   = "hard",
            soft_tolerance:    float = None,
            threshold:         float = None) -> Dict:
        """
        Detect the dataset boundary type and apply the chosen preservation mode.

        Call AFTER find_all_junctions_with_faults() (or find_all_junctions())
        because this method relies on self.shared_boundaries being populated.

        Parameters
        ----------
        polygon_file      : path to the polygon shapefile
        boundary_preserve : "hard" (default) or "soft"
        soft_tolerance    : area threshold for soft-exterior vertices (m²).
                            Default = threshold / 10 when not specified.
        threshold         : main simplification threshold (used to derive
                            soft_tolerance default and the area scale factor)

        Returns
        -------
        Dict with detection and pinning statistics.
        """
        if soft_tolerance is None:
            soft_tolerance = (threshold / 10.0) if threshold else 1.0

        stats = {
            'is_rectangular':   False,
            'boundary_preserve': boundary_preserve,
            'soft_tolerance':   soft_tolerance,
            'hard_pinned':      0,
            'soft_exterior':    0,
            'mode':             None,
        }

        # ── Step 1: detect rectangular vs non-rectangular ───────────────────
        is_rect, bounds = self._is_rectangular_dataset(polygon_file)
        stats['is_rectangular'] = is_rect

        if is_rect:
            # Always hard-pin regardless of boundary_preserve setting
            bbox_verts = self._collect_bbox_boundary_vertices(polygon_file, bounds)
            new_pins   = 0
            for v in bbox_verts:
                if v not in self.dict_junctions:
                    self.dict_junctions[v] = 1
                    new_pins += 1

            stats['mode']        = 'rectangular_bbox_hard_pin'
            stats['hard_pinned'] = new_pins
            print(f"  [boundary] Dataset is RECTANGULAR — bbox edges hard-pinned.")
            print(f"  [boundary] Bounding box: "
                  f"x=[{bounds[0]:.1f}, {bounds[2]:.1f}], "
                  f"y=[{bounds[1]:.1f}, {bounds[3]:.1f}]")
            print(f"  [boundary] New vertices pinned: {new_pins}  "
                  f"(already-junctions skipped)")

        else:
            # ── Non-rectangular: apply boundary_preserve mode ────────────────
            ext_verts = self._collect_exterior_vertices_from_shared_boundaries()

            if boundary_preserve == "hard":
                new_pins = 0
                for v in ext_verts:
                    if v not in self.dict_junctions:
                        self.dict_junctions[v] = 1
                        new_pins += 1

                stats['mode']        = 'exterior_arc_hard_pin'
                stats['hard_pinned'] = new_pins
                print(f"  [boundary] Dataset is NON-RECTANGULAR, mode='hard'.")
                print(f"  [boundary] Exterior arc vertices hard-pinned: {new_pins}")

            elif boundary_preserve == "soft":
                self.exterior_vertices = ext_verts

                # Scale factor: apparent_area = actual_area × soft_scale
                # Vertex removed when actual_area × soft_scale < threshold
                # → actual_area < threshold / soft_scale = soft_tolerance
                # → soft_scale = threshold / soft_tolerance
                if threshold and soft_tolerance > 0:
                    self.soft_scale = threshold / soft_tolerance
                else:
                    self.soft_scale = 10.0   # fallback 10× if no threshold given

                stats['mode']         = 'exterior_arc_soft_pin'
                stats['soft_exterior'] = len(ext_verts)
                print(f"  [boundary] Dataset is NON-RECTANGULAR, mode='soft'.")
                print(f"  [boundary] Exterior arc vertices (soft-scaled): "
                      f"{len(ext_verts)}")
                print(f"  [boundary] Soft tolerance: {soft_tolerance:.1f} m²  "
                      f"(scale = {self.soft_scale:.2f}×)")
            else:
                print(f"  [boundary] WARNING: unknown boundary_preserve='{boundary_preserve}'. "
                      f"No boundary action taken.")
                stats['mode'] = 'unknown'

        return stats

    # =========================================================================
    # TOPOLOGY ANALYSIS
    # Classifies every shared boundary segment as fault-polygon, polygon-
    # polygon, or other.  Marks both endpoints of topologically significant
    # segments (fault-polygon and polygon-polygon shared boundaries) as
    # critical junctions so they are never removed during simplification.
    # =========================================================================

    def _analyze_topological_significance(self):
        fault_polygon_shared = 0
        polygon_polygon_shared = 0
        other_shared = 0

        for segment_hash, feature_entries in self.shared_boundaries.items():
            if len(feature_entries) < 2:
                continue

            polygon_features = [f for f in feature_entries if
                                 (isinstance(f, dict) and f.get('type') == 'polygon') or
                                 (isinstance(f, str) and 'polygon' in f)]
            fault_features   = [f for f in feature_entries if
                                 (isinstance(f, dict) and f.get('type') == 'fault') or
                                 (isinstance(f, str) and 'fault' in f)]

            segment_coords = self.boundary_segments[segment_hash]
            p1 = self.quantitize(segment_coords[0])
            p2 = self.quantitize(segment_coords[1])

            if polygon_features and fault_features:
                fault_polygon_shared += 1
                self.fault_polygon_boundaries[segment_hash] = feature_entries
                self.topologically_significant_junctions.add(p1)
                self.topologically_significant_junctions.add(p2)
            elif len(polygon_features) >= 2:
                polygon_polygon_shared += 1
                self.polygon_polygon_boundaries[segment_hash] = feature_entries
                self.topologically_significant_junctions.add(p1)
                self.topologically_significant_junctions.add(p2)
            else:
                other_shared += 1

        print(f"    - Fault-polygon boundaries: {fault_polygon_shared}")
        print(f"    - Polygon-polygon boundaries: {polygon_polygon_shared}")
        print(f"    - Other shared boundaries (ignored): {other_shared}")
        print(f"    - Topologically significant junctions: "
              f"{len(self.topologically_significant_junctions)}")

        return {
            'fault_polygon_boundaries':  fault_polygon_shared,
            'polygon_polygon_boundaries': polygon_polygon_shared,
            'ignored_boundaries':         other_shared,
            'significant_junctions':      len(self.topologically_significant_junctions),
        }

    def find_all_junctions_with_faults(self, polygon_file, fault_file, junctions_dict):
        """Enhanced junction detection with refined constraint identification."""
        neighbors_dict = {}
        feature_count  = 0

        print(f"  Processing polygon dataset...")
        with fiona.open(polygon_file, 'r') as input_layer:
            print(f"    Found {len(input_layer)} polygon features")
            for record in input_layer:
                feature_id = f"polygon_{feature_count}"
                geometry   = record['geometry']
                geom_type  = geometry['type']
                self.geometry_registry[feature_id] = geometry

                if geom_type == 'Polygon':
                    self._append_junctions_typed(junctions_dict, neighbors_dict,
                                                 geometry['coordinates'][0],
                                                 f"{feature_id}_exterior", 'polygon')
                    for ri, ring in enumerate(geometry['coordinates'][1:]):
                        self._append_junctions_typed(junctions_dict, neighbors_dict,
                                                     ring,
                                                     f"{feature_id}_interior_{ri}", 'polygon')
                elif geom_type == 'MultiPolygon':
                    for pi, poly_coords in enumerate(geometry['coordinates']):
                        pfid = f"{feature_id}_poly_{pi}"
                        self._append_junctions_typed(junctions_dict, neighbors_dict,
                                                     poly_coords[0],
                                                     f"{pfid}_exterior", 'polygon')
                        for ri, ring in enumerate(poly_coords[1:]):
                            self._append_junctions_typed(junctions_dict, neighbors_dict,
                                                         ring,
                                                         f"{pfid}_interior_{ri}", 'polygon')
                feature_count += 1
                if feature_count % 100 == 0:
                    print(f"    Processed {feature_count} polygon features...")

        print(f"  Processing fault network dataset...")
        fault_count = 0
        with fiona.open(fault_file, 'r') as fault_layer:
            print(f"    Found {len(fault_layer)} fault features")
            for record in fault_layer:
                feature_id = f"fault_{fault_count}"
                geometry   = record['geometry']
                geom_type  = geometry['type']
                self.geometry_registry[feature_id] = geometry

                if geom_type == 'LineString':
                    self._append_junctions_typed(junctions_dict, neighbors_dict,
                                                 geometry['coordinates'],
                                                 feature_id, 'fault')
                elif geom_type == 'MultiLineString':
                    for li, line in enumerate(geometry['coordinates']):
                        self._append_junctions_typed(junctions_dict, neighbors_dict,
                                                     line,
                                                     f"{feature_id}_line_{li}", 'fault')
                elif geom_type == 'Polygon':
                    self._append_junctions_typed(junctions_dict, neighbors_dict,
                                                 geometry['coordinates'][0],
                                                 f"{feature_id}_exterior", 'fault')
                fault_count += 1
                if fault_count % 100 == 0:
                    print(f"    Processed {fault_count} fault features...")

        topology = self._analyze_topological_significance()
        print(f"Topology analysis complete:")
        print(f"  - Polygon features: {feature_count}, Fault features: {fault_count}")
        print(f"  - Total junctions: {len(junctions_dict)}")
        print(f"  - Topologically significant junctions: {topology['significant_junctions']}")

    def find_all_junctions(self, in_file, junctions_dict):
        """Junction detection for single-layer datasets."""
        neighbors_dict = {}
        feature_count  = 0

        with fiona.open(in_file, 'r') as input_layer:
            print(f"Analyzing {len(input_layer)} features for topology relationships...")
            for record in input_layer:
                feature_id = f"feature_{feature_count}"
                geometry   = record['geometry']
                geom_type  = geometry['type']
                self.geometry_registry[feature_id] = geometry

                if geom_type == 'LineString':
                    self._append_junctions(junctions_dict, neighbors_dict,
                                           geometry['coordinates'], feature_id)
                elif geom_type == 'MultiLineString':
                    for li, line in enumerate(geometry['coordinates']):
                        self._append_junctions(junctions_dict, neighbors_dict,
                                               line, f"{feature_id}_line_{li}")
                elif geom_type == 'Polygon':
                    self._append_junctions(junctions_dict, neighbors_dict,
                                           geometry['coordinates'][0],
                                           f"{feature_id}_exterior")
                    for ri, ring in enumerate(geometry['coordinates'][1:]):
                        self._append_junctions(junctions_dict, neighbors_dict,
                                               ring, f"{feature_id}_interior_{ri}")
                elif geom_type == 'MultiPolygon':
                    for pi, poly_coords in enumerate(geometry['coordinates']):
                        pfid = f"{feature_id}_poly_{pi}"
                        self._append_junctions(junctions_dict, neighbors_dict,
                                               poly_coords[0], f"{pfid}_exterior")
                        for ri, ring in enumerate(poly_coords[1:]):
                            self._append_junctions(junctions_dict, neighbors_dict,
                                                   ring, f"{pfid}_interior_{ri}")
                feature_count += 1
                if feature_count % 100 == 0:
                    print(f"  Processed {feature_count} features...")

        shared = {s: f for s, f in self.shared_boundaries.items() if len(f) > 1}
        print(f"Topology analysis complete:")
        print(f"  - Total junctions: {len(junctions_dict)}")
        print(f"  - Shared boundary segments: {len(shared)}")

    # =========================================================================
    # CONSTRAINT QUERIES
    # Given a list of vertex coordinates, returns the indices of those that
    # are present in dict_junctions (i.e. pinned and cannot be removed).
    # =========================================================================

    def get_shared_boundary_constraints_refined(self, points_list):
        constraint_indices = set()
        for i, point in enumerate(points_list):
            if self.quantitize(point) in self.dict_junctions:
                constraint_indices.add(i)
        return constraint_indices

    def get_shared_boundary_constraints(self, points_list):
        return self.get_shared_boundary_constraints_refined(points_list)

    # =========================================================================
    # ALGORITHMS 1–4: DECIMATION, DOUGLAS-PEUCKER, BEND SIMPLIFICATION
    # Simple non-topology-aware methods included for completeness.
    # These operate on raw geometry without junction pinning or arc caching.
    # =========================================================================

    def decimation(self, line, nth_point=2):
        if len(line.coords) <= 3 or nth_point <= 1:
            return line
        coords = list(line.coords)
        simplified = [coords[0]]
        for i in range(1, len(coords) - 1):
            if i % nth_point == 0:
                simplified.append(coords[i])
        simplified.append(coords[-1])
        return LineString(simplified)

    def douglas_peucker(self, geometry, tolerance):
        return geometry.simplify(tolerance, preserve_topology=False)

    def douglas_peucker_topology_preserving(self, geometry, tolerance):
        return geometry.simplify(tolerance, preserve_topology=True)

    def calculate_bend_compactness(self, p1, p2, p3):
        area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0
        s1 = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        s2 = math.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2)
        s3 = math.sqrt((p1[0]-p3[0])**2 + (p1[1]-p3[1])**2)
        perim = s1 + s2 + s3
        return (4 * math.pi * area) / (perim ** 2) if perim else 0

    def bend_simplify(self, line, compactness_threshold=0.7):
        if len(line.coords) <= 3:
            return line
        coords = list(line.coords)
        sig = [coords[0]]
        for i in range(1, len(coords) - 1):
            if self.calculate_bend_compactness(coords[i-1], coords[i], coords[i+1]) >= compactness_threshold:
                sig.append(coords[i])
        sig.append(coords[-1])
        return LineString(sig)

    # =========================================================================
    # ALGORITHM 5: VISVALINGAM-WHYATT (LINESTRING)
    # Area-based vertex removal for open line features.
    # Pinned vertices (in dict_junctions) get area = infinity and are skipped
    # by the heap entirely — they are never candidates for removal.
    # Soft-exterior vertices have their area multiplied by soft_scale so they
    # behave as if their triangle is larger, making them resistant to removal
    # until their actual triangle area falls below the soft_tolerance level.
    # =========================================================================

    def visvalingam_whyatt(self, line: LineString, threshold: float) -> LineString:
        """
        Visvalingam-Whyatt simplification for an open LineString.

        Repeatedly removes the vertex with the smallest triangle area formed
        by itself and its two neighbours until the smallest remaining area
        is ≥ threshold.  Pinned vertices (dict_junctions) are never removed.
        Soft-exterior vertices are scaled to resist removal until their
        triangle area falls below soft_tolerance.
        """
        if len(line.coords) <= 2:
            return line

        coords             = list(line.coords)
        constraint_indices = self.get_shared_boundary_constraints_refined(coords)
        use_soft           = bool(self.exterior_vertices)

        triangle_array = []
        for index in range(1, len(coords) - 1):
            is_constrained = index in constraint_indices
            # Apply soft-exterior scaling if this vertex is on the dataset boundary
            scale = 1.0
            if use_soft and not is_constrained:
                if self.quantitize(coords[index]) in self.exterior_vertices:
                    scale = self.soft_scale
            triangle_array.append(
                TriangleCalculator(coords[index], index, is_constrained, scale))

        if not triangle_array:
            return line

        start_tri = TriangleCalculator(coords[0],  0,               True)
        end_tri   = TriangleCalculator(coords[-1], len(coords) - 1, True)

        for i, t in enumerate(triangle_array):
            t.prevTriangle = triangle_array[i - 1] if i > 0 else start_tri
            t.nextTriangle = triangle_array[i + 1] if i < len(triangle_array) - 1 else end_tri
        if triangle_array:
            start_tri.nextTriangle = triangle_array[0]
            end_tri.prevTriangle   = triangle_array[-1]
        else:
            start_tri.nextTriangle = end_tri
            end_tri.prevTriangle   = start_tri

        unconstrained = [t for t in triangle_array if not t.is_constrained]
        print(f"    Line: {len(coords)} vertices, {len(constraint_indices)} constrained "
              f"(refined), {len(unconstrained)} removable")

        if not unconstrained:
            return line

        heapq.heapify(unconstrained)
        removed = 0

        while unconstrained:
            heapq.heapify(unconstrained)
            min_area = unconstrained[0].calcArea()
            if not math.isfinite(min_area) or min_area >= threshold:
                if min_area >= threshold:
                    print(f"    Cannot simplify further at threshold {threshold}: "
                          f"minimum remaining area = {unconstrained[0].calcArea():.4f}")
                break
            t = heapq.heappop(unconstrained)
            t.prevTriangle.nextTriangle = t.nextTriangle
            t.nextTriangle.prevTriangle = t.prevTriangle
            removed += 1

        print(f"    Removed {removed} vertices, threshold: {threshold}")

        simplified = [start_tri.point]
        node = start_tri.nextTriangle
        while node is not end_tri:
            simplified.append(node.point)
            node = node.nextTriangle
        simplified.append(end_tri.point)

        return LineString(simplified) if len(simplified) >= 2 else line

    def visvalingam_whyatt_ring(self, ring: LinearRing, threshold: float,
                                 minimum_points: int = 3) -> Optional[LinearRing]:
        """
        Visvalingam-Whyatt simplification for a closed LinearRing.

        Same area-based removal as the linestring variant but operates on a
        circular doubly-linked list so the ring closure is maintained.
        Pinned vertices are never removed.  Soft-exterior vertices are scaled.

        Self-intersection guard: before committing to any removal, checks
        whether the proposed replacement edge (prev → next) would cross any
        non-adjacent ring edge.  If it would, the vertex is kept for the
        remainder of the run (treated as constrained).

        Post-simplification validity check: if the finished ring is still not
        geometrically simple, repairs it with make_valid() and returns the
        exterior of the largest valid polygon part.  Returns None if repair
        fails (caller then writes the original geometry).
        """
        if len(ring.coords) - 1 <= minimum_points:
            return ring

        coords             = list(ring.coords[:-1])
        constraint_indices = self.get_shared_boundary_constraints_refined(coords)
        use_soft           = bool(self.exterior_vertices)

        triangle_ring = []
        for index, point in enumerate(coords):
            is_constrained = index in constraint_indices
            scale = 1.0
            if use_soft and not is_constrained:
                if self.quantitize(point) in self.exterior_vertices:
                    scale = self.soft_scale
            triangle_ring.append(
                TriangleCalculator(point, index, is_constrained, scale))

        n = len(triangle_ring)
        for i, t in enumerate(triangle_ring):
            t.prevTriangle = triangle_ring[(i - 1) % n]
            t.nextTriangle = triangle_ring[(i + 1) % n]

        unconstrained   = [t for t in triangle_ring if not t.is_constrained]
        constrained_cnt = sum(1 for t in triangle_ring if t.is_constrained)

        print(f"    Ring: {len(coords)} vertices, {len(constraint_indices)} constrained "
              f"(refined), {len(unconstrained)} removable")

        if not unconstrained:
            return ring

        heapq.heapify(unconstrained)
        removed = 0

        while unconstrained:
            heapq.heapify(unconstrained)
            if (len(unconstrained) + constrained_cnt) <= minimum_points:
                break
            min_area = unconstrained[0].calcArea()
            if not math.isfinite(min_area) or min_area >= threshold:
                if min_area >= threshold:
                    print(f"    Ring: cannot simplify further at threshold {threshold}: "
                          f"minimum remaining area = {unconstrained[0].calcArea():.4f}")
                break
            t = heapq.heappop(unconstrained)
            prev = t.prevTriangle
            nxt  = t.nextTriangle

            # ── Ring self-intersection guard ──────────────────────────────
            # Before committing to the removal of t, verify that the proposed
            # replacement edge (prev.point → nxt.point) does NOT cross any
            # non-adjacent edge in the current ring.  If it would, skip this
            # vertex — it stays in triangle_ring as an effectively-constrained
            # vertex and is never considered again (already popped from heap).
            #
            # This prevents the "bowtie" artefact in thin polygons where both
            # long edges, when simplified to near-straight lines, would cross
            # at the tapering tip.
            _new_p1 = prev.point
            _new_p2 = nxt.point
            _would_cross = False
            # Walk from nxt.next all the way around to prev (exclusive).
            # We skip the 4 edges directly adjacent to the new edge:
            #   prev.prev → prev   (shares endpoint prev)
            #   prev → t           (being removed)
            #   t → nxt            (being removed)
            #   nxt → nxt.next     (shares endpoint nxt)
            _chk = nxt.nextTriangle
            while _chk is not prev:
                _chk_nxt = _chk.nextTriangle
                if _chk_nxt is not prev:   # skip edge ending at prev (shares endpoint)
                    if _segments_cross(_new_p1, _new_p2, _chk.point, _chk_nxt.point):
                        _would_cross = True
                        break
                _chk = _chk_nxt
            if _would_cross:
                # Keep t in the ring; treat as constrained from now on.
                constrained_cnt += 1
                continue
            # ──────────────────────────────────────────────────────────────

            prev.nextTriangle = nxt
            nxt.prevTriangle  = prev
            for i, tr in enumerate(triangle_ring):
                if tr is t:
                    triangle_ring.pop(i)
                    break
            removed += 1

        print(f"    Ring removed {removed} vertices, threshold: {threshold}")

        if len(triangle_ring) < 3:
            return None

        start  = triangle_ring[0]
        coords_out = []
        node = start
        while True:
            coords_out.append(node.point)
            node = node.nextTriangle
            if node is start:
                break

        if len(coords_out) < 3:
            return None
        coords_out.append(coords_out[0])
        result_ring = LinearRing(coords_out)

        # ── Post-simplification ring validity check ────────────────────────
        # The per-removal guard above prevents IMMEDIATE crossings, but a
        # sequence of individually-safe removals from both sides of a thin
        # polygon can cumulatively produce a final ring where an early edge
        # and a late edge cross each other.
        #
        # If the finished ring is not geometrically simple, repair it with
        # make_valid():
        #   • For a bowtie: make_valid splits at the crossing and returns the
        #     larger of the two sub-areas (we accept a small area loss rather
        #     than outputting an invalid polygon)
        #   • Returns None only if repair fails; caller then writes the original
        if not result_ring.is_simple:
            try:
                _tmp_poly   = Polygon(result_ring)
                _repaired   = _make_valid(_tmp_poly)
                # make_valid may return MultiPolygon or GeometryCollection;
                # keep only the largest polygon part
                if _repaired and not _repaired.is_empty:
                    if _repaired.geom_type == 'Polygon':
                        result_ring = _repaired.exterior
                    elif _repaired.geom_type in ('MultiPolygon', 'GeometryCollection'):
                        _parts = [g for g in _repaired.geoms
                                  if g.geom_type == 'Polygon' and not g.is_empty]
                        if _parts:
                            _best = max(_parts, key=lambda g: g.area)
                            result_ring = _best.exterior
                        else:
                            return None
                    else:
                        return None
                else:
                    return None
            except Exception:
                return None   # safety: caller writes original

        return result_ring

    # =========================================================================
    # ALGORITHM 6: MODIFIED VISVALINGAM-WHYATT (TOPOLOGY-PRESERVING)
    # Cuts each polygon ring into arcs at junction points, simplifies each
    # arc independently using VW, caches the result so adjacent polygons
    # sharing the same arc use exactly the same simplified vertices
    # (watertightness), then reassembles the arcs into a closed ring.
    # Falls back to whole-ring simplification if the assembled ring is not
    # geometrically simple (two arcs crossing at a thin-polygon tip).
    # =========================================================================

    def _is_arc_shared(self, arc_coords):
        for i in range(len(arc_coords) - 1):
            seg = self.create_segment_hash(arc_coords[i], arc_coords[i + 1])
            if seg in self.shared_boundaries and len(self.shared_boundaries[seg]) >= 2:
                return True
        return False

    def _make_arc_cache_key(self, arc_coords):
        p_start = self.quantitize(arc_coords[0])
        p_end   = self.quantitize(arc_coords[-1])
        if p_start <= p_end:
            interior = self.quantitize(arc_coords[1]) if len(arc_coords) > 2 else None
            return (p_start, p_end, interior), False
        else:
            interior = self.quantitize(arc_coords[-2]) if len(arc_coords) > 2 else None
            return (p_end, p_start, interior), True

    def cut_line_by_junctions(self, line, junctions_dict):
        arcs, current = [], []
        for point in line.coords:
            qpt = self.quantitize(point)
            current.append(point)
            if qpt in junctions_dict and len(current) >= 2:
                arcs.append(current)
                current = [point]
        if len(current) > 1:
            arcs.append(current)
        return [LineString(a) for a in arcs if len(a) >= 2]

    def cut_polygon_by_junctions(self, polygon, junctions_dict):
        if not isinstance(polygon, Polygon):
            raise ValueError(f'Non-Polygon geometry: {polygon.geom_type}')
        junction_indices = [i for i, pt in enumerate(polygon.exterior.coords[:-1])
                            if self.quantitize(pt) in junctions_dict]
        if not junction_indices:
            return None, polygon
        return self.cut_line_by_junctions(polygon.exterior, junctions_dict), polygon

    def create_ring_from_arcs(self, arcs):
        if not arcs:
            return None
        ring_points = []
        for i, arc in enumerate(arcs):
            pts = list(arc.coords)
            ring_points.extend(pts if i == 0 else pts[1:])
        if len(ring_points) < 4:
            return None
        if self.quantitize(ring_points[0]) != self.quantitize(ring_points[-1]):
            ring_points.append(ring_points[0])
        return LinearRing(ring_points)

    def modified_visvalingam_whyatt(self, polygon: Polygon, threshold: float) -> Optional[Polygon]:
        """
        Topology-preserving VW simplification for a Polygon.

        If the polygon has junction constraints, the ring is cut into arcs at
        those junctions.  Each arc is looked up in the shared arc cache
        (watertightness: adjacent polygons reuse the same simplified arc rather
        than simplifying independently).  Uncached shared arcs are simplified
        and stored; non-shared arcs are simplified without caching.

        If the assembled ring is not geometrically simple, falls back to
        whole-ring simplification (which has the per-removal self-intersection
        guard).  The fallback result is not cached so the shared cache is not
        poisoned.  Soft-exterior scaling is propagated through the arc VW calls.
        """
        simplified_exterior = None

        if self.dict_junctions:
            arcs, _ = self.cut_polygon_by_junctions(polygon, self.dict_junctions)

            if arcs is None:
                simplified_exterior = self.visvalingam_whyatt_ring(polygon.exterior, threshold)
            else:
                simplified_arcs = []
                for arc in arcs:
                    arc_coords = list(arc.coords)
                    if len(arc_coords) < 2:
                        continue
                    cache_key, is_rev = self._make_arc_cache_key(arc_coords)
                    arc_shared = self._is_arc_shared(arc_coords)

                    if arc_shared and cache_key in self.dict_simple_arcs:
                        cached = self.dict_simple_arcs[cache_key]
                        simplified_arc = LineString(
                            list(reversed(cached)) if is_rev else cached)
                    else:
                        simplified_arc = self.visvalingam_whyatt(arc, threshold)
                        if arc_shared and simplified_arc and not simplified_arc.is_empty:
                            out = list(simplified_arc.coords)
                            # ── Arc cache poisoning guard ──────────────────────
                            # Only store the simplified arc in the shared cache
                            # if it is geometrically simple (no self-crossings).
                            # A self-intersecting arc stored here would be reused
                            # by every adjacent polygon that shares this boundary,
                            # propagating bowtie artefacts to multiple features.
                            # The post-assembly guard below (not is_simple →
                            # whole-ring fallback) handles the bad arc for THIS
                            # polygon without polluting the shared cache.
                            if simplified_arc.is_simple:
                                self.dict_simple_arcs[cache_key] = (
                                    list(reversed(out)) if is_rev else out)

                    if simplified_arc and not simplified_arc.is_empty:
                        simplified_arcs.append(simplified_arc)

                if not simplified_arcs:
                    return None
                simplified_exterior = self.create_ring_from_arcs(simplified_arcs)

                # ── Post-assembly self-intersection guard ──────────────────
                # Arcs are simplified independently so two arcs from the SAME
                # polygon can cross each other even though neither arc
                # self-intersects on its own.  This is the dominant cause of
                # bowtie artefacts in thin polygons: both long sides simplify
                # to near-straight lines that then cross at the tapering tip.
                #
                # If the assembled ring is not simple, fall back to whole-ring
                # simplification.  visvalingam_whyatt_ring() has its own
                # per-removal self-intersection guard and cannot produce a
                # crossing ring.  The fallback result is NOT cached so
                # adjacent polygons sharing this boundary are not affected.
                if (simplified_exterior is not None
                        and not simplified_exterior.is_simple):
                    simplified_exterior = self.visvalingam_whyatt_ring(
                        polygon.exterior, threshold)
        else:
            simplified_exterior = self.visvalingam_whyatt_ring(polygon.exterior, threshold)

        if simplified_exterior is None:
            return None

        simplified_interiors = []
        for ring in polygon.interiors:
            sr = self.visvalingam_whyatt_ring(ring, threshold)
            if sr is not None:
                simplified_interiors.append(sr)

        return Polygon(simplified_exterior, simplified_interiors)

    # =========================================================================
    # MAIN SIMPLIFICATION INTERFACE
    # Dispatches to the correct algorithm based on geometry type and method
    # name.  All six algorithms are accessible through simplify_geometry().
    # =========================================================================

    def simplify_geometry(self, geometry, method, threshold, **kwargs):
        if isinstance(geometry, LineString):
            return self._simplify_linestring(geometry, method, threshold, **kwargs)
        elif isinstance(geometry, MultiLineString):
            return self._simplify_multilinestring(geometry, method, threshold, **kwargs)
        elif isinstance(geometry, Polygon):
            return self._simplify_polygon(geometry, method, threshold, **kwargs)
        elif isinstance(geometry, MultiPolygon):
            return self._simplify_multipolygon(geometry, method, threshold, **kwargs)
        else:
            raise ValueError(f'Unsupported geometry type: {geometry.geom_type}')

    def _simplify_linestring(self, line, method, threshold, **kwargs):
        if method == 'decimation':
            nth = min(int(threshold), len(line.coords)//2) if threshold >= 1 else max(2, int(1/threshold))
            return self.decimation(line, nth)
        elif method == 'douglas_peucker':            return self.douglas_peucker(line, threshold)
        elif method == 'douglas_peucker_tp':         return self.douglas_peucker_topology_preserving(line, threshold)
        elif method == 'bend_simplify':              return self.bend_simplify(line, kwargs.get('compactness_threshold', threshold))
        elif method == 'visvalingam_whyatt':         return self.visvalingam_whyatt(line, threshold)
        elif method == 'modified_visvalingam_whyatt':
            if self.dict_junctions or self.shared_boundaries:
                arcs = self.cut_line_by_junctions(line, self.dict_junctions)
                simp = [self.visvalingam_whyatt(a, threshold) for a in arcs if a]
                if len(simp) == 1:   return simp[0]
                elif simp:           return MultiLineString(simp)
                return LineString()
            return self.visvalingam_whyatt(line, threshold)
        raise ValueError(f'Unknown method: {method}')

    def _simplify_multilinestring(self, mline, method, threshold, **kwargs):
        simp = [self._simplify_linestring(l, method, threshold, **kwargs)
                for l in mline.geoms]
        simp = [s for s in simp if s and not s.is_empty]
        return MultiLineString(simp) if simp else None

    def _simplify_polygon(self, polygon, method, threshold, **kwargs):
        if method in ('decimation', 'douglas_peucker', 'douglas_peucker_tp', 'bend_simplify',
                       'visvalingam_whyatt'):
            # Standard algorithms — operate directly on the polygon, no junction/arc logic
            if method == 'decimation':
                nth = min(int(threshold), len(polygon.exterior.coords)//2) if threshold >= 1 else max(2, int(1/threshold))
                ext = LinearRing(self.decimation(LineString(polygon.exterior.coords), nth).coords)
                ints = [LinearRing(self.decimation(LineString(r.coords), nth).coords) for r in polygon.interiors]
                return Polygon(ext, ints)
            elif method in ('douglas_peucker', 'douglas_peucker_tp'):
                return (self.douglas_peucker if method == 'douglas_peucker'
                        else self.douglas_peucker_topology_preserving)(polygon, threshold)
            elif method == 'bend_simplify':
                ct = kwargs.get('compactness_threshold', threshold)
                ext = LinearRing(self.bend_simplify(LineString(polygon.exterior.coords), ct).coords)
                ints = [LinearRing(self.bend_simplify(LineString(r.coords), ct).coords) for r in polygon.interiors]
                return Polygon(ext, ints)
            elif method == 'visvalingam_whyatt':
                ext = self.visvalingam_whyatt_ring(polygon.exterior, threshold)
                if ext is None: return None
                ints = [r for r in (self.visvalingam_whyatt_ring(i, threshold) for i in polygon.interiors) if r]
                return Polygon(ext, ints)
        elif method == 'modified_visvalingam_whyatt':
            return self.modified_visvalingam_whyatt(polygon, threshold)
        raise ValueError(f'Unknown method for polygons: {method}')

    def _simplify_multipolygon(self, mpoly, method, threshold, **kwargs):
        # Simplify each part of a MultiPolygon independently.  If any part
        # collapses to None or an empty geometry during simplification, the
        # original (unsimplified) part is written instead so the feature count
        # in the output always matches the input — no parts are silently dropped.
        result_parts = []
        for p in mpoly.geoms:
            s = self._simplify_polygon(p, method, threshold, **kwargs)
            if s and not s.is_empty:
                result_parts.append(s)
            else:
                result_parts.append(p)   # preserve original part
        return MultiPolygon(result_parts) if result_parts else None


# =============================================================================
# VALIDATION
# =============================================================================

def validate_topology_consistency(original_geoms, simplified_geoms):
    results = {'topology_preserved': True, 'errors': [], 'warnings': []}
    try:
        for i, (orig, simp) in enumerate(zip(original_geoms, simplified_geoms)):
            if simp is None:
                results['errors'].append(f'Geometry {i} simplified away')
                continue
            if not simp.is_valid:
                results['topology_preserved'] = False
                results['errors'].append(f'Geometry {i} invalid after simplification')
    except Exception as e:
        results['topology_preserved'] = False
        results['errors'].append(f'Validation error: {e}')
    return results


# =============================================================================
# STAGE 0: TOPOLOGY PRE-PROCESSING
# Runs before any simplification algorithm.  Snaps vertex coordinates to a
# decimal grid to eliminate floating-point mismatches, corrects hairline
# polygon overlaps and gaps, and bidirectionally snaps fault and polygon
# vertices that should coincide to exactly the same coordinate.
# =============================================================================

def _round_ring_coords(coords: list, decimals: int) -> list:
    """Round each (x, y) pair in a coordinate list to `decimals` decimal places."""
    return [(round(float(c[0]), decimals), round(float(c[1]), decimals))
            for c in coords]


def _round_geometry(geom, decimals: int = 7):
    """
    Return a new geometry with all vertex coordinates rounded to `decimals`
    decimal places.  make_valid() is applied afterwards to repair any
    degeneracies (e.g. collapsed rings) that rounding may introduce.
    """
    if geom is None or geom.is_empty:
        return geom

    if geom.geom_type == 'Polygon':
        ext   = _round_ring_coords(list(geom.exterior.coords), decimals)
        holes = [_round_ring_coords(list(r.coords), decimals)
                 for r in geom.interiors]
        try:
            g = Polygon(ext, holes)
        except Exception:
            return geom          # fallback: keep original if ring is degenerate
        return make_valid(g) if not g.is_valid else g

    elif geom.geom_type == 'MultiPolygon':
        parts = [_round_geometry(p, decimals) for p in geom.geoms]
        parts = [p for p in parts if p is not None and not p.is_empty]
        if not parts:
            return geom
        g = MultiPolygon(parts)
        return make_valid(g) if not g.is_valid else g

    return geom          # LineString, Point, etc — pass through


def _build_vertex_snap_map(
    source_coords: List[Tuple[float, float]],
    target_coords: List[Tuple[float, float]],
    snap_distance: float,
    snap_decimals: int,
) -> Dict[Tuple[float, float], Tuple[float, float]]:
    """
    For each SOURCE vertex within snap_distance of any TARGET vertex, return
    a mapping {source_coord: canonical_target_coord}.

    The canonical target coordinate is the nearest target vertex rounded to
    snap_decimals decimal places.  SOURCE vertices with no target within
    snap_distance are not included in the map (they keep their own coordinate).

    Parameters
    ----------
    source_coords : list of (x, y) — vertices to potentially move
    target_coords : list of (x, y) — authoritative anchor vertices
    snap_distance : maximum distance (CRS units) to snap across
    snap_decimals : decimal places to round the canonical coordinate to
    """
    if not source_coords or not target_coords:
        return {}

    from shapely.geometry import Point as _Pt

    # Build spatial index on target vertices
    tgt_pts  = [_Pt(x, y) for x, y in target_coords]
    tgt_tree = STRtree(tgt_pts)

    snap_map: Dict = {}
    for sx, sy in source_coords:
        if (sx, sy) in snap_map:
            continue                      # already resolved
        src_pt  = _Pt(sx, sy)
        hits    = tgt_tree.query(src_pt.buffer(snap_distance), predicate='intersects')
        if len(hits) == 0:
            continue
        # Nearest target vertex within snap_distance
        nearest = min(hits,
                      key=lambda i: (target_coords[i][0] - sx) ** 2
                                  + (target_coords[i][1] - sy) ** 2)
        tx, ty  = target_coords[nearest]
        dist    = ((tx - sx) ** 2 + (ty - sy) ** 2) ** 0.5
        if dist <= snap_distance:
            snap_map[(sx, sy)] = (
                round(tx, snap_decimals),
                round(ty, snap_decimals),
            )
    return snap_map


def _build_midpoint_snap_maps(
    fault_coords:  List[Tuple[float, float]],
    poly_coords:   List[Tuple[float, float]],
    snap_distance: float,
    snap_decimals: int,
) -> Tuple[Dict, Dict]:
    """
    For every fault vertex within snap_distance of a polygon vertex, compute
    the midpoint between the two and return snap maps that move BOTH vertices
    to that shared midpoint.

    WHY MIDPOINT:
    One-sided bidirectional snapping (fault → polygon, then polygon → fault)
    causes vertices that are millimetres apart to swap positions: the fault
    ends up where the polygon was, and the polygon ends up where the fault was.
    Snapping both to the midpoint eliminates the swap — each vertex travels
    half the gap, and they both arrive at exactly the same coordinate.

    Parameters
    ----------
    fault_coords  : list of (x, y) fault vertex coordinates
    poly_coords   : list of (x, y) polygon vertex coordinates
    snap_distance : maximum separation (CRS units) to snap across
    snap_decimals : decimal places to round the shared midpoint coordinate to

    Returns
    -------
    (fault_snap_map, poly_snap_map)
      fault_snap_map : {fault_coord → midpoint}  for every paired fault vertex
      poly_snap_map  : {poly_coord  → midpoint}  for every paired polygon vertex
    """
    if not fault_coords or not poly_coords:
        return {}, {}

    from shapely.geometry import Point as _Pt

    # Build a spatial index on polygon vertices for fast proximity queries
    poly_pts  = [_Pt(x, y) for x, y in poly_coords]
    poly_tree = STRtree(poly_pts)

    fault_snap_map: Dict = {}
    poly_snap_map:  Dict = {}

    for fx, fy in fault_coords:
        if (fx, fy) in fault_snap_map:
            continue  # this fault vertex was already matched to a polygon vertex
        f_pt = _Pt(fx, fy)
        hits = poly_tree.query(f_pt.buffer(snap_distance), predicate='intersects')
        if not len(hits):
            continue  # no polygon vertex is within snap_distance of this fault vertex

        # Choose the nearest polygon vertex among candidates
        nearest = min(hits,
                      key=lambda i: (poly_coords[i][0] - fx) ** 2
                                  + (poly_coords[i][1] - fy) ** 2)
        px, py = poly_coords[nearest]
        dist   = ((px - fx) ** 2 + (py - fy) ** 2) ** 0.5
        if dist > snap_distance:
            continue  # nearest is actually outside tolerance (buffer overestimates)

        # Midpoint: both vertices move here — no swap, no one-sided bias
        mx = round((fx + px) / 2.0, snap_decimals)
        my = round((fy + py) / 2.0, snap_decimals)

        fault_snap_map[(fx, fy)] = (mx, my)
        poly_snap_map[(px, py)]  = (mx, my)

    return fault_snap_map, poly_snap_map


def _apply_snap_map_to_ring(coords: list,
                             snap_map: Dict,
                             snap_decimals: int) -> list:
    """Apply a snap map to a coordinate list, rounding everything to snap_decimals."""
    out = []
    for c in coords:
        xy = (float(c[0]), float(c[1]))
        if xy in snap_map:
            out.append(snap_map[xy])
        else:
            out.append((round(xy[0], snap_decimals), round(xy[1], snap_decimals)))
    return out


def _apply_snap_map_to_geom(geom, snap_map: Dict, snap_decimals: int):
    """Apply a snap map to all vertices of a Shapely polygon geometry."""
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == 'Polygon':
        ext   = _apply_snap_map_to_ring(list(geom.exterior.coords),
                                        snap_map, snap_decimals)
        holes = [_apply_snap_map_to_ring(list(r.coords), snap_map, snap_decimals)
                 for r in geom.interiors]
        try:
            g = Polygon(ext, holes)
            return make_valid(g) if not g.is_valid else g
        except Exception:
            return geom
    elif geom.geom_type == 'MultiPolygon':
        parts = [_apply_snap_map_to_geom(p, snap_map, snap_decimals)
                 for p in geom.geoms]
        parts = [p for p in parts if p is not None and not p.is_empty]
        g = MultiPolygon(parts) if parts else geom
        return make_valid(g) if not g.is_valid else g
    return geom


def _collect_all_coords_from_features(
        features: List[Dict]) -> List[Tuple[float, float]]:
    """Extract all unique (x, y) vertex coordinates from a list of feature dicts."""
    seen: set = set()
    out: List = []
    for feat in features:
        g = feat.get('geom')
        if g is None or g.is_empty:
            continue
        rings = []
        if g.geom_type == 'Polygon':
            rings = [g.exterior] + list(g.interiors)
        elif g.geom_type == 'MultiPolygon':
            for part in g.geoms:
                rings += [part.exterior] + list(part.interiors)
        for ring in rings:
            for c in ring.coords:
                xy = (float(c[0]), float(c[1]))
                if xy not in seen:
                    seen.add(xy)
                    out.append(xy)
    return out


def _segments_cross(p1: tuple, p2: tuple, q1: tuple, q2: tuple) -> bool:
    """
    Return True if segments p1-p2 and q1-q2 PROPERLY cross.

    "Properly" means they intersect at an interior point — touching at
    an endpoint is NOT considered a crossing (returns False), so adjacent
    ring edges that share a vertex are never flagged.

    Uses the parametric / cross-product test: O(1), no Shapely overhead.
    """
    p1x, p1y = p1; p2x, p2y = p2
    q1x, q1y = q1; q2x, q2y = q2

    rx = p2x - p1x;   ry = p2y - p1y      # direction of p1→p2
    sx = q2x - q1x;   sy = q2y - q1y      # direction of q1→q2
    denom = rx * sy - ry * sx              # r × s

    if abs(denom) < 1e-12:
        return False   # parallel or collinear

    qpx = q1x - p1x;  qpy = q1y - p1y    # q1 - p1
    t = (qpx * sy - qpy * sx) / denom     # parameter along p
    u = (qpx * ry - qpy * rx) / denom     # parameter along q

    # Strict interior crossing — exclude the [0, 1] endpoints so that
    # segments sharing a vertex (e.g. consecutive ring edges) don't trigger.
    return 1e-10 < t < 1.0 - 1e-10 and 1e-10 < u < 1.0 - 1e-10


def _collect_fault_coords(fault_file: str) -> List[Tuple[float, float]]:
    """Extract all unique (x, y) vertex coordinates from a fault shapefile."""
    seen: set = set()
    out: List = []
    with fiona.open(fault_file, 'r') as src:
        for feat in src:
            g = feat['geometry']
            if g is None:
                continue
            t = g['type']
            if t == 'LineString':
                raw = [g['coordinates']]
            elif t == 'MultiLineString':
                raw = g['coordinates']
            else:
                raw = []
            for line in raw:
                for c in line:
                    xy = (float(c[0]), float(c[1]))
                    if xy not in seen:
                        seen.add(xy)
                        out.append(xy)
    return out


# =============================================================================
# UNIT FIELD INSPECTION
# Reads the polygon shapefile schema, identifies which field contains the
# geological unit identifier (e.g. CODE), and prints a formatted report
# of all fields with sample values and candidate suggestions.
# =============================================================================

def inspect_unit_field(polygon_file: str, unit_field: str = None) -> Dict:
    """
    Inspect a polygon shapefile's attribute schema and identify which field
    contains the geological unit identifier (the 'rock unit field').

    Parameters
    ----------
    polygon_file : path to polygon shapefile
    unit_field   : the field name to validate.  If None, auto-detect candidates.

    Returns
    -------
    dict with keys:
        'fields'          : list of all field names in the schema
        'field_types'     : {field_name: fiona_type_string}
        'sample_values'   : {field_name: [up to 5 unique sample values]}
        'candidates'      : list of field names that look like unit identifiers
        'unit_field_ok'   : True if unit_field exists in the schema
        'unit_field'      : the validated (or best-candidate) field name
        'suggestion'      : human-readable suggestion string

    Side-effect
    -----------
    Prints a formatted report to stdout so the user can see what is available.
    """
    with fiona.open(polygon_file, 'r') as layer:
        schema     = layer.schema
        prop_schema = schema.get('properties', {})
        n_features = len(layer)

        # Collect field info and sample values
        field_types: Dict[str, str] = dict(prop_schema)
        fields: List[str] = list(prop_schema.keys())
        sample_values: Dict[str, List] = {f: [] for f in fields}
        seen_vals: Dict[str, set] = {f: set() for f in fields}

        for rec in layer:
            props = rec['properties']
            for f in fields:
                v = props.get(f)
                sv = str(v).strip() if v is not None else ''
                if sv and sv not in seen_vals[f] and len(sample_values[f]) < 5:
                    sample_values[f].append(sv)
                    seen_vals[f].add(sv)

    # Heuristic: fields that look like unit/lithology identifiers
    _UNIT_KEYWORDS = {
        'code', 'unit', 'lith', 'rock', 'type', 'formation',
        'strat', 'map', 'symbol', 'abbrev', 'short',
    }
    candidates: List[str] = []
    for f in fields:
        f_lower = f.lower()
        # Must be a string-type field
        if not field_types[f].startswith('str'):
            continue
        if any(kw in f_lower for kw in _UNIT_KEYWORDS):
            candidates.append(f)
        # Also include short string fields that look like codes
        elif len(sample_values[f]) >= 2:
            avg_len = sum(len(v) for v in sample_values[f]) / max(len(sample_values[f]), 1)
            if avg_len <= 20:          # short values → likely a code
                candidates.append(f)

    # Remove duplicates while preserving order
    seen_c: set = set()
    candidates = [c for c in candidates if not (c in seen_c or seen_c.add(c))]

    # Validate specified unit_field
    unit_field_ok = unit_field in fields if unit_field else False

    # Best suggestion
    if unit_field_ok:
        suggestion = f"'{unit_field}' found in schema — OK."
    elif unit_field:
        if candidates:
            suggestion = (f"'{unit_field}' NOT FOUND.  "
                          f"Did you mean one of: {candidates}?")
        else:
            suggestion = (f"'{unit_field}' NOT FOUND and no obvious candidates detected.  "
                          f"Check the field list above.")
    else:
        suggestion = (f"No unit_field specified.  "
                      f"Likely candidates: {candidates}" if candidates
                      else "No obvious unit-code field found — check field list above.")

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Polygon attribute schema  ({polygon_file.split('/')[-1].split(chr(92))[-1]})")
    print(f"  {n_features} features   |   {len(fields)} fields")
    print(f"{'─'*70}")
    print(f"  {'Field name':25s}  {'Type':10s}  Sample values")
    print(f"  {'─'*25}  {'─'*10}  {'─'*30}")
    for f in fields:
        samples = ', '.join(sample_values[f][:4]) if sample_values[f] else '(empty)'
        marker  = '  ◄ candidate' if f in candidates else ''
        if unit_field and f == unit_field:
            marker = '  ◄ SELECTED' if unit_field_ok else '  ✗ NOT FOUND'
        print(f"  {f:25s}  {field_types[f]:10s}  {samples}{marker}")
    print(f"{'─'*70}")
    print(f"  {suggestion}")
    if not unit_field_ok and candidates:
        print(f"  Pass  unit_field='{candidates[0]}'  (or whichever is correct)")
    print(f"{'─'*70}\n")

    return {
        'fields':        fields,
        'field_types':   field_types,
        'sample_values': sample_values,
        'candidates':    candidates,
        'unit_field_ok': unit_field_ok,
        'unit_field':    unit_field if unit_field_ok else (candidates[0] if candidates else None),
        'suggestion':    suggestion,
    }


def preprocess_topology(
    input_file:    str,
    output_file:   str   = None,
    snap_decimals: int   = 7,
    fix_overlaps:  bool  = True,
    fix_gaps:      bool  = True,
    fault_file:    str   = None,       # if given, bidirectionally snap fault↔polygon vertices
    snap_distance: float = 0.01,       # maximum distance (m) for fault↔polygon vertex snap
    verbose:       bool  = True,
) -> Tuple[str, Optional[str], Dict]:
    """
    Topology pre-processing (Stage 0): runs four steps in this order —

      Step 1  Snap polygon coordinates to decimal grid
      Step 2  Fix polygon–polygon overlaps
      Step 3  Fix polygon mosaic gaps
      Step 4  Bidirectional fault↔polygon vertex snap

    The cross-dataset snap (Step 4) intentionally runs LAST — after the
    topology fix has settled the final positions of all polygon vertices.
    Snapping before topology fixing caused the fault to align with polygon
    positions that were subsequently moved by the overlap/gap correction,
    leaving the fault misaligned with the corrected boundary.

    This runs BEFORE any of the six vector simplification methods and
    directly addresses:
      (a) Polygon-only topology:  hairline overlaps and gaps due to
          floating-point mismatches between adjacent polygon rings.
      (b) Fault-polygon alignment (MVW only):  fault and polygon vertices
          that should coincide differ by up to snap_distance metres due to
          digitising in separate sessions or different CRS precision.

    Cross-dataset snap logic (Step 4, when fault_file is provided)
    ---------------------------------------------------------------
    Two passes are made, each using ORIGINAL vertex coordinates as anchors
    so that the canonical coordinate for every mismatched pair is consistent:

      Pass 1  fault → polygon
              For every fault vertex within snap_distance of a polygon vertex,
              the fault vertex is moved to exactly match the polygon vertex
              (rounded to snap_decimals decimal places).
              The polygon vertex does NOT move — it is the authoritative anchor.

      Pass 2  polygon → fault  (using original, pre-pass-1 fault coordinates)
              For every polygon vertex within snap_distance of an ORIGINAL fault
              vertex, the polygon vertex is moved to exactly match that fault
              vertex (rounded to snap_decimals decimal places).
              This handles the complementary case where the polygon boundary is
              slightly off from where the fault runs.

    After both passes every fault-polygon vertex pair that was within
    snap_distance now shares exactly the same coordinate at snap_decimals
    decimal-place precision (≤ 0.0000001 m with the default of 7).

    Parameters
    ----------
    input_file    : polygon shapefile
    output_file   : corrected polygon output path
                    (default: <stem>_topo_fixed.shp alongside input)
    snap_decimals : grid precision for all coordinate rounding (default 7)
    fix_overlaps  : detect and fix polygon–polygon overlaps (default True)
    fix_gaps      : detect and fill polygon mosaic gaps (default True)
    fault_file    : fault network shapefile — enables cross-dataset snap
    snap_distance : max distance (m) between fault and polygon vertices to
                    treat as the same point (default 0.01 m = 1 cm)
    verbose       : print progress (default True)

    Returns
    -------
    (poly_output_path : str,
     fault_output_path : str or None,
     stats : dict)

    stats keys
    ----------
    n_features           — polygon features loaded
    n_poly_grid_snapped  — polygon features whose coordinates changed on grid snap
    n_overlaps           — polygon–polygon overlaps corrected
    overlap_area_total   — total overlap area (m²)
    n_gaps               — polygon mosaic gaps corrected
    gap_area_total       — total gap area (m²)
    n_fault_snapped      — fault vertices snapped to polygon (pass 1)
    n_poly_snapped       — polygon vertices snapped to fault  (pass 2)
    elapsed_s            — wall-clock seconds
    """
    t0 = time.time()
    noise_area    = 10 ** (-snap_decimals)
    snapped_fault = None          # will be set if fault_file is provided
    n_fault_snapped = 0
    n_poly_snapped  = 0

    if verbose:
        print(f"\n{'='*70}")
        print(f"  STAGE 0: Topology pre-processing")
        print(f"{'='*70}")
        print(f"  Input         : {os.path.basename(input_file)}")
        print(f"  Snap decimals : {snap_decimals}  "
              f"(grid = {10**-snap_decimals:.0e} m)")
        print(f"  Noise floor   : {noise_area:.0e} m²")
        if fault_file:
            print(f"  Fault file    : {os.path.basename(fault_file)}")
            print(f"  Snap distance : {snap_distance} m  "
                  f"(cross-dataset fault↔polygon snap)")

    # ── Step 1: Load polygons and snap vertices to decimal grid ───────────────
    features          = []
    n_poly_grid_snapped = 0

    with fiona.open(input_file, 'r') as src:
        meta = src.meta.copy()
        for feat in src:
            try:
                from shapely.geometry import shape as _shape
                g_orig = _shape(feat['geometry'])
                g_snap = _round_geometry(g_orig, snap_decimals)
                if not g_snap.equals(g_orig):
                    n_poly_grid_snapped += 1
                features.append({
                    'fid':   feat.get('id', str(len(features))),
                    'props': dict(feat['properties']),
                    'geom':  g_snap,
                })
            except Exception as exc:
                if verbose:
                    print(f"  Warning: skipped polygon feature — {exc}")

    n = len(features)
    if verbose:
        print(f"  Polygon features loaded      : {n:,}")
        print(f"  Polygon grid-snapped         : {n_poly_grid_snapped:,}")

    # ── Step 2: Polygon–polygon overlap detection and correction ─────────────
    n_overlaps       = 0
    overlap_area_sum = 0.0

    if fix_overlaps and n > 1:
        if verbose:
            print(f"  Checking polygon overlaps …", flush=True)

        geom_list = [f['geom'] for f in features]
        tree      = STRtree(geom_list)

        for i in range(n):
            gi = features[i]['geom']
            if gi is None or gi.is_empty:
                continue
            for j in tree.query(gi, predicate='intersects'):
                if j <= i:
                    continue
                gj = features[j]['geom']
                if gj is None or gj.is_empty:
                    continue
                try:
                    inter = gi.intersection(gj)
                except Exception:
                    continue
                if inter.is_empty or inter.area <= noise_area:
                    continue
                n_overlaps       += 1
                overlap_area_sum += inter.area
                try:
                    fixed_j = gj.difference(inter)
                    if not fixed_j.is_empty:
                        features[j]['geom'] = (make_valid(fixed_j)
                                               if not fixed_j.is_valid else fixed_j)
                    if verbose and n_overlaps <= 5:
                        print(f"    Overlap {n_overlaps:>3}: features {i}↔{j}  "
                              f"area={inter.area:.4e} m²  → fixed")
                    elif verbose and n_overlaps == 6:
                        print(f"    … (further overlaps suppressed)")
                except Exception as exc:
                    if verbose:
                        print(f"    Overlap fix error ({i},{j}): {exc}")

        if verbose:
            print(f"  Overlaps found : {n_overlaps:,}  "
                  f"({overlap_area_sum:.4e} m² total)")

    # ── Step 3: Gap detection and correction ──────────────────────────────────
    n_gaps       = 0
    gap_area_sum = 0.0

    if fix_gaps and n > 1:
        if verbose:
            print(f"  Checking polygon gaps …", flush=True)

        union = unary_union([f['geom'] for f in features
                             if f['geom'] is not None and not f['geom'].is_empty])
        raw_gaps: List = []
        if union.geom_type == 'Polygon':
            raw_gaps = [Polygon(hole) for hole in union.interiors]
        elif union.geom_type == 'MultiPolygon':
            for poly in union.geoms:
                raw_gaps.extend([Polygon(hole) for hole in poly.interiors])

        real_gaps    = [g for g in raw_gaps if g.area > noise_area]
        n_gaps       = len(real_gaps)
        gap_area_sum = sum(g.area for g in real_gaps)

        if verbose:
            print(f"  Gaps found : {n_gaps:,}  ({gap_area_sum:.4e} m² total)")

        if real_gaps:
            tree2 = STRtree([f['geom'] for f in features])
            for gi_idx, gap in enumerate(real_gaps):
                cands       = tree2.query(gap, predicate='intersects')
                best_idx    = -1
                best_length = 0.0
                gap_bdy     = gap.boundary
                for idx in cands:
                    feat_g = features[idx]['geom']
                    if feat_g is None or feat_g.is_empty:
                        continue
                    try:
                        shared = feat_g.boundary.intersection(gap_bdy)
                        length = shared.length if not shared.is_empty else 0.0
                    except Exception:
                        length = 0.0
                    if length > best_length:
                        best_length = length
                        best_idx    = idx
                if best_idx >= 0:
                    try:
                        merged = features[best_idx]['geom'].union(gap)
                        features[best_idx]['geom'] = (make_valid(merged)
                                                      if not merged.is_valid else merged)
                        if verbose and gi_idx < 5:
                            print(f"    Gap {gi_idx+1:>3}: "
                                  f"area={gap.area:.4e} m²  "
                                  f"→ assigned to feature {best_idx}")
                    except Exception as exc:
                        if verbose:
                            print(f"    Gap assign error "
                                  f"(gap {gi_idx}, feat {best_idx}): {exc}")

    # ── Step 4: Cross-dataset fault↔polygon vertex snap (midpoint) ───────────
    # Runs AFTER overlap/gap fixing so the snap targets the final, corrected
    # polygon positions rather than positions that will subsequently be moved.
    #
    # MIDPOINT APPROACH (replaces two-pass bidirectional snap):
    # The previous two-pass approach (fault → polygon, then polygon → fault)
    # caused a position-swap bug: when a fault vertex F and a polygon vertex P
    # were millimetres apart, Pass 1 moved F to P, and Pass 2 moved P to the
    # original F position — they simply traded places.  The fix is to compute
    # the midpoint M = (F+P)/2 and move both vertices to M in a single pass.
    if fault_file:
        if verbose:
            print(f"  Cross-dataset snap (midpoint) …", flush=True)

        # Collect coordinates from the topology-corrected polygon features
        poly_coords  = _collect_all_coords_from_features(features)
        fault_coords = _collect_fault_coords(fault_file)

        if verbose:
            print(f"    Polygon vertices (unique) : {len(poly_coords):,}")
            print(f"    Fault vertices  (unique)  : {len(fault_coords):,}")
            print(f"    Snapping paired vertices to midpoint  "
                  f"(snap_distance = {snap_distance} m) …", flush=True)

        # Build both snap maps in one pass — every matched pair goes to its midpoint
        fault_snap_map, poly_snap_map = _build_midpoint_snap_maps(
            fault_coords  = fault_coords,
            poly_coords   = poly_coords,
            snap_distance = snap_distance,
            snap_decimals = snap_decimals,
        )
        n_fault_snapped = len(fault_snap_map)
        n_poly_snapped  = len(poly_snap_map)

        if verbose:
            print(f"    Fault vertices snapped to midpoint   : {n_fault_snapped:,}")
            print(f"    Polygon vertices snapped to midpoint : {n_poly_snapped:,}")

        # Write snapped fault shapefile using the fault midpoint map
        fault_stem    = os.path.splitext(os.path.basename(fault_file))[0]
        out_dir       = os.path.dirname(os.path.abspath(
                            output_file if output_file else input_file))
        snapped_fault = os.path.join(out_dir, f"{fault_stem}_sn.shp")

        with fiona.open(fault_file, 'r') as fsrc:
            fmeta = fsrc.meta.copy()
            with fiona.open(snapped_fault, 'w', **fmeta) as fdst:
                for frec in fsrc:
                    g  = frec['geometry']
                    gt = g['type']
                    if gt == 'LineString':
                        new_coords = _apply_snap_map_to_ring(
                            g['coordinates'], fault_snap_map, snap_decimals)
                        new_geom = {'type': 'LineString', 'coordinates': new_coords}
                    elif gt == 'MultiLineString':
                        new_lines = [_apply_snap_map_to_ring(
                            line, fault_snap_map, snap_decimals)
                            for line in g['coordinates']]
                        new_geom = {'type': 'MultiLineString',
                                    'coordinates': new_lines}
                    else:
                        new_geom = g
                    fdst.write({'geometry': new_geom,
                                'properties': dict(frec['properties'])})

        # Apply the polygon midpoint snap map to all in-memory polygon features
        if poly_snap_map:
            for feat in features:
                feat['geom'] = _apply_snap_map_to_geom(
                    feat['geom'], poly_snap_map, snap_decimals)

    # ── Step 5: Write corrected polygon shapefile ─────────────────────────────
    if output_file is None:
        stem        = os.path.splitext(input_file)[0]
        output_file = f"{stem}_topo_fixed.shp"

    has_multi = any(
        f['geom'] is not None and f['geom'].geom_type == 'MultiPolygon'
        for f in features
    )
    if has_multi:
        meta['schema']['geometry'] = 'MultiPolygon'

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    written = 0
    with fiona.open(output_file, 'w', **meta) as dst:
        for feat in features:
            g = feat['geom']
            if g is None or g.is_empty:
                continue
            try:
                if has_multi and g.geom_type == 'Polygon':
                    g = MultiPolygon([g])
                dst.write({'geometry': mapping(g), 'properties': feat['props']})
                written += 1
            except Exception as exc:
                if verbose:
                    print(f"  Write error (fid={feat['fid']}): {exc}")

    elapsed = time.time() - t0

    if verbose:
        print(f"\n  ── Stage 0 Summary ──────────────────────────────────────")
        print(f"  Polygon features loaded      : {n:,}")
        print(f"  Polygon grid-snapped         : {n_poly_grid_snapped:,} features")
        if fault_file:
            print(f"  Fault  → polygon snap        : {n_fault_snapped:,} vertices moved")
            print(f"  Polygon → fault snap         : {n_poly_snapped:,} vertices moved")
        print(f"  Polygon overlaps corrected   : {n_overlaps:,}  "
              f"({overlap_area_sum:.4e} m²)")
        print(f"  Polygon gaps corrected       : {n_gaps:,}  "
              f"({gap_area_sum:.4e} m²)")
        print(f"  Features written             : {written:,}")
        print(f"  Elapsed                      : {elapsed:.1f} s")
        print(f"  Polygon output → {output_file}")
        if snapped_fault:
            print(f"  Fault output   → {snapped_fault}")
        print(f"{'='*70}\n")

    stats = {
        'n_features':          n,
        'n_poly_grid_snapped': n_poly_grid_snapped,
        'n_fault_snapped':     n_fault_snapped,
        'n_poly_snapped':      n_poly_snapped,
        'n_overlaps':          n_overlaps,
        'overlap_area_total':  overlap_area_sum,
        'n_gaps':              n_gaps,
        'gap_area_total':      gap_area_sum,
        'elapsed_s':           elapsed,
    }
    return output_file, snapped_fault, stats


# =============================================================================
# STAGE 1: FAULT NETWORK SIMPLIFICATION
# Simplifies the fault line network with Modified Visvalingam-Whyatt.
# Before simplification, scans every polygon vertex and pins any fault vertex
# that coincides with a polygon boundary so fault-polygon alignment is
# maintained through to Stage 2.  Also detects and pins fault-fault junction
# points so the network topology is preserved.
# =============================================================================

def simplify_fault_network(fault_file: str, threshold: float,
                            polygon_file: str = None,
                            output_dir: str = None) -> str:
    """
    Simplify the fault network line layer using Modified Visvalingam-Whyatt.

    Pins two categories of vertices before simplification:
      1. Fault-fault junctions — detected by scanning all fault vertices for
         points where local connectivity changes (more than two neighbours).
      2. Polygon-boundary coincident fault vertices — all polygon vertices are
         collected and any fault vertex that matches one is pinned so it cannot
         be removed, preserving exact fault-polygon crossing coordinates.
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()

    fault_basename      = os.path.splitext(os.path.basename(fault_file))[0]
    simplified_fault_file = os.path.join(
        output_dir, f"{fault_basename}_si_{threshold}.shp")

    print(f"=== STEP 1: SIMPLIFYING FAULT NETWORK ===")
    print(f"Input: {fault_file}  |  Threshold: {threshold}")

    engine = SimplificationEngine()
    engine.set_quantitization_factor(0.1)

    print("  Detecting fault network junctions...")
    engine.find_all_junctions(fault_file, engine.dict_junctions)
    print(f"  Fault-fault junctions: {len(engine.dict_junctions)}")

    # Pin fault vertices that coincide with any polygon boundary vertex.
    # This covers both triple junctions and ordinary mid-arc polygon boundary
    # points, ensuring every fault-polygon crossing is preserved through Stage 1.
    pinned_count = 0
    if polygon_file:
        print("  Scanning polygon dataset for all vertices to pin on fault network...")
        poly_all_verts = engine._collect_all_polygon_vertices(polygon_file)
        print(f"  Polygon vertices found: {len(poly_all_verts)}")
        with fiona.open(fault_file, 'r') as fl:
            for record in fl:
                geom   = record['geometry']
                coords: List = []
                if geom['type'] == 'LineString':
                    coords = geom['coordinates']
                elif geom['type'] == 'MultiLineString':
                    for line in geom['coordinates']:
                        coords.extend(line)
                elif geom['type'] == 'Polygon':
                    coords = geom['coordinates'][0]
                for pt in coords:
                    qpt = engine.quantitize(pt)
                    if qpt in poly_all_verts and qpt not in engine.dict_junctions:
                        engine.dict_junctions[qpt] = 1
                        pinned_count += 1
        print(f"  Pinned {pinned_count} fault nodes at polygon boundary vertices")
        print(f"  Total fault constraints: {len(engine.dict_junctions)}")
    else:
        print("  No polygon_file supplied — skipping polygon-vertex pinning.")

    with fiona.open(fault_file, 'r') as src:
        meta = src.meta.copy()
        with fiona.open(simplified_fault_file, 'w', **meta) as dst:
            print(f"Processing {len(src)} fault features...")
            processed = simplified = errors = 0
            for record in src:
                try:
                    gd   = record['geometry']
                    if gd['type'] == 'LineString':
                        geom = LineString(gd['coordinates'])
                    elif gd['type'] == 'MultiLineString':
                        geom = MultiLineString(gd['coordinates'])
                    elif gd['type'] == 'Polygon':
                        geom = Polygon(gd['coordinates'][0], gd['coordinates'][1:])
                    else:
                        continue
                    sg = engine.simplify_geometry(geom, 'modified_visvalingam_whyatt', threshold)
                    if sg and not sg.is_empty:
                        dst.write({'geometry': mapping(sg), 'properties': record['properties']})
                        simplified += 1
                    processed += 1
                    if processed % 100 == 0:
                        print(f"  Processed {processed} fault features...")
                except Exception as e:
                    print(f"Error on fault feature {processed}: {e}")
                    errors += 1

    print(f"Fault simplification complete: {simplified}/{processed} simplified, "
          f"{errors} errors, {pinned_count} triple-junctions pinned.")
    return simplified_fault_file


# =============================================================================
# MAIN ENTRY POINT: THREE-STAGE POLYGON SIMPLIFICATION
# Orchestrates the full pipeline:
#   Stage 0 — Snap coordinates, fix overlaps/gaps, snap fault↔polygon vertices
#   Stage 1 — Simplify fault network with polygon-vertex pinning
#   Stage 2 — Simplify polygons with junction constraints, boundary
#              preservation, and unique geological contact pinning
# =============================================================================

def vector_simplify_file_two_stage(
        input_file,
        output_file,
        method,
        threshold,
        fault_file         = None,
        boundary_preserve  = "hard",   # "hard" | "soft"
        soft_tolerance     = None,     # default = threshold / 10
        preprocess         = True,     # run Stage 0 topology pre-processing
        snap_decimals      = 7,        # coordinate grid decimal places
        unit_field         = 'CODE',   # attribute field for geological unit ID
        **kwargs):
    """
    Three-stage polygon simplification pipeline.

    Parameters
    ----------
    input_file        : polygon shapefile
    output_file       : output shapefile
    method            : simplification algorithm name (see module docstring)
    threshold         : Visvalingam-Whyatt area threshold (m²)
    fault_file        : fault network shapefile (enables two-stage MVW pipeline)
    boundary_preserve : "hard" — exterior arc vertices pinned as infinite constraints
                        "soft" — exterior arc vertices scaled by threshold/soft_tolerance
                        For rectangular datasets the bbox edges are always hard-pinned.
    soft_tolerance    : area threshold for soft-exterior vertices (m²).
                        Default = threshold / 10.
    preprocess        : if True (default), run Stage 0 topology pre-processing
                        before simplification.
    snap_decimals     : decimal-place grid for coordinate snapping in Stage 0
                        (default 7 = 0.0000001 m).  Only used when preprocess=True.
    unit_field        : attribute field identifying the geological unit code.
                        Used to detect and pin unique geological contacts.
                        Default "CODE" for GSWA datasets.
    """
    valid_methods = [
        'decimation', 'douglas_peucker', 'douglas_peucker_tp',
        'bend_simplify', 'visvalingam_whyatt', 'modified_visvalingam_whyatt',
    ]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of: {', '.join(valid_methods)}")

    if soft_tolerance is None:
        soft_tolerance = threshold / 10.0

    if fault_file:
        print(f"Initializing THREE-STAGE {method} simplification "
              f"(topo pre-process + all-vertex pinning + boundary preservation):")
        print(f"  Polygon dataset : {input_file}")
        print(f"  Fault network   : {fault_file}")
        print(f"  Threshold       : {threshold:,} m²")
        print(f"  boundary_preserve = '{boundary_preserve}'  "
              f"| soft_tolerance = {soft_tolerance:,.1f} m²")
        print(f"  preprocess = {preprocess}  | snap_decimals = {snap_decimals}")
    else:
        print(f"Initializing {method} simplification for {input_file}...")

    # ── Stage 0: Topology pre-processing ──────────────────────────────────────
    # Snaps all coordinates to a decimal grid, fixes polygon overlaps and gaps,
    # and bidirectionally snaps fault↔polygon vertices to shared coordinates.
    # The corrected polygon and fault files are used for all subsequent stages.
    topo_stats = None
    if preprocess:
        out_dir         = os.path.dirname(os.path.abspath(output_file))
        topo_fixed_path = os.path.join(
            out_dir,
            os.path.splitext(os.path.basename(input_file))[0] + '_topo_fixed.shp',
        )
        # Pass fault_file only for MVW (cross-dataset snap only needed there)
        _fault_for_preprocess = (
            fault_file
            if method == 'modified_visvalingam_whyatt' else None
        )
        topo_poly_path, topo_fault_path, topo_stats = preprocess_topology(
            input_file    = input_file,
            output_file   = topo_fixed_path,
            snap_decimals = snap_decimals,
            fix_overlaps  = True,
            fix_gaps      = True,
            fault_file    = _fault_for_preprocess,
            snap_distance = 0.01,
            verbose       = True,
        )
        input_file = topo_poly_path    # polygon stages use corrected file
        if topo_fault_path and os.path.exists(topo_fault_path):
            fault_file = topo_fault_path   # Stage 1 uses snapped fault

    if method == 'modified_visvalingam_whyatt' and fault_file:

        # ── Stage 1: Simplify fault network ───────────────────────────────
        # Pins all fault-fault junctions and all fault vertices that coincide
        # with polygon boundary vertices, then runs MVW on every fault feature.
        try:
            simplified_fault_file = simplify_fault_network(
                fault_file, threshold,
                polygon_file=input_file,
                output_dir=os.path.dirname(output_file),
            )
        except Exception as e:
            print(f"Error in Stage 1: {e}  — falling back to original fault network.")
            simplified_fault_file = fault_file

        # ── Stage 2: Simplify polygons ────────────────────────────────────
        print(f"\n=== STEP 2: SIMPLIFYING POLYGONS WITH FAULT CONSTRAINTS + BOUNDARY PRESERVATION ===")
        print(f"Using simplified fault network: {simplified_fault_file}")

        engine = SimplificationEngine()
        engine.set_quantitization_factor(0.1)

        print("Performing topology analysis...")
        engine.find_all_junctions_with_faults(
            input_file, simplified_fault_file, engine.dict_junctions)

        print(f"Topology results:")
        print(f"  - Junctions : {len(engine.dict_junctions)}")
        shared_count = len([s for s, f in engine.shared_boundaries.items() if len(f) > 1])
        print(f"  - Shared segments : {shared_count}")

        # ── Boundary preservation ─────────────────────────────────────────
        # Detect whether the dataset is rectangular and pin or scale exterior
        # arc vertices according to the chosen boundary_preserve mode.
        print(f"\nApplying boundary preservation (mode='{boundary_preserve}')...")
        bp_stats = engine.detect_and_apply_boundary_preservation(
            polygon_file      = input_file,
            boundary_preserve = boundary_preserve,
            soft_tolerance    = soft_tolerance,
            threshold         = threshold,
        )
        print(f"  Boundary mode : {bp_stats['mode']}")
        print(f"  Hard-pinned   : {bp_stats['hard_pinned']} vertices")
        print(f"  Soft-exterior : {bp_stats['soft_exterior']} vertices")
        print(f"  Junctions (total after boundary pin) : {len(engine.dict_junctions)}")

        # ── Validate unit_field, then pin unique geological contact representatives ──
        contacts: Dict     = {}
        contact_rep_verts: Set = set()
        if method == 'modified_visvalingam_whyatt':
            # Check the unit_field exists before trying to use it
            _field_info = inspect_unit_field(input_file, unit_field=unit_field)
            _effective_unit_field = _field_info['unit_field']  # may be auto-corrected

            if not _field_info['unit_field_ok']:
                if _effective_unit_field:
                    print(f"  ⚠ unit_field='{unit_field}' not found.  "
                          f"Using best candidate '{_effective_unit_field}' instead.")
                    print(f"    To suppress this warning, pass "
                          f"unit_field='{_effective_unit_field}' explicitly.")
                    unit_field = _effective_unit_field
                else:
                    print(f"  ⚠ unit_field='{unit_field}' not found and no candidates detected.")
                    print(f"    Skipping unique contact pinning.")
                    print(f"    Re-run with the correct unit_field= parameter "
                          f"(see field list above).")
                    _effective_unit_field = None

            if _effective_unit_field:
                print(f"\nPinning unique geological contact representatives...")
                print(f"  Using unit identifier field : '{_effective_unit_field}'")
                contacts, contact_rep_verts = engine._collect_unique_contact_representatives(
                    input_file, unit_field=_effective_unit_field,
                )
            print(f"  Unique CODE-pair contacts found    : {len(contacts)}")
            print(f"  New representative vertices pinned : {len(contact_rep_verts)}")
            print(f"  Junctions (total after contact pin): {len(engine.dict_junctions)}")
            for key, info in sorted(contacts.items(), key=lambda kv: kv[1]['code_a']):
                print(f"    {info['code_a']:20s} | {info['code_b']:20s}  "
                      f"len={info['shared_length']:8.0f} m  "
                      f"verts={info['n_shared_verts']:4d}  "
                      f"rep=({info['representative'][0]:.2f}, {info['representative'][1]:.2f})")

        # ── Pre-simplification thin-body detection ───────────────────────
        # A narrow polygon (fold stripe, dyke outline, thin sliver) whose
        # approximate minimum width is smaller than sqrt(threshold)/2 is at
        # risk of collapsing to a degenerate geometry during MVW simplification.
        # Warn the user BEFORE any polygon is processed so they can lower the
        # threshold if shape preservation of thin bodies is required.
        #
        # Approximate minimum width  ≈  2 × area / perimeter
        # (exact for a rectangle, conservative for other shapes)
        _risk_width = (threshold ** 0.5) * 0.5
        _thin_bodies: List[tuple] = []
        _unit_field_for_scan = unit_field if unit_field else None
        with fiona.open(input_file, 'r') as _scan_src:
            for _rec in _scan_src:
                _gd = _rec['geometry']
                if _gd is None or _gd['type'] not in ('Polygon', 'MultiPolygon'):
                    continue
                _uid = (_rec['properties'].get(_unit_field_for_scan, f"fid={_rec['id']}")
                        if _unit_field_for_scan else f"fid={_rec['id']}")
                _geom_s = (Polygon(_gd['coordinates'][0], _gd['coordinates'][1:])
                           if _gd['type'] == 'Polygon'
                           else MultiPolygon([Polygon(p[0], p[1:])
                                              for p in _gd['coordinates']]))
                _polys_s = ([_geom_s] if _geom_s.geom_type == 'Polygon'
                            else list(_geom_s.geoms))
                for _p in _polys_s:
                    if _p.is_empty or _p.area == 0:
                        continue
                    _perim = _p.exterior.length + sum(r.length for r in _p.interiors)
                    if _perim > 0:
                        _min_w = 2.0 * _p.area / _perim
                        if _min_w < _risk_width:
                            # 5-tuple: uid, area, min_width, perimeter, geometry
                            # (geometry is needed below to pre-register thin-arc keys)
                            _thin_bodies.append((_uid, _p.area, _min_w, _perim, _p))

        if _thin_bodies:
            print(f"\n  ⚠ THIN BODY WARNING — {len(_thin_bodies)} narrow polygon "
                  f"body/-ies detected (min_width < {_risk_width:.0f} m).")
            print(f"    At threshold {threshold:,} m² these features may collapse "
                  f"during simplification.")
            print(f"    If any fold stripe or thin unit disappears in the output,")
            print(f"    reduce the threshold or increase snap_decimals.")
            print(f"    At-risk features (first 10):")
            for _uid, _area, _mw, _per, *_ in _thin_bodies[:10]:
                print(f"      '{_uid}'  area={_area:.0f} m²  "
                      f"perim={_per:.0f} m  min_width≈{_mw:.1f} m")
            if len(_thin_bodies) > 10:
                print(f"      … and {len(_thin_bodies) - 10} more thin bodies")
        else:
            print(f"\n  ✓ No thin bodies detected "
                  f"(all min_widths ≥ {_risk_width:.0f} m at this threshold).")

        # ── Pre-simplify thin-body arcs and pin surviving vertices ──────────
        # Thin bodies (narrow polygons such as fold stripes or dyke outlines)
        # share boundary arcs with their neighbours.  If the main simplification
        # threshold is too coarse, a shared arc may be reduced to just its two
        # endpoint junctions, collapsing the thin polygon to a line or point.
        #
        # To prevent this, each thin-body arc is pre-simplified at a conservative
        # "safe" threshold, and every intermediate vertex that survives is pinned
        # into dict_junctions (assigned infinite area weight so it can never be
        # removed by Visvalingam-Whyatt).
        #
        # Topology guarantee:
        #   During the main simplification run every polygon — the thin body
        #   itself and every neighbour sharing any of those arcs — sees the
        #   pinned vertices as immovable hard constraints.  Every vertex between
        #   two consecutive pins has a VW triangle area smaller than the safe
        #   threshold, so it is automatically removed by all adjacent polygons
        #   at the main threshold.  The result is that the shared arc simplifies
        #   to exactly [J1, P1, P2, …, J2] for every polygon that touches it,
        #   regardless of processing order and without relying on cache lookups.
        #   No gaps or overlaps can arise on those shared boundaries by construction.
        #
        # Safe threshold derivation:
        #   A rectangle of width w collapses when T ≈ w² / 4 (a single triangle
        #   covers the entire interior).  Setting safe_thr = area / 4 for the
        #   narrowest thin body guarantees at least one interior triangle survives
        #   the pre-simplification, so a non-degenerate arc with at least one
        #   interior vertex is available to extract pins from.
        if _thin_bodies and engine.dict_junctions:
            _safe_thr = max(1.0, min(tb[1] for tb in _thin_bodies) / 4.0)
            print(f"\n  Pre-simplify-and-pin thin-body arcs:")
            print(f"    Safe threshold : {_safe_thr:,.0f} m²  "
                  f"(= smallest thin-body area / 4)")
            print(f"    Full threshold : {threshold:,} m²")
            _n_pinned_new = 0
            for _uid_tb, _area_tb, _mw_tb, _per_tb, _pg_tb in _thin_bodies:
                try:
                    _arcs_tb, _ = engine.cut_polygon_by_junctions(
                        _pg_tb, engine.dict_junctions)
                except Exception:
                    _arcs_tb = None
                if _arcs_tb is None:
                    continue
                for _arc_tb in _arcs_tb:
                    if len(list(_arc_tb.coords)) < 3:
                        # Arc is just endpoints — nothing to pin between them
                        continue
                    try:
                        # Pre-simplify at the safe threshold
                        _pre_simp = engine.visvalingam_whyatt(_arc_tb, _safe_thr)
                    except Exception:
                        continue
                    if _pre_simp is None or _pre_simp.is_empty:
                        continue
                    # Pin every surviving intermediate vertex (skip the two
                    # endpoint junctions — they are already in dict_junctions)
                    _pre_coords = list(_pre_simp.coords)
                    for _pvx, _pvy in _pre_coords[1:-1]:
                        _qpv = engine.quantitize((_pvx, _pvy))
                        if _qpv not in engine.dict_junctions:
                            engine.dict_junctions[_qpv] = 1
                            _n_pinned_new += 1
            print(f"    New vertices pinned : {_n_pinned_new}")
            print(f"    Total dict_junctions after pin : "
                  f"{len(engine.dict_junctions)}")

        # ── Simplify each polygon ─────────────────────────────────────────
        original_geoms   = []
        simplified_geoms = []

        with fiona.open(input_file, 'r') as src:
            meta         = src.meta.copy()
            _all_records = list(src)   # read entirely so we can reorder

        # Thin-body features are sorted to the END of the processing queue.
        # Reason: when a thin body collapses and the clip fallback runs, it
        # subtracts the already-simplified neighbours from the original geom.
        # If the thin body were processed first, its neighbours would not yet
        # be in simplified_geoms, so the clip would miss them and the result
        # would still overlap polygons simplified later.  Processing all
        # non-thin-body features first guarantees every neighbour's simplified
        # geometry is available before the clip fallback runs.
        _thin_codes = {uid for uid, *_ in _thin_bodies}
        _non_thin_records = [r for r in _all_records
                             if r['properties'].get(unit_field) not in _thin_codes]
        _thin_records     = [r for r in _all_records
                             if r['properties'].get(unit_field) in _thin_codes]

        # Within the thin-body group, sort by polygon area DESCENDING.
        # Larger thin-body rings simplify successfully in arc mode; smaller
        # ones are more likely to collapse and need the clip fallback.
        # Processing larger ones first means they are in simplified_geoms
        # (and therefore available for the clip neighbour check) before the
        # smaller, potentially-collapsing rings are processed.
        def _rec_area(r):
            gd = r['geometry']
            if gd is None:
                return 0.0
            try:
                if gd['type'] == 'Polygon':
                    return Polygon(gd['coordinates'][0],
                                   gd['coordinates'][1:]).area
                elif gd['type'] == 'MultiPolygon':
                    return MultiPolygon([Polygon(p[0], p[1:])
                                         for p in gd['coordinates']]).area
            except Exception:
                pass
            return 0.0

        _thin_records.sort(key=_rec_area, reverse=True)  # largest-area first
        _ordered_records  = _non_thin_records + _thin_records

        with fiona.open(output_file, 'w', **meta) as dst:
            _n_thin_recs = len(_thin_records)
            print(f"\nProcessing {len(_all_records)} polygon features "
                  f"({_n_thin_recs} thin-body record(s) deferred to end)...")
            processed = simplified = errors = collapsed = heavy_loss = 0
            collapsed_ring = 0   # collapsed but rescued by ring-mode retry
            collapsed_clip = 0   # collapsed + ring-mode overlapped; clipped by neighbours
            collapsed_orig = 0   # collapsed + clip empty; original written (overlap unavoidable)

            for record in _ordered_records:
                    try:
                        gd   = record['geometry']
                        if gd['type'] == 'Polygon':
                            geom = Polygon(gd['coordinates'][0], gd['coordinates'][1:])
                        elif gd['type'] == 'MultiPolygon':
                            geom = MultiPolygon([
                                Polygon(p[0], p[1:]) for p in gd['coordinates']])
                        else:
                            continue

                        original_geoms.append(geom)
                        sg = engine.simplify_geometry(geom, method, threshold, **kwargs)

                        # ── Main-loop make_valid() safety net ──────────────────
                        # The ring-mode guard (Fix B) and arc-assembly guard
                        # (Fix C) prevent self-intersections for the common cases,
                        # but cumulative effects across many removals may slip
                        # through in degenerate geometry.  This final check ensures
                        # no invalid polygon ever reaches the output file.
                        if sg and not sg.is_empty and not sg.is_valid:
                            try:
                                sg_fixed = _make_valid(sg)
                                if sg_fixed and not sg_fixed.is_empty:
                                    # keep only polygon parts (make_valid may
                                    # split a bowtie into polygons + lines)
                                    if sg_fixed.geom_type == 'Polygon':
                                        sg = sg_fixed
                                    elif sg_fixed.geom_type in (
                                            'MultiPolygon', 'GeometryCollection'):
                                        _polys = [g for g in sg_fixed.geoms
                                                  if g.geom_type == 'Polygon'
                                                  and not g.is_empty]
                                        if _polys:
                                            sg = (max(_polys, key=lambda g: g.area)
                                                  if len(_polys) == 1 else
                                                  MultiPolygon(_polys))
                                        else:
                                            sg = None
                                    else:
                                        sg = None
                            except Exception:
                                sg = None  # collapse fallback below will handle it

                        if sg and not sg.is_empty:
                            simplified_geoms.append(sg)
                            dst.write({'geometry': mapping(sg), 'properties': record['properties']})
                            simplified += 1

                            # ── Vertex-retention check ─────────────────────────
                            # Warn when a polygon loses > 75% of its vertices.
                            # The polygon is NOT dropped (contact pinning and
                            # boundary preservation keep it valid), but heavy
                            # vertex loss (>75%) means the characteristic shape
                            # of thin or intricate features — fold stripes, dyke
                            # outlines, embayments — may be visually unrecognisable
                            # in the output even though the polygon area is preserved.
                            # Example: a concentric fold stripe with 42 original
                            # vertices may simplify to 9 at 50,000 m², losing the
                            # concentric arc shape even though the polygon exists.
                            def _count_verts_geom(g):
                                if g.geom_type == 'Polygon':
                                    return (len(g.exterior.coords) +
                                            sum(len(r.coords) for r in g.interiors))
                                elif g.geom_type == 'MultiPolygon':
                                    return sum(len(p.exterior.coords) +
                                               sum(len(r.coords) for r in p.interiors)
                                               for p in g.geoms)
                                return 0

                            nv_orig = _count_verts_geom(geom)
                            nv_simp = _count_verts_geom(sg)
                            if nv_orig >= 10 and nv_simp < nv_orig * 0.25:
                                heavy_loss += 1
                                _uid_h = (record['properties'].get(unit_field,
                                                                    f"fid={processed+1}")
                                          if unit_field else f"fid={processed+1}")
                                if heavy_loss <= 10:
                                    pct_kept = nv_simp / nv_orig * 100
                                    print(f"  ⚠ Heavy shape loss: '{_uid_h}' "
                                          f"{nv_orig}v → {nv_simp}v "
                                          f"({pct_kept:.0f}% verts kept).  "
                                          f"Feature visible but shape is significantly altered. "
                                          f"Reduce threshold to preserve shape detail.")
                                elif heavy_loss == 11:
                                    print(f"  ⚠ … further heavy-loss warnings suppressed.")

                        else:
                            # ── Collapse fallback: retry in whole-ring mode ─────────────
                            # WHY ARC MODE COLLAPSES FOR THIS POLYGON:
                            # modified_visvalingam_whyatt() cuts the ring into arcs at every
                            # junction vertex.  At high tolerances the simplified fault has
                            # fewer vertices, so the polygon may have fewer junction pins —
                            # but those pins can still divide the ring into very short arcs
                            # (2–3 vertices each) that cannot be simplified further.  The
                            # assembled ring then collapses to < 3 usable vertices.
                            #
                            # WHY RING MODE FIXES IT:
                            # visvalingam_whyatt_ring() treats the whole exterior as one
                            # continuous ring.  Pinned vertices still get infinite area weight
                            # and cannot be removed, but there is no arc splitting — MVW can
                            # remove non-pinned vertices freely across the whole ring until the
                            # threshold is met.  Starting at the original threshold and
                            # stepping down by 10 % finds the highest tolerance at which a
                            # valid simplified ring can be produced ("max tolerance possible").
                            _uid_c = (record['properties'].get(unit_field, f"fid={processed+1}")
                                      if unit_field else f"fid={processed+1}")

                            # Start at the original threshold — ring mode often succeeds
                            # where arc mode failed, so no need to step down immediately.
                            _retry_thr   = threshold
                            _retry_min   = max(threshold * 0.001, 1.0)  # floor: 0.1 % or 1 m²
                            _retry_sg    = None
                            _retry_steps = 0

                            # Build a spatial index of already-simplified polygons so
                            # the retry loop can reject ring-mode candidates that
                            # overlap neighbours.  At high tolerances, whole-ring
                            # simplification can produce a coarser shape that bulges
                            # into an adjacent polygon's territory — the result is
                            # geometrically valid on its own but topologically
                            # incorrect relative to the dataset.  Stepping the
                            # threshold down until no neighbour overlap is detected
                            # finds the highest tolerance at which the output is
                            # both valid and non-overlapping.
                            # Only polygons already written are indexed — features
                            # processed later in the loop are handled by their own
                            # simplification pass.
                            _RETRY_OVERLAP_TOL = 1.0          # m² noise floor
                            _nbr_geoms = list(simplified_geoms)
                            _nbr_tree  = STRtree(_nbr_geoms) if _nbr_geoms else None

                            while _retry_thr >= _retry_min:
                                _retry_steps += 1

                                # Whole-ring simplification — bypasses arc splitting
                                _ring_result = engine.visvalingam_whyatt_ring(
                                    geom.exterior, _retry_thr)

                                if _ring_result is not None and not _ring_result.is_empty:
                                    try:
                                        _retry_poly = Polygon(_ring_result)
                                        if not _retry_poly.is_valid:
                                            _retry_poly = _make_valid(_retry_poly)
                                        if (_retry_poly is not None
                                                and not _retry_poly.is_empty
                                                and _retry_poly.geom_type == 'Polygon'):
                                            # Reject candidates that overlap an already-
                                            # simplified neighbour — ring mode with a high
                                            # threshold can produce a coarse polygon that
                                            # is valid on its own but crosses into a
                                            # neighbour's territory.  Step the threshold
                                            # down until the result fits cleanly.
                                            _has_nbr_overlap = False
                                            if _nbr_tree is not None:
                                                for _nc in _nbr_tree.query(_retry_poly):
                                                    _ni  = _nbr_geoms[_nc]
                                                    _nix = _retry_poly.intersection(_ni)
                                                    if (not _nix.is_empty
                                                            and _nix.area
                                                            > _RETRY_OVERLAP_TOL):
                                                        _has_nbr_overlap = True
                                                        break
                                            if not _has_nbr_overlap:
                                                _retry_sg = _retry_poly
                                                break  # highest viable threshold found
                                    except Exception:
                                        pass

                                _retry_thr *= 0.9  # step down 10 % and try again

                            if _retry_sg and not _retry_sg.is_empty:
                                # Ring-mode retry succeeded at the highest possible tolerance
                                simplified_geoms.append(_retry_sg)
                                dst.write({'geometry': mapping(_retry_sg),
                                           'properties': record['properties']})
                                simplified    += 1
                                collapsed     += 1
                                collapsed_ring += 1
                                if collapsed <= 10:
                                    print(f"  ⚠ Polygon '{_uid_c}' (area≈{geom.area:.0f} m²) "
                                          f"arc-mode collapsed at {threshold:,} m² — "
                                          f"ring-mode retry succeeded at "
                                          f"{_retry_thr:,.0f} m² "
                                          f"({_retry_steps} step(s)).")
                                elif collapsed == 11:
                                    print(f"  ⚠ … further collapse-retry warnings suppressed.")
                            else:
                                # ── Neighbour-clip fallback ─────────────────────────
                                # Ring-mode retry was rejected at every threshold because
                                # the simplified ring always overlapped an already-
                                # simplified neighbour.  This occurs when the neighbour's
                                # arc-mode simplification has shifted the shared boundary
                                # far into this polygon's original territory (the
                                # neighbour's simplified boundary "eats" much of this
                                # polygon's area).
                                #
                                # The topologically correct output is:
                                #   this_poly = original − union(overlapping neighbours)
                                #
                                # This trims this polygon's original geometry to fit
                                # exactly within the space not already claimed by the
                                # simplified neighbours.  The resulting shared boundary
                                # is IDENTICAL to the neighbour's simplified boundary —
                                # no overlap, no gap.  The polygon's other boundaries
                                # (non-shared sides) remain as the original geometry.
                                _clip_result = None
                                _n_clipped_by = 0
                                if _nbr_tree is not None:
                                    _overlap_nbrs = []
                                    for _nc in _nbr_tree.query(geom):
                                        _ni  = _nbr_geoms[_nc]
                                        _nix = geom.intersection(_ni)
                                        if (not _nix.is_empty
                                                and _nix.area > _RETRY_OVERLAP_TOL):
                                            _overlap_nbrs.append(_ni)
                                    if _overlap_nbrs:
                                        _n_clipped_by = len(_overlap_nbrs)
                                        try:
                                            _diff = geom.difference(
                                                unary_union(_overlap_nbrs))
                                            if not _diff.is_empty:
                                                if not _diff.is_valid:
                                                    _diff = _make_valid(_diff)
                                                # Extract polygon parts (difference
                                                # may produce a GeometryCollection
                                                # if slivers are split off)
                                                if _diff.geom_type == 'Polygon':
                                                    _clip_result = _diff
                                                elif _diff.geom_type in (
                                                        'MultiPolygon',
                                                        'GeometryCollection'):
                                                    _pps = [g for g in _diff.geoms
                                                            if g.geom_type == 'Polygon'
                                                            and not g.is_empty]
                                                    if _pps:
                                                        _clip_result = (
                                                            _pps[0] if len(_pps) == 1
                                                            else MultiPolygon(_pps))
                                        except Exception:
                                            _clip_result = None

                                if (_clip_result is not None
                                        and not _clip_result.is_empty):
                                    simplified_geoms.append(_clip_result)
                                    dst.write({'geometry': mapping(_clip_result),
                                               'properties': record['properties']})
                                    collapsed      += 1
                                    collapsed_clip += 1
                                    if collapsed <= 10:
                                        print(
                                            f"  ⚠ Polygon '{_uid_c}' "
                                            f"(area≈{geom.area:.0f} m²) "
                                            f"arc-mode collapsed and ring-mode overlapped "
                                            f"{_n_clipped_by} neighbour(s) — clipped to "
                                            f"remaining space "
                                            f"(area≈{_clip_result.area:.0f} m² after clip)."
                                        )
                                    elif collapsed == 11:
                                        print("  ⚠ … further collapse warnings suppressed.")
                                else:
                                    # Absolute last resort: every threshold produced an
                                    # overlap AND the difference is empty or failed.
                                    # This means the polygon's entire territory has been
                                    # absorbed by its simplified neighbours — the polygon
                                    # simply cannot be represented without overlap at this
                                    # threshold.  Write original to preserve the feature;
                                    # the resulting overlap is geometrically unavoidable
                                    # at this tolerance.  Reduce the threshold to fix.
                                    simplified_geoms.append(geom)
                                    dst.write({'geometry': mapping(geom),
                                               'properties': record['properties']})
                                    collapsed      += 1
                                    collapsed_orig += 1
                                    if collapsed <= 10:
                                        print(
                                            f"  ⚠ Polygon '{_uid_c}' "
                                            f"(area≈{geom.area:.0f} m²) "
                                            f"cannot be represented at {threshold:,} m² "
                                            f"without overlap — simplified neighbours absorb "
                                            f"its entire territory.  Original geometry written; "
                                            f"overlap is unavoidable at this tolerance.  "
                                            f"Reduce the threshold to resolve."
                                        )
                                    elif collapsed == 11:
                                        print(
                                            "  ⚠ … further collapse warnings suppressed."
                                        )

                        processed += 1
                        if processed % 100 == 0:
                            print(f"  Processed {processed} polygon features...")

                    except Exception as e:
                        print(f"Error on polygon feature {processed}: {e}")
                        errors += 1
                        simplified_geoms.append(None)

        success_rate = (simplified / processed * 100) if processed else 0.0

        # ── Post-simplification contact status check ────────────────────────────
        # Re-run contact detection on the OUTPUT to count surviving shared
        # boundary vertices per contact pair.  The representative is always
        # pinned so it survives, but if n_shared_verts_output == 1 the
        # contact boundary has collapsed to a single point — relationship
        # preserved, shape information lost.
        #
        # Status categories
        #   'preserved'         : ≥ 3 shared verts → meaningful boundary shape
        #   'minimal'           : exactly 2 shared verts → barely a line
        #   'shape_lost'        : 1 shared vert (only the pin) → point contact;
        #                         further simplification CANNOT reduce further
        #                         but shape is already gone
        #   'relationship_lost' : contact not found in output at all
        #                         (should NEVER happen with pinning)

        contact_status: Dict = {}
        n_contact_preserved = n_contact_minimal = n_contact_shape_lost = \
            n_contact_diverged = 0

        if contacts and os.path.exists(output_file):
            # Use a fresh engine so dict_junctions stays clean
            _check_eng = SimplificationEngine()
            _check_eng.set_quantitization_factor(0.1)
            out_contacts, _ = _check_eng._collect_unique_contact_representatives(
                output_file, unit_field=unit_field,
            )

            for key, orig_info in contacts.items():
                n_orig   = orig_info['n_shared_verts']
                out_info = out_contacts.get(key)

                if out_info is None:
                    # The intersection test found no shared boundary vertices.
                    # At high tolerances this means the independently-simplified
                    # polygon boundaries have diverged — both polygons still carry
                    # the pinned representative vertex, but Shapely's intersection
                    # can no longer detect a common edge.  The contact record
                    # (which units touch which) is preserved by the pin; only
                    # the shared-boundary geometry is gone.
                    n_out      = 0
                    status_str = 'boundary_diverged'
                    n_contact_diverged += 1
                else:
                    n_out = out_info['n_shared_verts']
                    if n_out >= 3:
                        status_str = 'preserved'
                        n_contact_preserved += 1
                    elif n_out == 2:
                        status_str = 'minimal'
                        n_contact_minimal += 1
                    else:
                        status_str = 'shape_lost'
                        n_contact_shape_lost += 1

                contact_status[key] = {
                    **orig_info,
                    'status':       status_str,
                    'n_orig_verts': n_orig,
                    'n_out_verts':  n_out,
                }

        print(f"\n=== THREE-STAGE SIMPLIFICATION SUMMARY ===")
        print(f"Method    : {method}")
        print(f"Threshold : {threshold:,} m²")
        if topo_stats:
            print(f"Stage 0 (topo pre-process):")
            print(f"  Polygon grid-snapped : {topo_stats['n_poly_grid_snapped']}")
            print(f"  Fault  → polygon     : {topo_stats['n_fault_snapped']} vertices")
            print(f"  Polygon → fault      : {topo_stats['n_poly_snapped']} vertices")
            print(f"  Overlaps corrected   : {topo_stats['n_overlaps']}  "
                  f"({topo_stats['overlap_area_total']:.4e} m²)")
            print(f"  Gaps corrected      : {topo_stats['n_gaps']}  "
                  f"({topo_stats['gap_area_total']:.4e} m²)")
        print(f"boundary_preserve = '{boundary_preserve}'")
        print(f"  Mode            : {bp_stats['mode']}")
        print(f"  Rect dataset    : {bp_stats['is_rectangular']}")
        print(f"  Hard-pinned     : {bp_stats['hard_pinned']}")
        print(f"  Soft-exterior   : {bp_stats['soft_exterior']}")
        if contacts:
            n_total = len(contacts)
            print(f"Unique geological contacts (tol={threshold:,} m²):")
            print(f"  Total unique contacts         : {n_total}")
            print(f"  Boundary shape preserved (≥3v): {n_contact_preserved}")
            print(f"  Boundary minimal     (2v only): {n_contact_minimal}")
            print(f"  Boundary shape lost  (1v only): {n_contact_shape_lost}"
                  f"{'  ← contact relationship still recorded' if n_contact_shape_lost else ''}")
            print(f"  Boundaries diverged  (0v/none): {n_contact_diverged}"
                  f"{'  ← reduce tolerance' if n_contact_diverged else ''}")

            if n_contact_diverged:
                print(f"  ⚠ WARNING — {n_contact_diverged} contact boundary/-ies are no longer "
                      f"geometrically detectable at this tolerance.")
                print(f"    Independent polygon simplification has caused adjacent boundaries to")
                print(f"    diverge.  The pinned representative vertex still records WHICH units")
                print(f"    are in contact, but the shared boundary shape is gone.")
                print(f"    Reduce the tolerance to preserve boundary geometry.")
                for key, cs in sorted(contact_status.items(),
                                      key=lambda kv: kv[1]['status']):
                    if cs['status'] == 'boundary_diverged':
                        print(f"    [DIVERGED] {cs['code_a']} | {cs['code_b']}  "
                              f"(orig {cs['n_orig_verts']}v → 0v detectable)")

            if n_contact_shape_lost or n_contact_minimal:
                print(f"  ⚠ WARNING — {n_contact_shape_lost + n_contact_minimal} contact "
                      f"boundary/-ies are now nearly or fully degenerate at this tolerance.")
                print(f"    The contact relationship is preserved (representative vertex pinned),")
                print(f"    but the shared boundary shape has been simplified away.")
                print(f"    Reduce the tolerance if boundary shape detail is required.")
                for key, cs in sorted(contact_status.items(),
                                      key=lambda kv: kv[1]['status']):
                    if cs['status'] in ('shape_lost', 'minimal'):
                        pct = (cs['n_out_verts'] / max(cs['n_orig_verts'], 1)) * 100
                        print(f"    [{cs['status'].upper():10s}] "
                              f"{cs['code_a']} | {cs['code_b']:20s}  "
                              f"{cs['n_orig_verts']:4d}v → {cs['n_out_verts']:4d}v "
                              f"({pct:.0f}% retained)")

            if not (n_contact_diverged or n_contact_shape_lost or n_contact_minimal):
                print(f"  ✓ All {n_total} contact boundaries have ≥ 3 vertices — shape is meaningful")
        print(f"Polygon features processed  : {processed}")
        print(f"Polygon features simplified : {simplified}")
        if collapsed:
            print(f"Polygon features collapsed  : {collapsed}  (breakdown below)")
            if collapsed_ring:
                print(f"  ↳ Rescued by ring-mode retry          : {collapsed_ring}")
            if collapsed_clip:
                print(f"  ↳ Clipped to remaining space           : {collapsed_clip}")
            if collapsed_orig:
                print(f"  ↳ Original written (overlap unavoidable): {collapsed_orig}"
                      f"  ← reduce threshold to resolve")
        else:
            print(f"Polygon features collapsed  : 0")
        print(f"Heavy shape loss (>75% verts removed): {heavy_loss}"
              f"{'  ← visible but shape may be unrecognisable' if heavy_loss else ''}")
        print(f"Errors                      : {errors}")
        print(f"Success rate                : {success_rate:.1f}%")
        if _thin_bodies:
            print(f"Thin bodies detected (pre-run): {len(_thin_bodies)}"
                  f"  ← reduce threshold if shape matters")
        if heavy_loss:
            print(f"  ⚠ {heavy_loss} polygon(s) lost more than 75% of their vertices.")
            print(f"    These features are present in the output with correct area, but their")
            print(f"    boundary shape (e.g. concentric fold arc, embayment, thin stripe)")
            print(f"    has been simplified beyond visual recognition at map scale.")
            print(f"    Reduce the threshold to preserve shape detail for these features.")
        if collapsed:
            print(f"  ⚠ {collapsed} polygon body/-ies could not be simplified at this threshold.")
            print(f"    Their original geometries were written to preserve all features.")
            print(f"    These are typically thin stripes (fold layers, dyke outlines) whose")
            print(f"    triangles are all smaller than {threshold:,} m².")
            print(f"    Reduce the threshold to obtain simplified (not original) output for these.")
        print(f"Output : {output_file}")

        return {
            'method':              method,
            'threshold':           threshold,
            'features_processed':  processed,
            'features_simplified': simplified,
            'features_collapsed':  collapsed,              # polygons written unsimplified (thin-body fallback)
            'features_heavy_loss': heavy_loss,             # polygons that lost >75% of vertices
            'errors_encountered':  errors,
            'success_rate':        success_rate,
            'output_file':         output_file,
            'status':              'SUCCESS',
            'topo_preprocess':          topo_stats is not None,   # True if Stage 0 ran
            'topo_stats':               topo_stats,               # Stage 0 diagnostic dict
            'triple_junction_pinning':  True,                     # always enabled for MVW
            'all_vertex_pinning':       True,                     # fault vertices pinned on polygon boundary
            'boundary_preservation':    True,                     # exterior arc vertices pinned/scaled
            'unique_contact_pinning':   True,                     # one representative per CODE-pair contact
            'thin_body_detection':      True,                     # narrow polygons flagged before simplification
            'n_thin_bodies_detected':   len(_thin_bodies),        # count of at-risk narrow bodies
            'thin_bodies':              [t[:4] for t in _thin_bodies],  # list of (uid, area, min_width, perimeter)
            'boundary_mode':       bp_stats['mode'],
            'is_rectangular':      bp_stats['is_rectangular'],
            'hard_pinned':         bp_stats['hard_pinned'],
            'soft_exterior':       bp_stats['soft_exterior'],
            'junctions_preserved': len(engine.dict_junctions),
            'shared_boundaries':   shared_count,
            'contacts':               contacts,              # dict of unique CODE-pair contact info
            'contact_status':         contact_status,        # per-contact status after simplification
            'n_contact_preserved':    n_contact_preserved,   # contacts with ≥3 shared verts (shape OK)
            'n_contact_minimal':      n_contact_minimal,     # contacts with exactly 2 shared verts
            'n_contact_shape_lost':   n_contact_shape_lost,  # contacts with 1 shared vert (point only)
            'n_contact_diverged':     n_contact_diverged,    # contacts where boundaries no longer meet
        }

    else:
        return vector_simplify_file(input_file, output_file, method, threshold,
                                    fault_file, **kwargs)


# =============================================================================
# SINGLE-STAGE ENTRY POINT
# Lightweight wrapper used when no fault file is provided.  Runs Stage 2
# (polygon simplification) only, without topology pre-processing or fault
# network alignment.  Uses the same simplification engine as the three-stage
# pipeline.  For full topology preservation pass a fault_file to
# vector_simplify_file_two_stage() instead.
# =============================================================================

def vector_simplify_file(input_file, output_file, method, threshold,
                          fault_file=None, **kwargs):
    """Single-stage polygon simplification without fault-network alignment."""
    valid_methods = [
        'decimation', 'douglas_peucker', 'douglas_peucker_tp',
        'bend_simplify', 'visvalingam_whyatt', 'modified_visvalingam_whyatt',
    ]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of: {', '.join(valid_methods)}")

    engine = SimplificationEngine()

    if method == 'modified_visvalingam_whyatt':
        engine.set_quantitization_factor(0.1)
        if fault_file:
            engine.find_all_junctions_with_faults(input_file, fault_file, engine.dict_junctions)
        else:
            engine.find_all_junctions(input_file, engine.dict_junctions)

    original_geoms   = []
    simplified_geoms = []

    with fiona.open(input_file, 'r') as src:
        meta = src.meta.copy()
        with fiona.open(output_file, 'w', **meta) as dst:
            processed = simplified = errors = 0
            for record in src:
                try:
                    gd = record['geometry']
                    if gd['type'] == 'LineString':
                        geom = LineString(gd['coordinates'])
                    elif gd['type'] == 'MultiLineString':
                        geom = MultiLineString(gd['coordinates'])
                    elif gd['type'] == 'Polygon':
                        geom = Polygon(gd['coordinates'][0], gd['coordinates'][1:])
                    elif gd['type'] == 'MultiPolygon':
                        geom = MultiPolygon([Polygon(p[0], p[1:]) for p in gd['coordinates']])
                    else:
                        continue
                    original_geoms.append(geom)
                    sg = engine.simplify_geometry(geom, method, threshold, **kwargs)
                    if sg and not sg.is_empty:
                        simplified_geoms.append(sg)
                        dst.write({'geometry': mapping(sg), 'properties': record['properties']})
                        simplified += 1
                    else:
                        simplified_geoms.append(None)
                    processed += 1
                    if processed % 100 == 0:
                        print(f"  Processed {processed} features...")
                except Exception as e:
                    print(f"Error on feature {processed}: {e}")
                    errors += 1
                    simplified_geoms.append(None)

    return {
        'method': method, 'threshold': threshold,
        'features_processed': processed, 'features_simplified': simplified,
        'errors_encountered': errors,
        'success_rate': (simplified / processed * 100) if processed else 0.0,
        'output_file': output_file, 'status': 'SUCCESS',
    }


if __name__ == "__main__": 
    print("\n=== RUNNING BATCH PROCESSING WITH REFINED CONSTRAINTS ===")
    
    output_folder="./output_data_v13"
    input_file="./Input/500KClipped4_M2LFIX.shp"
    fault_file="./Input/GEOS_GEOLOGY_LINEARSTRUCTURE_500K_GSD.shp"    
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Fixed batch processing with proper syntax
    tolerance_list = [0.2, 2.0, 200000.0]
    tolerance_list = [ 50000.0,100000.0,150000.0,200000.0,2000000.0]
    #tolerance_list = [ 2000000.0]
    methods = ["decimation", "douglas_peucker", "douglas_peucker_tp", "bend_simplify", "visvalingam_whyatt", "modified_visvalingam_whyatt"]
    methods = ["modified_visvalingam_whyatt"]
    
    # Store results for CSV summary
    batch_results = []

    for method in methods:
        for tolerance in tolerance_list:
            try:
                # Construct output filename based on input, method, and tolerance
                input_basename = os.path.splitext(os.path.basename(input_file))[0]
                tolerance_str = str(tolerance).replace('.', '_')
                output_file = f'{output_folder}/{input_basename}_{method}_{tolerance_str}.shp'
                
                print(f"\n{'='*60}")
                print(f"Processing: {method} with threshold {tolerance}")
                print(f"Output file: {output_file}")
                print(f"{'='*60}")
                
                # Use two-stage processing for modified_visvalingam_whyatt
                if method == 'modified_visvalingam_whyatt':
                    result = vector_simplify_file_two_stage(
                        input_file=input_file,
                        output_file=output_file,
                        method=method,
                        threshold=tolerance,
                        fault_file=fault_file
                    )
                else:
                    result = vector_simplify_file(
                        input_file=input_file,
                        output_file=output_file,
                        method=method,
                        threshold=tolerance
                    )
                    
                if result:
                    batch_results.append(result)
                    print(f"✓ Successfully processed {method} with threshold {tolerance}")
                    
            except Exception as e:
                print(f"✗ Error occurred with {method} at {tolerance}: {e}")
                # Add error result to batch results
                batch_results.append({
                    'method': method,
                    'threshold': tolerance,
                    'features_processed': 0,
                    'features_simplified': 0,
                    'errors_encountered': 1,
                    'success_rate': 0.0,
                    'output_file': 'ERROR',
                    'status': 'ERROR',
                    'error_message': str(e),
                    'topological_junctions': 0
                })
                continue
    
    # Save batch results to CSV
    if batch_results:
        df_results = pd.DataFrame(batch_results)
        csv_filename = './output_data/refined_constraint_simplification_batch_summary.csv'
        df_results.to_csv(csv_filename, index=False)
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE - REFINED CONSTRAINTS")
        print(f"{'='*60}")
        print(f"Total combinations processed: {len(batch_results)}")
        successful_results = [r for r in batch_results if r['status'] == 'SUCCESS']
        print(f"Successful simplifications: {len(successful_results)}")
        if successful_results:
            avg_success_rate = sum(r['success_rate'] for r in successful_results) / len(successful_results)
            print(f"Average success rate: {avg_success_rate:.1f}%")
            
            # Calculate average constraint improvements for MVW
            mvw_results = [r for r in successful_results if r['method'] == 'modified_visvalingam_whyatt']
            if mvw_results:
                avg_topological_junctions = sum(r.get('topological_junctions', 0) for r in mvw_results) / len(mvw_results)
                print(f"Average topologically significant junctions preserved: {avg_topological_junctions:.1f}")
                
        print(f"Refined constraint results saved to: {csv_filename}")
        
        # Display summary table
        print(f"\nSummary of results:")
        print(f"{'Method':<25} {'Threshold':<12} {'Status':<10} {'Features':<10} {'Success Rate':<12} {'Topo Junctions':<15}")
        print(f"{'-'*25} {'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*15}")
        for result in batch_results:
            status = result['status']
            features = result['features_simplified']
            success_rate = f"{result['success_rate']:.1f}%" if result['success_rate'] > 0 else "0.0%"
            topo_junctions = result.get('topological_junctions', 0)
            print(f"{result['method']:<25} {result['threshold']:<12} {status:<10} {features:<10} {success_rate:<12} {topo_junctions:<15}")
    else:
        print("No results to save - all combinations failed.")