# -*- coding: utf-8 -*-
"""
FaultLineMerger - Merges connected fault line segments based on distance and angle criteria.

This version uses Shapely's STRtree instead of rtree to avoid DLL conflicts with QGIS.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import linemerge
from shapely import STRtree
import math
from typing import List, Tuple, Optional


class FaultLineMerger:
    def __init__(
        self,
        distance_tolerance: float = 10.0,
        angle_tolerance: float = 30.0,
        min_join_angle: float = 150.0,
    ):
        """
        Initialize the fault line merger.

        Parameters:
        distance_tolerance: Maximum distance between endpoints to consider for merging (in map units)
        angle_tolerance: Maximum angle difference between line segments to allow merging (in degrees)
        min_join_angle: Minimum angle at the join point (in degrees). Angles less than this will be rejected.
        """
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance = angle_tolerance
        self.min_join_angle = min_join_angle

    def read_shapefile(self, filepath: str) -> gpd.GeoDataFrame:
        """Read shapefile and return GeoDataFrame."""
        return gpd.read_file(filepath)

    def explode_multilinestrings(self, gdf: gpd.GeoDataFrame) -> List[LineString]:
        """
        Explode MultiLineStrings to individual LineStrings.

        Returns:
        List of LineString geometries
        """
        lines = []
        for geom in gdf.geometry:
            if isinstance(geom, MultiLineString):
                lines.extend(list(geom.geoms))
            elif isinstance(geom, LineString):
                lines.append(geom)
        return lines

    def build_spatial_index(self, lines: List[LineString]) -> Tuple[STRtree, List[LineString]]:
        """
        Build spatial index for efficient spatial queries using Shapely's STRtree.

        Returns:
        Tuple of (STRtree spatial index, list of lines)
        """
        tree = STRtree(lines)
        return tree, lines

    def get_endpoint_bounds(
        self, line: LineString, endpoint: str
    ) -> Tuple[float, float, float, float]:
        """
        Get bounding box around a specific endpoint of a line.

        Parameters:
        line: LineString geometry
        endpoint: 'start' or 'end'

        Returns:
        Bounding box tuple (minx, miny, maxx, maxy) around the endpoint
        """
        coords = list(line.coords)

        if endpoint == "start":
            point = coords[0]
        else:  # endpoint == 'end'
            point = coords[-1]

        # Create small bounding box around the endpoint
        minx = point[0] - self.distance_tolerance
        miny = point[1] - self.distance_tolerance
        maxx = point[0] + self.distance_tolerance
        maxy = point[1] + self.distance_tolerance

        return (minx, miny, maxx, maxy)

    def get_line_angle(self, line: LineString, from_end: bool = True) -> float:
        """
        Calculate the angle of a line segment at either end.

        Parameters:
        line: LineString geometry
        from_end: If True, calculate angle from the end; if False, from the start

        Returns:
        Angle in radians
        """
        coords = list(line.coords)

        if from_end:
            # Angle from second-to-last to last point
            if len(coords) >= 2:
                p1, p2 = coords[-2], coords[-1]
            else:
                return 0.0
        else:
            # Angle from first to second point
            if len(coords) >= 2:
                p1, p2 = coords[0], coords[1]
            else:
                return 0.0

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.atan2(dy, dx)

    def angle_difference(self, angle1: float, angle2: float) -> float:
        """
        Calculate the minimum angle difference between two angles.

        Returns:
        Angle difference in degrees
        """
        diff = abs(angle1 - angle2)
        diff = min(diff, 2 * math.pi - diff)  # Handle wraparound
        return math.degrees(diff)

    def calculate_join_angle_acb(
        self, line1: LineString, line2: LineString, endpoint1: str, endpoint2: str
    ) -> float:
        """Calculate the angle ACB at the common node C (public API kept for compatibility)."""
        return self._calculate_join_angle_from_coords(
            list(line1.coords), list(line2.coords), endpoint1, endpoint2
        )

    def _calculate_join_angle_from_coords(
        self, coords1, coords2, endpoint1: str, endpoint2: str
    ) -> float:
        """Calculate angle ACB from pre-fetched coordinate lists."""
        if endpoint1 == "start" and endpoint2 == "start":
            C = coords1[0]
            A = coords1[1] if len(coords1) > 1 else coords1[0]
            B = coords2[1] if len(coords2) > 1 else coords2[0]
        elif endpoint1 == "start" and endpoint2 == "end":
            C = coords1[0]
            A = coords1[1] if len(coords1) > 1 else coords1[0]
            B = coords2[-2] if len(coords2) > 1 else coords2[-1]
        elif endpoint1 == "end" and endpoint2 == "start":
            C = coords1[-1]
            A = coords1[-2] if len(coords1) > 1 else coords1[-1]
            B = coords2[1] if len(coords2) > 1 else coords2[0]
        else:
            C = coords1[-1]
            A = coords1[-2] if len(coords1) > 1 else coords1[-1]
            B = coords2[-2] if len(coords2) > 1 else coords2[-1]

        CAx, CAy = A[0] - C[0], A[1] - C[1]
        CBx, CBy = B[0] - C[0], B[1] - C[1]

        mag_CA = math.hypot(CAx, CAy)
        mag_CB = math.hypot(CBx, CBy)

        if mag_CA == 0 or mag_CB == 0:
            return 180.0

        cos_angle = max(-1.0, min(1.0, (CAx * CBx + CAy * CBy) / (mag_CA * mag_CB)))
        return math.degrees(math.acos(cos_angle))

    def lines_can_merge(
        self, line1: LineString, line2: LineString, endpoint1: str, endpoint2: str
    ) -> Tuple[bool, float]:
        """
        Check if two lines can be merged based on distance and angle criteria.

        Parameters:
        line1, line2: LineString geometries
        endpoint1, endpoint2: 'start' or 'end' indicating which endpoints to check

        Returns:
        (can_merge: bool, straightness_score: float)
        """
        coords1 = list(line1.coords)
        coords2 = list(line2.coords)

        # Extract endpoints and compute angles directly from cached coords
        if endpoint1 == "start":
            px1, py1 = coords1[0]
            angle1 = math.atan2(coords1[1][1] - coords1[0][1], coords1[1][0] - coords1[0][0]) if len(coords1) >= 2 else 0.0
        else:
            px1, py1 = coords1[-1]
            angle1 = math.atan2(coords1[-1][1] - coords1[-2][1], coords1[-1][0] - coords1[-2][0]) if len(coords1) >= 2 else 0.0

        if endpoint2 == "start":
            px2, py2 = coords2[0]
            angle2 = math.atan2(coords2[1][1] - coords2[0][1], coords2[1][0] - coords2[0][0]) if len(coords2) >= 2 else 0.0
        else:
            px2, py2 = coords2[-1]
            angle2 = math.atan2(coords2[-1][1] - coords2[-2][1], coords2[-1][0] - coords2[-2][0]) if len(coords2) >= 2 else 0.0

        # Check distance without creating Point objects
        distance = math.hypot(px2 - px1, py2 - py1)
        if distance > self.distance_tolerance:
            return False, float("inf")

        # Check angle - lines should be roughly parallel (or anti-parallel)
        angle_diff = self.angle_difference(angle1, angle2)
        angle_diff_anti = self.angle_difference(angle1, angle2 + math.pi)

        min_angle_diff = min(angle_diff, angle_diff_anti)

        if min_angle_diff > self.angle_tolerance:
            return False, float("inf")

        # Check the angle ACB at the join point (pass cached coords to avoid re-reading)
        join_angle = self._calculate_join_angle_from_coords(coords1, coords2, endpoint1, endpoint2)

        if join_angle < self.min_join_angle:
            return False, float("inf")

        angle_penalty = (180.0 - join_angle) / (180.0 - self.min_join_angle)
        straightness_score = (
            distance
            + (min_angle_diff / self.angle_tolerance) * self.distance_tolerance
            + angle_penalty * self.distance_tolerance
        )

        return True, straightness_score

    def merge_two_lines(
        self, line1: LineString, line2: LineString, endpoint1: str, endpoint2: str
    ) -> LineString:
        """
        Merge two lines into one, handling coordinate order properly.

        Parameters:
        line1, line2: LineString geometries to merge
        endpoint1, endpoint2: Which endpoints are being connected

        Returns:
        Merged LineString
        """
        coords1 = list(line1.coords)
        coords2 = list(line2.coords)

        # Determine the correct order and orientation
        if endpoint1 == "end" and endpoint2 == "start":
            # line1_end -> line2_start: concatenate normally
            merged_coords = coords1 + coords2[1:]  # Skip duplicate point
        elif endpoint1 == "end" and endpoint2 == "end":
            # line1_end -> line2_end: reverse line2
            merged_coords = coords1 + coords2[-2::-1]  # Reverse line2, skip duplicate
        elif endpoint1 == "start" and endpoint2 == "start":
            # line1_start -> line2_start: reverse line1
            merged_coords = coords1[::-1] + coords2[1:]  # Reverse line1, skip duplicate
        else:  # endpoint1 == 'start' and endpoint2 == 'end'
            # line1_start -> line2_end: reverse both or reorder
            merged_coords = coords2 + coords1[1:]  # line2 + line1, skip duplicate

        return LineString(merged_coords)

    def merge_fault_lines(self, lines: List[LineString]) -> List[LineString]:
        """
        Merging algorithm: collects all non-conflicting merges in one pass, applies
        them all at once, then rebuilds the spatial index once per pass rather than
        once per merge.

        Returns:
        List of merged LineString geometries
        """
        working_lines = lines.copy()

        changes_made = True
        iteration = 0

        while changes_made and len(working_lines) > 1:
            changes_made = False
            iteration += 1
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Processing {len(working_lines)} lines")

            # Build spatial index once per pass
            spatial_tree, indexed_lines = self.build_spatial_index(working_lines)

            consumed = set()   # indices merged away this pass
            new_lines = []     # replacement lines produced this pass

            for i in range(len(working_lines) - 1, -1, -1):
                if i in consumed:
                    continue

                candidates = self.find_merge_candidates(i, working_lines, spatial_tree)

                # Filter out any candidate whose line was already consumed this pass
                candidates = [c for c in candidates if c[0] not in consumed]

                if not candidates:
                    continue

                candidates.sort(key=lambda x: x[3])
                candidate_idx, endpoint1, endpoint2, score = candidates[0]

                if candidate_idx >= len(working_lines):
                    continue

                merged_line = self.merge_two_lines(
                    working_lines[i], working_lines[candidate_idx], endpoint1, endpoint2
                )

                consumed.add(i)
                consumed.add(candidate_idx)
                new_lines.append(merged_line)
                changes_made = True

            if consumed:
                working_lines = [l for idx, l in enumerate(working_lines) if idx not in consumed]
                working_lines.extend(new_lines)
                print(f"  Pass {iteration}: {len(consumed)//2} merges → {len(working_lines)} lines remaining")

        print(f"Merging complete after {iteration} passes")
        print(f"Final result: {len(working_lines)} lines")

        return working_lines

    def find_merge_candidates(
        self, target_idx: int, lines: List[LineString], spatial_tree: STRtree
    ) -> List[Tuple[int, str, str, float]]:
        """
        Find all potential merge candidates for a given line using endpoint-based spatial queries.

        Returns:
        List of (line_index, target_endpoint, candidate_endpoint, straightness_score)
        """
        target_line = lines[target_idx]
        candidates = []

        # Check each endpoint of the target line
        for target_endpoint in ["start", "end"]:
            # Get small bounding box around this endpoint
            endpoint_bounds = self.get_endpoint_bounds(target_line, target_endpoint)
            
            # Create a box geometry for the query
            query_box = box(*endpoint_bounds)
            
            # Find lines that intersect with this endpoint area using STRtree
            # STRtree.query returns indices of geometries that intersect with the query geometry
            potential_indices = spatial_tree.query(query_box)

            for candidate_idx in potential_indices:
                if candidate_idx == target_idx:
                    continue

                candidate_line = lines[candidate_idx]

                # Check both endpoints of the candidate line
                for candidate_endpoint in ["start", "end"]:
                    can_merge, score = self.lines_can_merge(
                        target_line, candidate_line, target_endpoint, candidate_endpoint
                    )

                    if can_merge:
                        candidates.append(
                            (candidate_idx, target_endpoint, candidate_endpoint, score)
                        )

        return candidates

    def process_shapefile(self, gdf):
        """
        Main processing function.

        Parameters:
        gdf: GeoDataFrame with line geometries
        
        Returns:
        GeoDataFrame with merged line geometries
        """
        print(f"Original features: {len(gdf)}")

        # Explode multilinestrings
        print("Exploding MultiLineStrings...")
        lines = self.explode_multilinestrings(gdf)
        print(f"Individual line segments: {len(lines)}")

        # Merge lines
        print("Merging fault lines...")
        merged_lines = self.merge_fault_lines(lines)

        # Create new GeoDataFrame
        print("Creating output GeoDataFrame...")
        output_gdf = gpd.GeoDataFrame({"geometry": merged_lines}, crs=gdf.crs)

        return output_gdf


# Usage example
if __name__ == "__main__":
    # Configuration
    input_shapefile = "waxi_faults.shp"
    output_shapefile = "waxi_merged_faults_fast2bb.shp"

    # Create merger with custom tolerances
    merger = FaultLineMerger(
        distance_tolerance=10.0,  # 10 map units
        angle_tolerance=30.0,  # 30 degrees for line parallelism
        min_join_angle=150.0,  # Minimum 150° angle at join point (rejects angles < 150°)
    )
    gdf = gpd.read_file(input_shapefile)
    # Process the shapefile
    output_gdf = merger.process_shapefile(gdf)
    output_gdf.to_file(output_shapefile, driver="ESRI Shapefile")