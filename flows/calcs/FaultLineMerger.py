import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge
import networkx as nx
from rtree import index
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

    def build_spatial_index(self, lines: List[LineString]) -> index.Index:
        """
        Build spatial index for efficient spatial queries.
        Uses full bounding boxes for the spatial index entries.

        Returns:
        Rtree spatial index
        """
        idx = index.Index()
        for i, line in enumerate(lines):
            bounds = line.bounds
            idx.insert(i, bounds)
        return idx

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
        """
        Calculate the angle ACB at the common node C.
        A is the adjacent node in line1, C is the common node, B is the adjacent node in line2.

        Returns:
        Angle ACB in degrees
        """
        coords1 = list(line1.coords)
        coords2 = list(line2.coords)

        # Get the three points: A, C, B
        if endpoint1 == "start" and endpoint2 == "start":
            # Common node C is at start of both lines
            C = coords1[0]  # Common node
            A = coords1[1] if len(coords1) > 1 else coords1[0]  # Next node in line1
            B = coords2[1] if len(coords2) > 1 else coords2[0]  # Next node in line2
        elif endpoint1 == "start" and endpoint2 == "end":
            # Common node C is at start of line1 and end of line2
            C = coords1[0]  # Common node
            A = coords1[1] if len(coords1) > 1 else coords1[0]  # Next node in line1
            B = (
                coords2[-2] if len(coords2) > 1 else coords2[-1]
            )  # Previous node in line2
        elif endpoint1 == "end" and endpoint2 == "start":
            # Common node C is at end of line1 and start of line2
            C = coords1[-1]  # Common node
            A = (
                coords1[-2] if len(coords1) > 1 else coords1[-1]
            )  # Previous node in line1
            B = coords2[1] if len(coords2) > 1 else coords2[0]  # Next node in line2
        else:  # endpoint1 == 'end' and endpoint2 == 'end'
            # Common node C is at end of both lines
            C = coords1[-1]  # Common node
            A = (
                coords1[-2] if len(coords1) > 1 else coords1[-1]
            )  # Previous node in line1
            B = (
                coords2[-2] if len(coords2) > 1 else coords2[-1]
            )  # Previous node in line2

        # Calculate direction vectors CA and CB
        CA = (A[0] - C[0], A[1] - C[1])
        CB = (B[0] - C[0], B[1] - C[1])

        # Calculate magnitudes
        mag_CA = math.sqrt(CA[0] ** 2 + CA[1] ** 2)
        mag_CB = math.sqrt(CB[0] ** 2 + CB[1] ** 2)

        if mag_CA == 0 or mag_CB == 0:
            return 180.0  # Default to straight line if no length

        # Calculate dot product
        dot_product = CA[0] * CB[0] + CA[1] * CB[1]

        # Calculate cosine of angle
        cos_angle = dot_product / (mag_CA * mag_CB)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range

        # Calculate angle in degrees
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

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
        # Get endpoint coordinates
        coords1 = list(line1.coords)
        coords2 = list(line2.coords)

        if endpoint1 == "start":
            p1 = Point(coords1[0])
            angle1 = self.get_line_angle(line1, from_end=False)
        else:
            p1 = Point(coords1[-1])
            angle1 = self.get_line_angle(line1, from_end=True)

        if endpoint2 == "start":
            p2 = Point(coords2[0])
            angle2 = self.get_line_angle(line2, from_end=False)
        else:
            p2 = Point(coords2[-1])
            angle2 = self.get_line_angle(line2, from_end=True)

        # Check distance
        distance = p1.distance(p2)
        if distance > self.distance_tolerance:
            return False, float("inf")

        # Check angle - lines should be roughly parallel (or anti-parallel)
        angle_diff = self.angle_difference(angle1, angle2)
        angle_diff_anti = self.angle_difference(angle1, angle2 + math.pi)

        min_angle_diff = min(angle_diff, angle_diff_anti)

        if min_angle_diff > self.angle_tolerance:
            return False, float("inf")

        # NEW: Check the angle ACB at the join point
        join_angle = self.calculate_join_angle_acb(line1, line2, endpoint1, endpoint2)

        if join_angle < self.min_join_angle:
            return False, float("inf")  # Reject sharp angles at the join

        # Calculate straightness score (lower is better)
        # Combines distance and angle penalties
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
        Working merging algorithm with simple optimization.

        Returns:
        List of merged LineString geometries
        """
        # Create working copy
        working_lines = lines.copy()

        changes_made = True
        iteration = 0

        while changes_made and len(working_lines) > 1:
            changes_made = False
            iteration += 1
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Processing {len(working_lines)} lines")

            # Rebuild spatial index for current lines
            spatial_idx = self.build_spatial_index(working_lines)

            # Simple optimization: process larger indices first to minimize index shifting
            # when we remove elements
            for i in range(len(working_lines) - 1, -1, -1):
                if i >= len(working_lines):  # Safety check
                    continue

                target_line = working_lines[i]

                # Find merge candidates
                candidates = self.find_merge_candidates(i, working_lines, spatial_idx)

                if candidates:
                    # Sort by straightness score (best first)
                    candidates.sort(key=lambda x: x[3])

                    # Take the best candidate
                    best_candidate = candidates[0]
                    candidate_idx, endpoint1, endpoint2, score = best_candidate

                    # Skip if candidate index is out of range (shouldn't happen but safety first)
                    if candidate_idx >= len(working_lines):
                        continue

                    # Merge the lines
                    candidate_line = working_lines[candidate_idx]
                    merged_line = self.merge_two_lines(
                        target_line, candidate_line, endpoint1, endpoint2
                    )

                    # Remove old lines and add merged line
                    # Remove higher index first to maintain correct indices
                    if candidate_idx > i:
                        working_lines.pop(candidate_idx)
                        working_lines.pop(i)
                    else:
                        working_lines.pop(i)
                        working_lines.pop(candidate_idx)

                    working_lines.append(merged_line)

                    changes_made = True
                    if i % 100 == 0:
                        print(f"  Merged lines: {len(working_lines)} remaining")
                    break  # Restart the loop

        print(f"Merging complete after {iteration} iterations")
        print(f"Final result: {len(working_lines)} lines")

        return working_lines

    def find_merge_candidates(
        self, target_idx: int, lines: List[LineString], spatial_idx: index.Index
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

            # Find lines that intersect with this endpoint area
            potential_indices = list(spatial_idx.intersection(endpoint_bounds))

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
        input_path: Path to input shapefile
        output_path: Path to output shapefile
        """
        # print(f"Reading shapefile: {input_path}")
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
        """# Save to shapefile
        print(f"Saving to: {output_path}")
        output_gdf.to_file(output_path)
        print("Processing complete!")"""


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
    output_gdf = merger.process_shapefile(gdf, input_shapefile)
    output_gdf.to_file(output_shapefile, driver="ESRI Shapefile")
