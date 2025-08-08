import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from scipy.spatial import Delaunay
import pandas as pd
import random
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")


class PolygonTriangulator:
    def __init__(
        self,
        shapefile_path=None,
        gdf=None,
        id_column=None,
        min_area_threshold=1.0,
        distance_threshold=1.0,
        strat1="",
        strat2="",
        strat3="",
        strat4="",
        lithoname="",
    ):
        """
        Initialize the triangulator from a shapefile or GeoDataFrame.

        Args:
            shapefile_path: Path to shapefile, or
            gdf: GeoDataFrame with polygon data
            id_column: Column name containing polygon IDs (if None, uses index)
        """
        if gdf is not None:
            self.gdf = gdf.copy()
        elif shapefile_path is not None:
            self.gdf = gpd.read_file(shapefile_path)
        else:
            raise ValueError("Must provide either shapefile_path or gdf")

        # Set up ID column
        if id_column is None:
            self.id_column = "mesh_id"
            if "mesh_id" not in self.gdf.columns:
                self.gdf[self.id_column] = self.gdf.index
        else:
            self.id_column = id_column
            if id_column not in self.gdf.columns:
                raise ValueError(f"ID column '{id_column}' not found in data")

        # Ensure IDs are strings
        self.gdf[self.id_column] = self.gdf[self.id_column].astype(str)

        self.strat1 = strat1
        self.strat2 = strat2
        self.strat3 = strat3
        self.strat4 = strat4
        self.lithoname = lithoname
        self.min_area_threshold = min_area_threshold

    def triangulate_polygons(self, target_ids, field_config=None):
        """
        Triangulate specified polygons and combine with untriangulated polygons.

        Args:
            target_ids: List of polygon IDs to triangulate (or single ID)
            field_config: Dict specifying field names and types for random values
                         Format: {'field_name': 'string'|'float'|'int', ...}
                         Defaults to 5 fields if None provided

        Returns:
            GeoDataFrame containing both triangulated and untriangulated polygons
        """
        # Handle single ID
        if not isinstance(target_ids, (list, tuple)):
            target_ids = [target_ids]

        # Convert to strings
        target_ids = [str(tid) for tid in target_ids]

        # Default field configuration
        if field_config is None:
            field_config = {
                "fld_str": "string",
                "fld_float": "float",
                "fld_int1": "int",
                "fld_int2": "int",
                "fld_float2": "float",
            }

        print(f"Triangulating {len(target_ids)} polygon IDs...")

        all_records = []
        tri_id = 0

        # First, process triangulated polygons
        triangulated_indices = set()

        for target_id in target_ids:
            # Find polygons with this ID
            target_mask = self.gdf[self.id_column] == target_id
            target_polygons = self.gdf[target_mask]

            if len(target_polygons) == 0:
                print(f"Warning: No polygons found with ID {target_id}")
                continue

            print(f"Processing {len(target_polygons)} polygon(s) with ID {target_id}")

            # Triangulate each polygon with this ID
            for idx, row in target_polygons.iterrows():
                if row.geometry.area < self.min_area_threshold:
                    triangulated_indices.add(
                        idx
                    )  # Track which polygons were triangulated
                    polygon = row.geometry
                    triangles = self._triangulate_single_polygon(polygon)

                    print(
                        f"  Created {len(triangles)} triangles from polygon at index {idx}"
                    )

                    # Create triangle records
                    for triangle in triangles:
                        triangle_record = {
                            "rec_id": f"tri_{tri_id}",
                            "rec_type": "triangle",
                            "tri_id": tri_id,
                            "src_id": target_id,
                            "src_index": idx,
                            "orig_id": str(row[self.id_column]),
                            "area": triangle.area,
                            "geometry": triangle,
                            self.strat1: f"{tri_id}",
                            self.strat2: f"{tri_id}",
                            self.strat3: f"{tri_id}",
                            self.strat4: f"{tri_id}",
                            self.lithoname: f"{tri_id}",
                        }

                        # Add random field values
                        for field_name, field_type in field_config.items():
                            triangle_record[field_name] = self._generate_random_value(
                                field_type
                            )

                        # Copy other original fields if they exist
                        for col in self.gdf.columns:
                            if col not in ["geometry"] and col not in triangle_record:
                                triangle_record[f"{col}"] = row[col]

                        all_records.append(triangle_record)
                        tri_id += 1

        # Second, add untriangulated polygons
        untriangulated_count = 0
        for idx, row in self.gdf.iterrows():
            if idx not in triangulated_indices:
                untriangulated_count += 1
                untriangulated_record = {
                    "rec_id": f"orig_{idx}",
                    "rec_type": "original",
                    "tri_id": None,
                    "src_id": None,
                    "src_index": idx,
                    "orig_id": str(row[self.id_column]),
                    "area": row.geometry.area,
                    "geometry": row.geometry,
                }

                # For untriangulated polygons, preserve original field values if they exist
                # Otherwise use None/null values for the random fields
                for field_name, field_type in field_config.items():
                    # Check if this field exists in original data
                    if field_name in self.gdf.columns:
                        untriangulated_record[field_name] = row[field_name]
                    else:
                        # Set to None/null for fields that don't exist in original
                        untriangulated_record[field_name] = None

                # Copy other original fields
                for col in self.gdf.columns:
                    if col not in ["geometry"] and col not in untriangulated_record:
                        untriangulated_record[f"{col}"] = row[col]

                all_records.append(untriangulated_record)

        if not all_records:
            print("No records were created")
            return gpd.GeoDataFrame()

        # Create combined GeoDataFrame
        result_gdf = gpd.GeoDataFrame(all_records)

        # Set CRS to match original
        if hasattr(self.gdf, "crs") and self.gdf.crs is not None:
            result_gdf.crs = self.gdf.crs

        triangles_count = len([r for r in all_records if r["rec_type"] == "triangle"])

        print(f"Created combined dataset:")
        print(f"  - {triangles_count} triangles")
        print(f"  - {untriangulated_count} untriangulated polygons")
        print(f"  - {len(result_gdf)} total records")

        return result_gdf

    def _triangulate_single_polygon(self, polygon):
        """
        Triangulate a single polygon using Delaunay triangulation with hole awareness.
        """
        try:
            # Extract coordinates
            if isinstance(polygon, Polygon):
                # Check for holes
                if len(polygon.interiors) > 0:
                    print(
                        f"    Polygon has {len(polygon.interiors)} interior holes - using hole-aware triangulation"
                    )
                    return self._triangulate_polygon_with_holes(polygon)
                else:
                    coords = list(polygon.exterior.coords)[
                        :-1
                    ]  # Remove duplicate last point
            elif isinstance(polygon, MultiPolygon):
                # Use largest part for MultiPolygon
                largest_poly = max(polygon.geoms, key=lambda p: p.area)
                if len(largest_poly.interiors) > 0:
                    print(
                        f"    MultiPolygon largest part has {len(largest_poly.interiors)} holes - using hole-aware triangulation"
                    )
                    return self._triangulate_polygon_with_holes(largest_poly)
                else:
                    coords = list(largest_poly.exterior.coords)[:-1]
            else:
                print(f"Unsupported geometry type: {polygon.geom_type}")
                return []

            if len(coords) < 3:
                print("Polygon has too few points for triangulation")
                return []

            # Convert to numpy array
            points = np.array(coords)

            # Create Delaunay triangulation
            tri = Delaunay(points)

            # Create triangle polygons
            triangles = []
            for simplex in tri.simplices:
                triangle_coords = points[simplex]
                triangle_poly = Polygon(triangle_coords)

                # Check if triangle is inside original polygon
                if self._triangle_inside_polygon(triangle_poly, polygon):
                    triangles.append(triangle_poly)

            return triangles

        except Exception as e:
            print(f"Error triangulating polygon: {e}")
            return []

    def _triangulate_polygon_with_holes(self, polygon):
        """
        Triangulate a polygon that has interior holes.
        """
        try:
            # Try Triangle library first (most robust)
            try:
                import triangle as tr

                return self._triangulate_with_triangle_library(polygon)
            except ImportError:
                print("    Triangle library not available, using fallback method")
                pass

            # Fallback: Use Delaunay and filter out triangles in holes
            return self._triangulate_and_filter_holes(polygon)

        except Exception as e:
            print(f"    Error in hole-aware triangulation: {e}")
            return []

    def _triangulate_with_triangle_library(self, polygon):
        """
        Use Triangle library for constrained triangulation with holes.
        """
        import triangle as tr
        import numpy as np

        # Prepare vertices (exterior + all holes)
        vertices = []
        segments = []
        holes = []

        # Add exterior boundary
        exterior_coords = list(polygon.exterior.coords)[:-1]
        start_idx = 0
        vertices.extend(exterior_coords)

        # Create segments for exterior
        for i in range(len(exterior_coords)):
            segments.append(
                [start_idx + i, start_idx + ((i + 1) % len(exterior_coords))]
            )

        # Add interior holes
        for hole in polygon.interiors:
            hole_coords = list(hole.coords)[:-1]
            hole_start_idx = len(vertices)
            vertices.extend(hole_coords)

            # Create segments for this hole
            for i in range(len(hole_coords)):
                segments.append(
                    [hole_start_idx + i, hole_start_idx + ((i + 1) % len(hole_coords))]
                )

            # Add hole point (any point inside the hole)
            hole_poly = Polygon(hole)
            hole_centroid = hole_poly.centroid
            holes.append([hole_centroid.x, hole_centroid.y])

        # Prepare Triangle input
        tri_input = {
            "vertices": np.array(vertices, dtype=np.float64),
            "segments": np.array(segments, dtype=np.int32),
        }

        if holes:
            tri_input["holes"] = np.array(holes, dtype=np.float64)

        # Generate constrained triangulation
        try:
            # Try with quality mesh first
            mesh = tr.triangulate(tri_input, "pq20")  # Min angle 20 degrees
        except:
            try:
                # Fallback to basic constrained triangulation
                mesh = tr.triangulate(tri_input, "p")
            except:
                print("    Triangle library triangulation failed")
                return []

        # Extract triangles
        if "triangles" in mesh and "vertices" in mesh:
            mesh_vertices = mesh["vertices"]
            mesh_triangles = mesh["triangles"]

            triangles = []
            for tri_indices in mesh_triangles:
                try:
                    tri_coords = mesh_vertices[tri_indices]
                    triangle_poly = Polygon(tri_coords)

                    # Validate triangle
                    if triangle_poly.is_valid and triangle_poly.area > 1e-10:
                        triangles.append(triangle_poly)
                except:
                    continue

            print(f"    Triangle library created {len(triangles)} triangles")
            return triangles

        return []

    def _triangulate_and_filter_holes(self, polygon):
        """
        Fallback method: triangulate exterior and filter out triangles in holes.
        """
        # Get exterior coordinates
        coords = list(polygon.exterior.coords)[:-1]
        points = np.array(coords)

        # Create Delaunay triangulation of exterior
        tri = Delaunay(points)

        triangles = []
        for simplex in tri.simplices:
            triangle_coords = points[simplex]
            triangle_poly = Polygon(triangle_coords)
            triangle_centroid = triangle_poly.centroid

            # Check if triangle is in the polygon exterior
            if polygon.contains(triangle_centroid):
                # Additional check: ensure triangle doesn't overlap significantly with holes
                valid_triangle = True

                for hole in polygon.interiors:
                    hole_poly = Polygon(hole)

                    # Check if triangle centroid is in hole
                    if hole_poly.contains(triangle_centroid):
                        valid_triangle = False
                        break

                    # Check for significant overlap with hole
                    try:
                        overlap = triangle_poly.intersection(hole_poly)
                        if (
                            hasattr(overlap, "area")
                            and overlap.area > triangle_poly.area * 0.1
                        ):
                            valid_triangle = False
                            break
                    except:
                        pass

                if valid_triangle:
                    triangles.append(triangle_poly)

        print(
            f"    Fallback method created {len(triangles)} triangles (filtered for holes)"
        )
        return triangles

    def _triangle_inside_polygon(self, triangle, polygon):
        """
        Check if triangle is inside the original polygon.
        """
        try:
            triangle_centroid = triangle.centroid

            # Primary check: centroid inside
            if polygon.contains(triangle_centroid):
                return True

            # Secondary check: significant overlap
            try:
                overlap = triangle.intersection(polygon)
                if hasattr(overlap, "area"):
                    overlap_ratio = overlap.area / triangle.area
                    return overlap_ratio > 0.5
            except:
                pass

            return False

        except:
            return False

    def _generate_random_value(self, field_type):
        """
        Generate random value based on field type.
        """
        if field_type == "string":
            return str(random.randint(1, 100000))
        elif field_type == "float":
            return round(random.uniform(1.0, 100000.0), 3)
        elif field_type == "int":
            return random.randint(1, 100000)
        else:
            return random.randint(1, 100000)  # Default to int

    def visualize_triangulation(
        self, result_gdf, title="Triangulated Polygons", figsize=(15, 10)
    ):
        """
        Visualize the triangulation result showing both triangles and original polygons.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot original polygons
        self.gdf.plot(
            ax=ax1,
            alpha=0.6,
            edgecolor="black",
            linewidth=1,
            column=self.id_column,
            cmap="Set3",
            legend=False,
        )
        ax1.set_title("Original Polygons")
        ax1.set_aspect("equal")

        # Plot combined result
        if not result_gdf.empty:
            # Different colors for triangles vs original polygons
            triangles = result_gdf[result_gdf["rec_type"] == "triangle"]
            originals = result_gdf[result_gdf["rec_type"] == "original"]

            if not triangles.empty:
                triangles.plot(
                    ax=ax2,
                    alpha=0.7,
                    edgecolor="red",
                    linewidth=0.5,
                    color="lightcoral",
                    label="Triangles",
                )

            if not originals.empty:
                originals.plot(
                    ax=ax2,
                    alpha=0.6,
                    edgecolor="blue",
                    linewidth=1,
                    color="lightblue",
                    label="Original",
                )

            ax2.legend()
            ax2.set_title(
                f"Combined Result ({len(triangles)} triangles + {len(originals)} originals)"
            )
        else:
            ax2.set_title("No Results Created")
        ax2.set_aspect("equal")

        plt.tight_layout()
        return fig, (ax1, ax2)

    def save_result(self, result_gdf, output_path):
        """
        Save combined result to file.
        """
        try:
            if result_gdf.empty:
                print("No data to save")
                return

            if result_gdf.crs is None:
                print("Warning: No CRS defined, setting to EPSG:4326")
                result_gdf = result_gdf.set_crs("EPSG:4326")

            # Try shapefile first
            if output_path.endswith(".shp"):
                result_gdf.to_file(output_path, driver="ESRI Shapefile")
                print(f"Saved {len(result_gdf)} records to {output_path}")
            else:
                # Default to GeoJSON
                result_gdf.to_file(output_path, driver="GeoJSON")
                print(f"Saved {len(result_gdf)} records to {output_path}")

            # Print summary
            triangles = len(result_gdf[result_gdf["rec_type"] == "triangle"])
            originals = len(result_gdf[result_gdf["rec_type"] == "original"])
            print(f"  - {triangles} triangulated polygons")
            print(f"  - {originals} original polygons")

        except Exception as e:
            print(f"Error saving result: {e}")
            # Try alternative format
            try:
                alt_path = str(output_path).replace(".shp", ".geojson")
                result_gdf.to_file(alt_path, driver="GeoJSON")
                print(f"Saved as GeoJSON to {alt_path}")
            except Exception as e2:
                print(f"Failed to save in any format: {e2}")


def create_sample_data():
    """Create sample polygon data for testing."""
    polygons = []
    ids = []

    # Create a few test polygons
    size = 100

    # Square
    square = Polygon([(0, 0), (size, 0), (size, size), (0, size)])
    polygons.append(square)
    ids.append("A")

    # Triangle
    triangle = Polygon([(150, 0), (200, 80), (100, 80)])
    polygons.append(triangle)
    ids.append("B")

    # Pentagon
    pentagon = Polygon([(250, 50), (280, 20), (320, 40), (310, 80), (260, 85)])
    polygons.append(pentagon)
    ids.append("C")

    gdf = gpd.GeoDataFrame(
        {"poly_id": ids, "area": [p.area for p in polygons], "geometry": polygons},
        crs="EPSG:4326",
    )

    return gdf


# Example usage
if __name__ == "__main__":
    print("Polygon Triangulation Tool")

    # Option 1: Use existing shapefile
    shapefile_path = "wa100k/100k_IGB_buffered.shp"

    try:
        if Path(shapefile_path).exists():
            print(f"Loading {shapefile_path}...")
            triangulator = PolygonTriangulator(
                shapefile_path=shapefile_path, id_column="CODE"
            )
        else:
            print("Creating sample data...")
            sample_gdf = create_sample_data()
            triangulator = PolygonTriangulator(gdf=sample_gdf, id_column="poly_id")
    except Exception as e:
        print(f"Using sample data due to error: {e}")
        sample_gdf = create_sample_data()
        triangulator = PolygonTriangulator(gdf=sample_gdf, id_column="poly_id")

    print(f"Loaded {len(triangulator.gdf)} polygons")
    print(
        "Available polygon IDs:",
        list(triangulator.gdf[triangulator.id_column].unique()),
    )

    # Define custom field configuration
    field_config = {
        "random_str": "string",
        "value_float": "float",
        "count_int": "int",
        "weight": "float",
        "rank": "int",
    }

    # Triangulate specific polygons
    target_polygons = ["P_-_wz-om"]  # Change to your target IDs, e.g., [3874, 3875]

    try:
        result_gdf = triangulator.triangulate_polygons(
            target_ids=target_polygons, field_config=field_config
        )

        if not result_gdf.empty:
            print(f"\nTriangulation Results:")
            print(f"Total records: {len(result_gdf)}")

            # Show breakdown
            triangles = result_gdf[result_gdf["rec_type"] == "triangle"]
            originals = result_gdf[result_gdf["rec_type"] == "original"]

            print(f"  - {len(triangles)} triangulated polygons")
            print(f"  - {len(originals)} original polygons")
            print(f"Field columns: {list(result_gdf.columns)}")

            # Show sample data
            print("\nSample triangle data:")
            if not triangles.empty:
                print(
                    triangles[
                        ["rec_id", "tri_id", "src_id", "area"]
                        + list(field_config.keys())
                    ].head(3)
                )

            print("\nSample original polygon data:")
            if not originals.empty:
                print(
                    originals[
                        ["rec_id", "orig_id", "area"] + list(field_config.keys())
                    ].head(3)
                )

            # Visualize
            fig, axes = triangulator.visualize_triangulation(result_gdf)

            # Save result
            output_file = "combined_result_holes.shp"
            triangulator.save_result(result_gdf, output_file)

            plt.show()
        else:
            print("No results were created")

    except Exception as e:
        print(f"Error during triangulation: {e}")
        import traceback

        traceback.print_exc()
