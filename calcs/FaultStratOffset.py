import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
import warnings

warnings.filterwarnings("ignore")

class FaultStratOffset:
    def __init__(
        self
    ):
        """
        Initialize the Fault Strat Offset Calculation.

        
        """
        pass

    def read_shapefiles(self,fault_path, geology_path):
        """
        Read fault polylines and geology polygons from shapefiles
        """
        print("Reading shapefiles...")
        faults_gdf = gpd.read_file(fault_path)
        geology_gdf = gpd.read_file(geology_path)

        # Ensure same CRS
        if faults_gdf.crs != geology_gdf.crs:
            geology_gdf = geology_gdf.to_crs(faults_gdf.crs)
            print(f"Reprojected geology to match fault CRS: {faults_gdf.crs}")

        return faults_gdf, geology_gdf
    
    def get_sampling_points(self,faults_gdf, offset_distance=50):
        """
        Create offset points at midpoint and both endpoints of each fault

        Parameters:
        - faults_gdf: GeoDataFrame of fault polylines
        - offset_distance: Distance in map units to offset points from fault line

        Returns:
        - GeoDataFrame of points with fault IDs and location info
        """
        print("Creating offset points at fault midpoints and endpoints...")

        points_list = []

        for idx, fault in faults_gdf.iterrows():
            # Get fault geometry
            fault_line = fault.geometry
            fault_id = idx  # or use a specific ID column if available

            # Define sampling locations: start (0), middle (0.5), end (1)
            sampling_locations = [(0.0, "start"), (0.5, "midpoint"), (1.0, "end")]

            for position, location_name in sampling_locations:
                # Get point at this position along the line
                sample_point = fault_line.interpolate(position, normalized=True)

                # Get the bearing (direction) of the line at this point
                # Sample points slightly before and after to get direction
                if position == 0.0:
                    # At start, look forward
                    point_before = sample_point
                    point_after = fault_line.interpolate(0.01, normalized=True)
                elif position == 1.0:
                    # At end, look backward
                    point_before = fault_line.interpolate(0.99, normalized=True)
                    point_after = sample_point
                else:
                    # At middle, look both ways
                    point_before = fault_line.interpolate(position - 0.01, normalized=True)
                    point_after = fault_line.interpolate(position + 0.01, normalized=True)

                # Calculate perpendicular direction
                dx = point_after.x - point_before.x
                dy = point_after.y - point_before.y

                # Normalize the direction vector
                length = np.sqrt(dx**2 + dy**2)
                if length == 0:
                    continue

                dx_norm = dx / length
                dy_norm = dy / length

                # Perpendicular vectors (90 degrees rotated)
                perp_dx = -dy_norm
                perp_dy = dx_norm

                # Create 2 points: one on each side of fault at this location
                # Side 1 (positive perpendicular direction)
                p1 = Point(
                    sample_point.x + perp_dx * offset_distance,
                    sample_point.y + perp_dy * offset_distance,
                )

                # Side 2 (negative perpendicular direction)
                p2 = Point(
                    sample_point.x - perp_dx * offset_distance,
                    sample_point.y - perp_dy * offset_distance,
                )

                # Store points with fault ID, location, and side information
                points_list.extend(
                    [
                        {
                            "fault_id": fault_id,
                            "location": location_name,
                            "point_type": "side1",
                            "geometry": p1,
                        },
                        {
                            "fault_id": fault_id,
                            "location": location_name,
                            "point_type": "side2",
                            "geometry": p2,
                        },
                    ]
                )

        # Create GeoDataFrame of points
        points_gdf = gpd.GeoDataFrame(points_list, crs=faults_gdf.crs)

        return points_gdf


    def assign_geology_to_points(self,points_gdf, geology_gdf, strat_columns):
        """
        Spatial join points with geology polygons to get stratigraphic information

        Parameters:
        - points_gdf: GeoDataFrame of offset points
        - geology_gdf: GeoDataFrame of geology polygons
        - strat_columns: List of column names representing stratigraphic hierarchy (most detailed first)

        Returns:
        - GeoDataFrame with geology information joined
        """
        print("Performing spatial join with geology...")

        # Spatial join - each point gets attributes from intersecting polygon
        points_with_geology = gpd.sjoin(
            points_gdf, geology_gdf, how="left", predicate="within"
        )

        # Keep only the columns we need
        keep_columns = ["fault_id", "location", "point_type", "geometry"] + strat_columns
        if "index_right" in points_with_geology.columns:
            points_with_geology = points_with_geology.drop("index_right", axis=1)

        # Select only available columns
        available_columns = [
            col for col in keep_columns if col in points_with_geology.columns
        ]
        points_with_geology = points_with_geology[available_columns]

        return points_with_geology


    def calculate_stratigraphic_offset(self,points_with_geology, strat_columns):
        """
        Optimized calculation of stratigraphic offset using vectorized operations

        Parameters:
        - points_with_geology: GeoDataFrame with points and geology info
        - strat_columns: List of stratigraphic columns (most detailed first)

        Returns:
        - DataFrame with fault offsets
        """
        print("Calculating stratigraphic offsets (optimized)...")

        # Filter out points with no geology data
        valid_points = points_with_geology.dropna(subset=strat_columns, how="all").copy()

        if len(valid_points) == 0:
            print("Warning: No valid geology data found!")
            return pd.DataFrame(), pd.DataFrame()

        # Pivot to get side1 and side2 data in separate columns
        print("Restructuring data...")
        pivot_data = []

        for (fault_id, location), group in valid_points.groupby(["fault_id", "location"]):
            side1_data = group[group["point_type"] == "side1"]
            side2_data = group[group["point_type"] == "side2"]

            # Check if we have data on both sides
            side1_has_data = len(side1_data) > 0 and any(
                pd.notna(side1_data[col].iloc[0]) for col in strat_columns
            )
            side2_has_data = len(side2_data) > 0 and any(
                pd.notna(side2_data[col].iloc[0]) for col in strat_columns
            )

            row_data = {
                "fault_id": fault_id,
                "location": location,
                "side1_has_data": side1_has_data,
                "side2_has_data": side2_has_data,
            }

            # Add stratigraphic data for both sides
            for col in strat_columns:
                row_data[f"side1_{col}"] = (
                    side1_data[col].iloc[0] if len(side1_data) > 0 else None
                )
                row_data[f"side2_{col}"] = (
                    side2_data[col].iloc[0] if len(side2_data) > 0 else None
                )

            pivot_data.append(row_data)

        if len(pivot_data) == 0:
            print("Warning: No valid fault-location pairs found!")
            return pd.DataFrame(), pd.DataFrame()

        pivot_df = pd.DataFrame(pivot_data)

        print("Computing offset levels...")
        # Calculate offset levels vectorized
        offset_levels = []
        offset_descriptions = []

        for _, row in pivot_df.iterrows():
            offset_level = None
            offset_desc = "No offset calculated"

            # Check if we have data on both sides
            if not row["side1_has_data"] and not row["side2_has_data"]:
                # No data on either side
                offset_level = None
                offset_desc = "No geological data found"
            elif not row["side1_has_data"] or not row["side2_has_data"]:
                # Data on only one side - fault at edge of map
                offset_level = -1
                offset_desc = "Fault at edge of map"
            else:
                # Data on both sides - check for differences
                # Check each priority level (most general to most detailed)
                for priority_level, strat_col in enumerate(reversed(strat_columns), 1):
                    side1_val = row[f"side1_{strat_col}"]
                    side2_val = row[f"side2_{strat_col}"]

                    # Skip if either side has no data for this column
                    if pd.isna(side1_val) or pd.isna(side2_val):
                        continue

                    # Check if different
                    if side1_val != side2_val:
                        actual_priority = len(strat_columns) - priority_level + 1
                        offset_level = actual_priority
                        offset_desc = (
                            f"Offset at {strat_col} level (priority {actual_priority})"
                        )
                        break

                # If no differences found but we have data on both sides
                if offset_level is None:
                    offset_level = 0
                    offset_desc = "No offset - same at all levels"

            offset_levels.append(offset_level)
            offset_descriptions.append(offset_desc)

        # Add results to pivot_df
        pivot_df["offset_level"] = offset_levels
        pivot_df["offset_description"] = offset_descriptions

        # Create location results DataFrame
        location_results = pivot_df[
            ["fault_id", "location", "offset_level", "offset_description"]
        ].copy()

        # Add side unit information
        location_results["side1_units"] = [
            [row[f"side1_{col}"] for col in strat_columns if pd.notna(row[f"side1_{col}"])]
            for _, row in pivot_df.iterrows()
        ]
        location_results["side2_units"] = [
            [row[f"side2_{col}"] for col in strat_columns if pd.notna(row[f"side2_{col}"])]
            for _, row in pivot_df.iterrows()
        ]

        print("Finding maximum offsets per fault...")
        # Find maximum offset for each fault using groupby operations
        fault_max_offsets = (
            location_results.groupby("fault_id")
            .agg({"offset_level": ["max", "idxmax"]})
            .reset_index()
        )

        # Flatten column names
        fault_max_offsets.columns = ["fault_id", "max_offset_level", "max_idx"]

        # Create final results
        final_results = []

        for _, fault_row in fault_max_offsets.iterrows():
            fault_id = fault_row["fault_id"]
            max_offset_level = fault_row["max_offset_level"]

            if pd.isna(max_offset_level):
                fault_summary = {
                    "fault_id": fault_id,
                    "max_offset_level": None,
                    "max_offset_location": "no_data",
                    "max_offset_description": "No geological data found",
                    "side1_units": [],
                    "side2_units": [],
                }
            elif max_offset_level == -1:
                # Find which location has the edge condition
                edge_locations = location_results[
                    (location_results["fault_id"] == fault_id)
                    & (location_results["offset_level"] == -1)
                ]
                edge_location = (
                    edge_locations["location"].iloc[0]
                    if len(edge_locations) > 0
                    else "unknown"
                )

                fault_summary = {
                    "fault_id": fault_id,
                    "max_offset_level": -1,
                    "max_offset_location": edge_location,
                    "max_offset_description": "Fault at edge of map",
                    "side1_units": [],
                    "side2_units": [],
                }
            else:
                max_idx = fault_row["max_idx"]
                max_row = location_results.loc[max_idx]

                fault_summary = {
                    "fault_id": fault_id,
                    "max_offset_level": max_row["offset_level"],
                    "max_offset_location": max_row["location"],
                    "max_offset_description": max_row["offset_description"],
                    "side1_units": max_row["side1_units"],
                    "side2_units": max_row["side2_units"],
                }

            # Add location-specific details
            fault_locations = location_results[location_results["fault_id"] == fault_id]
            for _, loc_row in fault_locations.iterrows():
                location = loc_row["location"]
                fault_summary[f"{location}_offset"] = loc_row["offset_level"]
                fault_summary[f"{location}_desc"] = loc_row["offset_description"]

            final_results.append(fault_summary)

        return pd.DataFrame(final_results), location_results


    def calculate_stratigraphic_offset_slow(self,points_with_geology, strat_columns):
        """
        Calculate stratigraphic offset for each fault at each sampling location,
        then return the maximum offset for each fault

        Parameters:
        - points_with_geology: GeoDataFrame with points and geology info
        - strat_columns: List of stratigraphic columns (most detailed first)

        Returns:
        - DataFrame with fault offsets
        """
        print("Calculating stratigraphic offsets...")

        all_location_results = []

        # Group by fault_id and location to process each sampling point
        for (fault_id, location), group in points_with_geology.groupby(
            ["fault_id", "location"]
        ):

            if len(group) != 2:
                print(
                    f"Warning: Fault {fault_id} at {location} doesn't have exactly 2 points"
                )
                continue

            # Get stratigraphic values for each priority level
            offset_info = {
                "fault_id": fault_id,
                "location": location,
                "offset_level": None,
                "offset_description": "No offset calculated",
                "side1_units": [],
                "side2_units": [],
            }

            # Separate points by side
            side1_point = group[group["point_type"] == "side1"]
            side2_point = group[group["point_type"] == "side2"]

            # Try each priority level (most general to most detailed)
            for priority_level, strat_col in enumerate(reversed(strat_columns), 1):

                if strat_col not in group.columns:
                    continue

                # Get unique units on each side for this priority level
                side1_units = set(side1_point[strat_col].dropna().unique())
                side2_units = set(side2_point[strat_col].dropna().unique())

                # Remove any None/NaN values
                side1_units = {unit for unit in side1_units if pd.notna(unit)}
                side2_units = {unit for unit in side2_units if pd.notna(unit)}

                offset_info["side1_units"] = list(side1_units)
                offset_info["side2_units"] = list(side2_units)

                # Skip if no data on either side
                if len(side1_units) == 0 or len(side2_units) == 0:
                    continue

                # Check if units are different at this level
                if side1_units != side2_units:
                    # Found difference at this level - this is the offset level
                    actual_priority = len(strat_columns) - priority_level + 1
                    offset_info["offset_level"] = actual_priority
                    offset_info["offset_description"] = (
                        f"Offset at {strat_col} level (priority {actual_priority})"
                    )
                    break

                # If same at this level, continue to more detailed level

            # If we got through all levels without finding differences
            if offset_info["offset_level"] is None and len(offset_info["side1_units"]) > 0:
                offset_info["offset_level"] = 0
                offset_info["offset_description"] = "No offset - same at all levels"

            all_location_results.append(offset_info)

        # Convert to DataFrame
        location_results_df = pd.DataFrame(all_location_results)

        # Now find the maximum offset for each fault
        final_results = []

        for fault_id, fault_group in location_results_df.groupby("fault_id"):

            # Find the location with maximum offset level - handle NaN values
            valid_offsets = fault_group.dropna(subset=["offset_level"])

            if len(valid_offsets) == 0:
                # No valid offsets found for this fault
                fault_summary = {
                    "fault_id": fault_id,
                    "max_offset_level": None,
                    "max_offset_location": "no_data",
                    "max_offset_description": "No geological data found",
                    "side1_units": [],
                    "side2_units": [],
                }
            else:
                # Find the location with maximum offset level
                max_offset_idx = valid_offsets["offset_level"].idxmax()
                max_offset_row = valid_offsets.loc[max_offset_idx]

                # Create summary for this fault
                fault_summary = {
                    "fault_id": fault_id,
                    "max_offset_level": max_offset_row["offset_level"],
                    "max_offset_location": max_offset_row["location"],
                    "max_offset_description": max_offset_row["offset_description"],
                    "side1_units": max_offset_row["side1_units"],
                    "side2_units": max_offset_row["side2_units"],
                }

            # Add details for each location
            for _, row in fault_group.iterrows():
                fault_summary[f'{row["location"]}_offset'] = row["offset_level"]
                fault_summary[f'{row["location"]}_desc'] = row["offset_description"]

            final_results.append(fault_summary)

            return pd.DataFrame(final_results), location_results_df


    def create_fault_offset_shapefile(self,faults_gdf, offset_results, output_path):
        """
        Join offset results with original fault polylines and save as shapefile

        Parameters:
        - faults_gdf: Original fault GeoDataFrame
        - offset_results: DataFrame with offset calculations
        - output_path: Path for output shapefile
        """
        print("Creating output shapefile...")

        # Reset index to use as fault_id if needed
        if "fault_id" not in faults_gdf.columns:
            faults_with_id = faults_gdf.copy()
            faults_with_id["fault_id"] = faults_with_id.index
        else:
            faults_with_id = faults_gdf.copy()

        # Merge with offset results
        faults_with_offset = faults_with_id.merge(offset_results, on="fault_id", how="left")

        # Clean up column names for shapefile (limit to 10 characters)
        column_mapping = {
            "max_offset_level": "max_off_lv",
            "max_offset_location": "max_off_loc",
            "max_offset_description": "max_off_ds",
            "side1_units": "side1_unit",
            "side2_units": "side2_unit",
            "strat_offset": "strat_off",
            "midpoint_offset": "mid_off",
            "end_offset": "end_off",
            "strat_desc": "strat_desc",
            "midpoint_desc": "mid_desc",
            "end_desc": "end_desc",
        }

        # Only rename columns that exist
        existing_mapping = {
            k: v for k, v in column_mapping.items() if k in faults_with_offset.columns
        }
        faults_with_offset = faults_with_offset.rename(columns=existing_mapping)

        # Convert lists to strings for shapefile compatibility
        for col in ["side1_unit", "side2_unit"]:
            if col in faults_with_offset.columns:
                faults_with_offset[col] = faults_with_offset[col].apply(
                    lambda x: str(x) if isinstance(x, list) else str(x)
                )

        # Save to shapefile
        faults_with_offset.to_file(output_path)
        print(f"Output saved to: {output_path}")

        return faults_with_offset


    # Main execution function
    def CalcFaultStratOffset(
        self,fault_shapefile,
        geology_shapefile,
        output_shapefile,
        strat_columns,
        offset_distance=50,
    ):
        """
        Main function to execute the complete workflow

        Parameters:
        - fault_shapefile: Path to fault polylines shapefile
        - geology_shapefile: Path to geology polygons shapefile
        - output_shapefile: Path for output shapefile
        - strat_columns: List of stratigraphic column names (most detailed first)
        - offset_distance: Distance to offset points from fault midpoint
        """

        try:
            # Step 0: Read shapefiles
            faults_gdf, geology_gdf = self.read_shapefiles(fault_shapefile, geology_shapefile)

            print(f"Loaded {len(faults_gdf)} fault segments")
            print(f"Loaded {len(geology_gdf)} geology polygons")
            print(f"Stratigraphic columns: {strat_columns}")

            # Step 1-2: Create offset points at fault midpoints
            points_gdf = self.get_sampling_points(faults_gdf, offset_distance)
            print(
                f"Created {len(points_gdf)} offset points ({len(points_gdf)//6} faults × 6 points each)"
            )
            # Step 3: Spatial join with geology
            points_with_geology = self.assign_geology_to_points(
                points_gdf, geology_gdf, strat_columns
            )

            # Step 4: Calculate stratigraphic offsets
            offset_results, location_details = self.calculate_stratigraphic_offset(
                points_with_geology, strat_columns
            )

            # Step 5: Create output shapefile
            result_gdf = self.create_fault_offset_shapefile(
                faults_gdf, offset_results, output_shapefile
            )

            print("\nSummary of maximum offsets:")
            print(offset_results["max_offset_level"].value_counts().sort_index())
            print("\nOffset locations:")
            print(offset_results["max_offset_location"].value_counts())

            return result_gdf, offset_results, location_details

        except Exception as e:
            print(f"Error in processing: {str(e)}")
            raise


  