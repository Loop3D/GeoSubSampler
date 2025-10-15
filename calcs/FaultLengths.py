import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from typing import Union, Optional
import warnings


class FaultLengths:
    """
    A class to systematically remove polylines shorter than a defined length.

    Supports filtering LineString and MultiLineString geometries based on their
    length in the coordinate reference system units.
    """

    def __init__(self, min_length: float, crs_units: str = "unknown"):
        """
        Initialize the PolylineFilter.

        Parameters:
        -----------
        min_length : float
            Minimum length threshold. Polylines shorter than this will be removed.
        crs_units : str, optional
            Units of the coordinate reference system (e.g., 'meters', 'feet', 'degrees').
            Used for informational purposes only.
        """
        self.min_length = min_length
        self.crs_units = crs_units
        self.stats = {
            "original_count": 0,
            "filtered_count": 0,
            "removed_count": 0,
            "total_original_length": 0,
            "total_filtered_length": 0,
        }

    def filter_geodataframe(
        self,
        gdf: gpd.GeoDataFrame,
        length_column: Optional[str] = None,
        inplace: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Filter a GeoDataFrame to remove polylines shorter than min_length.

        Parameters:
        -----------
        gdf : gpd.GeoDataFrame
            Input GeoDataFrame containing polyline geometries
        length_column : str, optional
            Name of column to store calculated lengths. If None, lengths are not stored.
        inplace : bool, default False
            If True, modify the original GeoDataFrame. If False, return a copy.

        Returns:
        --------
        gpd.GeoDataFrame
            Filtered GeoDataFrame with polylines >= min_length
        """
        if gdf.empty:
            warnings.warn("Input GeoDataFrame is empty")
            return gdf.copy() if not inplace else gdf

        # Work on copy unless inplace=True
        result_gdf = gdf if inplace else gdf.copy()

        # Calculate lengths
        lengths = result_gdf.geometry.length

        # Store lengths in column if requested
        if length_column:
            result_gdf[length_column] = lengths

        # Update statistics
        self.stats["original_count"] = len(result_gdf)
        self.stats["total_original_length"] = lengths.sum()

        # Create filter mask
        mask = lengths >= self.min_length

        # Apply filter
        if inplace:
            result_gdf.drop(result_gdf[~mask].index, inplace=True)
            result_gdf.reset_index(drop=True, inplace=True)
        else:
            result_gdf = result_gdf[mask].reset_index(drop=True)

        # Update statistics
        self.stats["filtered_count"] = len(result_gdf)
        self.stats["removed_count"] = (
            self.stats["original_count"] - self.stats["filtered_count"]
        )
        self.stats["total_filtered_length"] = (
            result_gdf.geometry.length.sum() if not result_gdf.empty else 0
        )

        return result_gdf

    def filter_multilinestring(
        self, multiline: MultiLineString
    ) -> Optional[Union[LineString, MultiLineString]]:
        """
        Filter individual LineStrings within a MultiLineString geometry.

        Parameters:
        -----------
        multiline : MultiLineString
            Input MultiLineString geometry

        Returns:
        --------
        LineString, MultiLineString, or None
            Filtered geometry with only LineStrings >= min_length
        """
        if not isinstance(multiline, MultiLineString):
            raise TypeError("Input must be a MultiLineString")

        # Filter individual LineStrings
        filtered_lines = [
            line for line in multiline.geoms if line.length >= self.min_length
        ]

        if not filtered_lines:
            return None
        elif len(filtered_lines) == 1:
            return filtered_lines[0]
        else:
            return MultiLineString(filtered_lines)

    def batch_filter(
        self,
        gdf: gpd.GeoDataFrame,
        handle_multilinestring: bool = True,
        length_column: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Advanced filtering that handles MultiLineString geometries by filtering
        their constituent LineStrings.

        Parameters:
        -----------
        gdf : gpd.GeoDataFrame
            Input GeoDataFrame
        handle_multilinestring : bool, default True
            If True, filter individual LineStrings within MultiLineString geometries
        length_column : str, optional
            Name of column to store calculated lengths

        Returns:
        --------
        gpd.GeoDataFrame
            Filtered GeoDataFrame
        """
        if gdf.empty:
            return gdf.copy()

        result_gdf = gdf.copy()

        if handle_multilinestring:
            # Process MultiLineString geometries
            multiline_mask = result_gdf.geometry.type == "MultiLineString"

            if multiline_mask.any():
                filtered_multilines = result_gdf[multiline_mask].geometry.apply(
                    self.filter_multilinestring
                )

                # Remove None geometries (completely filtered out)
                valid_mask = filtered_multilines.notna()

                # Update geometries
                result_gdf.loc[multiline_mask, "geometry"] = filtered_multilines

                # Remove rows with None geometries
                result_gdf = result_gdf[~(multiline_mask & ~valid_mask)]

        # Apply standard length filter to all remaining geometries
        return self.filter_geodataframe(result_gdf, length_column=length_column)

    def get_statistics(self) -> dict:
        """
        Get filtering statistics from the last operation.

        Returns:
        --------
        dict
            Dictionary containing filtering statistics
        """
        stats_copy = self.stats.copy()
        stats_copy["removal_percentage"] = (
            (self.stats["removed_count"] / self.stats["original_count"] * 100)
            if self.stats["original_count"] > 0
            else 0
        )
        stats_copy["crs_units"] = self.crs_units
        stats_copy["min_length_threshold"] = self.min_length
        return stats_copy

    def print_statistics(self):
        """Print filtering statistics in a readable format."""
        stats = self.get_statistics()

        print(f"Polyline Filtering Statistics:")
        print(f"  Minimum length threshold: {self.min_length} {self.crs_units}")
        print(f"  Original polylines: {stats['original_count']}")
        print(f"  Filtered polylines: {stats['filtered_count']}")
        print(f"  Removed polylines: {stats['removed_count']}")
        print(f"  Removal percentage: {stats['removal_percentage']:.2f}%")
        print(
            f"  Original total length: {stats['total_original_length']:.2f} {self.crs_units}"
        )
        print(
            f"  Filtered total length: {stats['total_filtered_length']:.2f} {self.crs_units}"
        )


# Example usage
if __name__ == "__main__":
    # Create sample data
    from shapely.geometry import Point

    # Sample polylines of different lengths
    lines = [
        LineString([(0, 0), (1, 0)]),  # length = 1
        LineString([(0, 0), (5, 0)]),  # length = 5
        LineString([(0, 0), (0.5, 0)]),  # length = 0.5
        LineString([(0, 0), (10, 0)]),  # length = 10
    ]

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {"id": range(len(lines)), "name": [f"line_{i}" for i in range(len(lines))]},
        geometry=lines,
    )

    print("Original GeoDataFrame:")
    print(f"Number of polylines: {len(gdf)}")
    print(f"Lengths: {gdf.geometry.length.tolist()}")

    # Filter polylines
    filter_obj = FaultLengths(min_length=2.0, crs_units="units")
    filtered_gdf = filter_obj.filter_geodataframe(gdf, length_column="length")

    print(f"\nFiltered GeoDataFrame (min_length >= 2.0):")
    print(f"Number of polylines: {len(filtered_gdf)}")
    if not filtered_gdf.empty:
        print(f"Lengths: {filtered_gdf['length'].tolist()}")

    # Print statistics
    print()
    filter_obj.print_statistics()
