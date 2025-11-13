import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from typing import Union, Optional
import warnings
import numpy as np

class FaultLengths:
    """
    A class to systematically remove polylines shorter than a defined length.

    Supports filtering LineString and MultiLineString geometries based on their
    length in the coordinate reference system units.
    """

    def __init__(self):
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
        pass
            
    def add_polyline_length(self,shapefile_path, output_path=None, length_field='line_length', 
                        unit='meters', force_projection=None):
        """
        Load a polyline shapefile and add a field with the true length of each polyline
        (cumulative length of all segments).
        
        Parameters:
        -----------
        shapefile_path : str
            Path to the input polyline shapefile
        output_path : str, optional
            Path to save the modified shapefile. If None, returns GeoDataFrame only
        length_field : str
            Name of the new field to store length values (default: 'line_length')
        unit : str
            Desired unit for length ('meters', 'kilometers', 'feet', 'miles', 'degrees')
            Note: 'degrees' unit uses centroid latitude for approximate conversion to meters
        force_projection : str or int, optional
            EPSG code to reproject to before calculating length (e.g., 'EPSG:3857')
            Useful for geographic (lat/lon) data to get accurate lengths
        
        Returns:
        --------
        geopandas.GeoDataFrame
            GeoDataFrame with the new length field added
        """
        
        # Load the shapefile
        gdf = gpd.read_file(shapefile_path)
        original_crs = gdf.crs
        
        # Store original index to maintain order
        gdf['_original_index'] = gdf.index
        
        # Check if we're using degrees unit with geographic data
        if unit.lower() == 'degrees':
            if not (gdf.crs and gdf.crs.is_geographic):
                warnings.warn(
                    "Using 'degrees' unit but CRS doesn't appear to be geographic. "
                    "The latitude-based conversion may not be appropriate."
                )
            # For degrees unit, we'll calculate in the original CRS
            gdf_calc = gdf.copy()
        else:
            # Check if CRS is geographic and warn if no projection specified
            if gdf.crs and gdf.crs.is_geographic and not force_projection:
                warnings.warn(
                    "The shapefile uses a geographic CRS (lat/lon). "
                    "Lengths will be in degrees unless you specify force_projection. "
                    "Consider using force_projection='EPSG:3857' or a local projected CRS "
                    "for accurate metric lengths, or use unit='degrees' for approximate conversion."
                )
            
            # Create a working copy for length calculation
            if force_projection:
                print(f"Reprojecting to {force_projection} for length calculation...")
                gdf_calc = gdf.to_crs(force_projection)
            else:
                gdf_calc = gdf.copy()
        
        def calculate_polyline_length_degrees(geometry):
            """
            Calculate length in degrees and convert to meters using centroid latitude.
            Returns (length_in_degrees, centroid_latitude)
            """
            if geometry is None or geometry.is_empty:
                return np.nan, np.nan
            
            try:
                # Get length in degrees
                if geometry.geom_type in ['LineString', 'MultiLineString']:
                    length_degrees = geometry.length
                    # Get centroid for latitude
                    centroid = geometry.centroid
                    latitude = centroid.y
                    return length_degrees, latitude
                else:
                    return np.nan, np.nan
                    
            except Exception as e:
                print(f"Error calculating length: {e}")
                return np.nan, np.nan
        
        def calculate_polyline_length(geometry):
            """
            Calculate the true length of a LineString or MultiLineString.
            """
            if geometry is None or geometry.is_empty:
                return np.nan
            
            try:
                if geometry.geom_type in ['LineString', 'MultiLineString']:
                    return geometry.length
                else:
                    return np.nan
                    
            except Exception as e:
                print(f"Error calculating length: {e}")
                return np.nan
        
        # Calculate length based on unit type
        if unit.lower() == 'degrees':
            # Calculate length in degrees and convert using latitude
            results = gdf_calc['geometry'].apply(calculate_polyline_length_degrees)
            lengths_degrees = [r[0] for r in results]
            latitudes = [r[1] for r in results]
            
            # Convert degrees to meters using the formula: meters = degrees * 111139 * cos(latitude)
            # 111,139 meters is approximately the length of one degree at the equator
            lengths_meters = []
            for length_deg, lat in zip(lengths_degrees, latitudes):
                if np.isnan(length_deg) or np.isnan(lat):
                    lengths_meters.append(np.nan)
                else:
                    # Convert latitude to radians for cos calculation
                    lat_rad = np.radians(lat)
                    # This formula is more accurate for E-W distances, but provides a reasonable approximation
                    meters = length_deg * 111139 * np.cos(lat_rad)
                    lengths_meters.append(meters)
            
            gdf[length_field] = lengths_meters
            gdf['centroid_lat'] = latitudes  # Add this for reference
            
            print(f"\nUsing degree-to-meter conversion with centroid latitudes")
            valid_lats = [l for l in latitudes if not np.isnan(l)]
            if valid_lats:
                print(f"  Latitude range: {min(valid_lats):.4f}° to {max(valid_lats):.4f}°")
                print(f"  Mean latitude: {np.mean(valid_lats):.4f}°")
                print(f"  Conversion factor at mean latitude: {111139 * np.cos(np.radians(np.mean(valid_lats))):.1f} m/degree")
            
        else:
            # Standard length calculation
            gdf_calc['_calculated_length'] = gdf_calc['geometry'].apply(calculate_polyline_length)
            
            # Convert units if needed
            conversion_factors = {
                'meters': 1.0,
                'kilometers': 0.001,
                'feet': 3.28084,
                'miles': 0.000621371
            }
            
            if unit.lower() in conversion_factors:
                factor = conversion_factors[unit.lower()]
                gdf[length_field] = gdf_calc['_calculated_length'] * factor
            else:
                gdf[length_field] = gdf_calc['_calculated_length']
                warnings.warn(f"Unknown unit '{unit}'. No conversion applied.")
        
        # Save if output path is provided
        if output_path:
            gdf.to_file(output_path)
            print(f"Shapefile with length field saved to: {output_path}")
        
        # Print summary statistics
        valid_lengths = gdf[length_field].dropna()
        unit_display = 'meters (from degrees)' if unit.lower() == 'degrees' else unit
        print(f"\nLength statistics ({unit_display}):")
        print(f"  Total features: {len(gdf)}")
        print(f"  Valid lengths: {len(valid_lengths)}")
        print(f"  Invalid/null: {len(gdf) - len(valid_lengths)}")
        if len(valid_lengths) > 0:
            print(f"  Min length: {valid_lengths.min():.3f}")
            print(f"  Max length: {valid_lengths.max():.3f}")
            print(f"  Mean length: {valid_lengths.mean():.3f}")
            print(f"  Total length: {valid_lengths.sum():.3f}")
        
        return gdf


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
