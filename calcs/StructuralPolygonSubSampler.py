import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

class StructuralPolygonSubSampler:
    """
    A comprehensive class for subsampling structural polygon data using
    various methods described in geological modeling literature.
    Currently implements a method to clean small polygons and holes.
    This method merges interrior polygons with their host, and boundary 
    polygons with their neighbors based on the largest shared boundary length.
    
    Should also implement a constraint to merge first with units of same 
    unit/formation/group/supergroup in that order.

    If no strat info available, should merge on same/similar rock type.

    If no strat or rock type info available, could try to split down the middle?
    
    """
    
    def __init__(self, gdf: gpd.GeoDataFrame):
        """
        Initialize the subsampler with a GeoDataFrame of polygons.
        
        Parameters:
        -----------
        gdf : gpd.GeoDataFrame
            Input GeoDataFrame containing structural orientation measurements.
            Must contain columns for strike/dip or dip direction/dip.
        """
        
        self.gdf = gdf.copy()
        
    def clean_small_polygons_and_holes_new(self, gdf, min_area_threshold, distance_threshold=1e-6, 
                                    lithoname=None, strat1=None, strat2=None, strat3=None, strat4=None):
        """
        Remove interior holes smaller than threshold and merge small polygons 
        with adjacent polygons using hierarchical preferences:
        1. Same lithology code (lithoname) + longest shared boundary
        2. Same strat1 code + longest shared boundary
        3. Same strat2 code + longest shared boundary
        4. Same strat3 code + longest shared boundary
        5. Same strat4 code + longest shared boundary
        6. Longest shared boundary (fallback)
        
        Also tracks polygon merging history via 'primaryid' and 'inherit' fields.
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Input GeoDataFrame containing polygon geometries
        min_area_threshold : float
            Minimum area threshold for both holes and polygons
        distance_threshold : float, optional
            Maximum distance for considering nodes as "shared" (default: 1e-6)
        lithoname : str, optional
            Column name containing lithology codes
        strat1 : str, optional
            Column name containing finest stratigraphy level
        strat2 : str, optional
            Column name containing second stratigraphy level
        strat3 : str, optional
            Column name containing third stratigraphy level
        strat4 : str, optional
            Column name containing coarsest stratigraphy level
        
        Returns:
        --------
        GeoDataFrame
            Cleaned GeoDataFrame with small holes removed and small polygons merged
        """
        
        def filter_large_polygons(gdf,  min_area_threshold=1000):
            """
            Filter polygons in a GeoDataFrame based on a minimum area.

            Parameters:
            -----------
            input_path : str
                Path to the input polygon/multipolygon GeoDataFrame (e.g., .shp, .gpkg).
            output_path : str, optional
                If provided, the filtered GeoDataFrame will be saved to this path.
            min_area : float
                Minimum area threshold. Polygons with area >= min_area will be kept.

            Returns:
            --------
            gpd.GeoDataFrame
                Filtered GeoDataFrame with only polygons larger than or equal to min_area.
            """

            # Ensure geometry is valid
            gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]

            # Compute area and filter
            gdf['area'] = gdf.geometry.area

            filtered_gdf = gdf[gdf['area'] >= min_area_threshold].copy()

            # Drop temporary area column if desired
            filtered_gdf.drop(columns='area', inplace=True)

            # Reset index for cleanliness
            filtered_gdf.reset_index(drop=True, inplace=True)

            return filtered_gdf
                
        def remove_small_holes(geometry, min_area):
            """Remove interior holes smaller than min_area from a geometry."""
            if geometry.geom_type == 'Polygon':
                # Keep only holes larger than threshold
                if geometry.interiors:
                    large_holes = [hole for hole in geometry.interiors 
                                if Polygon(hole).area >= min_area]
                    return Polygon(geometry.exterior.coords, holes=large_holes)
                return geometry
            elif geometry.geom_type == 'MultiPolygon':
                # Process each polygon in the MultiPolygon
                cleaned_polys = []
                for poly in geometry.geoms:
                    if poly.interiors:
                        large_holes = [hole for hole in poly.interiors 
                                    if Polygon(hole).area >= min_area]
                        cleaned_polys.append(Polygon(poly.exterior.coords, holes=large_holes))
                    else:
                        cleaned_polys.append(poly)
                return MultiPolygon(cleaned_polys)
            else:
                return geometry
        
        def calculate_shared_boundary_length(geom1, geom2, distance_threshold):
            """Calculate the length of shared boundary between two geometries."""
            from shapely.geometry import LineString
            
            def get_boundary(geometry):
                """Get boundary from any geometry type."""
                if geometry.geom_type in ['Polygon', 'MultiPolygon']:
                    return geometry.boundary
                elif geometry.geom_type in ['LineString', 'MultiLineString']:
                    return geometry
                else:
                    return None
            
            boundary1 = get_boundary(geom1)
            boundary2 = get_boundary(geom2)
            
            if boundary1 is None or boundary2 is None:
                return 0
            
            try:
                # Direct intersection works well for most cases
                intersection = boundary1.intersection(boundary2)
                
                # Calculate total length
                if intersection.geom_type == 'LineString':
                    return intersection.length
                elif intersection.geom_type == 'MultiLineString':
                    return sum(line.length for line in intersection.geoms)
                elif intersection.geom_type == 'Point':
                    return 0  # Points don't contribute to length
                elif intersection.geom_type == 'MultiPoint':
                    return 0  # Points don't contribute to length
                elif intersection.geom_type == 'GeometryCollection':
                    total_length = 0
                    for geom in intersection.geoms:
                        if geom.geom_type == 'LineString':
                            total_length += geom.length
                        elif geom.geom_type == 'MultiLineString':
                            total_length += sum(line.length for line in geom.geoms)
                    return total_length
                else:
                    return 0
                    
            except Exception:
                # Fallback to buffered approach if direct intersection fails
                try:
                    buffer_dist = max(distance_threshold, 1e-10)
                    buffered = boundary1.buffer(buffer_dist)
                    intersection = buffered.intersection(boundary2)
                    
                    if intersection.geom_type == 'LineString':
                        return intersection.length
                    elif intersection.geom_type == 'MultiLineString':
                        return sum(line.length for line in intersection.geoms)
                    else:
                        return 0
                except Exception:
                    return 0
        
        def get_field_value(row, field_name):
            """Safely get field value, handling None/NaN values."""
            if field_name is None or field_name not in row.index:
                return None
            value = row[field_name]
            if pd.isna(value):
                return None
            return str(value)  # Convert to string for comparison
        
        def find_best_merge_candidate_with_preferences(small_poly_idx, gdf_temp, distance_threshold,
                                                    lithoname, strat1, strat2, strat3, strat4):
            """
            Find the polygon that shares the most boundary length with the small polygon,
            using hierarchical preferences for lithology and stratigraphy.
            """
            small_geom = gdf_temp.iloc[small_poly_idx].geometry
            small_row = gdf_temp.iloc[small_poly_idx]
            
            # Get attribute values for the small polygon
            small_litho = get_field_value(small_row, lithoname)
            small_strat1 = get_field_value(small_row, strat1)
            small_strat2 = get_field_value(small_row, strat2)
            small_strat3 = get_field_value(small_row, strat3)
            small_strat4 = get_field_value(small_row, strat4)
            
            # Get spatial index for efficient neighbor finding
            sindex = gdf_temp.sindex
            
            # Find potential neighbors using bounding box intersection
            possible_matches_index = list(sindex.intersection(small_geom.bounds))
            
            # Remove self from possible matches
            possible_matches_index = [idx for idx in possible_matches_index if idx != small_poly_idx]
            
            # Collect candidates with their attributes and shared boundary lengths
            candidates = []
            
            for idx in possible_matches_index:
                candidate_geom = gdf_temp.iloc[idx].geometry
                candidate_row = gdf_temp.iloc[idx]
                
                # Quick check: do bounding boxes actually intersect?
                def bounds_intersect(geom1, geom2):
                    """Check if two geometries' bounding boxes intersect."""
                    bounds1 = geom1.bounds  # (minx, miny, maxx, maxy)
                    bounds2 = geom2.bounds
                    return not (bounds1[2] < bounds2[0] or bounds2[2] < bounds1[0] or 
                            bounds1[3] < bounds2[1] or bounds2[3] < bounds1[1])
                
                if not bounds_intersect(small_geom, candidate_geom):
                    continue
                    
                # More precise check: do geometries actually touch or intersect?
                if not (small_geom.touches(candidate_geom) or small_geom.intersects(candidate_geom)):
                    continue
                
                shared_length = calculate_shared_boundary_length(small_geom, candidate_geom, distance_threshold)
                
                if shared_length > 0:  # Only consider candidates with shared boundary
                    # Get candidate attributes
                    cand_litho = get_field_value(candidate_row, lithoname)
                    cand_strat1 = get_field_value(candidate_row, strat1)
                    cand_strat2 = get_field_value(candidate_row, strat2)
                    cand_strat3 = get_field_value(candidate_row, strat3)
                    cand_strat4 = get_field_value(candidate_row, strat4)
                    
                    candidates.append({
                        'idx': idx,
                        'shared_length': shared_length,
                        'litho': cand_litho,
                        'strat1': cand_strat1,
                        'strat2': cand_strat2,
                        'strat3': cand_strat3,
                        'strat4': cand_strat4
                    })
            
            if not candidates:
                return None, 0, "no_neighbors"
            
            # Apply hierarchical preferences
            best_candidate = None
            merge_reason = "fallback_boundary_length"
            
        
            # Priority 1: Same strat1 + longest boundary
            if  small_strat1 is not None:
                strat1_matches = [c for c in candidates if c['strat1'] == small_strat1]
                if strat1_matches:
                    best_candidate = max(strat1_matches, key=lambda x: x['shared_length'])
                    merge_reason = f"same_strat1_{small_strat1}"
            
            # Priority 2: Same strat2 + longest boundary
            if best_candidate is None and small_strat2 is not None:
                strat2_matches = [c for c in candidates if c['strat2'] == small_strat2]
                if strat2_matches:
                    best_candidate = max(strat2_matches, key=lambda x: x['shared_length'])
                    merge_reason = f"same_strat2_{small_strat2}"
            
            # Priority 3: Same strat3 + longest boundary
            if best_candidate is None and small_strat3 is not None:
                strat3_matches = [c for c in candidates if c['strat3'] == small_strat3]
                if strat3_matches:
                    best_candidate = max(strat3_matches, key=lambda x: x['shared_length'])
                    merge_reason = f"same_strat3_{small_strat3}"
            
            # Priority 4: Same strat4 + longest boundary
            if best_candidate is None and small_strat4 is not None:
                strat4_matches = [c for c in candidates if c['strat4'] == small_strat4]
                if strat4_matches:
                    best_candidate = max(strat4_matches, key=lambda x: x['shared_length'])
                    merge_reason = f"same_strat4_{small_strat4}"

            # Priority 5: Same lithology + longest boundary
            if best_candidate is None and small_litho is not None:
                litho_matches = [c for c in candidates if c['litho'] == small_litho]
                if litho_matches:
                    best_candidate = max(litho_matches, key=lambda x: x['shared_length'])
                    merge_reason = f"same_lithology_{small_litho}"
            
            # Fallback: Longest boundary regardless of attributes
            if best_candidate is None:
                best_candidate = max(candidates, key=lambda x: x['shared_length'])
                merge_reason = "fallback_boundary_length"
            
            return best_candidate['idx'], best_candidate['shared_length'], merge_reason
        
        # Create a copy of the GeoDataFrame
        gdf_cleaned = gdf.copy()
        
        # Step 0: Initialize primaryid and inherit fields
        print("Initializing tracking fields...")
        if 'primaryid' not in gdf_cleaned.columns:
            print("  Creating 'primaryid' field...")
            gdf_cleaned['primaryid'] = range(len(gdf_cleaned))
        else:
            print("  'primaryid' field already exists.")
        
        if 'inherit' not in gdf_cleaned.columns:
            print("  Creating 'inherit' field...")
            gdf_cleaned['inherit'] = gdf_cleaned['primaryid'].astype(str)
        else:
            print("  'inherit' field already exists.")
        
        # Validate that specified columns exist
        available_columns = set(gdf_cleaned.columns)
        for col_name, col_var in [('lithoname', lithoname), ('strat1', strat1), ('strat2', strat2), 
                                ('strat3', strat3), ('strat4', strat4)]:
            if col_var is not None and col_var not in available_columns:
                print(f"Warning: Column '{col_var}' specified for {col_name} not found in GeoDataFrame. "
                    f"Available columns: {list(available_columns)}")
        
        # Step 1: Remove small interior holes
        print("Removing small interior holes...")
        gdf_cleaned['geometry'] = gdf_cleaned['geometry'].apply(
            lambda geom: remove_small_holes(geom, min_area_threshold)
        )
        
        # Step 2: Merge small polygons with neighbors using hierarchical preferences
        print("Merging small polygons with neighbors using lithology/stratigraphy preferences...")
        
        # Reset index to ensure proper indexing
        gdf_cleaned = gdf_cleaned.reset_index(drop=True)
        
        # Keep track of merge statistics
        merge_stats = {
            'same_lithology': 0,
            'same_strat1': 0,
            'same_strat2': 0,
            'same_strat3': 0,
            'same_strat4': 0,
            'fallback_boundary_length': 0,
            'no_neighbors': 0
        }
        
        # Keep iterating until no more merges are possible
        iteration = 0
        total_merged = 0
        
        while True:
            iteration += 1
            print(f"Merge iteration {iteration}...")
            
            # Rebuild spatial index for current iteration (since geometries may have changed)
            if hasattr(gdf_cleaned, 'sindex'):
                # Force rebuild of spatial index
                gdf_cleaned._sindex = None
            
            # Track which polygons have been merged in this iteration
            merged_indices = set()
            merge_targets = {}  # Maps small polygon index to (target polygon index, merge reason)
            
            # Find all small polygons and their best merge candidates
            small_polygon_count = 0
            for idx in gdf_cleaned.index:
                if idx in merged_indices:
                    continue
                    
                current_geom = gdf_cleaned.iloc[idx].geometry
                
                # Check if polygon is smaller than threshold
                if current_geom.area < min_area_threshold:
                    small_polygon_count += 1
                    # Find the best merge candidate using hierarchical preferences
                    best_candidate_idx, shared_length, merge_reason = find_best_merge_candidate_with_preferences(
                        idx, gdf_cleaned, distance_threshold, lithoname, strat1, strat2, strat3, strat4)
                    
                    if best_candidate_idx is not None and shared_length > 0:
                        merge_targets[idx] = (best_candidate_idx, merge_reason)
            
            print(f"Found {small_polygon_count} small polygons, {len(merge_targets)} with merge candidates")
            
            # If no merges found, break
            if not merge_targets:
                break
            
            # Group merges by target polygon to handle multiple small polygons merging to same target
            target_groups = {}
            for small_idx, (target_idx, merge_reason) in merge_targets.items():
                if target_idx not in target_groups:
                    target_groups[target_idx] = []
                target_groups[target_idx].append((small_idx, merge_reason))
            
            # Perform merges
            for target_idx, small_data in target_groups.items():
                if target_idx in merged_indices:
                    continue
                    
                # Collect all geometries to merge
                geometries_to_merge = [gdf_cleaned.iloc[target_idx].geometry]
                merge_reasons = []
                
                for small_idx, merge_reason in small_data:
                    if small_idx not in merged_indices:
                        geometries_to_merge.append(gdf_cleaned.iloc[small_idx].geometry)
                        
                        # Concatenate inherit field
                        small_inherit = str(gdf_cleaned.iloc[small_idx]['inherit'])
                        target_inherit = str(gdf_cleaned.iloc[target_idx]['inherit'])
                        new_inherit = target_inherit + ' ' + small_inherit
                        gdf_cleaned.iloc[target_idx, gdf_cleaned.columns.get_loc('inherit')] = new_inherit
                        
                        merged_indices.add(small_idx)
                        merge_reasons.append(merge_reason)
                        
                        # Update merge statistics
                        if merge_reason.startswith('same_lithology'):
                            merge_stats['same_lithology'] += 1
                        elif merge_reason.startswith('same_strat1'):
                            merge_stats['same_strat1'] += 1
                        elif merge_reason.startswith('same_strat2'):
                            merge_stats['same_strat2'] += 1
                        elif merge_reason.startswith('same_strat3'):
                            merge_stats['same_strat3'] += 1
                        elif merge_reason.startswith('same_strat4'):
                            merge_stats['same_strat4'] += 1
                        elif merge_reason == 'fallback_boundary_length':
                            merge_stats['fallback_boundary_length'] += 1
                        elif merge_reason == 'no_neighbors':
                            merge_stats['no_neighbors'] += 1
                
                # Merge the geometries
                if len(geometries_to_merge) > 1:
                    merged_geom = unary_union(geometries_to_merge)
                    
                    # Update the target polygon with merged geometry
                    gdf_cleaned.iloc[target_idx, gdf_cleaned.columns.get_loc('geometry')] = merged_geom
            
            # Remove merged polygons
            if merged_indices:
                gdf_cleaned = gdf_cleaned.drop(index=list(merged_indices))
                gdf_cleaned = gdf_cleaned.reset_index(drop=True)
                total_merged += len(merged_indices)
            else:
                break
        
        gdf_cleaned = filter_large_polygons(gdf_cleaned, min_area_threshold=min_area_threshold)

        # Step 3: Clean up any invalid geometries
        print("Cleaning up geometries...")
        gdf_cleaned['geometry'] = gdf_cleaned['geometry'].apply(
            lambda geom: geom.buffer(0) if not geom.is_valid else geom
        )
        
        # Print summary statistics
        print(f"\nMerge Summary:")
        print(f"Original polygons: {len(gdf)}")
        print(f"Final polygons: {len(gdf_cleaned)}")
        print(f"Total polygons merged: {total_merged}")
        print(f"Merge iterations: {iteration}")
        print(f"\nMerge reasons:")
        for reason, count in merge_stats.items():
            if count > 0:
                print(f"  {reason}: {count} polygons")
        
        return gdf_cleaned

    def clean_small_polygons_and_holes_new_x(self, gdf, min_area_threshold, distance_threshold=1e-6, 
                                    lithoname=None, strat1=None, strat2=None, strat3=None, strat4=None):
        """
        Remove interior holes smaller than threshold and merge small polygons 
        with adjacent polygons using hierarchical preferences:
        1. Same lithology code (lithoname) + longest shared boundary
        2. Same strat1 code + longest shared boundary
        3. Same strat2 code + longest shared boundary
        4. Same strat3 code + longest shared boundary
        5. Same strat4 code + longest shared boundary
        6. Longest shared boundary (fallback)
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Input GeoDataFrame containing polygon geometries
        min_area_threshold : float
            Minimum area threshold for both holes and polygons
        distance_threshold : float, optional
            Maximum distance for considering nodes as "shared" (default: 1e-6)
        lithoname : str, optional
            Column name containing lithology codes
        strat1 : str, optional
            Column name containing finest stratigraphy level
        strat2 : str, optional
            Column name containing second stratigraphy level
        strat3 : str, optional
            Column name containing third stratigraphy level
        strat4 : str, optional
            Column name containing coarsest stratigraphy level
        
        Returns:
        --------
        GeoDataFrame
            Cleaned GeoDataFrame with small holes removed and small polygons merged
        """
        
        def filter_large_polygons(gdf,  min_area_threshold=1000):
            """
            Filter polygons in a GeoDataFrame based on a minimum area.

            Parameters:
            -----------
            input_path : str
                Path to the input polygon/multipolygon GeoDataFrame (e.g., .shp, .gpkg).
            output_path : str, optional
                If provided, the filtered GeoDataFrame will be saved to this path.
            min_area : float
                Minimum area threshold. Polygons with area >= min_area will be kept.

            Returns:
            --------
            gpd.GeoDataFrame
                Filtered GeoDataFrame with only polygons larger than or equal to min_area.
            """

            # Ensure geometry is valid
            gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]

            # Compute area and filter
            gdf['area'] = gdf.geometry.area

            filtered_gdf = gdf[gdf['area'] >= min_area_threshold].copy()

            # Drop temporary area column if desired
            filtered_gdf.drop(columns='area', inplace=True)

            # Reset index for cleanliness
            filtered_gdf.reset_index(drop=True, inplace=True)

            return filtered_gdf
                
        def remove_small_holes(geometry, min_area):
            """Remove interior holes smaller than min_area from a geometry."""
            if geometry.geom_type == 'Polygon':
                # Keep only holes larger than threshold
                if geometry.interiors:
                    large_holes = [hole for hole in geometry.interiors 
                                if Polygon(hole).area >= min_area]
                    return Polygon(geometry.exterior.coords, holes=large_holes)
                return geometry
            elif geometry.geom_type == 'MultiPolygon':
                # Process each polygon in the MultiPolygon
                cleaned_polys = []
                for poly in geometry.geoms:
                    if poly.interiors:
                        large_holes = [hole for hole in poly.interiors 
                                    if Polygon(hole).area >= min_area]
                        cleaned_polys.append(Polygon(poly.exterior.coords, holes=large_holes))
                    else:
                        cleaned_polys.append(poly)
                return MultiPolygon(cleaned_polys)
            else:
                return geometry
        
        def calculate_shared_boundary_length(geom1, geom2, distance_threshold):
            """Calculate the length of shared boundary between two geometries."""
            from shapely.geometry import LineString
            
            def get_boundary(geometry):
                """Get boundary from any geometry type."""
                if geometry.geom_type in ['Polygon', 'MultiPolygon']:
                    return geometry.boundary
                elif geometry.geom_type in ['LineString', 'MultiLineString']:
                    return geometry
                else:
                    return None
            
            boundary1 = get_boundary(geom1)
            boundary2 = get_boundary(geom2)
            
            if boundary1 is None or boundary2 is None:
                return 0
            
            try:
                # Direct intersection works well for most cases
                intersection = boundary1.intersection(boundary2)
                
                # Calculate total length
                if intersection.geom_type == 'LineString':
                    return intersection.length
                elif intersection.geom_type == 'MultiLineString':
                    return sum(line.length for line in intersection.geoms)
                elif intersection.geom_type == 'Point':
                    return 0  # Points don't contribute to length
                elif intersection.geom_type == 'MultiPoint':
                    return 0  # Points don't contribute to length
                elif intersection.geom_type == 'GeometryCollection':
                    total_length = 0
                    for geom in intersection.geoms:
                        if geom.geom_type == 'LineString':
                            total_length += geom.length
                        elif geom.geom_type == 'MultiLineString':
                            total_length += sum(line.length for line in geom.geoms)
                    return total_length
                else:
                    return 0
                    
            except Exception:
                # Fallback to buffered approach if direct intersection fails
                try:
                    buffer_dist = max(distance_threshold, 1e-10)
                    buffered = boundary1.buffer(buffer_dist)
                    intersection = buffered.intersection(boundary2)
                    
                    if intersection.geom_type == 'LineString':
                        return intersection.length
                    elif intersection.geom_type == 'MultiLineString':
                        return sum(line.length for line in intersection.geoms)
                    else:
                        return 0
                except Exception:
                    return 0
        
        def get_field_value(row, field_name):
            """Safely get field value, handling None/NaN values."""
            if field_name is None or field_name not in row.index:
                return None
            value = row[field_name]
            if pd.isna(value):
                return None
            return str(value)  # Convert to string for comparison
        
        def find_best_merge_candidate_with_preferences(small_poly_idx, gdf_temp, distance_threshold,
                                                    lithoname, strat1, strat2, strat3, strat4):
            """
            Find the polygon that shares the most boundary length with the small polygon,
            using hierarchical preferences for lithology and stratigraphy.
            """
            small_geom = gdf_temp.iloc[small_poly_idx].geometry
            small_row = gdf_temp.iloc[small_poly_idx]
            
            # Get attribute values for the small polygon
            small_litho = get_field_value(small_row, lithoname)
            small_strat1 = get_field_value(small_row, strat1)
            small_strat2 = get_field_value(small_row, strat2)
            small_strat3 = get_field_value(small_row, strat3)
            small_strat4 = get_field_value(small_row, strat4)
            
            # Get spatial index for efficient neighbor finding
            sindex = gdf_temp.sindex
            
            # Find potential neighbors using bounding box intersection
            possible_matches_index = list(sindex.intersection(small_geom.bounds))
            
            # Remove self from possible matches
            possible_matches_index = [idx for idx in possible_matches_index if idx != small_poly_idx]
            
            # Collect candidates with their attributes and shared boundary lengths
            candidates = []
            
            for idx in possible_matches_index:
                candidate_geom = gdf_temp.iloc[idx].geometry
                candidate_row = gdf_temp.iloc[idx]
                
                # Quick check: do bounding boxes actually intersect?
                def bounds_intersect(geom1, geom2):
                    """Check if two geometries' bounding boxes intersect."""
                    bounds1 = geom1.bounds  # (minx, miny, maxx, maxy)
                    bounds2 = geom2.bounds
                    return not (bounds1[2] < bounds2[0] or bounds2[2] < bounds1[0] or 
                            bounds1[3] < bounds2[1] or bounds2[3] < bounds1[1])
                
                if not bounds_intersect(small_geom, candidate_geom):
                    continue
                    
                # More precise check: do geometries actually touch or intersect?
                if not (small_geom.touches(candidate_geom) or small_geom.intersects(candidate_geom)):
                    continue
                
                shared_length = calculate_shared_boundary_length(small_geom, candidate_geom, distance_threshold)
                
                if shared_length > 0:  # Only consider candidates with shared boundary
                    # Get candidate attributes
                    cand_litho = get_field_value(candidate_row, lithoname)
                    cand_strat1 = get_field_value(candidate_row, strat1)
                    cand_strat2 = get_field_value(candidate_row, strat2)
                    cand_strat3 = get_field_value(candidate_row, strat3)
                    cand_strat4 = get_field_value(candidate_row, strat4)
                    
                    candidates.append({
                        'idx': idx,
                        'shared_length': shared_length,
                        'litho': cand_litho,
                        'strat1': cand_strat1,
                        'strat2': cand_strat2,
                        'strat3': cand_strat3,
                        'strat4': cand_strat4
                    })
            
            if not candidates:
                return None, 0, "no_neighbors"
            
            # Apply hierarchical preferences
            best_candidate = None
            merge_reason = "fallback_boundary_length"
            
           
            # Priority 1: Same strat1 + longest boundary
            if  small_strat1 is not None:
                strat1_matches = [c for c in candidates if c['strat1'] == small_strat1]
                if strat1_matches:
                    best_candidate = max(strat1_matches, key=lambda x: x['shared_length'])
                    merge_reason = f"same_strat1_{small_strat1}"
            
            # Priority 2: Same strat2 + longest boundary
            if best_candidate is None and small_strat2 is not None:
                strat2_matches = [c for c in candidates if c['strat2'] == small_strat2]
                if strat2_matches:
                    best_candidate = max(strat2_matches, key=lambda x: x['shared_length'])
                    merge_reason = f"same_strat2_{small_strat2}"
            
            # Priority 3: Same strat3 + longest boundary
            if best_candidate is None and small_strat3 is not None:
                strat3_matches = [c for c in candidates if c['strat3'] == small_strat3]
                if strat3_matches:
                    best_candidate = max(strat3_matches, key=lambda x: x['shared_length'])
                    merge_reason = f"same_strat3_{small_strat3}"
            
            # Priority 4: Same strat4 + longest boundary
            if best_candidate is None and small_strat4 is not None:
                strat4_matches = [c for c in candidates if c['strat4'] == small_strat4]
                if strat4_matches:
                    best_candidate = max(strat4_matches, key=lambda x: x['shared_length'])
                    merge_reason = f"same_strat4_{small_strat4}"

            # Priority 5: Same lithology + longest boundary
            if best_candidate is None and small_litho is not None:
                litho_matches = [c for c in candidates if c['litho'] == small_litho]
                if litho_matches:
                    best_candidate = max(litho_matches, key=lambda x: x['shared_length'])
                    merge_reason = f"same_lithology_{small_litho}"
             
            # Fallback: Longest boundary regardless of attributes
            if best_candidate is None:
                best_candidate = max(candidates, key=lambda x: x['shared_length'])
                merge_reason = "fallback_boundary_length"
            
            return best_candidate['idx'], best_candidate['shared_length'], merge_reason
        
        # Create a copy of the GeoDataFrame
        gdf_cleaned = gdf.copy()
        
        # Validate that specified columns exist
        available_columns = set(gdf_cleaned.columns)
        for col_name, col_var in [('lithoname', lithoname), ('strat1', strat1), ('strat2', strat2), 
                                ('strat3', strat3), ('strat4', strat4)]:
            if col_var is not None and col_var not in available_columns:
                print(f"Warning: Column '{col_var}' specified for {col_name} not found in GeoDataFrame. "
                    f"Available columns: {list(available_columns)}")
        
        # Step 1: Remove small interior holes
        print("Removing small interior holes...")
        gdf_cleaned['geometry'] = gdf_cleaned['geometry'].apply(
            lambda geom: remove_small_holes(geom, min_area_threshold)
        )
        
        # Step 2: Merge small polygons with neighbors using hierarchical preferences
        print("Merging small polygons with neighbors using lithology/stratigraphy preferences...")
        
        # Reset index to ensure proper indexing
        gdf_cleaned = gdf_cleaned.reset_index(drop=True)
        
        # Keep track of merge statistics
        merge_stats = {
            'same_lithology': 0,
            'same_strat1': 0,
            'same_strat2': 0,
            'same_strat3': 0,
            'same_strat4': 0,
            'fallback_boundary_length': 0,
            'no_neighbors': 0
        }
        
        # Keep iterating until no more merges are possible
        iteration = 0
        total_merged = 0
        
        while True:
            iteration += 1
            print(f"Merge iteration {iteration}...")
            
            # Rebuild spatial index for current iteration (since geometries may have changed)
            if hasattr(gdf_cleaned, 'sindex'):
                # Force rebuild of spatial index
                gdf_cleaned._sindex = None
            
            # Track which polygons have been merged in this iteration
            merged_indices = set()
            merge_targets = {}  # Maps small polygon index to (target polygon index, merge reason)
            
            # Find all small polygons and their best merge candidates
            small_polygon_count = 0
            for idx in gdf_cleaned.index:
                if idx in merged_indices:
                    continue
                    
                current_geom = gdf_cleaned.iloc[idx].geometry
                
                # Check if polygon is smaller than threshold
                if current_geom.area < min_area_threshold:
                    small_polygon_count += 1
                    # Find the best merge candidate using hierarchical preferences
                    best_candidate_idx, shared_length, merge_reason = find_best_merge_candidate_with_preferences(
                        idx, gdf_cleaned, distance_threshold, lithoname, strat1, strat2, strat3, strat4)
                    
                    if best_candidate_idx is not None and shared_length > 0:
                        merge_targets[idx] = (best_candidate_idx, merge_reason)
            
            print(f"Found {small_polygon_count} small polygons, {len(merge_targets)} with merge candidates")
            
            # If no merges found, break
            if not merge_targets:
                break
            
            # Group merges by target polygon to handle multiple small polygons merging to same target
            target_groups = {}
            for small_idx, (target_idx, merge_reason) in merge_targets.items():
                if target_idx not in target_groups:
                    target_groups[target_idx] = []
                target_groups[target_idx].append((small_idx, merge_reason))
            
            # Perform merges
            for target_idx, small_data in target_groups.items():
                if target_idx in merged_indices:
                    continue
                    
                # Collect all geometries to merge
                geometries_to_merge = [gdf_cleaned.iloc[target_idx].geometry]
                merge_reasons = []
                
                for small_idx, merge_reason in small_data:
                    if small_idx not in merged_indices:
                        geometries_to_merge.append(gdf_cleaned.iloc[small_idx].geometry)
                        merged_indices.add(small_idx)
                        merge_reasons.append(merge_reason)
                        
                        # Update merge statistics
                        if merge_reason.startswith('same_lithology'):
                            merge_stats['same_lithology'] += 1
                        elif merge_reason.startswith('same_strat1'):
                            merge_stats['same_strat1'] += 1
                        elif merge_reason.startswith('same_strat2'):
                            merge_stats['same_strat2'] += 1
                        elif merge_reason.startswith('same_strat3'):
                            merge_stats['same_strat3'] += 1
                        elif merge_reason.startswith('same_strat4'):
                            merge_stats['same_strat4'] += 1
                        elif merge_reason == 'fallback_boundary_length':
                            merge_stats['fallback_boundary_length'] += 1
                        elif merge_reason == 'no_neighbors':
                            merge_stats['no_neighbors'] += 1
                
                # Merge the geometries
                if len(geometries_to_merge) > 1:
                    merged_geom = unary_union(geometries_to_merge)
                    
                    # Update the target polygon with merged geometry
                    gdf_cleaned.iloc[target_idx, gdf_cleaned.columns.get_loc('geometry')] = merged_geom
                    
                    small_indices = [data[0] for data in small_data if data[0] not in merged_indices]
                    """print(f"Merged {len(small_indices)} small polygon(s) {small_indices} "
                        f"with target polygon {target_idx} (reasons: {merge_reasons})")"""
            
            # Remove merged polygons
            if merged_indices:
                gdf_cleaned = gdf_cleaned.drop(index=list(merged_indices))
                gdf_cleaned = gdf_cleaned.reset_index(drop=True)
                total_merged += len(merged_indices)
            else:
                break
        gdf_cleaned=filter_large_polygons(gdf_cleaned,  min_area_threshold=min_area_threshold)

        # Step 3: Clean up any invalid geometries
        print("Cleaning up geometries...")
        gdf_cleaned['geometry'] = gdf_cleaned['geometry'].apply(
            lambda geom: geom.buffer(0) if not geom.is_valid else geom
        )
        
        # Print summary statistics
        print(f"\nMerge Summary:")
        print(f"Original polygons: {len(gdf)}")
        print(f"Final polygons: {len(gdf_cleaned)}")
        print(f"Total polygons merged: {total_merged}")
        print(f"Merge iterations: {iteration}")
        print(f"\nMerge reasons:")
        for reason, count in merge_stats.items():
            if count > 0:
                print(f"  {reason}: {count} polygons")
        
        return gdf_cleaned

