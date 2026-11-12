import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
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
            A small polygon that touches no other polygon may merge into one within this
            distance (layer units); the gap between them is filled (default: 1e-6)
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
                
                # Calculate total length.
                # LinearRing arises when a polygon sits exactly inside a hole of
                # another polygon — the shared boundary is the closed ring.
                def _ring_or_line_length(geom):
                    if geom.geom_type in ('LineString', 'LinearRing'):
                        return geom.length
                    elif geom.geom_type == 'MultiLineString':
                        return sum(g.length for g in geom.geoms)
                    return 0

                if intersection.geom_type in ('LineString', 'LinearRing'):
                    return intersection.length
                elif intersection.geom_type == 'MultiLineString':
                    return sum(line.length for line in intersection.geoms)
                elif intersection.geom_type in ('Point', 'MultiPoint'):
                    return 0
                elif intersection.geom_type == 'GeometryCollection':
                    return sum(_ring_or_line_length(g) for g in intersection.geoms)
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
        
        def append_to_inherit_fields(gdf_temp, target_idx, value_to_add, max_inherit_field):
            """
            Append value to inherit fields, creating new fields (inherit1, inherit2, etc.)
            if the current field would exceed 254 characters.
            
            Returns the updated max_inherit_field number.
            """
            MAX_FIELD_LENGTH = 254
            remaining_value = ' ' + value_to_add
            current_field_num = 0
            
            while remaining_value:
                # Determine field name
                field_name = 'inherit' if current_field_num == 0 else f'inherit{current_field_num}'
                
                # Create field if it doesn't exist
                if field_name not in gdf_temp.columns:
                    gdf_temp[field_name] = ''
                    max_inherit_field = max(max_inherit_field, current_field_num)
                    print(f"  Creating '{field_name}' field for overflow...")
                
                # Get current value in this field
                current_value = str(gdf_temp.at[target_idx, field_name]) if pd.notna(gdf_temp.at[target_idx, field_name]) else ''
                
                # Calculate available space
                available_space = MAX_FIELD_LENGTH - len(current_value)
                
                if available_space > 0:
                    # Add as much as we can to this field
                    to_add = remaining_value[:available_space]
                    gdf_temp.at[target_idx, field_name] = current_value + to_add
                    remaining_value = remaining_value[available_space:]
                else:
                    # This field is full, move to next field
                    current_field_num += 1
            
            return max_inherit_field
        
        def find_merge_targets(frame, small_idx, distance_threshold,
                               lithoname, strat1, strat2, strat3, strat4, near_tol=None):
            """
            For every small polygon, the polygon it should merge into, using hierarchical
            preferences for lithology and stratigraphy and the longest shared boundary.

            All small polygons are handled together: one bulk spatial query finds every touching
            pair, the shared boundary lengths of all pairs are computed in one go, and the
            attribute rules then run on plain lists (no pandas row access per candidate).

            With ``near_tol`` the candidates are the polygons within that distance (not only the
            touching ones), and the "shared boundary" is the length of the small polygon's
            boundary lying within ``near_tol`` of the candidate's boundary.

            Returns {small index: (target index, shared length, merge reason)}; polygons with no
            neighbour sharing a boundary are left out.
            """
            geoms = frame.geometry.values
            tree = shapely.STRtree(geoms)
            if near_tol:
                si_l, ci = tree.query(geoms[small_idx], predicate='dwithin', distance=near_tol)
            else:
                si_l, ci = tree.query(geoms[small_idx], predicate='intersects')
            si = small_idx[si_l]
            keep = si != ci
            si, ci = si[keep], ci[keep]
            if len(si) == 0:
                return {}

            # shared boundary length of every touching (or, with near_tol, nearby) pair
            try:
                bnd = shapely.boundary(geoms)
                if near_tol:
                    lengths = shapely.length(shapely.intersection(bnd[si], shapely.buffer(bnd[ci], near_tol)))
                else:
                    lengths = shapely.length(shapely.intersection(bnd[si], bnd[ci]))
            except Exception:
                # an invalid geometry makes the bulk call fail: do the pairs one by one, with
                # the per-pair handling (buffer fallback) that always applied
                lengths = np.array([calculate_shared_boundary_length(geoms[a], geoms[b], distance_threshold)
                                    for a, b in zip(si, ci)], dtype=float)
            has = lengths > 0
            si, ci, lengths = si[has], ci[has], lengths[has]
            order = np.lexsort((ci, si))
            si, ci, lengths = si[order], ci[order], lengths[order]

            def attr(name):
                if name is None or name not in frame.columns:
                    return [None] * len(frame)
                return [None if pd.isna(v) else str(v) for v in frame[name].values]

            litho_a = attr(lithoname)
            strat_a = [attr(strat1), attr(strat2), attr(strat3), attr(strat4)]
            rec_a = attr('rec_type') if 'rec_type' in frame.columns else [None] * len(frame)

            targets = {}
            starts = np.flatnonzero(np.r_[True, si[1:] != si[:-1]]) if len(si) else []
            ends = np.r_[starts[1:], len(si)] if len(si) else []
            for s0, s1_ in zip(starts, ends):
                s = int(si[s0])
                cands = ci[s0:s1_]
                lens = lengths[s0:s1_]
                # Triangles come from triangulating an ignore-code polygon (e.g. a dyke)
                # so its area can be shared out locally between the real units either
                # side of it. If triangles were allowed to merge into each other first,
                # a long thin dyke would chain into one piece and go entirely to
                # whichever unit has most boundary, painting that unit across the
                # dyke even where it crosses other units. So prefer real neighbours;
                # only fall back to triangle neighbours when no real one is touching
                # (such triangles are picked up on a later iteration).
                real = [k for k in range(len(cands)) if rec_a[cands[k]] != 'triangle']
                use = real if real else list(range(len(cands)))

                best, reason = None, "fallback_boundary_length"
                # Priorities 1-4: same strat1..strat4 + longest boundary
                for level, arr in enumerate(strat_a, 1):
                    mine = arr[s]
                    if best is None and mine is not None:
                        match = [k for k in use if arr[cands[k]] == mine]
                        if match:
                            best = max(match, key=lambda k: lens[k])
                            reason = f"same_strat{level}_{mine}"
                # Priority 5: same lithology + longest boundary
                if best is None and litho_a[s] is not None:
                    match = [k for k in use if litho_a[cands[k]] == litho_a[s]]
                    if match:
                        best = max(match, key=lambda k: lens[k])
                        reason = f"same_lithology_{litho_a[s]}"
                # Fallback: longest boundary regardless of attributes
                if best is None:
                    best = max(use, key=lambda k: lens[k])
                    reason = "fallback_boundary_length"
                targets[s] = (int(cands[best]), float(lens[best]), reason)
            return targets

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
            print("  Creating 'inherit' field (max 254 chars per field)...")
            gdf_cleaned['inherit'] = gdf_cleaned['primaryid'].astype(str)
        else:
            print("  'inherit' field already exists.")
        
        # Track which additional inherit fields are created
        max_inherit_field = 0  # 0 means just 'inherit', 1 means 'inherit1', etc.
        
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
        
        # Step 1b: Filling a small hole leaves any island polygon that sat in it
        # overlapping its host (the shared boundary ring is gone, so the island
        # would have no merge candidate). Absorb such islands into the host now.
        gdf_cleaned = gdf_cleaned.reset_index(drop=True)
        geoms_now = gdf_cleaned.geometry.values
        areas = shapely.area(geoms_now)
        island_hosts = {}
        small_now = np.flatnonzero(~shapely.is_empty(geoms_now) & (areas < min_area_threshold))
        if len(small_now):
            tree = shapely.STRtree(geoms_now)
            si_l, ci = tree.query(geoms_now[small_now])
            si = small_now[si_l]
            # only larger polygons can host an island; how much of the island each covers
            cand_ok = (si != ci) & (areas[ci] > areas[si])
            si, ci = si[cand_ok], ci[cand_ok]
            overlap = shapely.area(shapely.intersection(geoms_now[si], geoms_now[ci]))
            order = np.lexsort((ci, si))
            si, ci, overlap = si[order], ci[order], overlap[order]
            starts = np.flatnonzero(np.r_[True, si[1:] != si[:-1]]) if len(si) else []
            ends = np.r_[starts[1:], len(si)] if len(si) else []
            for s0, s1_ in zip(starts, ends):
                idx = int(si[s0])
                best_host, best_area = None, None
                for k in range(s0, s1_):
                    cand = int(ci[k])
                    if cand in island_hosts:
                        continue
                    if overlap[k] >= 0.99 * areas[idx] and (best_area is None or areas[cand] < best_area):
                        best_host, best_area = cand, areas[cand]
                if best_host is not None:
                    island_hosts[idx] = best_host

        if island_hosts:
            print(f"Absorbing {len(island_hosts)} island polygons covered by filled holes...")
            for island_idx, host_idx in island_hosts.items():
                # Host may itself be an absorbed island; follow the chain to a kept polygon
                while host_idx in island_hosts:
                    host_idx = island_hosts[host_idx]
                values = []
                field_num = 0
                while True:
                    field_name = 'inherit' if field_num == 0 else f'inherit{field_num}'
                    if field_name not in gdf_cleaned.columns:
                        break
                    value = gdf_cleaned.iloc[island_idx][field_name]
                    if pd.notna(value) and str(value).strip():
                        values.append(str(value))
                    field_num += 1
                if values:
                    max_inherit_field = append_to_inherit_fields(
                        gdf_cleaned, host_idx, ' '.join(values), max_inherit_field)
            gdf_cleaned = gdf_cleaned.drop(index=list(island_hosts)).reset_index(drop=True)

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
            
            # Track which polygons have been merged in this iteration
            merged_indices = set()

            # Find all small polygons and their best merge candidates
            areas_now = shapely.area(gdf_cleaned.geometry.values)
            small_idx = np.flatnonzero(areas_now < min_area_threshold)
            small_polygon_count = len(small_idx)
            found = find_merge_targets(gdf_cleaned, small_idx, distance_threshold,
                                       lithoname, strat1, strat2, strat3, strat4)
            # A small polygon that touches no other polygon (e.g. it is separated from its
            # neighbours by a sliver gap) may merge into a polygon within the tolerance; the
            # gap between the two is filled when they are merged (see below).
            near_merges = set()
            orphans = np.array([int(i) for i in small_idx if int(i) not in found], dtype=np.int64)
            if len(orphans) and distance_threshold and distance_threshold > 1e-6:
                found_near = find_merge_targets(gdf_cleaned, orphans, distance_threshold,
                                                lithoname, strat1, strat2, strat3, strat4,
                                                near_tol=distance_threshold)
                found.update(found_near)
                near_merges = set(found_near)
            # Maps small polygon index to (target polygon index, merge reason)
            merge_targets = {int(i): (found[int(i)][0], found[int(i)][2])
                             for i in small_idx if int(i) in found}
            all_geoms = gdf_cleaned.geometry.values
            geom_tree = shapely.STRtree(all_geoms) if near_merges else None

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
                        if small_idx in near_merges:
                            # the two do not touch: also add the gap between them (the area
                            # within the tolerance of both), minus any other polygon
                            sg, tg = all_geoms[small_idx], all_geoms[target_idx]
                            bridge = sg.buffer(distance_threshold).intersection(tg.buffer(distance_threshold))
                            others = [int(o) for o in geom_tree.query(bridge, predicate='intersects')
                                      if int(o) not in (small_idx, target_idx)]
                            if others:
                                bridge = bridge.difference(unary_union(all_geoms[others]))
                            if not bridge.is_empty:
                                geometries_to_merge.append(bridge)
                        
                        # Collect all inherit values from small polygon (inherit, inherit1, inherit2, etc.)
                        small_inherit_values = []
                        field_num = 0
                        while True:
                            field_name = 'inherit' if field_num == 0 else f'inherit{field_num}'
                            if field_name in gdf_cleaned.columns:
                                value = gdf_cleaned.iloc[small_idx][field_name]
                                if pd.notna(value) and str(value).strip():
                                    small_inherit_values.append(str(value))
                                field_num += 1
                            else:
                                break
                        
                        # Concatenate all inherit values from small polygon
                        small_inherit_combined = ' '.join(small_inherit_values)
                        
                        # Append to target's inherit field(s), creating overflow fields if needed
                        if small_inherit_combined:
                            max_inherit_field = append_to_inherit_fields(gdf_cleaned, target_idx, small_inherit_combined, max_inherit_field)
                        
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

                    # unary_union can return GeometryCollection for edge cases
                    # (e.g. shared-boundary LinearRings); extract only polygon parts
                    if merged_geom.geom_type == 'GeometryCollection':
                        poly_parts = [g for g in merged_geom.geoms
                                      if g.geom_type in ('Polygon', 'MultiPolygon')]
                        if poly_parts:
                            merged_geom = unary_union(poly_parts)
                        else:
                            continue  # no polygon parts; skip this merge

                    # Fix any self-intersections before storing
                    if not merged_geom.is_valid:
                        merged_geom = merged_geom.buffer(0)

                    # Update the target polygon with merged geometry
                    gdf_cleaned.iloc[target_idx, gdf_cleaned.columns.get_loc('geometry')] = merged_geom
            
            # Remove merged polygons
            if merged_indices:
                gdf_cleaned = gdf_cleaned.drop(index=list(merged_indices))
                gdf_cleaned = gdf_cleaned.reset_index(drop=True)
                total_merged += len(merged_indices)
            else:
                break
        
        # Step 3: Fix any remaining invalid geometries and remove any zero-area
        # degenerate results.  We do NOT apply filter_large_polygons here:
        # when all input polygons are smaller than the threshold, the merge loop
        # produces the best possible result but the merged polygons may still be
        # below the threshold — removing them would produce an empty output.
        print("Cleaning up geometries...")
        gdf_cleaned['geometry'] = gdf_cleaned['geometry'].apply(
            lambda geom: geom.buffer(0) if not geom.is_valid else geom
        )
        # Drop only truly degenerate (zero-area) polygons
        gdf_cleaned = gdf_cleaned[gdf_cleaned.geometry.area > 0].reset_index(drop=True)
        
        # Print summary statistics
        below_threshold = (gdf_cleaned.geometry.area < min_area_threshold).sum()
        print(f"\nMerge Summary:")
        print(f"Original polygons: {len(gdf)}")
        print(f"Final polygons: {len(gdf_cleaned)}")
        print(f"  (of which {below_threshold} are still below the threshold — orphans that had no merge candidates)")
        print(f"Total polygons merged: {total_merged}")
        print(f"Merge iterations: {iteration}")
        if max_inherit_field > 0:
            print(f"Created {max_inherit_field} additional inherit field(s) for overflow: ", end="")
            print(", ".join([f"inherit{i}" for i in range(1, max_inherit_field + 1)]))
        print(f"\nMerge reasons:")
        for reason, count in merge_stats.items():
            if count > 0:
                print(f"  {reason}: {count} polygons")
        
        return gdf_cleaned
        
