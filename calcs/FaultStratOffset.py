import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import Point
import warnings

warnings.filterwarnings("ignore")

# Fallback sample locations along a fault, in the (alphabetical) order the first-maximum
# tie-break has always used
_FALLBACK = [("start", 0.0), ("midpoint", 0.5), ("end", 1.0)]
_FALLBACK_ORDER = {"end": 0, "midpoint": 1, "start": 2}


class FaultStratOffset:
    """
    Stratigraphic offset across faults.

    At each *sample location* on a fault the geology is read at one point on each side of the
    fault (a point stepped sideways, perpendicular to the fault, by a small offset).  If the
    stratigraphy differs across the fault the sample shows an offset, graded by the most
    general stratigraphic level (priority) at which the two sides differ.

    Sample locations are (see :meth:`get_sampling_points`):

    * every point where the fault crosses a geological contact (polygon boundary) - these
      include the places where a fault meets a triple junction.  Where a fault runs along a
      contact, the middle of the shared section is used.
    * the start and end of every fault (skipped where a contact sample is already there);
      for faults that meet no contact at all, the middle as well, so that every fault gets
      an answer.

    Each fault then gets its maximum offset level plus how many samples were used and how
    many of them showed an offset.
    """

    def __init__(self):
        pass

    def read_shapefiles(self, fault_path, geology_path):
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

    # ------------------------------------------------------------------ geometry helpers
    @staticmethod
    def _metres_per_unit(crs, y):
        """Metres per CRS unit along x and along y at latitude ``y`` (1, 1 for a projected CRS)."""
        y = np.asarray(y, dtype=float)
        if crs is not None and getattr(crs, "is_geographic", False):
            return 111320.0 * np.cos(np.radians(y)), np.full(y.shape, 110540.0)
        return np.ones(y.shape), np.ones(y.shape)

    def _contact_samples(self, faults, geology_gdf, crs):
        """
        Where each fault meets a geological contact.

        Returns arrays ``(fault_pos, x, y, kind)``: position of the fault in ``faults``, the
        location, and 'contact' (a crossing / touching point) or 'shared' (middle of a section
        where the fault runs along a contact).  Duplicates are removed: a contact shared by two
        polygons, or a triple junction, is in the data several times.
        """
        empty = (np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), np.zeros(0, dtype=object))
        parts = shapely.get_parts(geology_gdf.geometry.values)
        parts = parts[~shapely.is_missing(parts)]
        polys = parts[shapely.get_type_id(parts) == 3]
        if len(polys) == 0:
            return empty
        contacts = shapely.get_rings(polys)
        tree = shapely.STRtree(contacts)
        fi, ci = tree.query(faults, predicate="intersects")
        if len(fi) == 0:
            return empty
        inter = shapely.intersection(faults[fi], contacts[ci])
        pieces, which = shapely.get_parts(inter, return_index=True)
        tid = shapely.get_type_id(pieces)
        is_pt = tid == 0
        is_ln = (tid == 1) | (tid == 2)
        pt_xy = shapely.get_coordinates(pieces[is_pt])
        pt_f = fi[which[is_pt]]
        mid = shapely.line_interpolate_point(pieces[is_ln], 0.5, normalized=True)
        ln_xy = shapely.get_coordinates(mid)
        ln_f = fi[which[is_ln]]
        xy = np.concatenate([pt_xy, ln_xy])
        f = np.concatenate([pt_f, ln_f])
        kind = np.concatenate([np.full(len(pt_f), "contact", dtype=object),
                               np.full(len(ln_f), "shared", dtype=object)])
        if len(f) == 0:
            return empty
        # remove duplicates (to ~1 mm)
        tol = 1e-8 if getattr(crs, "is_geographic", False) else 1e-3
        key = np.stack([f.astype(float), np.round(xy[:, 0] / tol), np.round(xy[:, 1] / tol)], axis=1)
        _, first = np.unique(key, axis=0, return_index=True)
        first = np.sort(first)
        return f[first].astype(np.int64), xy[first, 0], xy[first, 1], kind[first]

    # ------------------------------------------------------------------ sampling
    def get_sampling_points(self, faults_gdf, offset_distance=50, geology_gdf=None,
                            crossing_offset=5.0, use_crossings=True, sample_ends=True):
        """
        Create the sample locations on the faults and, at each, one point on each side.

        Parameters
        ----------
        faults_gdf       : GeoDataFrame of fault polylines
        offset_distance  : sideways step (metres) at the start / middle / end fallback locations
        geology_gdf      : geology polygons; needed to find where faults meet contacts
        crossing_offset  : sideways step (metres) at contacts - small, so the two points stay
                           next to the contact
        use_crossings    : False reproduces the original sampling (start, middle and end of
                           every fault, nothing else)
        sample_ends      : also sample the start and end of faults that meet contacts (skipped
                           where a contact sample lies within ``crossing_offset`` of the end)

        Returns
        -------
        GeoDataFrame of side points: ``sample_id``, ``fault_id``, ``location`` ('contact',
        'shared', 'start', 'midpoint' or 'end'), ``point_type`` ('side1' / 'side2'),
        ``x``, ``y`` (the location on the fault) and the point geometry.
        """
        print("Creating sample locations on the faults...")
        crs = faults_gdf.crs
        faults = faults_gdf.geometry.values
        valid = ~shapely.is_missing(faults) & ~shapely.is_empty(faults)
        length = np.where(valid, shapely.length(faults), 0.0)

        # 1. locations at contacts
        if use_crossings and geology_gdf is not None:
            cf, cx, cy, ckind = self._contact_samples(faults, geology_gdf, crs)
            ok = valid[cf]
            cf, cx, cy, ckind = cf[ok], cx[ok], cy[ok], ckind[ok]
        else:
            cf = np.zeros(0, dtype=np.int64); cx = cy = np.zeros(0); ckind = np.zeros(0, dtype=object)
        # order along each fault
        if len(cf):
            along = shapely.line_locate_point(faults[cf], shapely.points(np.stack([cx, cy], axis=1)))
            order = np.lexsort((along, cf))
            cf, cx, cy, ckind, along = cf[order], cx[order], cy[order], ckind[order], along[order]
        else:
            along = np.zeros(0)
        # tangent over +/- the sideways step, in CRS units
        mx0, my0 = self._metres_per_unit(crs, cy if len(cy) else np.zeros(0))
        delta = crossing_offset / np.maximum(mx0, 1e-12) if len(cf) else np.zeros(0)
        c_before = np.clip(along - delta, 0.0, length[cf]) if len(cf) else along
        c_after = np.clip(along + delta, 0.0, length[cf]) if len(cf) else along
        c_off = np.full(len(cf), float(crossing_offset))

        has_contact = np.zeros(len(faults), dtype=bool)
        has_contact[cf] = True

        # 1b. the two ends of faults that do meet contacts (unless a contact sample already
        #     sits at that end), so the fault tips are tested too
        if sample_ends and len(cf):
            ends = np.flatnonzero(valid & has_contact)
            lo = np.searchsorted(cf, ends, side="left")
            hi = np.searchsorted(cf, ends, side="right") - 1
            L = length[ends]
            e_parts = []
            for name, pos in (("start", 0.0), ("end", 1.0)):
                if pos == 0.0:
                    near = along[lo] <= delta[lo]
                    b_, a_ = 0.0 * L, 0.01 * L
                else:
                    near = along[hi] >= L - delta[hi]
                    b_, a_ = 0.99 * L, 1.0 * L
                sel = ~near
                p = shapely.line_interpolate_point(faults[ends[sel]], pos * L[sel])
                e_parts.append((ends[sel], shapely.get_x(p), shapely.get_y(p),
                                np.full(sel.sum(), name, dtype=object),
                                b_[sel], a_[sel], pos * L[sel]))
            e_f, e_x, e_y, e_k, e_b, e_a, e_al = (np.concatenate(v) for v in zip(*e_parts))
            # merge with the contact samples, in order along each fault
            cf = np.concatenate([cf, e_f]).astype(np.int64)
            cx = np.concatenate([cx, e_x]); cy = np.concatenate([cy, e_y])
            ckind = np.concatenate([ckind, e_k]).astype(object)
            c_before = np.concatenate([c_before, e_b]); c_after = np.concatenate([c_after, e_a])
            c_off = np.concatenate([c_off, np.full(len(e_f), float(offset_distance))])
            along = np.concatenate([along, e_al])
            order = np.lexsort((along, cf))
            cf, cx, cy, ckind, c_before, c_after, c_off, along = (
                v[order] for v in (cf, cx, cy, ckind, c_before, c_after, c_off, along))

        # 2. fallback locations for faults that meet no contact
        fb = np.flatnonzero(valid & ~has_contact)
        ff, fx, fy, fkind, fbefore, fafter = [], [], [], [], [], []
        if len(fb):
            L = length[fb]
            for name, pos in _FALLBACK:
                p = shapely.line_interpolate_point(faults[fb], pos * L)
                if pos == 0.0:
                    b_, a_ = 0.0 * L, 0.01 * L
                elif pos == 1.0:
                    b_, a_ = 0.99 * L, 1.0 * L
                else:
                    b_, a_ = (pos - 0.01) * L, (pos + 0.01) * L
                ff.append(fb); fx.append(shapely.get_x(p)); fy.append(shapely.get_y(p))
                fkind.append(np.full(len(fb), name, dtype=object)); fbefore.append(b_); fafter.append(a_)
            ff = np.concatenate(ff); fx = np.concatenate(fx); fy = np.concatenate(fy)
            fkind = np.concatenate(fkind); fbefore = np.concatenate(fbefore); fafter = np.concatenate(fafter)
            # order: fault, then alphabetical location (the order the maximum has always used)
            o = np.lexsort((np.array([_FALLBACK_ORDER[k] for k in fkind]), ff))
            ff, fx, fy, fkind, fbefore, fafter = ff[o], fx[o], fy[o], fkind[o], fbefore[o], fafter[o]
        else:
            ff = np.zeros(0, dtype=np.int64); fx = fy = fbefore = fafter = np.zeros(0)
            fkind = np.zeros(0, dtype=object)
        f_off = np.full(len(ff), float(offset_distance))

        # 3. all samples, ordered by fault
        S_f = np.concatenate([cf, ff]).astype(np.int64)
        S_x = np.concatenate([cx, fx]); S_y = np.concatenate([cy, fy])
        S_kind = np.concatenate([ckind, fkind]).astype(object)
        S_b = np.concatenate([c_before, fbefore]); S_a = np.concatenate([c_after, fafter])
        S_off = np.concatenate([c_off, f_off])
        order = np.argsort(S_f, kind="stable")
        S_f, S_x, S_y, S_kind, S_b, S_a, S_off = (v[order] for v in (S_f, S_x, S_y, S_kind, S_b, S_a, S_off))

        # direction of the fault at each sample, then the two side points
        pb = shapely.line_interpolate_point(faults[S_f], S_b)
        pa = shapely.line_interpolate_point(faults[S_f], S_a)
        dx = shapely.get_x(pa) - shapely.get_x(pb)
        dy = shapely.get_y(pa) - shapely.get_y(pb)
        mx, my = self._metres_per_unit(crs, S_y)
        dxm, dym = dx * mx, dy * my
        ln = np.sqrt(dxm ** 2 + dym ** 2)
        keep = ln != 0
        S_f, S_x, S_y, S_kind, S_off, dxm, dym, ln, mx, my = (
            v[keep] for v in (S_f, S_x, S_y, S_kind, S_off, dxm, dym, ln, mx, my))
        dxn, dyn = dxm / ln, dym / ln
        perp_x, perp_y = -dyn, dxn                     # perpendicular, in metre space
        p1x = S_x + perp_x * S_off / mx
        p1y = S_y + perp_y * S_off / my
        p2x = S_x - perp_x * S_off / mx
        p2y = S_y - perp_y * S_off / my

        ns = len(S_f)
        sample_id = np.repeat(np.arange(ns), 2)
        fault_id = np.repeat(faults_gdf.index.values[S_f], 2)
        xs = np.empty(2 * ns); ys = np.empty(2 * ns)
        xs[0::2], xs[1::2] = p1x, p2x
        ys[0::2], ys[1::2] = p1y, p2y
        points = gpd.GeoDataFrame({
            "sample_id": sample_id,
            "fault_id": fault_id,
            "location": np.repeat(S_kind, 2),
            "point_type": np.tile(np.array(["side1", "side2"], dtype=object), ns),
            "x": np.repeat(S_x, 2),
            "y": np.repeat(S_y, 2),
        }, geometry=shapely.points(xs, ys), crs=crs)
        return points

    # ------------------------------------------------------------------ geology lookup
    def assign_geology_to_points(self, points_gdf, geology_gdf, strat_columns):
        """
        Stratigraphy of the polygon containing each point (the first, if polygons overlap).
        Points inside no polygon get nothing.
        """
        print("Looking up the geology at each point...")
        geoms = geology_gdf.geometry.values
        tree = shapely.STRtree(geoms)
        pi, gi = tree.query(points_gdf.geometry.values, predicate="within")
        first_poly = np.full(len(points_gdf), -1, dtype=np.int64)
        if len(pi):
            o = np.lexsort((gi, pi))
            pi, gi = pi[o], gi[o]
            keep = np.r_[True, pi[1:] != pi[:-1]]
            first_poly[pi[keep]] = gi[keep]
        out = points_gdf.copy()
        has = first_poly >= 0
        for col in strat_columns:
            if col in geology_gdf.columns:
                vals = geology_gdf[col].values
                arr = np.full(len(points_gdf), np.nan, dtype=object)
                arr[has] = vals[first_poly[has]]
                out[col] = arr
            else:
                out[col] = np.nan
        return out

    # ------------------------------------------------------------------ offsets
    def calculate_stratigraphic_offset(self, points_with_geology, strat_columns):
        """
        Offset level at every sample (from the units either side) and each fault's summary.

        Offset level, at a sample: ``-1`` geology on one side only (edge of the map); ``0`` the
        same units on both sides at every level; otherwise the priority level (a larger number
        is a more general level, so a bigger offset) at which the two sides first differ,
        working from the most general level down.  No row is made for a sample with no
        geology on either side.

        Returns (fault summary DataFrame, sample DataFrame).
        """
        print("Calculating stratigraphic offsets...")
        pts = points_with_geology
        s1 = pts[pts["point_type"] == "side1"].set_index("sample_id")
        s2 = pts[pts["point_type"] == "side2"].set_index("sample_id")
        ids = s1.index.values
        s2 = s2.reindex(ids)
        n = len(ids)
        if n == 0:
            print("Warning: no sample locations found!")
            return pd.DataFrame(), pd.DataFrame()

        v1 = {c: s1[c].values for c in strat_columns}
        v2 = {c: s2[c].values for c in strat_columns}
        na1 = {c: pd.isna(v1[c]) for c in strat_columns}
        na2 = {c: pd.isna(v2[c]) for c in strat_columns}
        has1 = np.zeros(n, dtype=bool)
        has2 = np.zeros(n, dtype=bool)
        for c in strat_columns:
            has1 |= ~na1[c]
            has2 |= ~na2[c]

        level = np.full(n, np.nan)
        desc = np.full(n, "", dtype=object)
        both = has1 & has2
        edge = has1 ^ has2
        level[edge] = -1
        desc[edge] = "Fault at edge of map"
        desc[~has1 & ~has2] = "No geological data found"
        for priority_level, col in enumerate(reversed(strat_columns), 1):
            usable = both & np.isnan(level) & ~na1[col] & ~na2[col]
            differ = usable & np.array([a != b for a, b in zip(v1[col], v2[col])], dtype=bool)
            actual = len(strat_columns) - priority_level + 1
            level[differ] = actual
            desc[differ] = f"Offset at {col} level (priority {actual})"
        same = both & np.isnan(level)
        level[same] = 0
        desc[same] = "No offset - same at all levels"

        def units(vals, na, i):
            return [vals[c][i] for c in strat_columns if not na[c][i]]

        samples = pd.DataFrame({
            "sample_id": ids,
            "fault_id": s1["fault_id"].values,
            "location": s1["location"].values,
            "x": s1["x"].values,
            "y": s1["y"].values,
            "offset_level": level,
            "offset_description": desc,
            "side1_units": [units(v1, na1, i) for i in range(n)],
            "side2_units": [units(v2, na2, i) for i in range(n)],
        })
        # a sample with no geology on either side makes no row (as in the original)
        samples = samples[~(~has1 & ~has2)].reset_index(drop=True)

        print("Finding maximum offsets per fault...")
        # first sample with the maximum level (ties: samples are in order along the fault)
        order_in_fault = samples.groupby("fault_id", sort=False).cumcount()
        samples["_pos"] = order_in_fault
        all_faults = pd.unique(s1["fault_id"].values)
        final = []
        grouped = {fid: g for fid, g in samples.groupby("fault_id", sort=False)}
        for fid in np.sort(all_faults) if pd.api.types.is_numeric_dtype(all_faults) else all_faults:
            g = grouped.get(fid)
            if g is None or g["offset_level"].notna().sum() == 0:
                summ = {"fault_id": fid, "max_offset_level": None, "max_offset_location": "no_data",
                        "max_offset_description": "No geological data found",
                        "side1_units": [], "side2_units": []}
                final.append(self._add_counts(summ, g))
                continue
            lv = g["offset_level"].values
            mx_level = np.nanmax(lv)
            row = g.iloc[int(np.flatnonzero(lv == mx_level)[0])]
            if mx_level == -1:
                summ = {"fault_id": fid, "max_offset_level": -1, "max_offset_location": row["location"],
                        "max_offset_description": "Fault at edge of map",
                        "side1_units": [], "side2_units": []}
            else:
                summ = {"fault_id": fid, "max_offset_level": row["offset_level"],
                        "max_offset_location": row["location"],
                        "max_offset_description": row["offset_description"],
                        "side1_units": row["side1_units"], "side2_units": row["side2_units"]}
            summ["max_off_x"] = row["x"]
            summ["max_off_y"] = row["y"]
            # per-location detail for the start / middle / end fallback samples
            for _, r in g.iterrows():
                if r["location"] in _FALLBACK_ORDER:
                    summ[f"{r['location']}_offset"] = r["offset_level"]
                    summ[f"{r['location']}_desc"] = r["offset_description"]
            final.append(self._add_counts(summ, g))
        samples = samples.drop(columns="_pos")
        return pd.DataFrame(final), samples

    @staticmethod
    def _add_counts(summary, g):
        """How many samples a fault had and how many of them show an offset."""
        if g is None or len(g) == 0:
            summary.update({"n_samples": 0, "n_offset": 0, "n_nooffset": 0, "n_edge": 0,
                            "sample_src": "none"})
            return summary
        lv = g["offset_level"].values
        summary.update({
            "n_samples": int(len(g)),
            "n_offset": int(np.sum(lv > 0)),
            "n_nooffset": int(np.sum(lv == 0)),
            "n_edge": int(np.sum(lv == -1)),
            "sample_src": "contacts" if g["location"].isin(["contact", "shared"]).any() else "ends",
        })
        return summary

    # ------------------------------------------------------------------ output
    def create_fault_offset_shapefile(self, faults_gdf, offset_results, output_path):
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

    def create_samples_shapefile(self, samples, crs, output_path):
        """Save the sample locations (one point per sample, on the fault) with their offsets."""
        out = gpd.GeoDataFrame({
            "fault_id": samples["fault_id"].values,
            "location": samples["location"].values,
            "off_level": samples["offset_level"].values,
            "off_desc": samples["offset_description"].values,
            "side1_unit": [str(u) for u in samples["side1_units"]],
            "side2_unit": [str(u) for u in samples["side2_units"]],
        }, geometry=shapely.points(samples["x"].values, samples["y"].values), crs=crs)
        out.to_file(output_path)
        print(f"Sample locations saved to: {output_path}")
        return out

    # Main execution function
    def CalcFaultStratOffset(
        self, fault_shapefile,
        geology_shapefile,
        output_shapefile,
        strat_columns,
        offset_distance=50,
        crossing_offset=5.0,
        use_crossings=True,
        samples_shapefile="auto",
    ):
        """
        Main function to execute the complete workflow

        Parameters:
        - fault_shapefile: Path to fault polylines shapefile
        - geology_shapefile: Path to geology polygons shapefile
        - output_shapefile: Path for output shapefile
        - strat_columns: List of stratigraphic column names (most detailed first)
        - offset_distance: Sideways step (metres) at the start / middle / end of faults that
          meet no contact
        - crossing_offset: Sideways step (metres) at contacts
        - use_crossings: False to sample only the start, middle and end of each fault
        - samples_shapefile: Where to save the sample points ("auto" = beside the output,
          ``*_samples.shp``; None = do not save)
        """

        try:
            # Step 0: Read shapefiles
            faults_gdf, geology_gdf = self.read_shapefiles(fault_shapefile, geology_shapefile)

            print(f"Loaded {len(faults_gdf)} fault segments")
            print(f"Loaded {len(geology_gdf)} geology polygons")
            print(f"Stratigraphic columns: {strat_columns}")

            # Step 1-2: Sample locations on the faults and a point either side of each
            points_gdf = self.get_sampling_points(
                faults_gdf, offset_distance, geology_gdf=geology_gdf,
                crossing_offset=crossing_offset, use_crossings=use_crossings)
            print(f"Created {len(points_gdf) // 2} sample locations "
                  f"({len(points_gdf)} points, one each side)")
            # Step 3: Look up the geology at the points
            points_with_geology = self.assign_geology_to_points(
                points_gdf, geology_gdf, strat_columns
            )

            # Step 4: Calculate stratigraphic offsets
            offset_results, location_details = self.calculate_stratigraphic_offset(
                points_with_geology, strat_columns
            )

            # Step 5: Create output shapefile(s)
            result_gdf = self.create_fault_offset_shapefile(
                faults_gdf, offset_results, output_shapefile
            )
            if samples_shapefile == "auto":
                samples_shapefile = output_shapefile.replace(".shp", "_samples.shp")
            if samples_shapefile and len(location_details):
                self.create_samples_shapefile(location_details, faults_gdf.crs, samples_shapefile)

            print("\nSummary of maximum offsets:")
            print(offset_results["max_offset_level"].value_counts().sort_index())
            print("\nSample locations used:")
            print(offset_results["sample_src"].value_counts())

            return result_gdf, offset_results, location_details

        except Exception as e:
            print(f"Error in processing: {str(e)}")
            raise
