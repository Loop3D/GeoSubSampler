# -*- coding: utf-8 -*-
"""
gss_processing.py  -  GUI-independent GeoSubSampler algorithms for the ArcGIS Pro version.

A port of the processing methods in GeoSubSampler.py (the QGIS plugin). Nothing here
needs QGIS or a running ArcGIS Pro: layers are described by plain "layer records"
(dicts, produced by the .pyt toolbox) and everything is done with geopandas on files.

Layer record keys:
    name        display name of the layer in the map
    source      data source path (shapefile, or <geodatabase>.gdb\\<feature class>)
    kind        "point" | "line" | "polygon"
    geographic  True if the layer's CRS is geographic (degrees)
    fields      list of [field name, field type]
"""

import csv
import math
import os
import random
import re
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import geopandas as gpd

from calcs.StructuralPolygonSubSampler import StructuralPolygonSubSampler
from calcs.PolygonTriangulator import PolygonTriangulator
from calcs.FaultLineMerger import FaultLineMerger
from calcs.FaultLengths import FaultLengths
from calcs.FaultsGraph import FaultsGraph
from calcs.FaultStratOffset import FaultStratOffset
from calcs.FaultClusterOrientation import FaultsOrientations
from calcs.FirstOrderOrientation import SubsamplingEngine, save_grid_to_shapefile
from calcs.TopferPillewizer import (dipdir_to_strike, topfer_count, scale_iterations,
                                    scale_points_tp, scale_lines_tp, scale_polygons_tp)
from calcs.PolygonSimplification import vector_simplify_file_two_stage

DEG_PER_M_INV = 110000.0     # metres per degree, as used by the QGIS plugin


class GSSError(Exception):
    """A problem to report to the user (not a bug)."""


def parse_codes(text):
    """Split a comma/newline separated code list, trimming whitespace and quotes."""
    codes = []
    for part in text.replace("\n", ",").replace(";", ",").split(","):
        part = part.strip().strip("'\"").strip()
        if part:
            codes.append(part)
    return codes


def safe_filename(text, max_len=80):
    """Layer name -> text that is safe inside a Windows file name."""
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', '_', str(text)).strip(' .')
    return cleaned[:max_len] or 'layer'


def shapefile_safe_columns(columns):
    """Map each column to a <=10 character DBF-safe name, de-duplicating collisions."""
    seen, mapping = {}, {}
    for col in columns:
        if col == 'geometry':
            mapping[col] = col
            continue
        candidate = str(col)[:10]
        n = 1
        while candidate in seen and seen[candidate] != col:
            suffix = f"_{n}"
            candidate = str(col)[:10 - len(suffix)] + suffix
            n += 1
        seen[candidate] = col
        mapping[col] = candidate
    return mapping


class GSSProcessor:
    """
    All GeoSubSampler operations.  `log(message, level)` receives progress / warning text
    (level is "info", "warning" or "error"); `add_output(path, like_layer_name)` is called
    for every result file that should be added to the map.
    """

    def __init__(self, log=print, add_output=None, fallback_dir=None):
        self._log = log
        self._add_output = add_output or (lambda path, like: None)
        self._fallback = fallback_dir
        self._field_maps = {}        # resolved shp path -> {original column: shapefile column}
        self._writable = {}
        self._copies = {}

    # ------------------------------------------------------------------ messages
    def log(self, message, level="info"):
        self._log(message, level)

    def warn(self, message):
        self.log(message, "warning")

    # ------------------------------------------------------------------ files
    def _dir_writable(self, folder):
        if not folder or not os.path.isdir(folder):
            return False
        if folder not in self._writable:
            try:
                with tempfile.NamedTemporaryFile(dir=folder):
                    pass
                self._writable[folder] = True
            except OSError:
                self._writable[folder] = False
        return self._writable[folder]

    def _fallback_dir(self):
        for folder in (self._fallback, tempfile.gettempdir()):
            if self._dir_writable(folder):
                return folder
        return tempfile.gettempdir()

    def _copy_to_writable(self, shp_path):
        if shp_path in self._copies and os.path.exists(self._copies[shp_path]):
            return self._copies[shp_path]
        base = os.path.splitext(shp_path)[0]
        dest_base = os.path.join(self._fallback_dir(), safe_filename(os.path.basename(base)))
        try:
            for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.qpj', '.sbn', '.sbx'):
                if os.path.exists(base + ext):
                    shutil.copyfile(base + ext, dest_base + ext)
        except OSError as exc:
            self.warn(f"'{os.path.dirname(shp_path)}' is not writable and a working copy "
                      f"could not be made: {exc}")
            return shp_path
        self._copies[shp_path] = dest_base + '.shp'
        self.log(f"'{os.path.dirname(shp_path)}' is not writable, so a working copy was made in "
                 f"{os.path.dirname(dest_base)}; results will be saved there.")
        return self._copies[shp_path]

    @staticmethod
    def _read_source(source):
        """Read a shapefile, or a feature class inside a file geodatabase."""
        m = re.match(r"(.*?\.gdb)[\\/](.+)$", source, re.IGNORECASE)
        if m:
            return gpd.read_file(m.group(1), layer=m.group(2))
        return gpd.read_file(source)

    def resolved_source(self, rec):
        """
        A .shp path for the layer.  Shapefiles are used in place (or copied if their folder
        is read-only); geodatabase feature classes are exported once to a shapefile, with
        field names pre-shortened to the 10-character DBF limit.
        """
        source = rec["source"]
        if source.lower().endswith('.shp'):
            if not os.path.exists(source) or self._dir_writable(os.path.dirname(source)):
                return source
            return self._copy_to_writable(source)

        out_dir = self._fallback_dir()
        gdb_match = re.match(r"(.*?\.gdb)", source, re.IGNORECASE)
        if not gdb_match and os.path.exists(source):
            out_dir = os.path.dirname(source) if self._dir_writable(os.path.dirname(source)) \
                else out_dir
        out_path = os.path.join(out_dir, safe_filename(rec["name"]) + ".shp")

        if out_path not in self._field_maps or not os.path.exists(out_path):
            try:
                gdf = self._read_source(source)
            except Exception as exc:
                raise GSSError(f"Could not read layer '{rec['name']}' ({source}): {exc}")
            field_map = shapefile_safe_columns(gdf.columns)
            self._field_maps[out_path] = field_map
            if any(k != v for k, v in field_map.items()):
                gdf = gdf.rename(columns=field_map)
            gdf.to_file(out_path, driver='ESRI Shapefile')
        return out_path

    def resolved_field(self, rec, field_name):
        """Map a field name as shown in the GUI to the column in the resolved shapefile."""
        if not field_name:
            return field_name
        field_map = self._field_maps.get(self.resolved_source(rec))
        if field_map and field_name in field_map:
            return field_map[field_name]
        return field_name

    def _output_directory(self, rec):
        folder = os.path.dirname(self.resolved_source(rec))
        return folder if self._dir_writable(folder) else self._fallback_dir()

    def _make_output_path(self, rec, suffix):
        base = os.path.join(self._output_directory(rec),
                            safe_filename(rec["name"]) + safe_filename(suffix))
        path = base + ".shp"
        if os.path.exists(path):
            path = f"{base}_{random.randint(10000, 99999)}.shp"
        return path

    def _save_gdf(self, gdf, path):
        """Save as a shapefile; on failure retry in the temp folder before raising."""
        try:
            gdf.to_file(path, driver="ESRI Shapefile")
            return path
        except Exception as exc:
            self.warn(f"Could not save {path}: {exc}. Trying the temp folder instead.")
        alt = os.path.join(tempfile.gettempdir(), os.path.basename(path))
        try:
            gdf.to_file(alt, driver="ESRI Shapefile")
        except Exception as exc2:
            raise GSSError(f"could not save the result to {alt}: {exc2}")
        self.warn(f"Result saved to {alt} instead.")
        return alt

    def _emit(self, path, like_rec):
        if not os.path.exists(path):
            self.warn(f"Expected output was not produced: {path}")
            return
        self.log(f"Created {path}")
        self._add_output(path, like_rec["name"] if like_rec else None)

    # ------------------------------------------------------------------ points
    def _setup_points(self, rec, dip_col, dipdir_col, is_dipdir, check_fields=True):
        """Read the points and build the engine. Returns (gdf, engine, dip_col, dipdir_col)."""
        if rec is None:
            raise GSSError("Select a points layer first.")
        gdf = gpd.read_file(self.resolved_source(rec))
        if (not dip_col or not dipdir_col):
            if check_fields:
                raise GSSError("Please define dip & dip dir/strike fields before continuing.")
            dip_col, dipdir_col = gdf.columns[1], gdf.columns[2]
        else:
            dip_col = self.resolved_field(rec, dip_col)
            dipdir_col = self.resolved_field(rec, dipdir_col)

        missing = [c for c in (dip_col, dipdir_col) if c not in gdf.columns]
        if missing:
            raise GSSError(f"field(s) {missing} not found. Available columns: {list(gdf.columns)}")

        gdf = gdf.copy()
        gdf['EASTING'] = gdf.geometry.x
        gdf['NORTHING'] = gdf.geometry.y

        n_before = len(gdf)
        gdf = gdf.dropna(subset=[dip_col, dipdir_col, 'EASTING', 'NORTHING']).copy()
        if n_before - len(gdf):
            self.warn(f"{n_before - len(gdf)} point(s) with null dip/dip-dir/coordinates "
                      f"dropped before subsampling.")

        if not is_dipdir:
            gdf['_eff_dipdir'] = (gdf[dipdir_col].astype(float) - 90) % 360
            eff = '_eff_dipdir'
        else:
            eff = dipdir_col
        engine = SubsamplingEngine(dip=dip_col, dipdir=eff, easting='EASTING', northing='NORTHING')
        return gdf, engine, dip_col, dipdir_col

    def _finalise_points(self, gdf2, rec, name, param=0):
        """Write a point result next to its source and queue it for the map."""
        suffix = "" if param == 0 else "_" + str(int(param))
        stem = safe_filename(rec["name"]) + "_" + safe_filename(name) + suffix
        folder = self._output_directory(rec)
        new_path = os.path.join(folder, stem + ".shp")
        if os.path.exists(new_path):
            new_path = os.path.join(folder, f"{stem}_{random.randint(10000, 99999)}.shp")

        if gdf2.empty:
            self.warn("No points were retained - an empty point layer was created. "
                      "Check that the polygon and point layers overlap.")
            try:
                gdf2.to_file(new_path, driver="ESRI Shapefile", engine="pyogrio",
                             geometry_type="Point")
            except Exception:
                schema = {"geometry": "Point", "properties": {
                    c: ("int" if pd.api.types.is_integer_dtype(gdf2[c].dtype)
                        else "float" if pd.api.types.is_float_dtype(gdf2[c].dtype) else "str")
                    for c in gdf2.columns if c != gdf2.geometry.name}}
                gdf2.to_file(new_path, driver="ESRI Shapefile", schema=schema)
        else:
            new_path = self._save_gdf(gdf2, new_path)
        self._emit(new_path, rec)

    @staticmethod
    def _drop_helpers(gdf):
        return gdf.drop(columns=[c for c in ['EASTING', 'NORTHING', '_eff_dipdir']
                                 if c in gdf.columns])

    def stochastic(self, rec, dip, dipdir, is_dipdir, percent):
        gdf, engine, _, _ = self._setup_points(rec, dip, dipdir, is_dipdir, check_fields=False)
        tmpdir = tempfile.mkdtemp()
        try:
            gdf2 = engine.stochastic(gdf, frac=percent / 100.0,
                                     random_state=int(time.time()), path_out=tmpdir)
            gdf2 = gdf2.set_crs(gdf.crs, allow_override=True)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        self._finalise_points(self._drop_helpers(gdf2), rec, "stochastic", percent)

    # --- grid-cell family (grid cell average, Kent, Kent outlier) ---
    def _count_gridcell_output(self, run_fn, gdf, bounds, cs):
        min_x, max_x, min_y, max_y = bounds
        tmpdir = tempfile.mkdtemp()
        try:
            file = run_fn(gdf, min_x, max_x, min_y, max_y, n=cs, path_out=tmpdir)
            df = pd.read_csv(os.path.join(tmpdir, file + ".csv"))
            return int((df['DIP'] != -999).sum()) if not df.empty else 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _find_grid_size_for_target(self, run_fn, gdf, bounds, target_n):
        """Cell size giving ~target_n points (within 15%): estimate, correct, then bisect."""
        min_x, max_x, min_y, max_y = bounds
        area = (max_x - min_x) * (max_y - min_y)
        tol = 0.15

        def count(cs):
            n = self._count_gridcell_output(run_fn, gdf, bounds, cs)
            self.log(f"  [target N] cell={cs:.2f} -> {n} points")
            return n

        cs1 = math.sqrt(area / max(1, target_n))
        n1 = count(cs1)
        if abs(n1 - target_n) <= tol * target_n:
            return cs1, n1
        cs2 = cs1 * math.sqrt(max(1, n1) / max(1, target_n))
        n2 = count(cs2)
        if abs(n2 - target_n) <= tol * target_n:
            return cs2, n2

        if cs1 < cs2:
            lo, lo_n, hi, hi_n = cs1, n1, cs2, n2
        else:
            lo, lo_n, hi, hi_n = cs2, n2, cs1, n1
        for _ in range(8):
            if lo_n >= target_n:
                break
            lo /= 2
            lo_n = count(lo)
        for _ in range(8):
            if hi_n <= target_n:
                break
            hi *= 2
            hi_n = count(hi)

        best_cs, best_n = (cs2, n2) if abs(n2 - target_n) < abs(n1 - target_n) else (cs1, n1)
        for _ in range(12):
            if hi - lo < 0.1:
                break
            mid = (lo + hi) / 2
            mid_n = count(mid)
            if abs(mid_n - target_n) < abs(best_n - target_n):
                best_cs, best_n = mid, mid_n
            if abs(mid_n - target_n) <= tol * target_n:
                break
            if mid_n > target_n:
                lo = mid
            else:
                hi = mid
        return best_cs, best_n

    def grid_method(self, kind, rec, dip, dipdir, is_dipdir, grid_size, target_n=None):
        """
        kind: "gridcell_average" | "spherical_kent" | "outlier_removal".
        Returns the grid size used (layer units / metres as displayed) so the GUI can show it.
        """
        gdf, engine, dip_col, dipdir_col = self._setup_points(rec, dip, dipdir, is_dipdir)
        run_fn = getattr(engine, kind)
        label = {"gridcell_average": "gridCellAveraging", "spherical_kent": "grid_cell_kent",
                 "outlier_removal": "grid_cell_kentOutlier"}[kind]
        geographic = bool(rec.get("geographic"))
        b = gdf.total_bounds
        bounds = (int(b[0]), int(b[2]) + 1, int(b[1]), int(b[3]) + 1)

        if target_n:
            size, _ = self._find_grid_size_for_target(run_fn, gdf, bounds, target_n)
            display = size * DEG_PER_M_INV if geographic else size
        else:
            display = float(grid_size)
            size = display / DEG_PER_M_INV if geographic else display

        tmpdir = tempfile.mkdtemp()
        try:
            file = run_fn(gdf, *bounds, n=size, path_out=tmpdir)
            gdf2 = save_grid_to_shapefile(tmpdir, file)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if gdf2 is not None:
            gdf2 = gdf2.set_crs(gdf.crs, allow_override=True)
            gdf2 = gdf2.drop(columns=['EASTING', 'NORTHING'], errors='ignore').rename(
                columns={'DIP': dip_col, 'DIP_DIR': dipdir_col})
            if not is_dipdir and dipdir_col in gdf2.columns:
                # engines average dip direction; convert back to strike
                gdf2 = gdf2.copy()
                gdf2[dipdir_col] = dipdir_to_strike(gdf2[dipdir_col], gdf.get(dipdir_col))
            self._finalise_points(gdf2, rec, label, display)
        return display

    def first_order(self, point_rec, poly_rec, dip, dipdir, is_dipdir, distance, angle):
        if point_rec is None or poly_rec is None:
            raise GSSError("1o Sampling needs both a points layer and a polygon layer.")
        if not dip or not dipdir:
            raise GSSError("First Order: please select DIP and DIP_DIR fields before running.")
        geology = gpd.read_file(self.resolved_source(poly_rec))
        contact_gdf = geology[['geometry']].copy()
        contact_gdf['geometry'] = geology.boundary
        orient = gpd.read_file(self.resolved_source(point_rec))
        dip_col = self.resolved_field(point_rec, dip)
        dipdir_col = self.resolved_field(point_rec, dipdir)
        eff = dipdir_col
        if not is_dipdir:
            orient['_eff_dipdir'] = (orient[dipdir_col].astype(float) - 90) % 360
            eff = '_eff_dipdir'
        sub = SubsamplingEngine(dip=dip_col, dipdir=eff)
        out = sub.subsample("firstorder", orient, path_out="", contact_gdf=contact_gdf,
                            dist_buffer=distance, angle_tol=angle)
        if out is None:
            return
        out = out.drop(columns=['_eff_dipdir'], errors='ignore')
        self._finalise_points(out, point_rec, f"first_order_{int(distance)}_{int(angle)}")

    # ------------------------------------------------------------------ polygons
    def rescale_map(self, poly_rec, p):
        """
        Remove polygons below a diameter threshold (km). p keys: node_tolerance, diameter_km,
        series, increment_km, prio (list of 4 fields), litho, ignore_field, ignore_codes.
        """
        if poly_rec is None:
            raise GSSError("Select a polygon layer first.")
        dist = float(p["node_tolerance"])
        strat = [self.resolved_field(poly_rec, f) for f in p["prio"]]
        lithoname = self.resolved_field(poly_rec, p["litho"])
        dyke_field = self.resolved_field(poly_rec, p["ignore_field"])
        dyke_codes = parse_codes(p["ignore_codes"])
        max_scale = float(p["diameter_km"])
        if p["series"]:
            inc, min_scale = float(p["increment_km"]), 0.0
        else:
            inc, min_scale = max_scale, max_scale
        geographic = bool(poly_rec.get("geographic"))
        if geographic:
            dist = dist / DEG_PER_M_INV

        gdf = None
        for up_scale in np.arange(min_scale, max_scale + 0.001, inc):
            min_area = ((up_scale / 2) ** 2) * np.pi          # km2
            if geographic:
                min_area = min_area / (DEG_PER_M_INV ** 2)
            new_path = self._make_output_path(poly_rec, "_min_area_" + str(up_scale))
            if gdf is None:
                gdf = gpd.read_file(self.resolved_source(poly_rec))

            if dyke_field and not dyke_codes:
                self.warn("Ignore field is set but the Ignore Codes box is empty - "
                          "no polygons will be triangulated.")
            if dyke_field and dyke_codes:
                tri = PolygonTriangulator(
                    gdf=gdf, id_column=dyke_field, min_area_threshold=min_area * 1e6,
                    distance_threshold=dist, strat1=strat[0] or None, strat2=strat[1] or None,
                    strat3=strat[2] or None, strat4=strat[3] or None,
                    lithoname=lithoname or None)
                gdf = tri.triangulate_polygons(target_ids=dyke_codes)
                self._report_ignore_codes(tri, gdf)

            sub = StructuralPolygonSubSampler(gdf)
            out = sub.clean_small_polygons_and_holes_new(
                gdf, min_area_threshold=min_area * 1e6, distance_threshold=dist,
                strat1=strat[0] or None, strat2=strat[1] or None,
                strat3=strat[2] or None, strat4=strat[3] or None,
                lithoname=lithoname or None)
            new_path = self._save_gdf(out, new_path)
            self._emit(new_path, poly_rec)

    def _report_ignore_codes(self, tri, gdf):
        s = getattr(tri, 'last_summary', None)
        if s is None:
            return
        if s['codes_not_found']:
            avail = sorted(set(str(v) for v in gdf[tri.id_column].unique()))
            self.warn(f"Ignore codes not found in field '{tri.id_column}': "
                      f"{', '.join(s['codes_not_found'])}. Values present: "
                      f"{', '.join(avail[:15])}{' ...' if len(avail) > 15 else ''}")
        if s.get('failed'):
            self.warn(f"Ignore codes: {s['failed']} polygon(s) could not be triangulated and "
                      f"were left unchanged.")
        if s['matched'] and not s['triangulated'] and not s.get('failed'):
            self.warn(f"Ignore codes matched {s['matched']} polygon(s), but none is smaller than "
                      f"the area threshold, so none was triangulated.")
        elif s['triangulated']:
            self.log(f"Ignore codes: {s['triangulated']} of {s['matched']} matching polygon(s) "
                     f"triangulated ({s['above_threshold']} above the area threshold left as is).")

    def simplify_map(self, poly_rec, fault_rec, area_tolerance):
        if poly_rec is None or fault_rec is None:
            raise GSSError("Map simplification: select both a polygon layer and a fault layer.")
        area_tolerance = int(area_tolerance)
        input_file = self.resolved_source(poly_rec)
        fault_file = self.resolved_source(fault_rec)
        output_file = input_file.replace(".shp", f"_MVW_{area_tolerance}.shp")
        fault_out = fault_file.replace(".shp", f"_sn_si_{area_tolerance}.shp")
        vector_simplify_file_two_stage(
            input_file=input_file, output_file=output_file,
            method="modified_visvalingam_whyatt", threshold=area_tolerance,
            fault_file=fault_file)
        self._emit(output_file, poly_rec)
        self._emit(fault_out, fault_rec)

    # ------------------------------------------------------------------ faults
    def merge_segments(self, fault_rec, tolerance, search_angle, join_angle):
        if fault_rec is None:
            raise GSSError("Select a fault layer first.")
        merger = FaultLineMerger(distance_tolerance=float(tolerance),
                                 angle_tolerance=float(search_angle),
                                 min_join_angle=float(join_angle))
        new_path = self._make_output_path(fault_rec, "_merged")
        gdf = gpd.read_file(self.resolved_source(fault_rec))
        out = merger.process_shapefile(gdf)
        self._emit(self._save_gdf(out, new_path), fault_rec)

    def process_faults(self, method, fault_rec, poly_rec=None, prio_fields=()):
        """method: "length" | "graph" | "strat_offset" | "clusters"."""
        if fault_rec is None:
            raise GSSError("Select a fault layer first.")
        src = self.resolved_source(fault_rec)

        if method == "length":
            out = src.replace(".shp", "_length.shp")
            FaultLengths().add_polyline_length(
                src, output_path=out, length_field="line_len",
                unit="degrees" if fault_rec.get("geographic") else "meters",
                force_projection=None)
            self._emit(out, fault_rec)

        elif method == "graph":
            FaultsGraph().CalcFaultsGraph(src)
            directory, filename = os.path.split(src)
            self._emit(os.path.join(directory, "simplified_full_" +
                                    filename.replace(".shp", "_edges.shp")), fault_rec)

        elif method == "strat_offset":
            if poly_rec is None:
                raise GSSError("Strat Offset needs a polygon layer as well as the fault layer.")
            cols = [self.resolved_field(poly_rec, f) for f in prio_fields if f]
            if not cols:
                raise GSSError("Fault Strat Offset: select at least one priority field.")
            out = src.replace(".shp", "_stratOffset.shp")
            FaultStratOffset().CalcFaultStratOffset(
                src, self.resolved_source(poly_rec), out, cols, offset_distance=50)
            self._emit(out, fault_rec)

        elif method == "clusters":
            out_az = src.replace(".shp", "_endpt_az.shp")
            fo = FaultsOrientations()
            fo.add_endpoint_azimuth(src, out_az, azimuth_field="endpt_az")
            safe_name = safe_filename(fault_rec["name"])
            best_n = fo.example_manual_clusters(out_az, safe_name, azimuth_field="endpt_az")
            if best_n is None:
                raise GSSError("Could not cluster faults - too few faults with distinct "
                               "orientations.")
            self._emit(os.path.join(os.path.dirname(out_az),
                                    safe_name + f"_fault_clusters_{best_n}.shp"), fault_rec)
        else:
            raise GSSError(f"Unknown fault method '{method}'.")

    # ------------------------------------------------------------------ Topfer & Pillewizer
    def _tp_prepare_fault_attrs(self, gdf, method, poly_rec, strat_cols):
        """Make sure gdf carries the attributes the chosen T&P fault method needs."""
        if method == 'graph' and 'edge_type' not in gdf.columns:
            tmpdir = tempfile.mkdtemp()
            try:
                tmp_in = os.path.join(tmpdir, 'tp_graph.shp')
                gdf.to_file(tmp_in)
                FaultsGraph().CalcFaultsGraph(tmp_in)
                tmp_edges = os.path.join(tmpdir, 'simplified_full_tp_graph_edges.shp')
                if os.path.exists(tmp_edges):
                    return gpd.read_file(tmp_edges).set_crs(gdf.crs, allow_override=True)
            except Exception as exc:
                self.warn(f"T&P: fault graph pre-compute failed: {exc}")
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        elif method == 'clusters' and not any('cluster' in c.lower() for c in gdf.columns):
            tmpdir = tempfile.mkdtemp()
            try:
                tmp_in = os.path.join(tmpdir, 'tp_cl.shp')
                tmp_az = os.path.join(tmpdir, 'tp_cl_az.shp')
                gdf.to_file(tmp_in)
                fo = FaultsOrientations()
                fo.add_endpoint_azimuth(tmp_in, tmp_az, azimuth_field='endpt_az')
                best_gdf, best_score = None, -1
                for n in range(2, 8):
                    try:
                        res = fo.analyze_shapefile_trends(
                            shapefile_path=tmp_az, trend_field='endpt_az', n_clusters=n)
                        sc = res.get('silhouette_score', -1)
                        if sc > best_score:
                            best_score, best_gdf = sc, res.get('gdf_with_clusters')
                    except Exception:
                        pass
                if best_gdf is not None:
                    return best_gdf.set_crs(gdf.crs, allow_override=True)
            except Exception as exc:
                self.warn(f"T&P: fault cluster pre-compute failed: {exc}")
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        elif method == 'strat_offset':
            has_strat = ('max_off_lv' in gdf.columns or any(
                'strat' in c.lower() or 'offset' in c.lower() for c in gdf.columns))
            if not has_strat and strat_cols and poly_rec is not None:
                tmpdir = tempfile.mkdtemp()
                try:
                    tmp_in = os.path.join(tmpdir, 'tp_so.shp')
                    tmp_out = os.path.join(tmpdir, 'tp_so_out.shp')
                    gdf.to_file(tmp_in)
                    FaultStratOffset().CalcFaultStratOffset(
                        tmp_in, self.resolved_source(poly_rec), tmp_out, strat_cols,
                        offset_distance=50)
                    if os.path.exists(tmp_out):
                        return gpd.read_file(tmp_out).set_crs(gdf.crs, allow_override=True)
                except Exception as exc:
                    self.warn(f"T&P: strat-offset pre-compute failed: {exc}")
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
        return gdf

    def topfer_pillewizer(self, point_rec, fault_rec, poly_rec, p):
        """
        Töpfer & Pillewizer scaling of whichever layers are given.
        p keys: ratio, increment, dip, dipdir, is_dipdir, point_method, fault_method,
                prio (4 fields), litho, ignore_field, ignore_codes, node_tolerance,
                grid_size, fo_distance, fo_angle
        Returns the path of the CSV report (or None).
        """
        if point_rec is None and fault_rec is None and poly_rec is None:
            raise GSSError("Select at least one layer.")
        ratio, increment = float(p["ratio"]), float(p["increment"])
        if ratio <= 0 or ratio > 1:
            raise GSSError("T&P Scaling: OS/TS ratio must be in the range (0, 1].")
        iterations = scale_iterations(ratio, increment)
        point_method, fault_method = p["point_method"], p["fault_method"]

        strat = [self.resolved_field(poly_rec, f) if poly_rec else f for f in p["prio"]]
        lithoname = self.resolved_field(poly_rec, p["litho"]) if poly_rec else p["litho"]
        dyke_field = self.resolved_field(poly_rec, p["ignore_field"]) if poly_rec \
            else p["ignore_field"]
        dyke_codes = parse_codes(p["ignore_codes"])
        try:
            dist_thresh = float(p["node_tolerance"])
        except ValueError:
            dist_thresh = 1.0

        cur_fault = cur_poly = None
        if fault_rec is not None:
            cur_fault = gpd.read_file(self.resolved_source(fault_rec))
            cur_fault = self._tp_prepare_fault_attrs(
                cur_fault, fault_method, poly_rec, [c for c in strat if c])
        if poly_rec is not None:
            cur_poly = gpd.read_file(self.resolved_source(poly_rec))
            if poly_rec.get("geographic"):
                dist_thresh = dist_thresh / DEG_PER_M_INV

        original_pt = pt_engine = pt_dip = pt_dipdir = None
        if point_rec is not None:
            original_pt, pt_engine, pt_dip, pt_dipdir = self._setup_points(
                point_rec, p["dip"], p["dipdir"], p["is_dipdir"])
        try:
            grid_sz = float(p["grid_size"])
        except ValueError:
            grid_sz = 5000.0

        fo_kwargs = None
        if point_method == 'firstorder' and original_pt is not None:
            if cur_poly is None:
                raise GSSError("T&P with 1o Sampling needs a polygon layer for the contacts.")
            contacts = cur_poly[['geometry']].copy()
            contacts['geometry'] = cur_poly.boundary
            fo_kwargs = {'contact_gdf': contacts, 'dist_buffer': float(p["fo_distance"]),
                         'angle_tol': float(p["fo_angle"])}

        rows = []
        orig = {'points': len(original_pt) if original_pt is not None else None,
                'faults': len(cur_fault) if cur_fault is not None else None,
                'polygons': len(cur_poly) if cur_poly is not None else None}
        exponent = {'points': 1, 'faults': 2, 'polygons': 3}

        def add_report(step, r, step_r, layer, method, n_in, n_out, info):
            rows.append({
                'step': step, 'OS/TS ratio': r, 'step ratio': step_r, 'layer': layer,
                'method': method, 'input count': n_in,
                'target count (from original)': topfer_count(orig[layer], r, exponent[layer]),
                'output count': n_out,
                'parameters used': '; '.join(
                    f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in info.items() if k != 'method')})

        # ratio = 1 baseline
        if original_pt is not None:
            self._finalise_points(self._drop_helpers(original_pt), point_rec, "tp_1")
            add_report('baseline', 1.0, 1.0, 'points', 'none (all kept)',
                       len(original_pt), len(original_pt), {})
        if cur_fault is not None:
            full = scale_lines_tp(cur_fault, 1.0, fault_method)
            add_report('baseline', 1.0, 1.0, 'faults', fault_method,
                       len(cur_fault), len(full), {})
            self._emit(self._save_gdf(full, self._make_output_path(fault_rec, "_tp_1")),
                       fault_rec)
        if cur_poly is not None:
            self._emit(self._save_gdf(cur_poly, self._make_output_path(poly_rec, "_tp_1")),
                       poly_rec)
            add_report('baseline', 1.0, 1.0, 'polygons', 'none (all kept)',
                       len(cur_poly), len(cur_poly), {})

        prev = 1.0
        for ir in iterations:
            step_ratio = ir / prev
            prev = ir
            tag = f"{ir:.6f}".rstrip('0').rstrip('.')
            if abs(ir - 1.0) < 1e-9:
                continue
            self.log(f"T&P ratio {tag} ...")

            if original_pt is not None:
                info = {}
                scaled = scale_points_tp(
                    original_pt, ir, pt_engine, point_method, pt_dip, pt_dipdir, grid_sz,
                    firstorder_kwargs=fo_kwargs, info=info, strike_input=not p["is_dipdir"])
                add_report(f"ratio {tag}", ir, ir, 'points', point_method,
                           len(original_pt), len(scaled), info)
                expected = topfer_count(len(original_pt), ir, x=1)
                if point_method == 'firstorder' and len(scaled) < expected:
                    self.warn(f"T&P ratio {ir}: only {len(scaled)} points pass the 1o "
                              f"distance/angle limits (target {expected}).")
                self._finalise_points(self._drop_helpers(scaled), point_rec, f"tp_{tag}")

            if cur_fault is not None:
                if fault_method == 'graph':
                    # topology changes as faults are removed, so recompute edge_type
                    cur_fault = cur_fault.drop(columns=['edge_type'], errors='ignore')
                    cur_fault = self._tp_prepare_fault_attrs(
                        cur_fault, fault_method, poly_rec, [c for c in strat if c])
                info, n_in = {}, len(cur_fault)
                scaled = scale_lines_tp(cur_fault, step_ratio, fault_method, info=info)
                add_report(f"ratio {tag}", ir, step_ratio, 'faults', fault_method,
                           n_in, len(scaled), info)
                cur_fault = scaled
                self._emit(self._save_gdf(
                    scaled, self._make_output_path(fault_rec, f"_tp_{tag}")), fault_rec)

            if cur_poly is not None:
                info, n_in = {}, len(cur_poly)
                try:
                    scaled = scale_polygons_tp(
                        cur_poly, step_ratio, StructuralPolygonSubSampler,
                        distance_threshold=dist_thresh, lithoname=lithoname or None,
                        strat1=strat[0] or None, strat2=strat[1] or None,
                        strat3=strat[2] or None, strat4=strat[3] or None,
                        dyke_field=dyke_field or None,
                        dyke_codes=dyke_codes if dyke_field else None,
                        triangulator_class=PolygonTriangulator if dyke_field else None,
                        info=info)
                    cur_poly = scaled
                except Exception as exc:
                    self.warn(f"T&P polygon scaling failed at ratio {ir}; "
                              f"unscaled polygons written: {exc}")
                    info['note'] = f"FAILED, unscaled polygons written: {exc}"
                    scaled = cur_poly.copy()
                add_report(f"ratio {tag}", ir, step_ratio, 'polygons',
                           'merge smallest into best-matching neighbour', n_in, len(scaled), info)
                self._emit(self._save_gdf(
                    scaled, self._make_output_path(poly_rec, f"_tp_{tag}")), poly_rec)

        settings = {
            'target OS/TS ratio': ratio, 'increment': increment,
            'points layer': point_rec["name"] if original_pt is not None else 'none',
            'point method': point_method if original_pt is not None else 'n/a',
            'fault layer': fault_rec["name"] if cur_fault is not None else 'none',
            'fault method': fault_method if cur_fault is not None else 'n/a',
            'polygon layer': poly_rec["name"] if poly_rec else 'none',
            'polygon priority fields': ', '.join(
                c for c in strat + [lithoname] if c) or 'none',
            'polygon node tolerance': dist_thresh}
        if fo_kwargs:
            settings['1o distance limit'] = fo_kwargs['dist_buffer']
            settings['1o angle limit'] = fo_kwargs['angle_tol']
        elif point_method in ('gridcell_average', 'spherical_kent', 'outlier_removal'):
            settings['grid size starting guess'] = grid_sz
        rows.insert(0, {'step': 'settings', 'OS/TS ratio': '', 'step ratio': '', 'layer': 'all',
                        'method': '', 'input count': '', 'target count (from original)': '',
                        'output count': '',
                        'parameters used': '; '.join(f"{k}={v}" for k, v in settings.items())})
        try:
            ref = next(r for r in (point_rec, fault_rec, poly_rec) if r is not None)
            report = os.path.join(self._output_directory(ref),
                                  f"TP_scaling_report_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            with open(report, 'w', newline='', encoding='utf-8-sig') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            self.log(f"T&P scaling complete. Report written to {report}")
            return report
        except Exception as exc:
            self.warn(f"T&P scaling finished but the report could not be written: {exc}")
            return None
