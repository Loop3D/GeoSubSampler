#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoSubSampler_GUI.py  -  Tkinter front end for GeoSubSampler (ArcGIS Pro).

Started by the "Launch GeoSubSampler GUI" tool in GeoSubSampler.pyt, which passes a JSON
file describing the layers in the open ArcGIS Pro project.  Results are appended to a
"pending" JSON file that the toolbox adds to the map as they appear.

Can also be run by hand:  python GeoSubSampler_GUI.py [info.json]
(with no file, layers can be added from disk with the "Add file..." buttons).
"""

import json
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TMP = tempfile.gettempdir()
PENDING_FILE = os.path.join(TMP, "geosubsampler_pending.json")
REFRESH_REQ = os.path.join(TMP, "geosubsampler_refresh_req.json")

BG, FG, SECTION_FG = "#f0f0f0", "#1c1c1c", "#0060b0"
FONT, FONT_BOLD, FONT_MONO = ("Segoe UI", 9), ("Segoe UI", 9, "bold"), ("Consolas", 8)

NONE = "(none)"


def section(parent, text):
    return tk.LabelFrame(parent, text=text, bg=BG, fg=SECTION_FG, font=FONT_BOLD,
                         relief="groove", bd=1, padx=4, pady=4)


def label(parent, text, **kw):
    return tk.Label(parent, text=text, bg=BG, fg=FG, font=FONT, **kw)


def entry(parent, var, width=8):
    return tk.Entry(parent, textvariable=var, width=width, font=FONT)


class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tw = widget, text, None
        widget.bind("<Enter>", self._show, add=True)
        widget.bind("<Leave>", self._hide, add=True)

    def _show(self, _=None):
        if self.tw or not self.text:
            return
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{self.widget.winfo_rootx() + 20}+"
                            f"{self.widget.winfo_rooty() + self.widget.winfo_height() + 4}")
        tk.Label(self.tw, text=self.text, bg="#ffffe0", fg=FG, font=("Segoe UI", 8),
                 relief="solid", bd=1, justify="left", padx=4, pady=2).pack()

    def _hide(self, _=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None


def tip(widget, text):
    """Attach (or replace) the tooltip of a widget."""
    existing = getattr(widget, "_gss_tip", None)
    if existing is not None:
        existing.text = text
    else:
        widget._gss_tip = ToolTip(widget, text)
    return widget


def _use_app_icon(app_id):
    """Before the first window: give this program its own Windows taskbar
    entry (app_id), so the taskbar shows its icon rather than python.exe's."""
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass  # not Windows, or an old Windows: the icon still shows on the window


def _set_window_icon(win, png):
    """The plugin's QGIS icon on this window and every window opened after it."""
    try:
        img = tk.PhotoImage(file=png)
        win.iconphoto(True, img)
        win._sgt_icon = img  # keep a reference, or Tk drops the image
    except Exception:
        pass


class App(tk.Tk):
    def __init__(self, info_file=None):
        _use_app_icon("WAXI.GeoSubSampler")
        super().__init__()
        _set_window_icon(self, os.path.join(_HERE, "..", "icon.png"))
        self.title("GeoSubSampler")
        self.configure(bg=BG)
        self.geometry("760x900")

        self.info_file = info_file
        self.layers = {"point": [], "line": [], "polygon": []}   # lists of layer records
        self.fallback_dir = None
        self.queue = queue.Queue()
        self.busy = False
        self.local_layers = []          # "Add file..." / result layers, kept across map refreshes

        self._load_info()
        self._build()
        self._refresh_combos()
        self.after(100, self._drain_queue)
        self.after(200, self._request_refresh)
        self.after(60000, self._periodic_refresh)

    def _periodic_refresh(self):
        """Pick up layers added to / removed from the map, as SGTool does (about every minute)."""
        if not self.busy:
            self._request_refresh()
        self.after(60000, self._periodic_refresh)

    # ------------------------------------------------------------------ layer info
    def _load_info(self):
        self.layers = {"point": [], "line": [], "polygon": []}
        for rec in self.local_layers:                       # files / results added in this session
            self.layers[rec["kind"]].append(rec)
        if not self.info_file or not os.path.exists(self.info_file):
            return
        try:
            with open(self.info_file) as f:
                info = json.load(f)
        except Exception:
            return
        aprx = info.get("aprx") or ""
        self.fallback_dir = os.path.dirname(aprx) if aprx else None
        local_sources = {r["source"] for r in self.local_layers}
        for rec in info.get("layers", []):
            if rec.get("kind") in self.layers and rec["source"] not in local_sources:
                self.layers[rec["kind"]].append(rec)

    def _rec(self, kind, var):
        """Layer record selected in combo `var`, or None."""
        name = var.get()
        for rec in self.layers[kind]:
            if rec["display"] == name:
                return rec
        return None

    @staticmethod
    def _field_names(rec, numeric_only=False):
        if rec is None:
            return []
        numeric = ("Double", "Integer", "SmallInteger", "Single", "BigInteger")
        return [n for n, t in rec["fields"] if not numeric_only or t in numeric]

    def _add_file(self, kind):
        """Add a shapefile from disk (stand-alone use, or data not in the map)."""
        path = filedialog.askopenfilename(filetypes=[("Shapefile", "*.shp"), ("All files", "*.*")])
        if not path:
            return
        try:
            import geopandas as gpd
            gdf = gpd.read_file(path, rows=1)
            gtype = str(gdf.geom_type.iloc[0]) if len(gdf) else ""
            found = ("point" if "Point" in gtype else "line" if "Line" in gtype
                     else "polygon" if "Polygon" in gtype else kind)
            geographic = bool(gdf.crs and gdf.crs.is_geographic)
            fields = [[c, "Double" if str(gdf[c].dtype).startswith(("float", "int")) else "String"]
                      for c in gdf.columns if c != "geometry"]
        except Exception as exc:
            messagebox.showerror("GeoSubSampler", f"Could not read {path}:\n{exc}")
            return
        if found != kind:
            messagebox.showwarning("GeoSubSampler", f"That file contains {found} features.")
        name = os.path.splitext(os.path.basename(path))[0]
        rec = {"name": name, "display": name + " [file]", "source": path,
               "kind": found, "geographic": geographic, "fields": fields}
        self.local_layers.append(rec)
        self.layers[found].append(rec)
        self._refresh_combos(select={found: name + " [file]"})

    # ------------------------------------------------------------------ UI
    def _build(self):
        canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="top", fill="both", expand=True)
        body = tk.Frame(canvas, bg=BG)
        win = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        self.bind_all("<MouseWheel>", lambda e: (
            None if isinstance(e.widget, tk.Text)
            else canvas.yview_scroll(int(-e.delta / 120), "units")))

        self._build_layers(body)
        self._build_orientation(body)
        self._build_faults(body)
        self._build_geology(body)
        self._build_tp(body)
        self._define_tips()

        # log
        logf = section(self, "Messages")
        logf.pack(side="bottom", fill="x", padx=6, pady=4)
        self.log_text = tk.Text(logf, height=8, font=FONT_MONO, wrap="word", state="disabled")
        self.log_text.pack(fill="x")
        self.log_text.tag_config("warning", foreground="#a05000")
        self.log_text.tag_config("error", foreground="#b00000")
        self.status = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status, bg=BG, fg=FG, font=FONT, anchor="w"
                 ).pack(side="bottom", fill="x", padx=8)

    def _build_layers(self, p):
        f = section(p, "Layers")
        f.pack(fill="x", padx=6, pady=4)
        self.v_points, self.v_faults, self.v_polys = (tk.StringVar() for _ in range(3))
        self.cb = {}
        for row, (text, kind, var) in enumerate([
                ("Structure Point Layer", "point", self.v_points),
                ("Fault Polyline Layer", "line", self.v_faults),
                ("Litho/strat Polygon Layer", "polygon", self.v_polys)]):
            label(f, text).grid(row=row, column=0, sticky="e", padx=2, pady=2)
            cb = ttk.Combobox(f, textvariable=var, state="readonly", width=55, font=FONT)
            cb.grid(row=row, column=1, sticky="we", padx=2)
            tk.Button(f, text="Add file...", font=FONT, command=lambda k=kind: self._add_file(k)
                      ).grid(row=row, column=2, padx=2)
            self.cb[kind] = cb
        tk.Button(f, text="Refresh from map", font=FONT, command=self._request_refresh
                  ).grid(row=3, column=1, sticky="w", padx=2, pady=2)
        f.columnconfigure(1, weight=1)
        self.v_points.trace_add("write", lambda *_: self._update_point_fields())
        self.v_polys.trace_add("write", lambda *_: self._update_poly_fields())

    def _build_orientation(self, p):
        f = section(p, "Orientation Points")
        f.pack(fill="x", padx=6, pady=4)
        r = tk.Frame(f, bg=BG)
        r.pack(fill="x")
        self.v_dip, self.v_dipdir = tk.StringVar(), tk.StringVar()
        self.v_is_dipdir = tk.BooleanVar(value=True)
        label(r, "Dip").pack(side="left")
        self.cb_dip = ttk.Combobox(r, textvariable=self.v_dip, state="readonly", width=14, font=FONT)
        self.cb_dip.pack(side="left", padx=4)
        label(r, "Dip Direction/Strike").pack(side="left")
        self.cb_dipdir = ttk.Combobox(r, textvariable=self.v_dipdir, state="readonly", width=14,
                                      font=FONT)
        self.cb_dipdir.pack(side="left", padx=4)
        tip(tk.Checkbutton(r, text="Dip Direction", variable=self.v_is_dipdir, bg=BG, font=FONT),
            "Ticked: the field holds dip direction. Unticked: it holds strike.").pack(side="left")

        self.v_method = tk.StringVar(value="stochastic")
        self.v_percent, self.v_grid_kent, self.v_grid_ko, self.v_grid_carm = (
            tk.StringVar(value=v) for v in ("50", "5000", "5000", "5000"))
        self.v_fo_dist, self.v_fo_angle = tk.StringVar(value="1000"), tk.StringVar(value="15")

        def radio(row, text, value, tiptext, fields):
            rr = tk.Frame(f, bg=BG)
            rr.pack(fill="x", pady=1)
            tip(tk.Radiobutton(rr, text=text, variable=self.v_method, value=value, bg=BG,
                               font=FONT, width=14, anchor="w"), tiptext).pack(side="left")
            for lbl_text, var in fields:
                label(rr, lbl_text).pack(side="left")
                entry(rr, var).pack(side="left", padx=3)

        radio(0, "Stochastic", "stochastic", "Keep a randomly selected percentage of points",
              [("Percent kept (%)", self.v_percent)])
        self.v_kappa_min = tk.StringVar(value="")
        radio(2, "Kent", "spherical_kent", "Grid cell averaging using Kent statistics",
              [("Grid Size (m)", self.v_grid_kent), ("Min kappa", self.v_kappa_min)])
        self.v_ko_iter = tk.BooleanVar(value=False)
        radio(3, "Kent Outlier", "outlier_removal",
              "Grid cell averaging using Kent statistics, removing the biggest outlier",
              [("Grid Size (m)", self.v_grid_ko)])
        tk.Checkbutton(f.winfo_children()[-1], text="Iterative", variable=self.v_ko_iter,
                       bg=BG, font=FONT).pack(side="left", padx=6)
        radio(4, "Carmichael", "outlier_carmichael",
              "Grid cell averaging using Kent statistics, removing several outliers per cell",
              [("Grid Size (m)", self.v_grid_carm)])

        tr = tk.Frame(f, bg=BG)
        tr.pack(fill="x", pady=1)
        self.v_target_on, self.v_target_n = tk.BooleanVar(), tk.StringVar(value="100")
        tip(tk.Checkbutton(tr, text="Auto-fit to N pts", variable=self.v_target_on, bg=BG,
                           font=FONT), "Kent, Kent Outlier and Carmichael only: search for the "
            "grid size that gives about N points (within 15%). Slower - several trial runs."
            ).pack(side="left", padx=(20, 4))
        entry(tr, self.v_target_n, 6).pack(side="left")
        label(tr, "points (+/-15%)").pack(side="left", padx=3)

        radio(5, "1o Sampling", "firstorder",
              "First order retention of points by distance and angle to contacts.\n"
              "Uses the boundaries of the polygon layer as contacts.",
              [("Distance", self.v_fo_dist), ("Angle", self.v_fo_angle)])

        tk.Button(f, text="Subsample Points", font=FONT_BOLD, width=20,
                  command=lambda: self._run("Point subsampling", self._do_subsample)
                  ).pack(anchor="e", pady=4)

    def _build_faults(self, p):
        f = section(p, "Fault Polylines")
        f.pack(fill="x", padx=6, pady=4)
        r = tk.Frame(f, bg=BG)
        r.pack(fill="x")
        self.v_mtol, self.v_msearch, self.v_mjoin = (tk.StringVar(value=v)
                                                     for v in ("10", "30", "150"))
        tip(tk.Button(r, text="Merge Segments", font=FONT, width=16,
                      command=lambda: self._run("Fault segment merging", self._do_merge)),
            "Merge connected polylines\nWARNING: VERY SLOW FOR LARGE DATASETS (>1s per object)"
            ).pack(side="left")
        for text, var in (("Dist.(m)", self.v_mtol), ("Search Angle", self.v_msearch),
                          ("Join Angle", self.v_mjoin)):
            label(r, text).pack(side="left", padx=(8, 0))
            entry(r, var, 5).pack(side="left", padx=2)
        ttk.Separator(f).pack(fill="x", pady=4)

        self.v_fault_method = tk.StringVar(value="length")
        for text, value, tp in (
                ("Length Only", "length", "Add fault length attribute"),
                ("Graph then Length", "graph", "Add graph data to faults"),
                ("Strat Offset then Length  [requires Geology Polygon Priorities]", "strat_offset",
                 "Needs both fault and polygon layers and the priority fields below"),
                ("Orientation Clusters then Length", "clusters",
                 "Add suite of orientation cluster data to faults (2-7 clusters)")):
            tip(tk.Radiobutton(f, text=text, variable=self.v_fault_method, value=value, bg=BG,
                               font=FONT, anchor="w"), tp).pack(fill="x")
        tk.Button(f, text="Process Faults", font=FONT_BOLD, width=20,
                  command=lambda: self._run("Fault processing", self._do_faults)
                  ).pack(anchor="e", pady=4)

    def _build_geology(self, p):
        f = section(p, "Geology Polygons")
        f.pack(fill="x", padx=6, pady=4)
        self.v_prio = [tk.StringVar() for _ in range(5)]
        self.v_ignore_field = tk.StringVar()
        self.cb_prio = []
        grid = tk.Frame(f, bg=BG)
        grid.pack(fill="x")
        names = ["Priority 1", "Priority 2", "Priority 3", "Priority 4", "Priority 5 (litho)"]
        for i, (nm, var) in enumerate(zip(names, self.v_prio)):
            label(grid, nm).grid(row=i // 3, column=(i % 3) * 2, sticky="e", padx=2, pady=1)
            cb = ttk.Combobox(grid, textvariable=var, state="readonly", width=14, font=FONT)
            cb.grid(row=i // 3, column=(i % 3) * 2 + 1, padx=2)
            tip(cb, f"Field for priority {i + 1} of polygon merging and fault strat offset")
            self.cb_prio.append(cb)
        label(grid, "Ignore Field").grid(row=1, column=4, sticky="e", padx=2)
        self.cb_ignore = ttk.Combobox(grid, textvariable=self.v_ignore_field, state="readonly",
                                      width=14, font=FONT)
        self.cb_ignore.grid(row=1, column=5, padx=2)

        r = tk.Frame(f, bg=BG)
        r.pack(fill="x", pady=2)
        label(r, "Ignore Codes (comma sep)").pack(side="left")
        self.codes = tk.Text(r, height=2, width=50, font=FONT)
        self.codes.pack(side="left", padx=4)

        r = tk.Frame(f, bg=BG)
        r.pack(fill="x", pady=2)
        self.v_tol, self.v_diam, self.v_inc = (tk.StringVar(value=v) for v in ("1", "5", "1.0"))
        self.v_series = tk.BooleanVar()
        for text, var in (("Tolerance (m)", self.v_tol), ("Diameter Threshold (km)", self.v_diam)):
            label(r, text).pack(side="left")
            entry(r, var, 6).pack(side="left", padx=3)
        tip(tk.Checkbutton(r, text="Series", variable=self.v_series, bg=BG, font=FONT),
            "Calculate the full series of rescaled maps between increment and threshold"
            ).pack(side="left", padx=4)
        label(r, "Increment (km)").pack(side="left")
        entry(r, self.v_inc, 6).pack(side="left", padx=3)
        tk.Button(f, text="Rescale Map", font=FONT_BOLD, width=20,
                  command=lambda: self._run("Polygon cleaning", self._do_rescale)
                  ).pack(anchor="e", pady=2)

        r = tk.Frame(f, bg=BG)
        r.pack(fill="x", pady=2)
        self.v_area_tol = tk.StringVar(value="10000")
        label(r, "Area Tolerance (m2)").pack(side="left")
        entry(r, self.v_area_tol, 9).pack(side="left", padx=3)
        tip(tk.Button(r, text="Simplify Map", font=FONT_BOLD, width=20,
                      command=lambda: self._run("Map simplification", self._do_simplify)),
            "Simplify map polygons by area tolerance, respecting fault topology"
            ).pack(side="right")

    def _build_tp(self, p):
        f = section(p, "Töpfer & Pillewizer Scaling")
        f.pack(fill="x", padx=6, pady=4)
        tip(tk.Label(f, text="TN = ON × (OS/TS)^(x/2)    x: 1=points  2=lines  3=polygons",
                     bg=BG, fg=FG, font=("Segoe UI", 9, "italic")),
            "Töpfer & Pillewizer (1966) selection equation\nTN target number of objects, "
            "ON original number,\nOS/TS original/target scale denominators").pack()
        r = tk.Frame(f, bg=BG)
        r.pack(fill="x", pady=2)
        self.v_ratio, self.v_incr = tk.StringVar(value="0.1"), tk.StringVar(value="0.1")
        self.v_steps = tk.StringVar()
        label(r, "OS/TS ratio").pack(side="left")
        entry(r, self.v_ratio, 7).pack(side="left", padx=3)
        label(r, "Increment").pack(side="left")
        entry(r, self.v_incr, 7).pack(side="left", padx=3)
        label(r, "", textvariable=self.v_steps).pack(side="left", padx=4)
        tk.Button(r, text="Run T&P Scaling", font=FONT_BOLD, width=20,
                  command=lambda: self._run("Töpfer & Pillewizer scaling", self._do_tp)
                  ).pack(side="right")
        self.v_ratio.trace_add("write", lambda *_: self._update_steps())
        self.v_incr.trace_add("write", lambda *_: self._update_steps())
        self._update_steps()
        label(f, "Uses the orientation method, fault method and polygon settings chosen above; "
                 "any layer left as (none) is skipped.").pack(anchor="w")

    # ------------------------------------------------------------------ tooltips
    def _define_tips(self):
        """The tooltips of the QGIS plugin (GeoSubSampler.defineTips), applied by widget."""
        by_var = {                       # entries / drop-downs, keyed by their Tk variable's name
            # (a Tk variable itself is unhashable, so it cannot be a dict key)
            str(self.v_points): "Points layer selected for processing",
            str(self.v_faults): "Fault polyline layer selected for processing",
            str(self.v_polys): "Layer map polygons selected for processing",
            str(self.v_dip): "Field containing Dip information",
            str(self.v_dipdir): "Field containing Dip Direction or Strike information",
            str(self.v_percent): "Percentage of points to keep, between 0 and 100",
            str(self.v_grid_kent): "Grid size in layer units for cell averaging using Kent statistics",
            str(self.v_grid_ko): "Grid size in layer units for Kent outlier averaging",
            str(self.v_grid_carm): "Grid size in layer units for Carmichael outlier averaging",
            str(self.v_kappa_min): (
                "Kent only. Keep a cell only if its Kent kappa (concentration) is at least this "
                "value.\nDrops cells with a single measurement and cells whose orientations are "
                "scattered.\nBlank (or 0) keeps every cell. The thesis models used 5."),
            str(self.v_target_n): (
                "Desired number of output points (used when 'Auto-fit to N pts' is ticked). "
                "The result is approximate (±15%) and cannot exceed the number of occupied "
                "grid cells."),
            str(self.v_fo_dist): "Distance buffer for first order retention of points",
            str(self.v_fo_angle): "Angle tolerance for first order retention of points",
            str(self.v_tol): "Tolerance for node snapping in layer units",
            str(self.v_ignore_field): "Field selected for ignore codes",
            str(self.v_diam): "Minimum threshold diameter of polygons to be retained (in km)",
            str(self.v_inc): "Increment for arithmetic series of maps",
            str(self.v_mtol): "Minimum distance between end points",
            str(self.v_msearch): "Minimum angle mismatch between end segments",
            str(self.v_mjoin): "Maximum angle for newly-formed angle defined by end segments",
            str(self.v_area_tol): "Tolerance for area-based filtering, in m2",
            str(self.v_ratio): "OS/TS ratio — e.g. 0.1 for going from 1:50 000 to 1:500 000",
            str(self.v_incr): (
                "Fractional reduction per step (0–1).\n"
                "Step n ratio = (1 - increment)^n\n"
                "e.g. increment=0.1 → steps 0.9, 0.81, 0.729 …\n"
                "Stops after the first step that drops below the OS/TS ratio."),
        }
        ordinals = ["1st", "2nd", "3rd", "4th", "5th"]
        for n, var in enumerate(self.v_prio):
            by_var[str(var)] = (f"Field selected for {ordinals[n]} priority of polygon merging "
                           f"and fault strat offset")

        by_text = {                      # buttons / radio buttons / check boxes, keyed by label
            "Dip Direction": "Dip Direction used instead of Strike \nProbably not needed?",
            "Stochastic": "Keep a randomly selected percentage of points",
            "Kent": ("Grid Cell Averaging of points using Kent statistics.\n"
                     "Output points sit at the mean position of each cell's measurements, with the\n"
                     "vector-mean orientation plus count, Kent kappa (concentration) and beta (ovalness)."),
            "Kent Outlier": "Grid Cell Averaging using Kent statistics, removing biggest outlier",
            "Carmichael": (
                "Grid Cell Averaging using Kent statistics, removing several outliers per cell\n"
                "(Carmichael & Ailleres 2016). Points are ranked by angle from the cell mean and\n"
                "removed one at a time while Kent kappa is recomputed; everything up to the\n"
                "biggest jump in kappa is treated as outliers. Cells with 3 or fewer points are "
                "skipped."),
            "Subsample Points": "Run the selected subsampling method on the point layer",
            "Auto-fit to N pts": (
                "Kent, Kent Outlier and Carmichael only.\n"
                "Instead of using the Grid Size box, search for the grid cell size that gives "
                "approximately N output points (within ±15%). The grid size found is written "
                "back into the Grid Size box of the selected method.\n"
                "Several trial runs are needed, so this is slower. Ignored by Stochastic and "
                "1o Sampling."),
            "1o Sampling": (
                "First order retention of points based on distance and angle to contacts.\n"
                "Requires a polygon layer (selected in the map polygons box) that overlaps the "
                "points - its boundaries are used as the contacts."),
            "Series": "Calculate full series of rescaling maps between increment and threshold",
            "Iterative": (
                "Kent Outlier only. Remove several outliers per cell instead of one: points are taken\n"
                "out in order of how much their removal raises Kent kappa (up to half the cell), and\n"
                "the sequence is cut at its biggest kappa jump - only if that jump stands out (5x the\n"
                "typical step), otherwise nothing is removed. Differs from Carmichael, which orders\n"
                "points by angle from the cell mean. Not applied by the Töpfer & Pillewizer tool."),
            "Rescale Map": "Rescale map polygons",
            "Merge Segments": ("Merge connected polylines\n"
                               "WARNING: VERY SLOW FOR LARGE DATASETS (>1s per object)"),
            "Length Only": "Add fault length attribute\nDoes not filter data, just adds length "
                           "attribute",
            "Graph then Length": "Add graph data to faults\nDoes not filter data, just adds "
                                 "graph attributes",
            "Orientation Clusters then Length": (
                "Add suite of orientation cluster data to faults (2-7 clusters)\n"
                "Does not filter data, just adds cluster attributes"),
            "Process Faults": "Run the selected fault attribute method",
            "Run T&P Scaling": ("Apply Töpfer & Pillewizer scaling to all selected layers.\n"
                                "Uses the orientation and fault methods selected above."),
            "Simplify Map": ("Simplify map polygons based on area tolerance, respecting fault "
                             "topology"),
        }
        strat_tip = ("Add stratigraphic offset data to faults\n"
                     "Requires both fault and map polygon layers\n"
                     "as well as Geology Map Priority Codes (see below) to be defined\n"
                     "Does not filter data, just adds Strat Offset attributes")

        def walk(w):
            yield w
            for c in w.winfo_children():
                yield from walk(c)

        for w in walk(self):
            cls = w.winfo_class()
            try:
                if cls in ("Entry", "TCombobox"):
                    text = by_var.get(str(w.cget("textvariable")))
                elif cls in ("Button", "Radiobutton", "Checkbutton"):
                    label_text = w.cget("text")
                    text = (strat_tip if label_text.startswith("Strat Offset")
                            else by_text.get(label_text))
                else:
                    continue
            except tk.TclError:
                continue
            if text:
                tip(w, text)
        tip(self.codes, "Comma separated list of ignore codes for special handling")

    # ------------------------------------------------------------------ combo plumbing
    def _refresh_combos(self, select=None):
        for kind, var in (("point", self.v_points), ("line", self.v_faults),
                          ("polygon", self.v_polys)):
            names = [NONE] + [r["display"] for r in self.layers[kind]]
            self.cb[kind]["values"] = names
            want = (select or {}).get(kind)
            if want in names:
                var.set(want)
            elif var.get() not in names:
                var.set(names[1] if len(names) > 1 else NONE)
        self._update_point_fields()
        self._update_poly_fields()

    def _update_point_fields(self):
        rec = self._rec("point", self.v_points)
        names = [""] + self._field_names(rec, numeric_only=True)
        self.cb_dip["values"] = self.cb_dipdir["values"] = names
        for var in (self.v_dip, self.v_dipdir):
            if var.get() not in names:
                var.set("")

    def _update_poly_fields(self):
        rec = self._rec("polygon", self.v_polys)
        names = [""] + self._field_names(rec)
        for cb, var in zip(self.cb_prio + [self.cb_ignore], self.v_prio + [self.v_ignore_field]):
            cb["values"] = names
            if var.get() not in names:
                var.set("")

    def _update_steps(self):
        try:
            from calcs.TopferPillewizer import scale_iterations
            n = len(scale_iterations(float(self.v_ratio.get()), float(self.v_incr.get())))
            self.v_steps.set(f"({n} step{'s' if n != 1 else ''})")
        except Exception:
            self.v_steps.set("")

    def _request_refresh(self):
        """Ask the toolbox (which owns the ArcGIS session) for a fresh layer list."""
        if not self.info_file:
            return
        self.status.set("Refreshing layer list from the map ...")
        threading.Thread(target=self._refresh_worker, daemon=True).start()

    def _refresh_worker(self):
        """Signal the toolbox, wait up to 5 s for it to rewrite the info file, then reload."""
        try:
            before = os.path.getmtime(self.info_file)
            with open(REFRESH_REQ, "w") as f:
                f.write("1")
            for _ in range(10):
                time.sleep(0.5)
                if os.path.getmtime(self.info_file) > before:
                    break
        except OSError:
            pass
        self.queue.put(("reload", None, ""))

    def _reload_layers(self):
        self._load_info()
        self._refresh_combos()
        self.status.set("Layer list refreshed")

    # ------------------------------------------------------------------ running
    def _log(self, message, level="info"):
        self.queue.put(("log", message, level))

    def _add_output(self, path, like):
        # also offer the result in the layer dropdowns
        try:
            import geopandas as gpd
            src = next((r for k in self.layers.values() for r in k if r["name"] == like), None)
            gdf = gpd.read_file(path, rows=1)
            kind = src["kind"] if src else "point"
            name = os.path.splitext(os.path.basename(path))[0]
            self.queue.put(("newlayer", {
                "name": name, "display": name + " [result]", "source": path, "kind": kind,
                "geographic": bool(gdf.crs and gdf.crs.is_geographic),
                "fields": [[c, "Double" if str(gdf[c].dtype).startswith(("float", "int"))
                            else "String"] for c in gdf.columns if c != "geometry"]}, ""))
        except Exception:
            pass
        try:
            try:
                with open(PENDING_FILE) as f:
                    pending = json.load(f)
            except Exception:
                pending = []
            pending.append({"path": path, "like": like})
            with open(PENDING_FILE, "w") as f:
                json.dump(pending, f)
        except OSError as exc:
            self._log(f"Could not queue {path} for the map: {exc}", "warning")

    def _processor(self):
        from gss_processing import GSSProcessor
        return GSSProcessor(log=self._log, add_output=self._add_output,
                            fallback_dir=self.fallback_dir)

    def _run(self, label_text, fn):
        if self.busy:
            messagebox.showinfo("GeoSubSampler", "A job is already running.")
            return
        try:
            args = fn(prepare=True)          # read widgets on the GUI thread
        except ValueError as exc:
            messagebox.showwarning("GeoSubSampler", f"Please enter valid numbers ({exc}).")
            return
        self.busy = True
        self.status.set(f"{label_text} running ...")
        self._log(f"--- {label_text} ---")

        def work():
            try:
                fn(prepare=False, args=args)
                self._log(f"{label_text} finished.")
            except Exception as exc:
                from gss_processing import GSSError
                if isinstance(exc, GSSError):
                    self._log(f"{label_text}: {exc}", "warning")
                else:
                    self._log(f"{label_text} failed: {exc}\n{traceback.format_exc()}", "error")
            finally:
                self.queue.put(("done", label_text, ""))

        threading.Thread(target=work, daemon=True).start()

    def _drain_queue(self):
        try:
            while True:
                kind, message, level = self.queue.get_nowait()
                if kind == "log":
                    self.log_text.configure(state="normal")
                    self.log_text.insert("end", message + "\n", level)
                    self.log_text.see("end")
                    self.log_text.configure(state="disabled")
                elif kind == "reload":
                    self._reload_layers()
                elif kind == "newlayer":                  # a result just written: offer it
                    self.local_layers.append(message)
                    self.layers[message["kind"]].append(message)
                    self._refresh_combos()
                elif kind == "set":                       # (var, value) from a worker thread
                    message[0].set(message[1])
                elif kind == "done":
                    self.busy = False
                    self.status.set("Ready")
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    # --- one handler per button: prepare=True reads widgets, prepare=False does the work ---
    def _point_args(self):
        return (self._rec("point", self.v_points), self.v_dip.get(), self.v_dipdir.get(),
                self.v_is_dipdir.get())

    def _do_subsample(self, prepare, args=None):
        if prepare:
            return {"pt": self._point_args(), "poly": self._rec("polygon", self.v_polys),
                    "method": self.v_method.get(),
                    "percent": self.v_percent.get(),
                    "kent": self.v_grid_kent.get(), "ko": self.v_grid_ko.get(),
                    "carm": self.v_grid_carm.get(), "ko_iter": self.v_ko_iter.get(),
                    "kappa_min": self.v_kappa_min.get().strip(),
                    "target": self.v_target_n.get() if self.v_target_on.get() else None,
                    "fo_dist": self.v_fo_dist.get(), "fo_angle": self.v_fo_angle.get()} \
                if self._validate_numbers(self.v_method.get()) else None
        proc, m = self._processor(), args["method"]
        rec, dip, dipdir, is_dd = args["pt"]
        if m == "stochastic":
            proc.stochastic(rec, dip, dipdir, is_dd, float(args["percent"]))
            return
        if m == "firstorder":
            proc.first_order(rec, args["poly"], dip, dipdir, is_dd,
                             float(args["fo_dist"]), float(args["fo_angle"]))
            return
        if not dip or not dipdir:
            from gss_processing import GSSError
            raise GSSError("Please define dip & dip dir/strike fields before continuing.")
        size_var = {"spherical_kent": (self.v_grid_kent, "kent"),
                    "outlier_removal": (self.v_grid_ko, "ko"),
                    "outlier_carmichael": (self.v_grid_carm, "carm")}[m]
        used = proc.grid_method(m, rec, dip, dipdir, is_dd, float(args[size_var[1]]),
                                target_n=int(args["target"]) if args["target"] else None,
                                kappa_min=float(args["kappa_min"]) if args["kappa_min"] else None,
                                iterative=args["ko_iter"])
        if args["target"]:
            self.queue.put(("set", (size_var[0], f"{used:.0f}"), ""))

    def _validate_numbers(self, method):
        """Raise ValueError (-> message box) for non-numeric entries of the chosen method."""
        checks = {"stochastic": [self.v_percent],
                  "spherical_kent": [self.v_grid_kent], "outlier_removal": [self.v_grid_ko],
                  "outlier_carmichael": [self.v_grid_carm],
                  "firstorder": [self.v_fo_dist, self.v_fo_angle]}[method]
        if self.v_target_on.get() and method in ("spherical_kent", "outlier_removal",
                                                 "outlier_carmichael"):
            int(self.v_target_n.get())
        for v in checks:
            float(v.get())
        if method == "spherical_kent" and self.v_kappa_min.get().strip():
            float(self.v_kappa_min.get())                    # blank = no kappa filter
        return True

    def _do_merge(self, prepare, args=None):
        if prepare:
            return (self._rec("line", self.v_faults), float(self.v_mtol.get()),
                    float(self.v_msearch.get()), float(self.v_mjoin.get()))
        self._processor().merge_segments(*args)

    def _do_faults(self, prepare, args=None):
        if prepare:
            return (self.v_fault_method.get(), self._rec("line", self.v_faults),
                    self._rec("polygon", self.v_polys), [v.get() for v in self.v_prio[:4]])
        method, fault, poly, prio = args
        self._processor().process_faults(method, fault, poly, prio)

    def _poly_params(self):
        float(self.v_tol.get())
        return {"node_tolerance": self.v_tol.get(), "diameter_km": self.v_diam.get(),
                "series": self.v_series.get(), "increment_km": self.v_inc.get(),
                "prio": [v.get() for v in self.v_prio[:4]], "litho": self.v_prio[4].get(),
                "ignore_field": self.v_ignore_field.get(),
                "ignore_codes": self.codes.get("1.0", "end").strip()}

    def _do_rescale(self, prepare, args=None):
        if prepare:
            float(self.v_diam.get())
            float(self.v_inc.get())
            return self._rec("polygon", self.v_polys), self._poly_params()
        self._processor().rescale_map(*args)

    def _do_simplify(self, prepare, args=None):
        if prepare:
            return (self._rec("polygon", self.v_polys), self._rec("line", self.v_faults),
                    int(float(self.v_area_tol.get())))
        self._processor().simplify_map(*args)

    def _do_tp(self, prepare, args=None):
        if prepare:
            float(self.v_ratio.get())
            float(self.v_incr.get())
            params = self._poly_params()
            params.update({
                "ratio": self.v_ratio.get(), "increment": self.v_incr.get(),
                "dip": self.v_dip.get(), "dipdir": self.v_dipdir.get(),
                "is_dipdir": self.v_is_dipdir.get(),
                "point_method": self.v_method.get(),       # the radio values are the engine names
                "fault_method": self.v_fault_method.get(),
                "grid_size": self.v_grid_kent.get(),
                "fo_distance": self.v_fo_dist.get(), "fo_angle": self.v_fo_angle.get()})
            return (self._rec("point", self.v_points), self._rec("line", self.v_faults),
                    self._rec("polygon", self.v_polys), params)
        self._processor().topfer_pillewizer(*args)


def main():
    info = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        App(info).mainloop()
    except Exception:
        # the console this runs in closes when the program ends, so keep the error readable
        log_path = os.path.join(TMP, "geosubsampler_gui_error.txt")
        text = traceback.format_exc()
        try:
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            pass
        print(text)
        print(f"\nGeoSubSampler failed to start (also saved to {log_path}).")
        input("Press Enter to close this window ...")


if __name__ == "__main__":
    main()
