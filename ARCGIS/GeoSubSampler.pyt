# -*- coding: utf-8 -*-
"""
GeoSubSampler.pyt  -  ArcGIS Pro Python Toolbox
ArcGIS Pro equivalent of the GeoSubSampler QGIS plugin.

"Launch GeoSubSampler GUI" opens the Tkinter interface (GeoSubSampler_GUI.py) as a separate
process, hands it the point / line / polygon layers of the open project, and then adds every
result the GUI writes to the active map while the GUI stays open.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import arcpy

_HERE = os.path.dirname(os.path.abspath(__file__))

_TMP = tempfile.gettempdir()
INFO_FILE = os.path.join(_TMP, "geosubsampler_info.json")
PENDING_FILE = os.path.join(_TMP, "geosubsampler_pending.json")
REFRESH_REQ = os.path.join(_TMP, "geosubsampler_refresh_req.json")

# Copy each new layer's symbology from the layer it was made from, by copying the renderer
# in the layer definition (CIM). Do NOT use arcpy.management.ApplySymbologyFromLayer here:
# running a geoprocessing tool inside the launch tool leaves the new layers locked (padlock).
COPY_SYMBOLOGY = True

_NUMERIC =("Double", "Integer", "SmallInteger", "Single", "BigInteger")
_KINDS = {"Point": "point", "Multipoint": "point", "Polyline": "line", "Polygon": "polygon"}


class Toolbox:
    def __init__(self):
        self.label = "GeoSubSampler"
        self.alias = "geosubsampler"
        self.tools = [LaunchGUI, LoadOutputs, SetUpPythonEnvironment]


def _collect_layers(proj):
    """Describe every feature layer in the project that GeoSubSampler can use."""
    layers, seen = [], set()
    for m in proj.listMaps():
        for lyr in m.listLayers():
            if not lyr.isFeatureLayer:
                continue
            try:
                desc = arcpy.Describe(lyr)
                kind = _KINDS.get(desc.shapeType)
                if kind is None:
                    continue
                display = f"{lyr.name}  [{m.name}]"
                if display in seen:
                    continue
                seen.add(display)
                sr = desc.spatialReference
                layers.append({
                    "name": lyr.name,
                    "display": display,
                    "source": lyr.dataSource,
                    "kind": kind,
                    "geographic": bool(sr and sr.type == "Geographic"),
                    "fields": [[f.name, f.type] for f in arcpy.ListFields(lyr)
                               if f.type not in ("Geometry", "OID", "Blob", "Raster", "GlobalID")],
                })
            except Exception:
                continue
    return layers


def _python_exe():
    """sys.executable is ArcGISPro.exe inside Pro; find the real python.exe."""
    for cand in (os.path.join(sys.exec_prefix, "python.exe"),
                 os.path.normpath(os.path.join(os.path.dirname(os.__file__), "..", "python.exe"))):
        if os.path.exists(cand):
            return cand
    return None


def _add_to_map(proj, entries, messages):
    """Add result files to the first map and copy symbology from the layer they came from."""
    maps = proj.listMaps()
    if not maps:
        return []
    m = maps[0]
    added = []
    for entry in entries:
        path, like = entry["path"], entry.get("like")
        try:
            new_layer = m.addDataFromPath(path)
            added.append(path)
            messages.addMessage(f"Added to map: {os.path.basename(path)}")
            if like and COPY_SYMBOLOGY:
                try:
                    src = next(l for l in m.listLayers(like) if l.isFeatureLayer)
                    # copy the renderer through the layer definition (CIM) - no geoprocessing
                    # tool is run, so the new layer is not locked while this tool is running
                    src_def = src.getDefinition("V3")
                    new_def = new_layer.getDefinition("V3")
                    new_def.renderer = src_def.renderer
                    new_layer.setDefinition(new_def)
                except Exception as exc:
                    messages.addWarningMessage(
                        f"Could not copy symbology to {os.path.basename(path)}: {exc}")
        except Exception as exc:
            messages.addWarningMessage(f"Could not add {path}: {exc}")
    return added


class LaunchGUI:
    def __init__(self):
        self.label = "Launch GeoSubSampler GUI"
        self.description = ("Open the GeoSubSampler window alongside ArcGIS Pro. Results are "
                            "added to the map as they are created. The tool shows as running "
                            "until the GeoSubSampler window is closed.")
        self.canRunInBackground = False
        self.category = "Launch"

    def getParameterInfo(self):
        return []

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        pass

    def updateMessages(self, parameters):
        pass

    def execute(self, parameters, messages):
        gui_path = os.path.join(_HERE, "GeoSubSampler_GUI.py")
        if not os.path.exists(gui_path):
            arcpy.AddError(f"GeoSubSampler_GUI.py not found at: {gui_path}")
            return
        python_exe = _python_exe()
        if python_exe is None:
            arcpy.AddError("Cannot locate python.exe for the active ArcGIS Pro environment.")
            return

        # geopandas etc. must be importable by that interpreter
        probe = subprocess.run([python_exe, "-c", "import geopandas, shapely, sklearn, networkx"],
                               capture_output=True, text=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)   # no console flash
        if probe.returncode != 0:
            arcpy.AddError(
                "A required Python package is missing from the ArcGIS Pro environment:\n"
                f"{probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else ''}\n"
                "Clone the default environment (Package Manager) and install geopandas, "
                "scikit-learn and networkx, then make the clone active.")
            return

        proj = arcpy.mp.ArcGISProject("CURRENT")
        info = {"aprx": proj.filePath or "", "layers": _collect_layers(proj)}
        with open(INFO_FILE, "w") as f:
            json.dump(info, f)
        with open(PENDING_FILE, "w") as f:
            json.dump([], f)
        if os.path.exists(REFRESH_REQ):
            os.remove(REFRESH_REQ)

        proc = subprocess.Popen([python_exe, gui_path, INFO_FILE],
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
        messages.addMessage(f"GeoSubSampler GUI launched ({len(info['layers'])} feature "
                            f"layer(s) passed). Close the GUI window to finish this tool.")

        added_paths = set()
        cycle = 0
        while proc.poll() is None:
            time.sleep(1.5)

            cycle += 1
            if os.path.exists(REFRESH_REQ) or cycle % 40 == 0:    # GUI request, or ~every minute
                try:
                    if os.path.exists(REFRESH_REQ):
                        os.remove(REFRESH_REQ)
                    info["layers"] = _collect_layers(arcpy.mp.ArcGISProject("CURRENT"))
                    with open(INFO_FILE, "w") as f:
                        json.dump(info, f)
                except Exception:
                    pass

            try:                                                    # results waiting for the map
                with open(PENDING_FILE) as f:
                    entries = json.load(f)
            except Exception:
                entries = []
            entries = [e for e in entries if e["path"] not in added_paths]
            if entries:
                try:
                    done = set(_add_to_map(arcpy.mp.ArcGISProject("CURRENT"), entries, messages))
                except Exception as exc:
                    done = set()
                    messages.addWarningMessage(f"Could not add results to the map: {exc}")
                # remember what was added so it can never be added twice, then drop it from
                # the queue (re-read first so results the GUI queued meanwhile are kept)
                done = {e["path"] for e in entries}      # a failed add is reported once, not retried
                added_paths |= done
                try:
                    with open(PENDING_FILE) as f:
                        current = json.load(f)
                    with open(PENDING_FILE, "w") as f:
                        json.dump([e for e in current if e["path"] not in done], f)
                except Exception:
                    pass
                try:
                    arcpy.RefreshTOC()
                except Exception:
                    pass
        messages.addMessage("GeoSubSampler GUI closed.")


class LoadOutputs:
    def __init__(self):
        self.label = "Add GeoSubSampler Outputs to Map"
        self.description = ("Add any results the GUI has written but that are not yet in the "
                            "map (normally done automatically while the GUI is open).")
        self.canRunInBackground = False
        self.category = "Launch"

    def getParameterInfo(self):
        return []

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        pass

    def updateMessages(self, parameters):
        pass

    def execute(self, parameters, messages):
        try:
            with open(PENDING_FILE) as f:
                entries = json.load(f)
        except Exception:
            entries = []
        if not entries:
            messages.addMessage("No pending GeoSubSampler outputs.")
            return
        done = _add_to_map(arcpy.mp.ArcGISProject("CURRENT"), entries, messages)
        with open(PENDING_FILE, "w") as f:
            json.dump([e for e in entries if e["path"] not in done], f)
        messages.addMessage(f"Added {len(done)} layer(s).")


class SetUpPythonEnvironment:
    """Run setup_arcgispro_env.py (in this folder): clone arcgispro-py3 and add the packages
    SGTool, Tomofast-x and GeoSubSampler need. The same tool is in all three toolboxes."""

    def __init__(self):
        self.label = "Set Up Python Environment"
        self.description = ("Clone the default ArcGIS Pro Python environment (arcgispro-py3), install "
                            "the packages SGTool, Tomofast-x and GeoSubSampler need, and make the clone "
                            "the active environment. Takes about 10 minutes and needs an internet "
                            "connection. Restart ArcGIS Pro when it has finished.")
        self.canRunInBackground = True
        self.category = "Setup"

    def getParameterInfo(self):
        name = arcpy.Parameter(displayName="Environment name", name="env_name",
                               datatype="GPString", parameterType="Required", direction="Input")
        name.value = "arcgispro-py3-geotools"
        activate = arcpy.Parameter(displayName="Make it the active ArcGIS Pro environment",
                                   name="activate", datatype="GPBoolean",
                                   parameterType="Optional", direction="Input")
        activate.value = True
        recreate = arcpy.Parameter(displayName="Delete the environment first if it already exists",
                                   name="recreate", datatype="GPBoolean",
                                   parameterType="Optional", direction="Input")
        recreate.value = False
        return [name, activate, recreate]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        pass

    def updateMessages(self, parameters):
        pass

    def execute(self, parameters, messages):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "setup_arcgispro_env", os.path.join(_HERE, "setup_arcgispro_env.py"))
        env_setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(env_setup)
        try:
            missing = env_setup.setup(parameters[0].valueAsText, bool(parameters[1].value),
                                      bool(parameters[2].value), log=messages.addMessage)
        except env_setup.SetupError as exc:
            messages.addErrorMessage(str(exc))
            raise arcpy.ExecuteError
        if missing:
            messages.addWarningMessage("Still missing: " + ", ".join(missing))
