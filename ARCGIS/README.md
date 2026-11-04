# GeoSubSampler for ArcGIS Pro

ArcGIS Pro equivalent of the GeoSubSampler QGIS plugin, modelled on `SGTool/ARCGIS`.

| File | Role |
|------|------|
| `GeoSubSampler.pyt` | Python toolbox. **Launch GeoSubSampler GUI** starts the Tkinter window as a separate process, passes it the project's point/line/polygon layers, and adds each result to the map as it is written (symbology copied from the source layer). |
| `GeoSubSampler_GUI.py` | Tkinter window - same groups and settings as the QGIS dock widget. |
| `gss_processing.py` | GUI-independent port of the plugin's processing methods. Uses the shared `../calcs` package. |
| `setup_arcgispro_env.py` | Sets up the Python environment (clone of the default one plus the packages needed); run by the toolbox's **Setup** tool. |

## Install
1. In ArcGIS Pro: Catalog > Toolboxes > Add Toolbox > `ARCGIS/GeoSubSampler.pyt`.
2. The ArcGIS Pro Python environment needs `geopandas`, `shapely`, `scikit-learn`, `networkx`,
   `scipy`, `matplotlib` (and `fiona` for map simplification, `triangle` for dyke triangulation).
   The easiest way is to run the toolbox's **Setup > Set Up Python Environment** tool, then restart
   ArcGIS Pro. It runs `setup_arcgispro_env.py` from this folder, which clones the default environment,
   installs everything, and activates the clone; it does not change `arcgispro-py3` or any other
   existing environment, and ArcGIS Pro must be restarted to use the new one (see the main
   [README](../README.md#setting-up-the-arcgis-pro-python-environment)).
   To do it by hand instead: clone the default environment in the
   Python Package Manager, add what is missing, and activate the clone. The launch tool checks this
   and says what is missing.
   > **Note:** geopandas is not included in the default ArcGIS Pro environment and has to be
   > added before the tool will run. In Python Package Manager, clone the default environment
   > (`arcgispro-py3` cannot be modified), search for `geopandas` under *Add Packages* and install it,
   > then activate the clone and restart ArcGIS Pro. Or from the Python Command Prompt:
   > `conda install -n <your-clone> geopandas`.
   >
   > Also install the following (if not already installed): `scikit-learn`, `networkx`, `scipy`,
   > `matplotlib` and `fiona`, e.g.
   > `conda install -n <your-clone> geopandas scikit-learn networkx scipy matplotlib fiona`.
3. Run **Launch GeoSubSampler GUI**. Keep the tool running; closing the GUI window ends it.

## What the window offers
Same groups as the QGIS plugin: orientation subsampling (Stochastic, Kent with an optional Min kappa, Kent Outlier
with an optional Iterative mode, Carmichael, 1o Sampling, with "Auto-fit to N pts"), fault tools (Merge Segments,
Length, Graph, Strat Offset, Orientation Clusters), geology polygons (Rescale Map, Simplify Map, **Densify
Nodes**) and Töpfer & Pillewizer scaling. Hover over a control for its tooltip.

## Notes
* **Densify Nodes** uses the selected polygon and/or fault layer, with a node spacing in metres; with both, new
  nodes are shared between them. Outputs are `<layer>_dense_<spacing>` shapefiles.
* **Strat Offset** also writes `<fault layer>_stratOffset_samples` (a point layer of where the offset was read,
  with the offset level and the units either side); it is added to the map without copying symbology.
* Feature classes in a geodatabase are exported once to a shapefile (field names shortened to
  10 characters) and results are written as shapefiles beside the source, or to the project folder
  if that folder is not writable - the same rules as the QGIS plugin.
* "Refresh from map" re-reads the layer list after you add layers to the project.
* Layer selections and definition queries in the map are not applied: the whole data source is used.
* `Add file...` lets the GUI use shapefiles that are not in the map. The GUI can also be run
  stand-alone: `python GeoSubSampler_GUI.py`.
