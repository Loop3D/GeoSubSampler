# GeoSubSampler

A QGIS and ArcGIS Pro plugin for subsampling geological orientations, faults, and polygons. This plugin provides various methods to reduce the complexity of geological datasets while preserving important structural information.

## Overview

GeoSubSampler is designed to help geologists and GIS professionals manage large geological datasets by providing intelligent subsampling methods. The plugin supports processing of:
- Structural orientation data (dip/strike measurements)
- Geological polygons (rock units, formations)
- Fault and contact lines

Beyond subsampling it can also clean a polygon and fault map so that it is topologically consistent (**Fix Map**),
add shared nodes to it (**Densify Nodes**), read the
stratigraphic offset across faults, simplify a map while preserving its topology, and rescale a whole map
following the Töpfer & Pillewizer rule. Every result records how it was made, in a metadata file beside it.

![The GeoSubSampler dock in QGIS, showing the Clean tab](GUI_clean.png)

## Installation (QGIS)

1. Install the [QPIP](https://plugins.qgis.org/plugins/a00_qpip/) plugin through QGIS Plugin Manager
2. Download the plugin files (works with QGIS 3.x and 4.x)
3. Place them in your QGIS plugins directory
4. Restart QGIS. QPIP will offer to install any missing Python packages listed in `requirements.txt`
5. Enable the plugin through QGIS Plugin Manager
6. The plugin will appear as a dockable widget in QGIS

## ArcGIS Pro version

An ArcGIS Pro equivalent of the plugin is in the [ARCGIS](ARCGIS/) folder. It is a Python toolbox
(`ARCGIS/GeoSubSampler.pyt`) whose **Launch GeoSubSampler GUI** tool opens a Tkinter window with the same
groups and settings as the QGIS dock widget (orientation subsampling, fault processing, geology polygons
including Densify Nodes, and Töpfer & Pillewizer scaling), using the same `calcs` code. The window lists the point, line and polygon
layers in your project, and results are added to the map automatically with the symbology of the layer they
were made from.

The ArcGIS Pro Python environment needs `geopandas` (not included by default) and, if not already installed,
`scikit-learn`, `networkx`, `scipy`, `matplotlib`, `fiona` and `triangle`. See [ARCGIS/README.md](ARCGIS/README.md) for
installation and usage notes.

### Setting up the ArcGIS Pro Python environment

The toolbox has a tool that sets up the environment for you. ArcGIS Pro's default environment
(arcgispro-py3) cannot be changed, so the tool clones it into a new environment called
`arcgispro-py3-geotools`, installs the missing packages, checks that they all import, and makes the new
environment the one ArcGIS Pro uses.

> **Note:** the tool creates a **new** environment, `arcgispro-py3-geotools`, as a copy of `arcgispro-py3`. It does not change `arcgispro-py3` or any other environment you already have. It then makes the new environment the one ArcGIS Pro uses, so **restart ArcGIS Pro** when it has finished. To go back, choose another environment in **Project > Package Manager** and restart ArcGIS Pro. The tool cannot change or delete the environment ArcGIS Pro is using at the time, so to run it again with *Delete the environment first*, switch to another environment and restart first.

1. In the ArcGIS Pro Catalog pane, go to **Toolboxes > Add Toolbox** and select `ARCGIS/GeoSubSampler.pyt`.
2. In the GeoSubSampler toolbox, open **Setup > Set Up Python Environment** and click **Run**. It needs an
   internet connection and takes about 10 minutes. Click **View Details** in the Geoprocessing pane to
   follow its progress.
3. When it has finished, **restart ArcGIS Pro**, then run **Launch GeoSubSampler GUI**.

The same tool is in the **SGTool**, **tomofast_x_q** and **GeoSubSampler** toolboxes and installs what all
three need, so you only need to run it once. Running it again only adds what is missing. Its options let you
choose a different environment name, leave ArcGIS Pro's active environment unchanged (activate it later in
Package Manager), or delete the environment and start again from a new clone.

The tool runs the script `ARCGIS/setup_arcgispro_env.py`, which can also be run from a Windows Command Prompt
(change the paths if ArcGIS Pro or this repository are somewhere else):
```
"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" "C:\path\to\GeoSubSampler\ARCGIS\setup_arcgispro_env.py"
```

## Using the plugin

The QGIS dock has four tabs, in the order a map is usually worked through:

| Tab | Tools |
|-----|-------|
| [Clean](#clean-tab) | Fix Map, Repair, Densify Nodes |
| [Points](#points-tab) | Orientation point subsampling |
| [Faults](#faults-tab) | Merge Segments, fault length, graph, stratigraphic offset and orientation clusters |
| [Polygons](#polygons-tab) | Rescale Map, Simplify Map, Töpfer & Pillewizer scaling |

The ArcGIS Pro window has the same tabs, tools and settings. Its metadata button, **Show Metadata...**, is next to
**Refresh from map**.

### Always visible

The section under the tab bar stays in place on every tab, and its settings are shared by all the tools:

- **Select Structure Point Layer**, **Select Fault Polyline Layer**, **Select Litho/strat Polygon Layer**: the
  layers the tools work on.
- **Show metadata of the selected layer in the layer panel**: shows how a result was made (see
  [Metadata](#metadata)). It uses the layer selected in the QGIS Layers panel, not the three selectors above it.
- **Geology Polygon Priorities**: the attribute fields that decide which polygons merge during Rescale Map and
  Töpfer & Pillewizer scaling.
  - **Priority 1-4**: stratigraphic fields, from most to least important, e.g. Priority 1 is the most specific
    unit. Fault Strat Offset also uses these four, so at least one must be set for it.
  - **Priority 5**: the lithology field.

### Clean tab

![The Clean tab](GUI_clean.png)

#### Fix Map
*A full explanation of the process, step by step, is in [How Fix Map works](docs/FixMap.md).*

Cleans up the small digitising errors that stop a geological map from being topologically perfect, within a
**Tolerance (m)** (degrees are converted approximately). No vertex ends up more than the tolerance from where it started.
It handles, in order:

1. invalid polygons are repaired (`buffer(0)`; also available on its own as **Repair (buffer 0)**);
2. exact duplicate polygons and faults are removed (the first is kept);
3. a polygon lying (at least 90% of its area) inside a larger polygon gets cut out of it as a hole;
4. small gaps and overlaps between polygons are closed, nodes missing on one side of a shared boundary are added,
   near-coincident nodes at triple junctions become one node, and faults that run almost along a contact or
   another fault are snapped onto it and share its nodes. Bigger polygons are never moved by smaller ones, and
   polygons are never moved by faults (longer faults win over shorter ones);
5. nodes are added where polygon boundaries and faults cross, so crossing features share a node (this is what
   Fault Graph needs to connect faults).

- **Remaining overlaps**: overlaps left at the end (of any width, e.g. two polygons that partly
  cover each other) are cut out of one of the two polygons, so that no polygons overlap any more. *Keep smaller
  polygon* (default) cuts the overlap out of the bigger polygon, as for a polygon inside another; *Keep larger
  polygon* cuts it out of the smaller one; *Leave overlaps* does nothing. A polygon that is entirely overlapped
  disappears (listed in the report).
- **Fill enclosed gaps** (on by default): every gap completely surrounded by polygons, whatever its width or
  size, is added to the neighbouring polygon that shares the longest boundary with it. This is done first, before
  anything is snapped (the filled polygon takes over its neighbours' own vertices along the gap), and again at the
  end for any gap made since. Gaps along the edge of the map are not touched. Untick it to keep wide unmapped
  areas (e.g. lakes) as holes.

No vertex is moved by more than the tolerance (the report gives the largest move). Two vertices that snapped onto
the same nodes can leave a straight chord where the original line had a short detour, so a new line can lie
further than the tolerance from the original at a few places; the report gives that distance too.

A polygon that lies on top of a bigger one has a hole cut in the bigger one. If smaller polygons completely
cover a bigger one, nothing is left of it and it is removed; the report lists these.

With *Fill enclosed gaps* and *Remaining overlaps* on, no enclosed gap or overlap is left, whatever its width;
with them off, gaps and overlaps wider than the tolerance remain. A text
report (before -> after counts of unmatched nodes, near-duplicate nodes, invalid polygons, duplicates, polygons
inside polygons, gaps and overlap area, plus the index of every removed feature) is shown, logged and saved as
`<layer>_fixed_report.txt`, in the run folder with the output layers (named `<layer>_fixed`).
Checked on a 1,900-polygon / 4,430-fault map (670,000 / 350,000 vertices, about 3 minutes): GEOS had to add 6,727
nodes to node the original linework and none to node the result.

#### Densify Nodes
Inserts nodes so that no segment is longer than a chosen spacing, **keeping polygons and faults topologically
consistent**: a new node on a boundary shared by two polygons, or run along by a fault, is the same node in
all of them. (Densifying each polygon separately does not give this: tiny floating-point differences between
the two directions in which a shared edge is traversed leave the two copies with slightly different nodes.)

- **Node Spacing (m)**: the largest distance between nodes. Each segment is divided into equal parts no longer
  than this, so new nodes are between half the spacing and the spacing apart. Shapes do not change. For
  geographic coordinates the distance is converted from degrees approximately (111 km per degree of latitude,
  scaled by cos(latitude) for longitude).
- **Shared node at fault crossing**: also adds a node, shared by both features, where a fault or polygon boundary
  crosses a polygon boundary, and where a vertex lies on another feature's segment (a T-junction).
- Vertices closer together than 1 micrometre are merged first: nodes that are "shared" in a file are often
  only shared to within rounding noise, which would otherwise leave the two sides with different nodes.
- Uses the selected polygon and/or fault layer; with both, nodes are shared between them. Output layers are
  named `<layer>_dense`. About half a second for 1,100 polygons and 1,400 faults.

### Points tab

![The Points tab](GUI_points.png)

Choose the fields and a method, then press **Subsample Points**.

#### Input fields
- **Dip**: field containing dip angle values
  - *Type*: Numeric field from the selected point layer
  - *Description*: Angle of dip from horizontal (0-90°). If this field holds values outside 0-90° the
    tools stop with a message, because the Dip and Dip Direction/Strike fields have most likely been swapped
- **Dip Direction/Strike**: field containing directional information
  - *Type*: Numeric field from the selected point layer
  - *Description*: Either dip direction (0-360°) or strike direction
- **Dip Direction** checkbox: tick if the field holds dip direction, untick if it holds strike

#### Stochastic
Randomly samples a percentage of orientation points. The random seed is recorded in the metadata.

**Parameters:**
- **%**: Percentage of points to keep (0-100%)

#### Grid methods
Kent, Kent Outlier and Carmichael each work on a grid of cells, with their own **Grid Size**.

- **Auto-fit to N pts**: instead of using the Grid Size, finds the size that gives about this many points
  (within 15%). The result cannot exceed the number of occupied grid cells. The size found is put in the Grid
  Size box and recorded in the metadata.

##### Kent Statistics Grid Averaging
Averages the orientations within each grid cell (the normalised vector mean) and reports Kent distribution
statistics for the cell. Each output point sits at the mean position of the cell's measurements and carries
`count`, `kappa` (concentration) and `beta` (ovalness) attributes. (A plain grid-cell average is the same
calculation without kappa and beta, so it is no longer offered separately.)

**Parameters:**
- **Grid Size (Kent)**: Grid cell size for Kent statistical averaging
  - *Type*: Float
  - *Units*: Layer units
  - *Description*: Grid size for applying Kent distribution statistics
- **Min kappa** (optional): Keep a cell only if its Kent kappa (concentration) is at least this value
  - *Type*: Float; leave blank (or 0) for no filter
  - *Description*: Drops cells with a single measurement and cells whose orientations are scattered
    (the thesis models used 5). Cells that survive carry `count`, `kappa` and `beta` attributes
    (-998 where undefined). Not applied by the Töpfer & Pillewizer tool.

##### Kent Statistics with Outlier Removal
Applies Kent statistics while removing the largest outliers within each grid cell.

**Parameters:**
- **Grid Size (Kent Outlier)**: Grid cell size for Kent averaging with outlier removal
  - *Type*: Float
  - *Units*: Layer units
- **Iterative** (experimental): Remove several outliers per cell instead of one. Points are taken out in order of
  how much their removal raises Kent kappa (up to half the cell) and the sequence is cut at its largest kappa
  jump, but only if that jump is at least 5 times the typical step; otherwise nothing is removed (removing any
  point from a small sample raises kappa a little, so a plain "stop when kappa stops rising" rule over-trims).
  This differs from the Carmichael method below, which orders points by angle from the cell mean.

##### Carmichael Iterative Outlier Removal
Kent averaging that can remove **several** outliers per grid cell (Carmichael & Ailleres, 2016), where
Kent Outlier removes at most one. In each cell the measurements are ranked by angle from the cell's mean
direction (farthest first) and removed one at a time while Kent kappa is recomputed; everything removed up
to the biggest jump in kappa is treated as outliers (at most half the cell). Cells with 3 or fewer
measurements are skipped. Output points carry `count`, `kappa`, `beta` and `n_removed` attributes.

**Parameters:**
- **Grid Size (Carmichael)**: Grid cell size, in layer units.

#### 1o Sampling
First order subsampling: retains orientation points based on their proximity to geological contacts and their
angle to them. The contacts are the boundaries of the polygons in the selected polygon layer, so that layer is needed too.

**Parameters:**
- **Distance**: Buffer distance around geological contacts
  - *Type*: Float
  - *Units*: Layer units
  - *Description*: Points within this distance of contacts are prioritized
- **Angle**: Angular tolerance for point retention
  - *Type*: Float
  - *Units*: Degrees
  - *Description*: Tolerance for angular relationships between measurements and contacts

### Faults tab

![The Faults tab](GUI_faults.png)

#### Merge Segments
Merges connected fault segments based on geometric criteria.
About half a second for 1,400 faults; the time grows roughly in proportion to the number of faults.

**Parameters:**
- **Dist.(m)**: Maximum distance between line endpoints for merging
  - *Type*: Float
  - *Units*: Layer units
  - *Description*: Endpoints closer than this distance are candidates for merging
- **Search Angle**: Maximum angle difference for parallel lines
  - *Type*: Float
  - *Units*: Degrees
  - *Description*: Lines with orientations within this tolerance are considered parallel
- **Join Angle**: Minimum angle at connection points
  - *Type*: Float
  - *Units*: Degrees
  - *Description*: Rejects connections that create angles smaller than this value

#### Process Faults
Choose a method and press **Process Faults**. None of these removes faults: each makes a new layer with extra
attributes. The method chosen here is also the ranking that [Töpfer & Pillewizer scaling](#töpfer--pillewizer-scaling)
uses to decide which faults to keep.

- **Length Only**: adds the length of each fault (`line_len`).
- **Graph then Length**: calculates network graph parameters for the fault network, including the `edge_type` of
  each fault.
- **Strat Offset then Length**: the stratigraphic offset across each fault. Needs the polygon layer and at least
  one of Priority 1-4.
  - The geology is read on each side of the fault (a few metres either side) at **every point where the fault
    crosses a geological contact** (polygon boundary), which includes triple junctions on the fault. Where a
    fault runs along a contact the middle of the shared section is used. Faults that meet no contact are
    sampled at their start, middle and end (stepping 50 m either side), so every fault gets an answer.
  - Each fault gets its maximum offset level (`max_off_lv`: -1 edge of map, 0 no offset, higher numbers a more
    general stratigraphic level at which the two sides differ), where it was found (`max_off_x`, `max_off_y`),
    how many samples were used (`n_samples`), and how many showed an offset (`n_offset`), no offset
    (`n_nooffset`) or the edge of the map (`n_edge`); `sample_src` is `contacts` or `ends`.
  - The sample points themselves are saved as `<output>_samples.shp` (fault, location, offset level and the
    units on each side) and added to the map.
- **Orientation Clusters then Length**: groups the faults by orientation, trying 2 to 30 clusters and keeping the
  best by silhouette score. The number of clusters chosen is recorded in the metadata.

### Polygons tab

![The Polygons tab](GUI_polygons.png)

#### Geology Polygons
**Rescale Map** removes small polygons and fills holes, merging each small polygon into its best-matching
neighbour using the [Geology Polygon Priorities](#always-visible).

**Parameters:**
- **Diameter Threshold (km)**: Threshold diameter for polygon retention
  - *Type*: Float
  - *Units*: Kilometers
  - *Description*: Polygons with equivalent diameter smaller than this are removed/merged
- **Tolerance (m)**: Tolerance for node snapping during processing
  - *Type*: Float
  - *Units*: Metres (converted for geographic coordinates)
  - *Description*: Distance tolerance for merging nearby vertices
- **Ignore Field**: Field identifying polygons that are removed by triangulation then merging, not by merging
  - *Type*: Attribute field
- **Ignore Codes**: Comma-separated list of codes identifying dykes/intrusions/alluvium that are best removed by triangulation, given their aspect ratio
  - *Format*: Text list (e.g., DY1, DY2, INTX-D )
  - *Description*: Special processing applied to polygons with these codes
- **Series**: When ticked, makes one output for each diameter from 0 up to the Diameter Threshold
- **Increment (km)**: Step between successive diameters in a series

#### Simplify Map
Modified Visvalingam-Whyatt topological map simplification. Needs both the polygon and fault layers.
- **Area Tolerance (m2)**: Area threshold for exterior vertices
  - *Description*: Decimates map polygons and associated faults composed of line segments to similar curves with fewer vertices. Vertices are removed while their triangular area (the area defined by the vertex and its two neighbours) is below the threshold. When a polygon vertex is to be removed that coincides with a fault vertex, the two will be removed together to maintain their topological relationships.

#### Töpfer & Pillewizer Scaling
Rescales all the selected layers together, following the Töpfer & Pillewizer (1966) rule: TN = ON * (OS/TS)^(x/2)
- TN  — target number of objects after generalisation
- ON  — original number of objects before generalisation
- OS  — original map scale denominator (e.g. 50 000)
- TS  — target map scale denominator (e.g. 500 000)
- x   — object type: 1 = points, 2 = lines, 3 = polygons

**Parameters:**
- **OS/TS ratio**: the target ratio, above 0 and up to 1.
- **Increment**: the step between ratios. With a ratio of 0.1 and an increment of 0.1 the steps are 0.9, 0.8 ...
  0.1; the number of steps is shown beside it.

It uses the settings of the other tabs, for whichever of the three layers are selected:
- **Points** use the method selected on the Points tab, and are always taken from the original points at the ratio
  of that step. Stochastic draws the target number directly. A grid method searches for the grid size that gives the
  target number, starting from its own Grid Size. 1o Sampling keeps the best-ranked points within its Distance and
  Angle limits, which can be fewer than the target, and needs the polygon layer. The %, Min kappa, Iterative and
  Auto-fit settings are not used.
- **Faults** are ranked by the method selected on the Faults tab, then the top ones kept: by length; by `edge_type`
  then length; by offset level then length; or by cluster size then length. The attributes are calculated
  automatically if the layer does not have them. Each step works from the result of the step before.
- **Polygons** merge the smallest polygons into their best-matching neighbours, using the priorities and the
  Tolerance, Ignore Field and Ignore Codes of the Geology Polygons group. Each step works from the result of the step before.

Each layer's outputs are numbered `_tp_1` (the unscaled data), `_tp_2`, `_tp_3` and so on, and the ratio of each is in
its metadata. A report of every step is saved in the run folder as `TP_scaling_report_<date>_<time>.csv`.

In the QGIS layer panel the results go in a **T&P** group at the top (created if it does not exist), not in Outputs. It
has one group per step, named after the ratio and the time of the run, holding that step's points, faults and polygons
in that order, so that they draw in the right order. Only one step shows at a time, so ticking a step switches the whole
map to that scale; the target scale is shown at the end of a run. The steps of the latest run are at the top.

## Results and metadata

### Where the results go
Every time you press a button, the results (shapefiles, reports, progress files) are written to a **new folder named
after the tool and the time**, e.g. `Fix_map_20261105_164049`, next to the input layer (or in the project / temp
folder if that is not writable). A series (Rescale Map, Töpfer & Pillewizer scaling) puts all its steps in one
folder; the next button press starts a new one, so earlier results are never overwritten or mixed up. In QGIS the
new layers are added to a layer-panel group called **Outputs** (created if it does not exist), newest on top;
Töpfer & Pillewizer results have [their own group](#töpfer--pillewizer-scaling). ArcGIS Pro does the same in the map's
Contents pane.

### Output names
File names give the tool, not its settings, which are in the metadata. `<layer>` is the name of the input layer.

| Tool | Output |
|------|--------|
| Fix Map | `<layer>_fixed`, `<layer>_fixed_report.txt`, `<layer>_fixed_progress.txt` |
| Repair (buffer 0) | `<layer>_valid` |
| Densify Nodes | `<layer>_dense` |
| Stochastic | `<layer>_stochastic` |
| Kent, Kent Outlier, Carmichael | `<layer>_grid_cell_kent`, `<layer>_grid_cell_kentOutlier` (`...Iter` if Iterative), `<layer>_grid_cell_carmichael` |
| 1o Sampling | `<layer>_first_order` |
| Merge Segments | `<layer>_merged` |
| Length Only | `<layer>_length` |
| Graph then Length | `simplified_full_<layer>_edges` (and the other graph files) |
| Strat Offset then Length | `<layer>_stratOffset`, `<layer>_stratOffset_samples` |
| Orientation Clusters then Length | `<layer>_fault_clusters` |
| Rescale Map | `<layer>_min_area` (a series: `_min_area_1`, `_min_area_2` ...) |
| Simplify Map | `<polygon layer>_MVW`, `<fault layer>_sn_si` |
| Töpfer & Pillewizer | `<layer>_tp_1`, `<layer>_tp_2` ..., `TP_scaling_report_<date>_<time>.csv` |

### Metadata
Every shapefile a tool writes gets a metadata file beside it, `<file>.shp.sgt.xml`, in the same format as SGTool's. It records:
- when it was made, and by which version;
- the tool and the settings it used, including what is only known once it has run: the grid size Auto-fit found, the
  random seed, and the ratio or diameter of each step of a series;
- the files it was made from. Tools that read two layers, such as Fix Map and 1o Sampling, list both.

To read it, select the layer in the Layers panel and press **Show metadata of the selected layer in the layer panel**.
The window has a **History** tab and an **XML** tab. SGTool can show GeoSubSampler's metadata, and this button shows
SGTool's. In ArcGIS Pro, press **Show Metadata...** and choose the layer (it starts with the newest result), or
**Browse...** for a shapefile. Layers made before version 0.5.1, and layers made by other software, have no metadata to
show.

## Technical Notes

### Coordinate System Handling
- **Geographic Coordinates**: Grid sizes and the Rescale Map tolerance are automatically converted from meters to degrees (÷110,000)
- **Projected Coordinates**: Grid sizes are used directly in the specified units

### File Format Requirements
- Input layers must be saved on disk, as a shapefile or in a GeoPackage
- A GeoPackage layer is first exported to a shapefile beside it (the GeoPackage itself is not changed), and field names
  longer than 10 characters are shortened
- The plugin cannot process temporary or memory layers

### Performance Considerations
Typical times on real data (28,600 bedding points; 1,121 polygons with 144,000 vertices; 1,403 faults):
- Kent / Kent Outlier / Carmichael grid methods: about a second at 5 km cells. The iterative Kent Outlier
  option grows with the square of the number of points in the largest cell, so very large cells (several
  thousand points) are slow.
- First Order sampling: about a second (it was ~100 s before it used a spatial index).
- Merge Segments, Fault Length, Fault Graph, Fault Strat Offset, Densify: from a fraction of a second to a few
  seconds. Cluster faults: several seconds.
- Rescale Map: a few seconds. Simplify Map: the slowest tool (about 10-15 s here); most of the time is in
  the topology preparation rather than in the simplification itself.
- Auto-fit to N points counts grid cells directly and no longer reruns the method for each trial size.
- The heavier libraries (matplotlib, scikit-learn, networkx) are loaded only when a tool that needs them is
  first used, so the plugin starts quickly.

## Error Handling

The plugin includes validation for:
- Missing required fields
- Invalid layer selections
- File access permissions
- Coordinate system compatibility

## Dependencies

- QGIS Python API (PyQt, qgis.core)
- Shapely (2.0 or later; its STRtree is used for polyline merging)
- NumPy
- pandas
- GeoPandas
- Fiona
- SciPy
- Matplotlib
- scikit-learn
- NetworkX
- triangle (only needed for polygon triangulation)

These are listed in `requirements.txt`. Most of them ship with QGIS; scikit-learn, NetworkX and triangle usually need to be installed. If the [QPIP](https://plugins.qgis.org/plugins/a00_qpip/) plugin is installed it adds them automatically from `requirements.txt`; otherwise install them with QPIP or pip before enabling GeoSubSampler. If any are missing, GeoSubSampler does not fail to load: it shows a warning naming them, and its button explains how to install them.

## License

This program is free software under the MIT License.   

## Future work   
- Add Ranee's code for simplification of fault/polygon sets preserving topology
- Simplify fault arrays suing topological constraints   
   
## Author

**Ranee Joshi-most calculation code**  
Email: raneejoshi@gmail.com   
**Mark Jessell-some calculation code & GUI**  
Email: mark.jessell@uwa.edu.au   
   
## Version Information

- Current version: 0.5.1
- Begin Date: 2025-07-28
- Generated by: Plugin Builder
- Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/

## Changelog

**0.5.1**
- Every result file gets a metadata file beside it (`<file>.shp.sgt.xml`, the same format as SGTool's), recording when and how it
  was made, the parameters and the files it was made from. SGTool can read and show these, and GeoSubSampler can show
  SGTool's.
- The metadata records the settings each tool actually used, including values only known once it has run: the fitted grid
  size of Auto-fit, the random seed, and the ratio or diameter of each step of a T&P or Rescale series. Tools that read two
  layers list both as sources.
- File names no longer contain parameter values (`geology_fixed.shp`, not `geology_fixed_20.shp`); the parameters are in the metadata.
- T&P: results go in a mutually exclusive T&P group in the layer panel, with one group per step, so ticking a step shows
  the whole map at that scale.
- T&P: each grid method now starts from its own Grid Size (all of them used Kent's); Rescale Map series on a geographic
  CRS no longer shrink the node tolerance at every step.
- The dock is split into Clean, Points, Faults and Polygons tabs. The layer selectors, the metadata button and the five
  priority fields stay at the top, under the tabs, and keep their values between tabs.
- **Show metadata** button: shows the history of the layer selected in the layer panel.
- ArcGIS Pro: the window has the same tabs and always-visible section as QGIS, a **Show Metadata...** button, and puts
  results in an Outputs group and T&P results in a T&P group (one step shown at a time) in the map.
- Fix Map: a final overlap sweep removes the small overlaps the last rounds can open; long folder paths no longer stop a run.

**0.5**
- *Faster:* First Order sampling (~100x, using a spatial index), Kent / Carmichael grid methods (Kent ~55x,
  Carmichael ~28x, Kent Outlier 5-6x, the iterative option ~2.5x), Visvalingam-Whyatt simplification, Rescale Map
  (~2x), Merge Segments (~7x), Fault Graph, Cluster faults. Results are identical to the previous version except
  where noted below. The heavier libraries (matplotlib, scikit-learn, networkx, fiona) load only when first
  needed, so the plugin starts faster.
- *New:* **Fix Map** (QGIS and ArcGIS Pro) - repairs invalid polygons, removes doubled polygons and faults, cuts holes
  for polygons inside polygons, closes small gaps/overlaps, matches nodes along shared boundaries and at triple
  junctions, snaps faults onto nearby contacts, and writes a text report.
- *New:* **Densify Nodes** (QGIS and ArcGIS Pro) - shared new nodes along polygon and fault boundaries.
- *Changed:* **Fault Strat Offset** samples every fault/contact crossing instead of only the start, middle and
  end of each fault, and saves the sample points and extra per-fault columns (see above). **Cluster faults**
  writes only the best result (`write_all=True` in the code writes one file per cluster count). **First Order**
  now uses the best-aligned of several equally near contact segments, so its result no longer depends on the
  order of the data (it can keep a few more points than before).

**0.4** - New Kent estimator with an optional Min kappa filter and count / kappa / beta attributes; Kent Outlier
with an optional iterative mode; new Carmichael outlier method; Grid Cell Avg removed (identical to Kent without
kappa); much faster auto-fit to N points; a check that stops the run when the Dip field holds values outside
0-90 degrees (Dip and Dip Direction fields swapped).

**0.3.1** - Loads on QGIS 4 (Qt6) as well as QGIS 3.

**0.3** - ArcGIS Pro version (Python toolbox with a Tkinter interface).

---

*Note: Layers must be saved to disk (as a shapefile or in a GeoPackage) before they can be processed; temporary and memory layers cannot be used.*