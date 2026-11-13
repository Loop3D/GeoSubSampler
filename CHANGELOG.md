# GeoSubSampler Changelog

All notable changes to GeoSubSampler, newest first. See the [README](README.md) for an overview and the [User Manual](docs/UserManual.md) for how to use each tool.

## Unreleased
- *New:* **Thin Nodes** (Clean tab, QGIS and ArcGIS Pro) - the opposite of Densify: removes nodes closer than a
  minimum spacing from polygon and fault layers together, so shared boundaries and faults along them lose the same
  nodes. Triple junctions, fault/contact junctions and crossings and line ends are kept, and no removal may create a
  self-intersection.
- Merge Segments no longer drops the end of the second fault at a join: when the two ends are apart (up to Dist.),
  both are kept and joined by a straight segment. Before, the merged fault cut across from one end to the second
  vertex of the other fault, moving it by up to the Dist. setting (up to 9 m in a test with Dist. = 10 m).
- Fix Map no longer leaves faults crossing themselves: where snapping made a fault pass through the same node twice
  (3 faults in the example data), the small loop between is cut out without moving anything, and any fault that still
  crosses itself is listed in the report. Report indexes print as plain numbers.
- QGIS: a GeoPackage layer's shapefile copy is redone whenever the GeoPackage has changed; before, the first copy was
  reused for good, so tools could run on out-of-date features.
- Fault results (Length, Graph, Strat Offset, Clusters) are coloured by their new attribute, in QGIS and ArcGIS Pro.
- Strat Offset also samples the start and end of faults that cross contacts.
- Simplify Map adds its simplified faults (`<faults>_sn_si_<tolerance>`) to the map; they were written but not loaded.
- Rescale Map / T&P: the Tolerance now lets a small polygon that touches nothing merge into a polygon within
  that distance, filling the gap; maps whose polygons touch are unaffected.
- T&P count merge uses the same priority order as Rescale Map (Priority 1-4, then 5).
- Grid methods write dip and dip direction / strike rounded to 0.1 degree instead of truncated to whole degrees.
- Ignore-code triangulation no longer leaves `rec_*`, `tri_id`, `src_*`, `orig_id`, `area` or random `field_*`
  columns in the output; each step of a Rescale Map series triangulates the original layer.
- T&P Increment tooltip corrected (steps are 1 - n x increment).

## 0.5.1
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

## 0.5
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
  end of each fault, and saves the sample points and extra per-fault columns (see the [Reference Manual](docs/ReferenceManual.md#54-strat-offset-then-length)). **Cluster faults**
  writes only the best result (`write_all=True` in the code writes one file per cluster count). **First Order**
  now uses the best-aligned of several equally near contact segments, so its result no longer depends on the
  order of the data (it can keep a few more points than before).

## 0.4

New Kent estimator with an optional Min kappa filter and count / kappa / beta attributes; Kent Outlier
with an optional iterative mode; new Carmichael outlier method; Grid Cell Avg removed (identical to Kent without
kappa); much faster auto-fit to N points; a check that stops the run when the Dip field holds values outside
0-90 degrees (Dip and Dip Direction fields swapped).

## 0.3.1

Loads on QGIS 4 (Qt6) as well as QGIS 3.

## 0.3

ArcGIS Pro version (Python toolbox with a Tkinter interface).
