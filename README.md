# GeoSubSampler

A QGIS plugin (with an ArcGIS Pro version) that simplifies geological maps while keeping their structure:
it subsamples structural orientation data, joins and ranks faults, cleans and merges geological polygons,
and scales a whole map to a smaller scale with the Töpfer & Pillewizer rule. Every result records how it
was made, in a metadata file beside it.

![The GeoSubSampler dock in QGIS](docs/images/gui_clean.png)

## Documentation

- **[User Manual](docs/UserManual.md)**: installing, a quick start, and every tool with worked examples and
  screenshots on a Western Australian 1:100 000 map.
- **[Reference Manual](docs/ReferenceManual.md)**: how every result is calculated (algorithms, formulas,
  constants, output fields, units, known issues and caveats).
- **[Changelog](CHANGELOG.md)**: what changed in each version.
- [ArcGIS Pro version](ARCGIS/README.md): the toolbox and its Python environment.

## What it does

| Tab | Tools |
|-----|-------|
| Clean | **Fix Map** (topological clean-up of polygons and faults, with a report), **Repair** (buffer 0), **Densify Nodes** (shared nodes at a fixed spacing) |
| Points | Orientation subsampling: **Stochastic**, **Kent** grid averaging (optional kappa filter), **Kent Outlier**, **Carmichael**, **1o Sampling**, with **Auto-fit to N points** |
| Faults | **Merge Segments**; fault attributes: **Length**, **Graph** (network edge types), **Strat Offset**, **Orientation Clusters** |
| Polygons | **Rescale Map** (merge small polygons by stratigraphic priority), **Simplify Map** (topology-preserving Visvalingam-Whyatt), **Töpfer & Pillewizer scaling** of points, faults and polygons together |

## Installation (QGIS 3.24 - 4.x)

1. Install the [QPIP](https://plugins.qgis.org/plugins/a00_qpip/) plugin through the QGIS Plugin Manager.
2. Put this folder in your QGIS plugins directory and restart QGIS. QPIP offers to install the missing
   packages listed in `requirements.txt`.
3. Enable GeoSubSampler in the Plugin Manager; it opens as a dock widget.

Dependencies: shapely 2.0+, NumPy, pandas, GeoPandas, Fiona, SciPy, Matplotlib, scikit-learn, NetworkX and
triangle (only for triangulating ignore-code polygons). For ArcGIS Pro see [ARCGIS/README.md](ARCGIS/README.md).

## License

MIT License.

## Authors

**Ranee Joshi** - most calculation code - raneejoshi@gmail.com  
**Mark Jessell** - some calculation code and GUI - mark.jessell@uwa.edu.au

## Version

- Current version: 0.5.1 (begun 2025-07-28, generated with [Plugin Builder](http://g-sherman.github.io/Qgis-Plugin-Builder/))
- Changes in each version, including unreleased ones: [CHANGELOG.md](CHANGELOG.md)

