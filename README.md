# GeoSubSampler

A QGIS plugin for subsampling geological orientations, faults, and polygons. This plugin provides various methods to reduce the complexity of geological datasets while preserving important structural information.

## Overview

GeoSubSampler is designed to help geologists and GIS professionals manage large geological datasets by providing intelligent subsampling methods. The plugin supports processing of:
- Structural orientation data (dip/strike measurements)
- Geological polygons (rock units, formations)
- Fault and contact lines

## Installation

1. Download the plugin files
2. Place them in your QGIS plugins directory
3. Enable the plugin through QGIS Plugin Manager
4. The plugin will appear as a dockable widget in QGIS

## Features

### 1. Structural Orientation Subsampling

#### Stochastic Subsampling
Randomly samples a percentage of orientation points.

**Parameters:**
- **Retain Percentage**: Percentage of points to keep (0-100%)
  - *Type*: Float
  - *Range*: 0.0 - 100.0
  - *Description*: Controls what fraction of the original dataset is retained

#### Grid Cell Averaging
Divides the area into grid cells and averages orientations within each cell.

**Parameters:**
- **Grid Size**: Size of grid cells for averaging
  - *Type*: Float
  - *Units*: Layer units (meters for projected, degrees for geographic)
  - *Description*: Larger values create coarser grids with more averaging

#### Kent Statistics Grid Averaging
Uses Kent distribution statistics for spherical averaging of orientations within grid cells.

**Parameters:**
- **Grid Size (Kent)**: Grid cell size for Kent statistical averaging
  - *Type*: Float
  - *Units*: Layer units
  - *Description*: Grid size for applying Kent distribution statistics

#### Kent Statistics with Outlier Removal
Applies Kent statistics while removing the largest outliers within each grid cell.

**Parameters:**
- **Grid Size (Kent Outlier)**: Grid cell size for Kent averaging with outlier removal
  - *Type*: Float
  - *Units*: Layer units
- **Outlier Threshold**: Threshold for outlier identification and removal
  - *Type*: Float
  - *Description*: Statistical threshold for identifying and removing outliers

#### First Order Subsampling
Retains orientation points based on proximity to geological contacts and angular relationships.

**Parameters:**
- **Distance Buffer**: Buffer distance around geological contacts
  - *Type*: Float
  - *Units*: Layer units
  - *Description*: Points within this distance of contacts are prioritized
- **Angle Tolerance**: Angular tolerance for point retention
  - *Type*: Float
  - *Units*: Degrees
  - *Description*: Tolerance for angular relationships between measurements and contacts

### 2. Input Field Configuration

#### Structural Data Fields
- **Layer Selection**: Choose the point layer containing structural measurements
- **Dip Field**: Field containing dip angle values
  - *Type*: Numeric field from selected layer
  - *Description*: Angle of dip from horizontal (0-90°)
- **Dip Direction/Strike Field**: Field containing directional information
  - *Type*: Numeric field from selected layer
  - *Description*: Either dip direction (0-360°) or strike direction
- **Dip Direction Checkbox**: Toggle between dip direction and strike convention
  - *Description*: Check if using dip direction, uncheck if using strike

#### Contact/Boundary Data
- **Polyline Layer**: Layer containing geological contacts or boundaries
  - *Type*: Line layer
  - *Description*: Used for first-order subsampling relative to geological boundaries

### 3. Polygon Processing

#### Minimum Area Filtering
Removes small polygons and fills holes based on area thresholds.

**Parameters:**
- **Minimum Polygon Diameter**: Threshold diameter for polygon retention
  - *Type*: Float
  - *Units*: Kilometers
  - *Description*: Polygons with equivalent diameter smaller than this are removed/merged
- **Node Tolerance**: Tolerance for node snapping during processing
  - *Type*: Float
  - *Units*: Layer units
  - *Description*: Distance tolerance for merging nearby vertices

#### Stratigraphic Priority Fields
Define the hierarchy for polygon merging decisions:
- **Priority 1-5 Fields**: Stratigraphic fields in order of importance
  - *Type*: Attribute fields from polygon layer
  - *Description*: Used to determine which polygon attributes are preserved during merging
  - *Order*: Priority 1 has highest importance, Priority 5 has lowest

#### Lithological Classification
- **Lithology Field**: Field containing lithological information
  - *Type*: Attribute field
  - *Description*: Used for lithological classification during polygon processing

#### Dyke/Intrusion Handling
- **Dyke Field**: Field identifying intrusive bodies
  - *Type*: Attribute field
- **Dyke Codes**: Comma-separated list of codes identifying dykes/intrusions
  - *Format*: Text list (e.g., DY1, DY2, INTX-D )
  - *Description*: Special processing applied to polygons with these codes

#### Series Processing Options
- **Series Checkbox**: Enable processing of multiple scales
  - *Description*: When checked, creates multiple outputs at different scales
- **Increment Value**: Step size for series processing
  - *Type*: Float
  - *Units*: Same as minimum diameter
  - *Description*: Increment between successive processing scales

### 4. Fault/Line Processing

#### Segment Merging
Merges connected fault segments based on geometric criteria.   
WARNING: This is not particulary optimised and may take hours to complete on large (>10k polylines) datasets   

**Parameters:**
- **Distance Tolerance**: Maximum distance between line endpoints for merging
  - *Type*: Float
  - *Units*: Layer units
  - *Description*: Endpoints closer than this distance are candidates for merging
- **Search Angle Tolerance**: Maximum angle difference for parallel lines
  - *Type*: Float
  - *Units*: Degrees
  - *Description*: Lines with orientations within this tolerance are considered parallel
- **Join Angle**: Minimum angle at connection points
  - *Type*: Float
  - *Units*: Degrees
  - *Description*: Rejects connections that create angles smaller than this value

#### Length-Based Subsampling
Filters fault lines based on minimum length criteria.

**Parameters:**
- **Minimum Fault Length**: Threshold length for fault retention
  - *Type*: Float
  - *Units*: Layer units
  - *Description*: Faults shorter than this length are removed

## Technical Notes

### Coordinate System Handling
- **Geographic Coordinates**: Grid sizes are automatically converted from meters to degrees (÷110,000)
- **Projected Coordinates**: Grid sizes are used directly in the specified units

### File Format Requirements
- Input layers must be saved as shapefiles on disk
- The plugin cannot process temporary or memory layers

### Output Naming Convention
Output files follow the pattern: `[original_name]_[method]_[parameters].shp`
- If a file with the same name exists, a random 5-digit suffix is added

### Performance Considerations
- **Segment Merging**: Very slow for large datasets (>1 second per object)
- **Large Datasets**: Consider using subsampling methods before applying computationally intensive operations

## Error Handling

The plugin includes validation for:
- Missing required fields
- Invalid layer selections
- File access permissions
- Coordinate system compatibility

## Dependencies

- QGIS Python API (PyQt, qgis.core)
- GeoPandas
- NumPy
- Random and time modules for statistical operations
- rtree for polyline merging

## License

This program is free software under the MIT License.

## Author

**Ranee Joshi**  
Email: raneejoshi@gmail.com

## Version Information

- Begin Date: 2025-07-28
- Generated by: Plugin Builder
- Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/

---

*Note: This plugin requires layers to be saved as shapefiles for processing. Ensure your data is properly saved before attempting to use the subsampling functions.*