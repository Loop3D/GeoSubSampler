# GeoSubSampler Reference Manual

How every GeoSubSampler tool calculates its result: the algorithms, formulas, constants, attribute
definitions and limits. For how to *use* the tools, see the [User Manual](UserManual.md).

Everything here was checked against the code in [`GeoSubSampler.py`](../GeoSubSampler.py) and
[`calcs/`](../calcs/) (version 0.5.1 plus the unreleased changes listed in the [changelog](../CHANGELOG.md)), and
the behaviour was confirmed by running every tool on the example data of the User Manual. Where a function
is named, it is in the file given in the section heading.

**Contents**

1. [Common behaviour](#1-common-behaviour)
2. [Clean tab](#2-clean-tab): [Fix Map](#21-fix-map), [Repair](#22-repair-buffer-0), [Densify Nodes](#23-densify-nodes)
3. [Orientation statistics](#3-orientation-statistics)
4. [Point subsampling](#4-point-subsampling)
5. [Fault tools](#5-fault-tools)
6. [Polygon tools](#6-polygon-tools)
7. [Töpfer & Pillewizer scaling](#7-töpfer--pillewizer-scaling)
8. [Metadata](#8-metadata)
9. [Units and coordinate systems](#9-units-and-coordinate-systems)
10. [Known issues and caveats](#10-known-issues-and-caveats)
11. [References](#11-references)

---

## 1. Common behaviour

### 1.1 Reading the input

Every tool reads the layers from disk with geopandas, not from QGIS's in-memory features
(`GeoSubSampler._resolved_source`):

- A **shapefile** is read directly. If its folder is not writable, it is first copied (all sidecar
  files) to the project folder or the temp folder, because results are written beside the input.
- A **GeoPackage** layer is exported once to a shapefile beside the GeoPackage
  (`<gpkg name>_<layer name>.shp`); the GeoPackage itself is never changed. Field names longer than 10
  characters are shortened (`abcdefghijkl` -> `abcdefghij`, a clash gets `_1`, `_2` ...), and the field
  combo boxes are translated to the shortened names.
- A field chosen in a combo box that shows an **alias** is translated back to the real field name.
- Memory and temporary layers cannot be used.

Unsaved edits in QGIS are therefore not seen; save the layer first.

### 1.2 Output folder and names

Each button press gets a new folder `<tool>_<YYYYMMDD_HHMMSS>` (with `_2`, `_3` ... if two presses fall
in the same second) beside the input (`_run_directory`). If the input is itself in a run folder, the new
folder is put beside that folder, not inside it. If the full path would exceed about 255 characters
(the Windows limit) the folder goes to the project or temp folder instead. A file name that already exists
gets a random 5-digit number appended. Names are made from the layer name with the characters illegal in
Windows file names replaced by `_`.

Each result is loaded as a layer, given the style of its input layer (QGIS `exportNamedStyle` /
`importNamedStyle`), and inserted at the top of the **Outputs** group. The fault attribute tools then
restyle their result by the new attribute (`_style_by_field`): categorised with the Tableau-10 colours,
in sorted order of the values (more than 10 values continue with golden-ratio hues), or, for `line_len`,
graduated in 10 quantile classes on the Viridis ramp.

### 1.3 Errors

A button's work is wrapped (`_guard`) so that an unexpected error is shown in the message bar (and the
traceback printed to the Python console) instead of failing silently.

---

## 2. Clean tab

### 2.1 Fix Map

*Code: [`calcs/FixMap.py`](../calcs/FixMap.py) (`fix_map`); QGIS entry `GeoSubSampler.fixMap`; ArcGIS entry
`GSSProcessor.fix_map` in [`ARCGIS/gss_processing.py`](../ARCGIS/gss_processing.py).*

Fix Map takes a polygon layer and/or a fault layer and makes them topologically consistent: every shared
boundary has exactly the same nodes on both sides, nothing overlaps, nothing is doubled, and faults share
nodes with the contacts and faults they touch or cross.

#### The problem it solves

| Defect | What it looks like | Why it matters |
|---|---|---|
| Unmatched nodes | A vertex of one polygon lies on (or within a few metres of) its neighbour's edge, which has no vertex there | Shared boundaries are not really shared; graph and topology tools break |
| Near-coincident nodes | The corner where 3 polygons meet is 2 or 3 nodes a few metres apart | Triple junctions are not single points |
| Gaps and overlaps | Slivers between, or on top of, neighbouring polygons | Area is lost or counted twice |
| Doubled features | The same polygon or fault is in the layer twice | Counts and areas are wrong |
| Polygons inside polygons | A small polygon sits on a bigger one that has no hole for it | Overlap |
| Faults near contacts | A fault runs almost, but not exactly, along a contact or another fault | No shared nodes, so Fault Graph cannot connect them |
| Invalid polygons | Self-intersecting rings | Many tools refuse them |

#### Tolerance and units

Features closer together than the **tolerance** are treated as meant to coincide, and **no vertex moves by
more than the tolerance** from where it started. In geographic coordinates the tolerance is converted to
degrees using 110.5 km per degree of latitude and 111.3 km × cos(mid-latitude of the layers) per degree of
longitude. Both layers must be in the same CRS, or the run stops with a message (`check_same_crs`).

#### The pipeline

```
 measure the starting state
   |
 1. repair invalid polygons                      (buffer(0), make_valid if that fails)
 2. remove exact duplicate polygons and faults
 3. polygons >= 90% inside a bigger polygon  ->  hole in the bigger one
 3b. gaps completely surrounded by polygons  ->  added to the neighbour with the longest shared boundary
   |
 ROUNDS (up to 5):
   4. CONFORM  - move/insert nodes until every boundary is shared      <- the heart of it
   rebuild the geometries, remove new duplicates
   check the polygons that changed:  repair invalid | holes for contained polygons | trim overlaps narrower than the tolerance
   (stop early when only a handful of polygons are still changing)
   |
 when settled:   remaining overlaps cut out of one polygon  ->  any new gaps filled     (then one more conform)
 final round:    nodes added where lines cross  ->  conform around the new nodes  ->  checks
   |
 repair anything still self-intersecting
 measure the final state  ->  report
```

#### Repair, duplicates, polygons inside polygons

* **Repair.** Invalid polygons are repaired with `buffer(0)`, falling back to `make_valid` if that would
  empty a polygon. This keeps the area of a self-intersecting ring; only a tiny loop is lost.
* **Duplicates.** Two features are duplicates if their geometry is identical, whatever the start point or
  direction. The first is kept (with its attributes) and the rest removed.
* **Polygons inside polygons.** If at least **90% of a smaller polygon's area** lies inside a bigger
  polygon, a hole is cut in the bigger one for it. If smaller polygons together cover a bigger one
  completely, nothing is left of it and it is removed (listed in the report; its attributes are lost).
* **Gaps first** (if *Fill enclosed gaps* is ticked). Every hole in the union of the polygons (minus any
  polygon sitting in it) is a gap. Each gap completely surrounded by polygons is added to the neighbouring
  polygon with which it shares the **longest boundary**. This is done before snapping, because the filled
  polygon then takes over its neighbours' own vertices along the gap, so those edges are already exact
  (on a test region 10-20% fewer vertices had to move). It also closes gaps of any width, which snapping
  cannot. Gaps along the edge of the map are not touched.

#### Conform: making every boundary shared

The conform step works on a flat list of coordinates (every polygon ring and every fault is a *path*),
fully vectorised, so it handles millions of vertices.

**Ranks.** Every path gets a rank: polygons before faults; bigger polygons before smaller ones; longer
faults before shorter ones. A better-ranked path is the *master* of a worse-ranked *slave*, and is never
moved by it. So **polygons are never moved by faults**, and a bigger polygon keeps its shape while smaller
ones conform to it.

**Sites and unique segments.** Identical vertices of different polygons (nodes already shared) are one
*site*; identical segments are one *unique segment*. A boundary shared by two polygons is therefore handled
once, so it stays shared, and a node is created once and used by everyone.

**One pass.** For every site, find the unique segments of *other* paths within the tolerance (ignoring
segments the site is already a node of, and segments of its own path). Then:

1. A **slave vertex** moves onto the **nearest node** of a master line within the tolerance. This merges
   the near-coincident nodes of a triple junction into one.
2. If no node is that close, the slave vertex moves to the **nearest point on the nearest master
   segment**, and that point is inserted into the master segment as a new node.
3. A **master vertex** within the tolerance of a slave segment is **inserted into that segment** (the
   slave line bends through it), so both lines contain it.
4. A vertex moves at most once per pass, and vertices of different polygons that share a site move
   together, so shared nodes stay shared.
5. Afterwards repeated vertices are dropped, spikes (A-B-A) removed, and a ring left with fewer than 4
   vertices, or an area under 0.001 m², is removed.

**Passes repeat until nothing changes** (typically 15-30 on a big map, 13 on the example). After the first
full pass, a pass only looks around the points that changed in the previous pass (including along any
segment that moved). When local passes find nothing, one full pass checks nothing was missed. If 12
passes in a row insert the same nodes and move nothing, the passes stop (reported). The program tracks
where every vertex started and refuses any move that would put it more than the tolerance from there,
however many passes it takes.

#### Rounds, validity and overlaps

After each conform the geometries are rebuilt and **only the polygons that changed** are checked:
invalid ones are repaired (and the next round conforms their neighbours to the repair); polygons that now
lie inside another get their hole; **thin overlaps** (narrower than the tolerance everywhere: the overlap
disappears when shrunk by half the tolerance) are cut out of the smaller polygon. Up to 5 rounds. A
polygon that changes the same way two rounds running (repair and snapping undoing each other) is left
alone, and from round 3, if 25 or fewer polygons are still changing, the loop stops.

#### When settled: remaining overlaps and gaps

Once, on the whole map:

* **Remaining overlaps.** Every pair of polygons that still overlaps has the overlap cut out of one of
  them by the *Remaining overlaps* rule (default: out of the **bigger** polygon). A polygon entirely
  overlapped disappears. Cutting only removes area, so it cannot create new overlaps.
* **Gaps.** The gap step is repeated for any gap made since.

Another short conform then gives the new edges shared nodes.

#### Crossings, final repair, report

* **Crossing nodes** (`densify_paths` in `calcs/Densify.py` with line-line crossings on): nodes are added
  wherever lines cross (polygon boundaries, faults against polygon boundaries, faults against each other)
  and at T-junctions. Both lines get the same node. A local conform then merges any crossing node within
  the tolerance of another node. Repeated once if it changed anything.
* **Final repair** of anything still self-intersecting.
* **Report**: before -> after counts and everything removed.

#### Report lines

| Line | Meaning |
|---|---|
| Nodes lying on another line but not shared with it | a vertex within the tolerance of the *middle* of another line's segment, where that line has no node |
| Pairs of nodes closer than the tolerance, not shared | two nodes of different lines within the tolerance that are not the same node |
| Invalid / duplicate / polygons inside a polygon | as named |
| Gaps (holes in the map) and gap area | holes in the union of the polygons, minus any polygon lying in them |
| Polygon overlap area | sum of the polygon areas minus the area of their union |

#### What it promises, and what it does not

Promises: no vertex moves more than the tolerance (the report gives the largest move); polygons are never
moved by faults and bigger polygons never by smaller ones; shared boundaries have identical nodes; nothing
overlaps (default rule); enclosed gaps are filled (if ticked); no duplicates; no invalid polygons;
attributes are never edited. When the result is noded with GEOS no node needs adding (checked on
1 121-polygon and 11 801-polygon / 3.9-million-vertex maps).

Limits:

* **Chords.** When several vertices of one line snap onto the same two nodes, the extra vertices are
  dropped and a straight line replaces a short detour; the new *line* can then lie further than the
  tolerance from the original at a few places (the report says how far).
* **Features can disappear:** duplicates; polygons completely covered by smaller ones; polygons or faults
  narrower than the tolerance that collapse.
* Overlaps and gaps are resolved by size and boundary length, not by geology.
* Faults always conform to polygons, not the other way round.
* A pair of polygons whose overlay GEOS cannot compute even after repair and snap-rounding is skipped and
  reported.

| Step | Function in `calcs/FixMap.py` |
|---|---|
| Pipeline, report | `fix_map` |
| Repair, duplicates, polygons inside polygons | `repair_polygons`, `remove_duplicates`, `cut_contained_polygons` |
| Conform | `_candidates`, `_conform_pass`, `_conform`, `_insert_nodes`, `_cleanup` |
| Thin overlaps, remaining overlaps, gaps | `resolve_thin_overlaps`, `remove_overlaps`, `fill_thin_gaps`, `_gap_polygons` |
| Counts | `topology_report`, `_count_defects` |
| GEOS robustness | `_robust`, `_robust_union_all` |

### 2.2 Repair (buffer 0)

`repair_polygons` in `calcs/FixMap.py`: every invalid polygon is replaced by `buffer(0)` (or `make_valid`
if that empties it). `<layer>_valid` is written only when at least one polygon was invalid.

### 2.3 Densify Nodes

*Code: [`calcs/Densify.py`](../calcs/Densify.py) (`densify_layers`, `densify_paths`).*

Densifying each polygon on its own does not keep shared boundaries shared: the two polygons traverse the
edge in opposite directions, and floating-point differences give them slightly different new nodes.
Densify avoids this by creating every new node once per **unique segment**:

1. All polygon rings and fault lines of all selected layers are flattened into one coordinate list.
   Vertices closer than `snap_tol` (1e-6 layer units, i.e. 1 µm; 1e-10 degrees in a geographic CRS) are
   **welded** to one exact coordinate (connected clusters take their lexicographically first member).
2. Each segment's end points are put in a fixed (lexicographic) order and duplicates removed, so a
   boundary shared by two polygons, or followed by a fault, is one row.
3. If *Shared node at fault crossing* is ticked, nodes to insert are found: (a) any vertex within
   `snap_tol` of the inside of another segment (T-junction), and (b) every proper crossing of two segments,
   at least one of them a polygon boundary (fault-fault crossings are not noded by this tool). The
   crossing point is computed once and inserted in both segments.
4. Each unique segment is split at those nodes, and every piece of length L is divided into
   ceil(L / spacing) equal parts, always computed in the fixed direction.
5. Every ring and line is rebuilt by reading each segment's nodes in its own direction.

Lengths in a geographic CRS are approximated with 110 540 m per degree of latitude and
111 320 m × cos(latitude) per degree of longitude. Shapes do not change: new nodes lie on the existing
segments.

---

## 3. Orientation statistics

*Code: [`calcs/FirstOrderOrientation.py`](../calcs/FirstOrderOrientation.py), class `SubsamplingEngine`.*

### 3.1 Direction cosines

Each plane with dip δ and dip direction α is represented by the unit vector

&nbsp;&nbsp;&nbsp;&nbsp;l = sin δ · sin α (east), m = sin δ · cos α (north), n = cos δ (vertical).

It is converted back (`dircos2ddd`) with α = atan2(l, m) mod 360 and δ = 90 − asin(n); if δ comes out
above 90 the antipodal vector is used (δ → 180 − δ, α → α + 180).

### 3.2 Strike input

If the direction field holds strike (**Dip Direction** unticked), it is converted before any calculation
with α = (strike − 90) mod 360, and grid-method results are converted back with
strike = (α + 90) mod 360, kept in 0-180 if all input strikes were 0-180 (`dipdir_to_strike`). Because the
same rotation is applied to every measurement and undone afterwards, the averaged strike and dip do not
depend on which strike convention (right- or left-hand) the data use.

### 3.3 Vector mean

The mean orientation of N measurements is the normalised resultant: x̄ = (Σl, Σm, Σn)/N,
R̄ = |x̄| (the mean resultant length, 0 to 1), mean direction ĝ = x̄ / R̄. If R̄ < 1e-12 (vectors cancel)
there is no mean and the cell is dropped. Output dip and dip direction are **rounded to 0.1°**
(`_fmt_dip`, `_fmt_dipdir`; a dip direction that rounds to 360.0 is written as 0.0). Strike restored from
dip direction (3.2) is rounded to 0.1° as well.

### 3.4 Kent statistics

Kent (1982) moment estimates, as given by Leong & Carlile (1998) (`_calc_kent`, vectorised in
`_kent_kappa_batch`):

1. S = (1/N) Σ xᵢxᵢᵀ, the scatter matrix.
2. H rotates the mean direction ĝ (polar angle θ, azimuth φ) onto the z axis; B = HᵀSH.
3. The sword angle ψ = ½ atan2(2B₀₁, B₀₀ − B₁₁); K rotates by ψ about z; V = (HK)ᵀ S (HK).
4. r₁ = R̄ and r₂ = V₀₀ − V₁₁.
5. **κ = 1/(2 − 2r₁ − r₂) + 1/(2 − 2r₁ + r₂)** (concentration) and
   **β = ½ [1/(2 − 2r₁ − r₂) − 1/(2 − 2r₁ + r₂)]** (ovalness).

κ and β are undefined, and written as **-998**, when N < 2, when the measurements are identical
(R̄ ≥ 1 − 1e-12), or when a denominator is ≤ 0. Nearly identical measurements give very large κ (the
example has cells with κ ≈ 4 × 10⁸). Output κ and β are rounded to 2 decimals.

---

## 4. Point subsampling

### 4.1 Cleaning the input

Before any method (`clean_orientations`): dip and direction are read as numbers; a measurement is
**ignored** if either is missing or non-numeric, dip is outside 0-90, or direction is outside 0-360. If more
than **30%** of the valid dips are outside 0-90 the fields are taken to be swapped and nothing is processed.
Otherwise the ignored measurements are listed (up to 500) in `<layer>_ignored_points_report.txt`, with the
reasons and the most common out-of-range values (e.g. 9999 no-data codes).

### 4.2 Stochastic

`DataFrame.sample(frac = % / 100, replace = False, random_state = seed)` with seed = the current Unix
time in seconds, recorded in the metadata as `random_seed`. All input fields are kept.

### 4.3 The grid

The grid methods share one grid (`_grid_groups`):

- bounds: x₀ = int(min x), x₁ = int(max x) + 1, same for y, from the (cleaned) points;
- cell of a point: (floor((x − x₀)/s), floor((y − y₀)/s)) for grid size s;
- s is the Grid Size in layer units; for a geographic CRS the value in the box is divided by **110 000**
  (metres to degrees);
- each output point is at the **mean x, y of the measurements used** in its cell (so a lone measurement
  stays where it was), not at the cell centre.

Only the computed fields (`DIP`/`STRIKE` renamed back to the input field names, `count`, `kappa`, `beta`,
...) are written; other input fields are not.

### 4.4 Kent

For every occupied cell: vector mean (3.3), N, κ, β (3.4). With **Min kappa** k > 0 a cell is kept only if
κ is finite and κ ≥ k (so single-measurement cells, which have no κ, are dropped).

### 4.5 Kent Outlier

`outlier_removal`. For a cell with N ≥ 3 measurements:

- **Single mode** (default): compute κ of the whole cell, then κ₋ᵢ with each measurement i left out in
  turn (`_loo_kappas`, an O(N) update of the scatter matrix rather than N recomputations). The measurement
  with the largest Δκ = κ₋ᵢ − κ is removed, **if Δκ > 0**; otherwise nothing is removed.
- **Iterative mode** (`_iterative_outliers`): repeatedly remove the measurement whose removal gives the
  highest κ, recording the relative gain gⱼ = (κⱼ − κⱼ₋₁)/κⱼ₋₁, until half the cell is removed (N // 2),
  at most down to 3 measurements, or κ becomes undefined. Let gₘ be the largest gain. The measurements
  removed up to and including step m are outliers **only if** gₘ > 0 and gₘ ≥ 5 × max(median of the other
  gains, 0.05); otherwise nothing is removed. (Removing any point from a small sample raises κ a little, so
  a "stop when κ stops rising" rule would trim clean cells by half; real outliers show up as one jump.)

Cells with 1 measurement keep it; with 2 they are averaged. The orientation, κ and β of the kept
measurements are written with `n_removed` and `REMOVED_IN` (index of the first removed feature, -1 if none).

### 4.6 Carmichael

`outlier_carmichael`, after Carmichael & Aillères (2016). Cells with **3 or fewer** measurements are not
output. For each other cell:

1. Mean direction ĝ; angle of each measurement from ĝ; rank farthest first.
2. κₖ = κ of the cell with the k farthest removed, for k = 0 ... N // 2 (all at once from suffix sums).
3. Δκₖ = κₖ − κₖ₋₁ (steps where either κ is undefined are skipped). The break is at the largest Δκ.
4. If that Δκ ≤ 0 nothing is removed; otherwise the k measurements up to the break are removed.

Differences from Kent Outlier: the order of removal is by angle from the mean (not by effect on κ), and
there is no test that the jump stands out.

### 4.7 Auto-fit to N points

`_find_grid_size_for_target`. The number of output points is counted directly (number of cells with at
least 1 measurement, or 4 for Carmichael; with a Min kappa filter the method itself is run):

1. s₁ = √(area of the bounds / N).
2. s₂ = s₁ √(n₁ / N) (count ∝ 1/s²).
3. If neither is within ±15% of N, bracket the target (halving / doubling, up to 8 times each) and
   bisect up to 12 times, keeping the size whose count is closest, stopping within 15% or when the bracket
   is narrower than 0.1 units.

The size found is written back to the Grid Size box (× 110 000 for geographic data) and to the metadata
as `grid_size_m`.

### 4.8 1o Sampling (first order)

`firstorder`. Contacts are the polygon boundaries of the selected polygon layer (all rings, split into
segments). For each measurement:

- the **nearest contact segment** (STRtree `query_nearest`, returning all equally near segments) and its
  distance d;
- the measurement's strike = (α − 90) mod 180 and the segment azimuth (0-180); the angular difference
  a = |((azimuth − strike + 90) mod 180) − 90|, using the **best-aligned** of equally near segments (so the
  result does not depend on the data order);
- kept if **d ≤ Distance** and **a ≤ Angle**.

Distance is in layer units (not converted for geographic data). All input fields are kept. In T&P, the
kept points are ranked by d/Distance + a/Angle and the best TN returned.

---

## 5. Fault tools

### 5.1 Merge Segments

*Code: [`calcs/FaultLineMerger.py`](../calcs/FaultLineMerger.py).*

1. Multi-part lines are exploded into single lines.
2. For each line end, candidate ends of other lines are found with an STRtree query of a box of ± **Dist.**
   around it. Two ends can join if:
   - their distance d ≤ Dist.;
   - the directions of the two end segments (first/last segment of each line) differ by at most
     **Search Angle**, parallel or anti-parallel;
   - the join angle ACB ≥ **Join Angle**, where C is the end of the first line, A the next vertex of the
     first line and B the next vertex of the second line (180° = straight on).
3. Among valid candidates the one with the lowest score is chosen:
   score = d + (Δangle / Search Angle) × Dist. + ((180 − ACB)/(180 − Join Angle)) × Dist.
4. Lines are processed from last to first; within a pass each line takes part in at most one merge. The
   two lines are concatenated (reversed as needed; the gap between the ends becomes a segment). Passes
   repeat until no merge is possible.

The output has only the geometry; attributes are not carried over. Distances are in layer units.

### 5.2 Length Only

`FaultLengths.add_polyline_length`. `line_len` = geometry length (sum of all parts). For a projected CRS
that is in layer units (metres). For a geographic CRS the length in degrees is multiplied by
111 139 × cos(latitude of the line's centroid), a rough conversion that is exact only for east-west lines,
and a `centroid_l` field is added.

### 5.3 Graph then Length

*Code: [`calcs/FaultsGraph.py`](../calcs/FaultsGraph.py) (`CalcFaultsGraph`).*

1. Multi-part lines are exploded. Every vertex (coordinates rounded to 10 decimals) becomes a graph node,
   every segment an edge carrying its line's attributes. Lines are therefore connected **only where they
   share a vertex** exactly; a crossing without a shared vertex is not a junction.
2. The node degree (number of edges at a node) classifies nodes: degree 1 **Z** (free end), 2 **Y**
   (pass-through), 3+ **X** (junction), 0 **I** (isolated).
3. The graph is simplified: nodes of degree 2 are removed and the chains of segments between the
   remaining nodes are merged into single edges (a MultiGraph, so two faults joining the same two junctions
   stay separate).
4. Each simplified edge gets `edge_type` = its two end-node types in alphabetical order: `X-X` (junction to
   junction), `X-Z` (junction to free end), `Z-Z` (isolated fault), with `st_type`, `end_type`,
   `start_node`, `end_node`, `length`, `seg_count`, `orig_id(s)` and the attributes of the original fault.

Written as `simplified_full_<layer>_edges.shp` (added to the map) and `_nodes.shp` (`node_id`, `x`, `y`,
`degree`). Example: 1 403 faults -> 2 066 edges (975 X-X, 730 X-Z, 361 Z-Z) and 2 269 nodes.

### 5.4 Strat Offset then Length

*Code: [`calcs/FaultStratOffset.py`](../calcs/FaultStratOffset.py).*

The priority fields used are those of Priority 1-4 that are set, in that order (most detailed first);
call them c₁ ... cₙ (an unset priority is skipped, so the numbers refer to positions in this list).

**Sample locations** (`get_sampling_points`):

1. *Contacts*: the polygon rings are intersected with each fault. Each crossing point is a `contact`
   sample; where the fault runs along a ring, the middle of that shared piece is a `shared` sample.
   Duplicates within ~1 mm on the same fault (a contact is in two polygons, triple junctions) are removed.
2. *Ends of faults that meet contacts*: the start and end, unless a contact sample is already within the
   crossing step of that end.
3. *Faults that meet no contact*: start, midpoint and end.

**Side points.** At each location the fault direction is taken from the fault just before and after it
(± the crossing step at contacts; 1% of the fault length at the ends/midpoint), and two points are placed
perpendicular to it, one each side, at **5 m** for contact samples and **50 m** for end/midpoint samples
(converted to degrees with 111 320 × cos(lat) and 110 540 m/degree in a geographic CRS).

**Geology at a point**: the attributes of the first polygon that contains it (`within`), or nothing.

**Offset level of a sample** (`calculate_stratigraphic_offset`):

- geology on neither side: no sample;
- on one side only: **-1** (edge of map);
- otherwise compare the two sides from the most general field cₙ down to c₁; the level is the index k of
  the **first (most general) field whose values differ**, among fields with values on both sides;
- same at every level: **0**.

So a higher level is a bigger offset: a fault separating different `GROUP_`s outranks one that only
separates different `CODE`s within one group.

**Per fault**: `max_off_lv` = the highest level of its samples (the first such sample along the fault
gives `max_off_lo`, `max_off_ds`, `max_off_x`, `max_off_y`, `side1_unit`, `side2_unit`); `n_samples`,
`n_offset` (level > 0), `n_nooffset` (0), `n_edge` (-1); `sample_src` = `contacts` if any contact/shared
sample exists, else `ends`; and for the start/midpoint/end samples their level and description
(`start_offs`, `start_desc`, `mid_off`, `mid_desc`, `end_off`, `end_desc`). A fault with no geology anywhere
gets no level ("No geological data found"). Field names are shortened to 10 characters for the shapefile.

The samples file has one point per sample location with `fault_id`, `location`, `off_level`, `off_desc`,
`side1_unit`, `side2_unit`.

### 5.5 Orientation Clusters then Length

*Code: [`calcs/FaultClusterOrientation.py`](../calcs/FaultClusterOrientation.py).*

1. `endpt_az`: azimuth of the straight line from the first to the last point of each fault
   (first point of the first part to last point of the last part for multi-part lines), clockwise from
   north, folded to 0-180. Written to `<layer>_endpt_az.shp`.
2. Axial data are clustered by **doubling the angles**: each azimuth θ becomes (cos 2θ, sin 2θ), so 1° and
   179° are close.
3. k-means (scikit-learn, `n_init = 10`, `random_state = 42`) for every k from 2 to
   min(30, number of faults − 1, number of distinct azimuths).
4. The silhouette score of each k (on the doubled-angle points; a random sample of 5 000 for larger sets)
   picks the best k. Cluster centres are converted back by halving the angle.

The output has `cluster` (label) and `cluster_ce` (cluster centre azimuth). Example: k = 2 (721 and 682
faults). When T&P computes clusters itself, it tries k = 2 to 7 only.

---

## 6. Polygon tools

### 6.1 Rescale Map

*Code: [`calcs/StructuralPolygonSubSampler.py`](../calcs/StructuralPolygonSubSampler.py)
(`clean_small_polygons_and_holes_new`), [`calcs/PolygonTriangulator.py`](../calcs/PolygonTriangulator.py).*

**Area threshold.** A = π (D/2)² km² × 10⁶ m² for diameter D (÷ 110 000² in a geographic CRS). A series uses
D = 0, inc, 2·inc, ... up to the threshold, each step starting from the original layer.

1. `primaryid` (row number) and `inherit` fields are added if absent.
2. **Holes** smaller than A are removed from every polygon.
3. **Islands**: a polygon smaller than A that is now ≥ 99% covered by a larger polygon (it sat in a hole
   just filled) is absorbed into the smallest such host.
4. **Merging**, repeated until nothing changes. Every polygon smaller than A finds its target among the
   polygons it touches with a shared boundary of length > 0 (shared boundary = length of the intersection
   of the two boundaries). Preference order:
   1. same Priority 1 value, longest shared boundary;
   2. same Priority 2 ... 4 value (in turn);
   3. same Priority 5 (lithology) value;
   4. longest shared boundary.
   Neighbours that are triangles (below) are only used if no real polygon touches. All small polygons
   merging into the same target are dissolved into it in one go (`unary_union`, repaired with `buffer(0)`
   if invalid), the target keeps its own attributes, and the merged polygons' `inherit` values are
   appended to the target's (overflowing into `inherit1`, `inherit2` ... at 254 characters).
5. **Near neighbours (Tolerance t).** A small polygon that touches no other polygon looks instead at the
   polygons within distance t (STRtree `dwithin`), with the same preference order; the "shared boundary"
   is then the length of its boundary lying within t of the candidate's boundary. When it is merged, the
   gap between the two is filled with the area within t of both (buffer(t) ∩ buffer(t)), minus any third
   polygon, so the result is one piece and nothing overlaps. Polygons that touch a neighbour never use
   this, so on a topologically clean map (e.g. after Fix Map) the Tolerance changes nothing. It is skipped
   for t ≤ 1e-6 (the default when the function is called without one).
6. Invalid results are repaired; zero-area polygons are dropped. Polygons still below A with no neighbour
   within t are kept.

The Tolerance is also the fallback buffer distance if the direct shared-boundary calculation fails.

**Ignore codes.** Before merging, every polygon whose Ignore Field value matches one of the codes
(case- and space-insensitive) **and** whose area is below A is triangulated: Delaunay triangulation of
its vertices, clipped to the polygon so the triangles tile it exactly (the `triangle` library is used for
polygons with holes). Each triangle is given a unique value in every priority field, so it never matches
a neighbour by attribute, and the merge step prefers real neighbours over triangles. Each triangle
therefore joins the unit it shares most boundary with, and a long dyke is shared out between the units on
either side of it instead of being given whole to one neighbour. Polygons above A are left as they are. The
summary (codes not found, matched, triangulated) is shown in the message bar. The bookkeeping columns the
triangulation adds while merging (`rec_id`, `rec_type`, `tri_id`, `src_id`, `src_index`, `orig_id`; listed
in `PolygonTriangulator.helper_columns`) are removed from the result (`drop_helper_columns`), so the output
has the input's fields plus `primaryid` and `inherit*`. Each triangle has its own `primaryid`, so the
`inherit` lists of polygons that absorb a triangulated dyke can be long.

### 6.2 Simplify Map

*Code: [`calcs/PolygonSimplification.py`](../calcs/PolygonSimplification.py)
(`vector_simplify_file_two_stage`, method `modified_visvalingam_whyatt`).*

**Visvalingam-Whyatt.** Each removable vertex has an effective area = area of the triangle it forms with
its two current neighbours. The vertex with the smallest area is removed and its neighbours' areas
recomputed (min-heap, O(n log n)), until the smallest area reaches the **Area Tolerance**. Constrained
vertices have infinite area and are never removed; line end vertices are never removed.

**Stage 0: topology pre-processing** (`preprocess_topology`): polygon coordinates are snapped to a
7-decimal grid; hairline overlaps and gaps between polygons are corrected; fault and polygon vertices
within 0.01 m of each other are moved to their **midpoint** so they coincide exactly. Written as
`<polygon>_topo_fixed.shp` and `<fault>_sn.shp`.

**Stage 1: faults** (`simplify_fault_network`): fault-fault junctions and every fault vertex that
coincides with a polygon vertex are pinned; each fault is then simplified with the same tolerance and
split into arcs at its constraints. Written as `<fault>_sn_si_<tolerance>.shp` (multi-part where cut;
junction vertices are stored once per arc, so the vertex count can rise: 28 218 -> 33 368 in the example).

**Stage 2: polygons.** Before simplification these vertices are pinned:

- junctions: every point where 3 or more features (polygons and simplified faults) meet;
- every polygon vertex that is also a vertex of the simplified faults;
- the outer boundary of the map (hard-pinned; rectangular datasets always keep their frame);
- for each pair of touching polygons with **different `CODE`** values, one representative vertex of their
  shared boundary (the one closest to its centroid), so the contact survives (only if the layer has a
  `CODE` field);
- for thin polygons at risk of collapsing (approximate width 2 × area / perimeter < √tolerance / 2), the
  vertices that survive a pre-simplification of their arcs at a safe tolerance (smallest thin-body area / 4),
  so the thin body and its neighbours keep identical shared arcs. Thin bodies are processed last.

Polygons are cut into arcs at the pinned junctions; each arc is simplified once and cached, so both
polygons sharing it get the same result. A removal is refused if the new edge would cross another edge of
the ring (self-intersection guard); if assembled arcs still cross, the whole ring is simplified with the
guard, and a remaining invalid ring is repaired with `make_valid`, keeping the largest part. A summary of how
many unit contacts kept ≥ 3, 2, 1 or 0 shared vertices is printed to the Python console.

The Area Tolerance is read as an integer. Example (10 000 m²): 143 561 -> 51 557 polygon vertices.

---

## 7. Töpfer & Pillewizer scaling

*Code: [`calcs/TopferPillewizer.py`](../calcs/TopferPillewizer.py), `GeoSubSampler.topferPillewizer`.*

**Target count.** TN = max(1, round(ON × r^(x/2))), r = OS/TS, x = 1 points, 2 lines, 3 polygons
(`topfer_count`).

**Steps** (`scale_iterations`): r₁ = 1 − i, r₂ = 1 − 2i, ... while above the target ratio R, then R itself
(i = increment; 12-decimal rounding). Increment ≤ 0 gives one step, R. A baseline at ratio 1 (`_tp_1`, all
features) is always written first.

**Points** are always reduced from the original (cleaned) points with TN from the original count and the
absolute ratio of the step (`scale_points_tp`):

- Stochastic: fraction TN/ON, new seed each step.
- Grid methods (Kent without kappa filter, Kent Outlier single mode, Carmichael): count cells at the
  method's Grid Size s₀ → n₀; s₁ = s₀ √(n₀ / TN); if not within ±15%, bracket by halving/doubling (up to 10
  times) and bisect (up to 12 times), keeping the closest. If the grid gives nothing, it falls back to
  Stochastic. The found size is in the report (`grid_size_final`).
- 1o Sampling: the points passing the Distance and Angle limits, ranked by d/Distance + a/Angle, the best
  TN kept (fewer if fewer pass; a warning says so).

**Faults** are reduced from the previous step's result with the step ratio rₖ/rₖ₋₁ (so TN relative to the
original is still ON × rₖ) (`scale_lines_tp`). Missing attributes are first computed
(`_tp_prepare_fault_attrs`). Faults are sorted and the first TN kept:

| Fault method | Sort order (kept first) |
|---|---|
| Length Only | longest |
| Graph then Length | `edge_type` rank: X-X (0), other types (1), missing (2), Z-Z (3); then longest. The graph is **recomputed every step**, since removing faults changes junctions. |
| Strat Offset then Length | highest `max_off_lv`; then 0; edge-of-map (-1) and missing last; then longest |
| Orientation Clusters then Length | faults of the largest cluster first; then longest |

`tp_sel_val` records the value sorted on and `line_len` is added if missing.

**Polygons** are reduced from the previous step's result with the step ratio (`scale_polygons_tp`):

1. Ignore-code polygons below the area of the (N − TN)-th smallest polygon are triangulated (6.1).
2. **Bulk merge**: Rescale Map (6.1) with the area threshold set just above the (N − TN)-th smallest area,
   so that exactly N − TN polygons start below it.
3. **Count merge** (`_merge_to_count`), if still above TN: repeatedly dissolve the single smallest polygon
   into the touching neighbour chosen by attribute match in the order **Priority 1, 2, 3, 4, 5** (as Rescale
   Map), then
   longest shared boundary (an isolated polygon goes to the nearest centroid among its bounding-box
   candidates).

Polygons are always merged, never deleted, so no holes appear.

**Report.** `TP_scaling_report_<date>_<time>.csv`: a settings row, then one row per layer per step with the
ratio, step ratio, method, input count, target count from the original, output count, and the parameters
actually used (grid size start/final, shortest fault kept, area threshold, counts after each polygon pass,
any fallback).

---

## 8. Metadata

*Code: [`calcs/gss_metadata.py`](../calcs/gss_metadata.py).*

Each shapefile written gets `<file>.shp.sgt.xml` (format shared with SGTool; a GeoTIFF would carry it
embedded instead):

```xml
<SGToolMetadata schema="1">
  <created>2026-11-12T08:03:25+10:30</created>
  <software name="GeoSubSampler" version="0.5.1" />
  <output><path>...</path><sizeBytes>...</sizeBytes><modified>...</modified></output>
  <operation>Fault processing</operation>
  <parameters>
    <parameter name="method">Orientation Clusters then Length</parameter>
    <parameter name="best_number_of_clusters">2</parameter>
  </parameters>
  <sources>
    <source index="0"><path>...100k_IGB_faults.shp</path><exists>true</exists>
      <metadata>... the source's own SGToolMetadata, if any ...</metadata></source>
  </sources>
</SGToolMetadata>
```

The parameters are the dock settings relevant to the tool (`_collect_parameters`), plus values found
during the run (`grid_size_m`, `random_seed`, `best_number_of_clusters`, `series_step` and
`diameter_threshold_km` of a series, T&P `step`, `os_ts_ratio`, `step_ratio`, `output_count`, `result_*`).
Tools reading two layers list both as sources. Each source's own metadata (and any `.aux.xml` or `.xml`
sidecar) is copied in, so the full history travels with the file. Shapefiles written by a press but not
added to the map (e.g. the graph nodes) get the same metadata. Writing metadata never stops a tool.

The size and modification time recorded let the viewer tell when a file was changed after it was made.

---

## 9. Units and coordinate systems

| Setting | Projected CRS | Geographic CRS |
|---|---|---|
| Fix Map Tolerance (m) | layer units | ÷ 110 500 (lat), ÷ 111 300 cos(lat) (lon) |
| Densify Node Spacing (m) | layer units | lengths × 110 540 / 111 320 cos(lat) |
| Grid Size (Kent, Kent Outlier, Carmichael) | layer units | ÷ 110 000 |
| 1o Distance | layer units | **degrees** (not converted) |
| Merge Segments Dist. | layer units | **degrees** (not converted) |
| `line_len` | layer units | degrees × 111 139 cos(centroid lat) |
| Strat Offset side steps (5 m, 50 m) | metres | 111 320 cos(lat), 110 540 |
| Rescale Diameter (km), Tolerance (m) | km, m | area ÷ 110 000², tolerance ÷ 110 000 |
| Simplify Area Tolerance (m²) | layer units² | **degrees²** (not converted) |

A projected CRS in metres is strongly recommended.

---

## 10. Known issues and caveats

The User Manual lists these in plain terms ([Known issues and caveats](UserManual.md#10-tips-messages-known-issues-and-caveats)).

### Known issues

| Issue | Where | Effect |
|---|---|---|
| Attributes not kept | Merge Segments (5.1) | `process_shapefile` builds the output from the merged geometries only. |
| Long `inherit` lists | Rescale Map / T&P with ignore codes (6.1) | Each triangle has its own `primaryid`, which is appended to the absorbing polygon's `inherit` fields. |
| Triangles counted as polygons | T&P polygons (7) | N for the bulk-merge threshold includes the triangles; the count merge still reaches TN. |

### Caveats

| Topic | Detail | Section |
|---|---|---|
| Input read from disk | Unsaved edits are not seen; GeoPackage layers are exported with 10-character field names. | 1.1 |
| Units | 1o Distance, Merge Dist. and Simplify Area Tolerance are in layer units (degrees if geographic); other conversions are approximate (110 000 m/degree for grids and Rescale; 111 139 cos(lat) for `line_len`). | 9 |
| Fix Map | Can remove features (duplicates, covered polygons, collapsed slivers); resolves gaps and overlaps by size and boundary length, not geology; chords can lie further than the tolerance from the original line. | 2.1 |
| Densify | Fault-fault crossings are not noded (`line_crossings=False`). | 2.3 |
| Grid methods | Only computed fields are written; points are at the mean position of the measurements; κ can be very large for near-identical measurements. | 3.4, 4.3 |
| Point targets | Auto-fit and T&P grid searches stop within ±15%, typically just under the target. | 4.7, 7 |
| Stochastic | Seed from the clock, so runs differ (the seed is in the metadata). | 4.2 |
| Fault Graph | Faults connect only at exactly shared vertices (10-decimal rounding); run Fix Map first. | 5.3 |
| Strat Offset | First containing polygon used if polygons overlap; side points outside all polygons give -1; side steps fixed at 5 m and 50 m. | 5.4 |
| Orientation Clusters | End-to-end azimuth ignores curvature; T&P computes k = 2-7 only. | 5.5 |
| Rescale Map Tolerance | Only for small polygons touching nothing; the gap filled is buffer(t) ∩ buffer(t) minus other polygons. | 6.1 |
| Simplify Map | Contact pinning uses a field named `CODE` (skipped if absent); faults are cut into arcs, so stored vertex counts can rise. | 6.2 |
| T&P | Points from the original at each step, faults and polygons chained; Min kappa, Iterative and Auto-fit are ignored. | 7 |
| ArcGIS Pro | Same calculation code, not re-run for these manuals. | — |

---

## 11. References

- Carmichael, T. & Aillères, L. (2016). Method and analysis for the upscaling of structural data.
  *Journal of Structural Geology* 83, 121-133.
- Kent, J. T. (1982). The Fisher-Bingham distribution on the sphere. *Journal of the Royal Statistical
  Society B* 44, 71-80.
- Leong, P. & Carlile, S. (1998). Methods for spherical data analysis and visualization. *Journal of
  Neuroscience Methods* 80, 191-200.
- Töpfer, F. & Pillewizer, W. (1966). The principles of selection. *The Cartographic Journal* 3, 10-16.
- Visvalingam, M. & Whyatt, J. D. (1993). Line generalisation by repeated elimination of points.
  *The Cartographic Journal* 30, 46-51.
