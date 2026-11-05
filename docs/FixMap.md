# How Fix Map works

Fix Map takes a **geology polygon layer** and/or a **fault line layer** and makes them *topologically
consistent*: every shared boundary has exactly the same nodes on both sides, nothing overlaps, nothing is doubled,
and faults share nodes with the contacts and faults they touch or cross. It is available in the QGIS plugin and in
the ArcGIS Pro version; both call the same code ([`calcs/FixMap.py`](../calcs/FixMap.py)).

This document explains what it does, in what order, what it promises and what it does not. For the quick
reference to the settings see the [main README](../README.md#fix-map).

## 1. The problem

Geological maps are usually digitised polygon by polygon. Even when two polygons are meant to share a boundary they
rarely share it exactly:

| Defect | What it looks like | Why it matters |
|---|---|---|
| Unmatched nodes | A vertex of one polygon lies on (or within a few metres of) the edge of its neighbour, which has no vertex there | Shared boundaries are not really shared; graph and topology tools break |
| Near-coincident nodes | The corner where 3 polygons meet is 2 or 3 nodes a few metres apart | Triple junctions are not single points |
| Gaps and overlaps | Slivers between, or on top of, neighbouring polygons | Area is lost or counted twice |
| Doubled features | The same polygon or fault is in the layer twice | Counts and areas are wrong |
| Polygons inside polygons | A small polygon sits on a bigger one that has no hole for it | Overlap |
| Faults near contacts | A fault runs almost, but not exactly, along a contact or another fault | No shared nodes, so Fault Graph cannot connect them |
| Invalid polygons | Self-intersecting rings | Many tools refuse them |

## 2. Settings

| Setting | Meaning |
|---|---|
| **Tolerance (m)** | Features closer together than this are treated as meant to coincide. **No vertex moves by more than the tolerance.** In geographic coordinates (degrees) it is converted to metres approximately, using 110.5 km per degree of latitude and 111.3 km per degree of longitude scaled by the cosine of the middle latitude of the layers. |
| **Fill enclosed gaps** | Fill every gap completely surrounded by polygons, whatever its width or size: first, before snapping, and again at the end. On by default. |
| **Remaining overlaps** | What to do with overlaps still left at the end (of any width): *Keep smaller polygon* (cut the overlap out of the bigger one; default), *Keep larger polygon*, or *Leave overlaps*. |
| **Repair (buffer 0)** (separate button) | Only repairs invalid polygons; the first step of Fix Map, on its own. |

Both layers must use the same coordinate system; if they do not, Fix Map stops with a message instead of producing
nonsense. Results are written as `<layer>_fixed_<tolerance>` plus a text report and a progress file, all in a new folder
(`Fix_map_<date>_<time>`) next to the input layer, so every run is kept separately; the originals are never changed.

## 3. The pipeline at a glance

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

## 4. The steps in detail

### 4.1 Repair, duplicates, polygons inside polygons

* **Repair.** Invalid polygons are repaired with `buffer(0)` (the same repair many people do by hand), falling back to
  `make_valid` if that would empty a polygon. This keeps the area of a self-intersecting ring; only a tiny loop is lost.
* **Duplicates.** Two features are duplicates if their geometry is identical, whatever the start point or direction.
  The first one is kept (its attributes are kept, the other's are not compared) and the rest are removed.
* **Polygons inside polygons.** If at least **90% of a smaller polygon's area** lies inside a bigger polygon, a hole is
  cut in the bigger one for it, so the smaller polygon stays exposed. The 90% is measured on area, so a multi-part
  polygon with a tiny remote part still counts. If smaller polygons together cover a bigger one completely, nothing is
  left of it and it is removed (the report lists these; **their attributes are lost**).

* **Gaps first.** Every hole in the union of the polygons (minus any polygon sitting in it) is a gap. Each gap completely
  surrounded by polygons is added to the neighbouring polygon with which it **shares the longest boundary**. This is
  done before any snapping because the filled polygon then takes over its neighbours' own vertices along the gap, so
  those edges are already exact and the conform step has less to move (on a test region it moved 10-20% fewer vertices
  and slightly shorter distances than filling gaps last). It also closes gaps of any width, which snapping cannot.
  Gaps along the edge of the map are not touched.

### 4.2 Conform: making every boundary shared

This is the core of Fix Map. It works on a flat list of coordinates (every polygon ring and every fault is a "path"),
fully vectorised, so it handles millions of vertices.

**Ranks.** Every path gets a rank: polygons come before faults; bigger polygons before smaller ones; longer faults
before shorter ones. A path with a better rank is the *master* of a worse-ranked one, the *slave*. The master is
never moved by the slave. In particular, **polygons are never moved by faults**, and a bigger polygon keeps its shape
while the smaller ones around it conform to it.

**Sites and unique segments.** Identical vertices of different polygons (the nodes that are already shared) are one
*site*; identical segments are one *unique segment*. Working on these means a boundary shared by two polygons is handled
once, so it stays shared, and a node is created once and used by everyone.

**One pass.** For every site, the program finds the unique segments of *other* paths within the tolerance (ignoring
segments the site is already a node of, and segments of its own path). Then:

1. A **slave vertex** moves onto the **nearest node** of a master line within the tolerance. Moving to a node is always
   safe (the node is on that line) and it merges the two or three near-coincident nodes of a triple junction into one.
2. If there is no node that close, the slave vertex moves to the **nearest point on the nearest master segment**,
   and that point becomes a new node of the master segment.
3. A **master vertex** that lies within the tolerance of a slave segment is **inserted into that segment** as a node (the
   slave line bends through it), so both lines contain it.
4. A vertex moves at most once per pass, and vertices of different polygons that share a site move together, so shared
   nodes stay shared.
5. Afterwards repeated vertices are dropped (when two consecutive vertices of one line land on the same point), spikes
   (A-B-A) are removed, and a ring left with fewer than 4 vertices, or an area under 0.001 m², is removed.

**Passes repeat until nothing changes.** Changes spread: a vertex moved onto a node can create a new near-pair with a
third polygon. So the passes repeat, each acting on the result of the previous one, typically 15 to 30 times on a big
map. Two things keep this fast and safe:

* **Local passes.** After the first full pass, a pass only looks around the points that changed in the previous pass
  (including along any segment that moved, since a long segment swings when one end moves). When local passes find
  nothing, **one full pass** checks that nothing was missed; if it finds something, local passes continue.
* **Stall guard.** If 12 passes in a row insert the same nodes and move nothing, the passes stop (reported).
* **Displacement limit.** The program tracks where every vertex started and refuses a move that would put it more than
  the tolerance from there, however many passes it takes.

### 4.3 Rounds, validity and overlaps

Moving vertices can make a polygon self-intersect, or leave a thin strip where two polygons now overlap. After each
conform, the geometries are rebuilt and **only the polygons that changed** are checked:

* invalid ones are repaired and the next round conforms their neighbours to the repair;
* polygons that now lie inside another get their hole (4.1);
* **thin overlaps** (overlap narrower than the tolerance everywhere, tested by shrinking the overlap by half the
  tolerance and seeing whether anything is left) are cut out of the smaller polygon.

This repeats for up to 5 rounds. Two safeguards stop it going round in circles: a polygon that changes the same way two
rounds running (repair and snapping undoing each other) is left alone, and from round 3, if only 25 or fewer polygons
are still changing, the loop stops. Whatever is still invalid is repaired at the very end.

### 4.4 Settled (or out of rounds): overlaps and gaps

Once the geometry has settled, or the 5 rounds are used up, and before the crossing nodes, two things happen once, on
the whole map:

* **Remaining overlaps.** Every pair of polygons that still overlaps has the overlap cut out of one of them, by the
  *Remaining overlaps* rule (the default cuts it out of the **bigger** polygon, as with a polygon inside another). A
  polygon that is entirely overlapped disappears. Cutting only removes area from the polygon that loses, so it cannot
  create new overlaps.
* **Gaps.** The gap step of 4.1 is repeated, for any gap made since (for example where a polygon narrower than the
  tolerance collapsed). If you keep real unmapped areas (lakes, say), untick *Fill enclosed gaps*.

Both change the geometry, so another short conform follows to give the new edges shared nodes.

### 4.5 Crossings, final repair, report

* **Crossing nodes.** (Done again once if this step changed anything.) Nodes are added wherever lines cross: polygon boundaries, faults against polygon boundaries, and
  faults against each other, plus T-junctions. Both lines get the same node, which is what Fault Graph needs to connect
  faults. A local conform then merges any crossing node that landed within the tolerance of another node.
* **Final repair.** Any polygon that is still self-intersecting is repaired.
* **Report.** The before and after counts, and everything that was removed, are written to a text report.

## 5. What it promises, and what it does not

**Promises**

* No vertex moves by more than the tolerance (the report gives the largest move).
* Polygons are never moved by faults; bigger polygons are never moved by smaller ones.
* Shared boundaries have identical nodes; nothing overlaps (with the default rule); enclosed gaps are filled
  (if ticked); no duplicates; no invalid polygons.
* Attributes are untouched. Fix Map only moves vertices, adds nodes, cuts geometry and removes features; it never merges
  polygons or edits a field. Removed features are listed by index in the report.
* Verified independently (see section 7): when the result is noded with GEOS (a standard topology engine) no
  node needs to be added.

**Limits**

* **Chords.** When several vertices of one line snap onto the same two nodes, the extra vertices are dropped and a
  straight line replaces a short detour. No vertex moved more than the tolerance, but the new *line* can lie further than that
  from the original at a few places (the report says how far).
* **Features can disappear:** duplicates; polygons completely covered by smaller ones; polygons or faults narrower than
  the tolerance that collapse. The report lists them.
* **Real differences are not "fixed" by geology.** Overlaps and gaps of any width are handled by the overlap and gap
  rules, which decide by polygon size and boundary length, not by geology. Check the polygons the report names.
* **Fault–polygon priority is fixed:** faults conform to polygons, not the other way round.
* A pair of polygons whose overlay GEOS cannot compute even after repair and snap-rounding is skipped and reported,
  not allowed to stop the run.

## 6. Reading the report

The report has three parts.

1. **Header:** tolerance and the layers read (with their file paths).
2. **WHAT WAS DONE:** repairs, duplicates removed, holes cut, conform passes, vertices moved and nodes added, the largest
   move, gaps filled, overlaps cut, polygons removed, and anything left (still invalid, left alone). Features are listed by
   their index in the input layer.
3. **BEFORE -> AFTER:**

| Line | Meaning |
|---|---|
| Nodes lying on another line but not shared with it | A vertex within the tolerance of the *middle* of another line's segment, where that line has no node |
| Pairs of nodes closer than the tolerance, not shared | Two nodes of different lines within the tolerance of each other that are not the same node |
| Invalid / duplicate / polygons inside a polygon | As named |
| Gaps (holes in the map) and gap area | Holes in the union of the polygons, minus any polygon lying in them |
| Polygon overlap area | The sum of the polygon areas minus the area of their union |

After a successful run the node, duplicate, invalid and containment counts are 0. Gap and overlap totals depend on the
settings; the report notes any that remain and why.

In the QGIS plugin the report opens in a window and is saved in the run folder as `<layer>_fixed_<tol>_report.txt`; progress lines (with clock
times) are also written, as the run goes, to `<layer>_fixed_<tol>_progress.txt`, because QGIS cannot redraw while it works. In
the ArcGIS Pro window the same lines appear in the log as the job runs.

## 7. How it was checked

* A synthetic map with jittered nodes, missing nodes, doubled and nested polygons, and faults near the contacts.
* Real maps: a 1,121-polygon / 1,403-fault map, and a 11,801-polygon / 4,430-fault map of West Africa
  (about 3.9 million vertices).
* After each run the result is noded with GEOS (the same engine behind most GIS topology tools), polygons and faults
  together; zero nodes added means every crossing and T-junction is already a shared node. Overlaps are also checked
  pair by pair.
* The ArcGIS Pro code path was run with ArcGIS Pro's own Python (3.13, shapely 2.1, geopandas 1.1).

Typical times: about 20 s for the 1,121-polygon map; about 27 minutes for the 3.9-million-vertex map, of which measuring
the starting and final state take about 4.5 minutes and the first conform round about 7 minutes.

## 8. Where things are in the code

| Step | Function in `calcs/FixMap.py` |
|---|---|
| Whole pipeline, report | `fix_map` |
| Repair, duplicates, polygons inside polygons | `repair_polygons`, `remove_duplicates`, `cut_contained_polygons` |
| Conform | `_candidates` (finds the near pairs), `_conform_pass` (one pass), `_conform` (passes, local mode, displacement limit), `_insert_nodes`, `_cleanup` |
| Thin overlaps, remaining overlaps, gaps | `resolve_thin_overlaps`, `remove_overlaps`, `fill_thin_gaps`, `_gap_polygons` |
| Crossing nodes | `densify_paths` in [`calcs/Densify.py`](../calcs/Densify.py) (with line-line crossings switched on) |
| Counts for the report | `topology_report`, `_count_defects` |
| Robustness against GEOS failures | `_robust`, `_robust_union_all` |
| QGIS / ArcGIS entry points | `GeoSubSampler.fixMap`, `GSSProcessor.fix_map` ([`ARCGIS/gss_processing.py`](../ARCGIS/gss_processing.py)) |
