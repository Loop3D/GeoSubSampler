"""
QGIS Python console script: explain why two faults were / were not flagged.
Select 2 or more faults in the active polyline layer, then run. Prints, for each
pair, the matrix distance, cluster, sort_order and survive of both faults.
"""
from itertools import combinations
from qgis.core import QgsProject
from qgis.utils import iface

ID_FIELD = "fid"   # matrix row-header field: QGIS feature id (OBJECTID is not unique)
RANK_FIELD = "sort_order"
ORIENT_FIELD = "cluster"
OUT_FIELD = "survive"
MATRIX_NAME = "Fault node distance matrix"

layer = iface.activeLayer()
sel = layer.selectedFeatures()
if len(sel) < 2 or len(sel) > 10:
    raise Exception("Select between 2 and 10 faults.")
matrix = QgsProject.instance().mapLayersByName(MATRIX_NAME)[-1]
print(f"Matrix layer: {matrix.featureCount()} rows; polyline layer CRS: {layer.crs().authid()} "
      f"({layer.crs().mapUnits()} map units; 0 = metres)")

mrows = {str(r[ID_FIELD]): r for r in matrix.getFeatures()}


def dist(a, b):
    """Matrix holds the upper triangle only, so try both orientations."""
    for x, y in ((a, b), (b, a)):
        r = mrows.get(str(x))
        if r is None or str(y) not in matrix.fields().names():
            continue
        try:
            return float(r[str(y)])
        except (TypeError, ValueError):
            continue          # NULL (lower triangle)
    return "not in matrix"


for fa, fb in combinations(sel, 2):
    ia, ib = fa.id(), fb.id()
    print(f"\n{ia} vs {ib}: matrix distance = {dist(ia, ib)}")
    for f in (fa, fb):
        print(f"  {f.id()}: cluster={f[ORIENT_FIELD]!r} sort_order={f[RANK_FIELD]!r} "
              f"survive={f[OUT_FIELD]!r}")
