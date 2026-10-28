"""
QGIS Python console script: rank faults in the active polyline layer.

Sort keys (in priority order):
  1. max_off_lv  - largest stratigraphic offset first
  2. edge_type   - X-X, then X-Z, then Z-Z
  3. length      - longest first
Writes the rank (1 = first) to a new integer field "sort_order".
Uses the selected features, or all features if none are selected; unselected
features are left NULL when a selection exists.
"""
from qgis.core import QgsField
from qgis.PyQt.QtCore import QVariant
from qgis.utils import iface

OFF_FIELD = "max_off_lv"
EDGE_FIELD = "edge_type"
OUT_FIELD = "sort_order"

# If max_off_lv is text rather than numeric, list its categories here from
# LARGEST to SMALLEST offset, e.g. ["High", "Medium", "Low"].
# Leave empty if the values are numeric (larger number = larger offset).
OFF_ORDER = []

EDGE_ORDER = {"X-X": 0, "X-Z": 1, "Z-Z": 2}

layer = iface.activeLayer()
if layer is None or layer.geometryType() != 1:
    raise Exception("Select a polyline layer in the Layers panel first.")
for fld in (OFF_FIELD, EDGE_FIELD):
    if fld not in layer.fields().names():
        raise Exception(f"Field '{fld}' not found in layer.")


def off_key(v):
    """Smaller key = larger offset (sorts first)."""
    if OFF_ORDER:
        s = str(v).strip()
        if s not in OFF_ORDER:
            raise Exception(f"max_off_lv value '{s}' not in OFF_ORDER {OFF_ORDER}")
        return OFF_ORDER.index(s)
    try:
        return -float(v)
    except (TypeError, ValueError):
        raise Exception(f"max_off_lv value '{v}' is not numeric; set OFF_ORDER.")


def edge_key(v):
    s = str(v).strip().upper()
    return EDGE_ORDER.get(s, len(EDGE_ORDER))  # unknown types go last


feats = layer.selectedFeatures() or list(layer.getFeatures())
keyed = [((off_key(f[OFF_FIELD]), edge_key(f[EDGE_FIELD]), -f.geometry().length()), f.id())
         for f in feats]
keyed.sort(key=lambda t: t[0])

layer.startEditing()
if OUT_FIELD not in layer.fields().names():
    layer.addAttribute(QgsField(OUT_FIELD, QVariant.Int))
    layer.updateFields()
idx = layer.fields().indexFromName(OUT_FIELD)
for rank, (_, fid) in enumerate(keyed, start=1):
    layer.changeAttributeValue(fid, idx, rank)
layer.commitChanges()
print(f"Ranked {len(keyed)} faults into '{OUT_FIELD}'.")
