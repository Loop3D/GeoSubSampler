# -*- coding: utf-8 -*-
"""
Qt5 / Qt6 (QGIS 3.x / QGIS 4.x) enum compatibility.

PyQt6 only accepts the fully scoped enum names (Qt.AlignmentFlag.AlignRight), which PyQt5
also accepts; QGIS 4 likewise only has the scoped Qgis / QgsMapLayerProxyModel names. Older
QGIS 3.x releases lack some of the scoped QGIS names, so each constant is looked up under
every spelling and the first that exists is used.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QFrame
from qgis.core import Qgis, QgsMapLayerProxyModel, QgsFieldProxyModel


def _find(name, *owners):
    """First owners[i].name that exists (owners may be None for a missing scope)."""
    for owner in owners:
        if owner is not None and hasattr(owner, name):
            return getattr(owner, name)
    raise AttributeError(name)


def _scope(owner, scope):
    return getattr(owner, scope, None)


# ---- Qt ----
ALIGN_RIGHT = _find("AlignRight", _scope(Qt, "AlignmentFlag"), Qt)
ALIGN_LEFT = _find("AlignLeft", _scope(Qt, "AlignmentFlag"), Qt)
ALIGN_VCENTER = _find("AlignVCenter", _scope(Qt, "AlignmentFlag"), Qt)
ALIGN_CENTER = _find("AlignCenter", _scope(Qt, "AlignmentFlag"), Qt)
SCROLLBAR_ALWAYS_ON = _find("ScrollBarAlwaysOn", _scope(Qt, "ScrollBarPolicy"), Qt)
RIGHT_DOCK_AREA = _find("RightDockWidgetArea", _scope(Qt, "DockWidgetArea"), Qt)

# ---- QFrame ----
FRAME_HLINE = _find("HLine", _scope(QFrame, "Shape"), QFrame)
FRAME_SUNKEN = _find("Sunken", _scope(QFrame, "Shadow"), QFrame)

# ---- QGIS message levels ----
_LEVELS = _scope(Qgis, "MessageLevel")
MSG_INFO = _find("Info", _LEVELS, Qgis)
MSG_WARNING = _find("Warning", _LEVELS, Qgis)
MSG_CRITICAL = _find("Critical", _LEVELS, Qgis)
MSG_SUCCESS = _find("Success", _LEVELS, Qgis)

# ---- layer / field combo-box filters ----
_LAYER = (_scope(Qgis, "LayerFilter"), _scope(QgsMapLayerProxyModel, "Filter"),
          QgsMapLayerProxyModel)
POINT_LAYER = _find("PointLayer", *_LAYER)
LINE_LAYER = _find("LineLayer", *_LAYER)
POLYGON_LAYER = _find("PolygonLayer", *_LAYER)
FIELD_NUMERIC = _find("Numeric", _scope(QgsFieldProxyModel, "Filter"), QgsFieldProxyModel)
