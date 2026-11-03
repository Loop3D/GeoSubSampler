"""
A stand-in for GeoSubSampler when Python packages it needs are not installed.

The plugin's modules import scikit-learn, NetworkX, geopandas ... as they load,
so without them the plugin would fail to load with an import error. Instead,
classFactory() checks requirements.txt first and, if packages are missing,
loads this stand-in: it keeps the GeoSubSampler button and menu entry, warns
which packages are missing, and explains how to install them. Once they are
installed and QGIS restarted, the full plugin loads.
"""

import importlib.util
import os
import re

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox

from .qt_compat import MSG_WARNING

HERE = os.path.dirname(os.path.abspath(__file__))

# pip name -> import name, where they differ
IMPORT_NAMES = {"scikit-learn": "sklearn"}
# imported only when the tool that needs it runs (polygon triangulation)
OPTIONAL = {"triangle"}


def required_packages():
    """The package names listed in requirements.txt."""
    try:
        with open(os.path.join(HERE, "requirements.txt"), encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    names = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0])
    return names


def _found(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # a package in a broken state counts as missing
        return False


def missing_packages():
    """Packages from requirements.txt needed to load the plugin that cannot be found."""
    return [name for name in required_packages()
            if name not in OPTIONAL and not _found(IMPORT_NAMES.get(name, name))]


class MissingPackagesPlugin:
    """Keeps GeoSubSampler's button and menu entry, and says what to install."""

    MENU = "&GeoSubSampler"

    def __init__(self, iface, missing, error=None):
        self.iface = iface
        self.missing = missing
        self.error = error  # an installed package that failed to import
        self.action = None

    def _summary(self):
        if self.error:
            return f"a required Python package could not be loaded ({self.error})"
        return "these Python packages are not installed: " + ", ".join(self.missing)

    def initGui(self):
        self.action = QAction(QIcon(os.path.join(HERE, "icon.png")), "GeoSubSampler",
                              self.iface.mainWindow())
        self.action.triggered.connect(self.explain)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu(self.MENU, self.action)
        self.iface.messageBar().pushMessage(
            "GeoSubSampler",
            f"GeoSubSampler cannot start: {self._summary()}. "
            "Click the GeoSubSampler button for how to install them.",
            level=MSG_WARNING, duration=0,
        )

    def explain(self):
        packages = " ".join(self.missing) if not self.error else " ".join(required_packages())
        QMessageBox.warning(
            self.iface.mainWindow(), "GeoSubSampler",
            f"GeoSubSampler cannot start: {self._summary()}.\n\n"
            "To install them, either:\n"
            "  - install the QPIP plugin (Plugins > Manage and Install Plugins), which adds the "
            "packages listed in GeoSubSampler's requirements.txt, or\n"
            "  - run in the OSGeo4W Shell (Windows) or a terminal using QGIS's Python:\n"
            f"        python -m pip install {packages}\n\n"
            "Then restart QGIS.",
        )

    def unload(self):
        if self.action is not None:
            self.iface.removePluginMenu(self.MENU, self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None
