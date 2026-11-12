# -*- coding: utf-8 -*-
"""
GeoSubSamplerDockWidget — pure-Python UI (no .ui file required).
"""

import sys

from qgis.PyQt import QtWidgets
from qgis.PyQt.QtCore import pyqtSignal, QSettings
from qgis.gui import QgsMapLayerComboBox, QgsFieldComboBox, QgsDoubleSpinBox
from .qt_compat import (ALIGN_RIGHT, ALIGN_LEFT, ALIGN_VCENTER, ALIGN_CENTER, SCROLLBAR_ALWAYS_ON,
                        FRAME_HLINE, FRAME_SUNKEN, FIELD_NUMERIC)

_SB_ALWAYS = SCROLLBAR_ALWAYS_ON

# The native macOS style pads buttons/combos heavily and the default font is
# larger than on Windows, so the same layout comes out ~30% bigger. On macOS
# the dock uses a smaller font and the compact Fusion style instead.
# Set QSettings "GeoSubSampler/compactMacUI" to false to keep the native look.
IS_MACOS = sys.platform == "darwin"
MACOS_FONT_SCALE = 0.85


def _compact_mac_ui():
    if not IS_MACOS:
        return False
    return str(QSettings().value("GeoSubSampler/compactMacUI", True)).lower() not in ("false", "0")


COMPACT_MAC = _compact_mac_ui()


class GeoSubSamplerDockWidget(QtWidgets.QDockWidget):

    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GeoSubSampler")
        self._build_ui()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lbl(text, align=ALIGN_RIGHT | ALIGN_VCENTER):
        w = QtWidgets.QLabel(text)
        w.setAlignment(align)
        return w

    @staticmethod
    def _hline():
        line = QtWidgets.QFrame()
        line.setFrameShape(FRAME_HLINE)
        line.setFrameShadow(FRAME_SUNKEN)
        return line

    # ------------------------------------------------------------------
    # Top-level build
    # ------------------------------------------------------------------

    def _build_ui(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(_SB_ALWAYS)
        scroll.setVerticalScrollBarPolicy(_SB_ALWAYS)

        content = QtWidgets.QWidget()
        content.setMinimumWidth(520)
        ml = QtWidgets.QVBoxLayout(content)
        ml.setSpacing(6)
        ml.setContentsMargins(6, 6, 6, 6)

        # The tab bar is separate from the pages so the shared selectors can sit between the
        # two: they are the same widgets whichever tab is showing, so they keep their values.
        self.tabBar_sections = QtWidgets.QTabBar()
        self.stackedWidget_sections = QtWidgets.QStackedWidget()
        self.tabBar_sections.currentChanged.connect(self.stackedWidget_sections.setCurrentIndex)

        ml.addWidget(self.tabBar_sections)
        self._build_layer_selectors(ml)
        self._build_priority_group(ml)
        ml.addWidget(self.stackedWidget_sections, 1)

        self._add_tab("Clean", self._build_fixmap_group, self._build_densify_group, self._build_thin_group)
        self._add_tab("Points", self._build_orientation_group)
        self._add_tab("Faults", self._build_fault_group)
        self._add_tab("Polygons", self._build_geology_group, self._build_simplify_group, self._build_tp_group)

        scroll.setWidget(content)
        self.setWidget(scroll)

        if COMPACT_MAC:
            self._apply_compact_mac_style(scroll)

    def _add_tab(self, title, *builders):
        page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(page)
        vbox.setSpacing(6)
        vbox.setContentsMargins(0, 0, 0, 0)
        for build in builders:
            build(vbox)
        vbox.addStretch()
        self.stackedWidget_sections.addWidget(page)
        self.tabBar_sections.addTab(title)

    def _apply_compact_mac_style(self, top):
        # QGIS sets the font size through an application stylesheet, which
        # overrides setFont(), so the smaller size has to be a stylesheet too.
        try:
            base = float(QSettings().value("qgis/stylesheet/fontPointSize", 0))
        except (TypeError, ValueError):
            base = 0
        if base <= 0:
            base = QtWidgets.QApplication.font().pointSizeF()
        if base > 0:
            top.setStyleSheet(
                "QWidget { font-size: %.1fpt; }" % (base * MACOS_FONT_SCALE)
            )
        # Keep a reference: QWidget.setStyle does not take ownership.
        self._compact_style = QtWidgets.QStyleFactory.create("Fusion")
        if self._compact_style is None:
            return
        top.setStyle(self._compact_style)
        for w in top.findChildren(QtWidgets.QWidget):
            w.setStyle(self._compact_style)

    # ------------------------------------------------------------------
    # Layer selectors and priority fields (top, shared by all the tabs)
    # ------------------------------------------------------------------

    def _build_layer_selectors(self, layout):
        self.mMapLayerComboBox_points          = QgsMapLayerComboBox()
        self.mMapLayerComboBox_fault_polylines  = QgsMapLayerComboBox()
        self.mMapLayerComboBox_maps_polygons    = QgsMapLayerComboBox()

        pairs = [
            ("Select Structure Point Layer",    self.mMapLayerComboBox_points),
            ("Select Fault Polyline Layer",      self.mMapLayerComboBox_fault_polylines),
            ("Select Litho/strat Polygon Layer", self.mMapLayerComboBox_maps_polygons),
        ]
        for lbl_text, combo in pairs:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self._lbl(lbl_text))
            row.addWidget(combo)
            layout.addLayout(row)
        self.pushButton_show_metadata = QtWidgets.QPushButton("Show metadata of the selected layer in the layer panel")
        layout.addWidget(self.pushButton_show_metadata)

    def _build_priority_group(self, layout):
        grp = QtWidgets.QGroupBox("Geology Polygon Priorities")
        grid = QtWidgets.QGridLayout(grp)
        self.mFieldComboBox_priority_1 = QgsFieldComboBox()
        self.mFieldComboBox_priority_2 = QgsFieldComboBox()
        self.mFieldComboBox_priority_3 = QgsFieldComboBox()
        self.mFieldComboBox_priority_4 = QgsFieldComboBox()
        self.mFieldComboBox_priority_5 = QgsFieldComboBox()
        cells = [
            ("Priority 1", self.mFieldComboBox_priority_1, 0, 0),
            ("2", self.mFieldComboBox_priority_2, 0, 2),
            ("3", self.mFieldComboBox_priority_3, 0, 4),
            ("4", self.mFieldComboBox_priority_4, 1, 0),
            ("5", self.mFieldComboBox_priority_5, 1, 2),
        ]
        for text, combo, row, col in cells:
            grid.addWidget(self._lbl(text), row, col)
            grid.addWidget(combo, row, col + 1)
        for col in (1, 3, 5):
            grid.setColumnStretch(col, 1)
        layout.addWidget(grp)

    # ------------------------------------------------------------------
    # Orientation Points group
    # ------------------------------------------------------------------

    def _build_orientation_group(self, layout):
        grp = QtWidgets.QGroupBox("Orientation Points")
        vbox = QtWidgets.QVBoxLayout(grp)

        # Dip / Dip Dir / convention row
        self.mFieldComboBox_dip     = QgsFieldComboBox()
        self.mFieldComboBox_dip_dir = QgsFieldComboBox()
        self.mFieldComboBox_dip.setFilters(FIELD_NUMERIC)
        self.mFieldComboBox_dip_dir.setFilters(FIELD_NUMERIC)
        self.checkBox_dip_dir = QtWidgets.QCheckBox("Dip Direction")
        self.checkBox_dip_dir.setChecked(True)

        dip_row = QtWidgets.QHBoxLayout()
        dip_row.addWidget(self._lbl("Dip"))
        dip_row.addWidget(self.mFieldComboBox_dip)
        dip_row.addWidget(self._lbl("Dip Direction/Strike"))
        dip_row.addWidget(self.mFieldComboBox_dip_dir)
        dip_row.addWidget(self.checkBox_dip_dir)
        vbox.addLayout(dip_row)

        vbox.addWidget(self._hline())

        # Radio group — subsampling algorithms
        self._subsample_group = QtWidgets.QButtonGroup(self)

        # (o) Stochastic
        self.radioButton_stochastic = QtWidgets.QRadioButton("Stochastic")
        self.radioButton_stochastic.setChecked(True)
        self.mQgsDoubleSpinBox_percent = QgsDoubleSpinBox()
        self.mQgsDoubleSpinBox_percent.setDecimals(1)
        self.mQgsDoubleSpinBox_percent.setSingleStep(5.0)
        self.mQgsDoubleSpinBox_percent.setValue(50.0)
        self.mQgsDoubleSpinBox_percent.setFixedWidth(75)
        row_stoch = QtWidgets.QHBoxLayout()
        row_stoch.addWidget(self.radioButton_stochastic)
        row_stoch.addWidget(self.mQgsDoubleSpinBox_percent)
        row_stoch.addWidget(self._lbl("%", ALIGN_LEFT | ALIGN_VCENTER))
        row_stoch.addStretch()
        vbox.addLayout(row_stoch)
        self._subsample_group.addButton(self.radioButton_stochastic)

        # (o) Kent
        self.radioButton_kent = QtWidgets.QRadioButton("Kent")
        self.lineEdit_grid_size_kent = QtWidgets.QLineEdit("5000")
        self.lineEdit_grid_size_kent.setFixedWidth(75)
        row_kent = QtWidgets.QHBoxLayout()
        row_kent.addWidget(self.radioButton_kent)
        row_kent.addWidget(self._lbl("Grid Size (m)"))
        row_kent.addWidget(self.lineEdit_grid_size_kent)
        self.lineEdit_kent_kappa_min = QtWidgets.QLineEdit("")
        self.lineEdit_kent_kappa_min.setFixedWidth(50)
        row_kent.addWidget(self._lbl("Min kappa"))
        row_kent.addWidget(self.lineEdit_kent_kappa_min)
        row_kent.addStretch()
        vbox.addLayout(row_kent)
        self._subsample_group.addButton(self.radioButton_kent)

        # (o) Kent Outlier
        self.radioButton_kent_outlier = QtWidgets.QRadioButton("Kent Outlier")
        self.lineEdit_grid_size_kent_2 = QtWidgets.QLineEdit("5000")
        self.lineEdit_grid_size_kent_2.setFixedWidth(75)
        self.lineEdit_kent_threshold = QtWidgets.QLineEdit("2.0")
        self.lineEdit_kent_threshold.setFixedWidth(50)
        self.lineEdit_kent_threshold.setEnabled(False)
        self.pushButton_subsample_points = QtWidgets.QPushButton("Subsample Points")
        row_ko = QtWidgets.QHBoxLayout()
        row_ko.addWidget(self.radioButton_kent_outlier)
        row_ko.addWidget(self._lbl("Grid Size (m)"))
        row_ko.addWidget(self.lineEdit_grid_size_kent_2)
        self.checkBox_kent_outlier_iter = QtWidgets.QCheckBox("Iterative")
        row_ko.addWidget(self.checkBox_kent_outlier_iter)
        row_ko.addStretch()
        row_ko.addWidget(self.pushButton_subsample_points)
        vbox.addLayout(row_ko)
        self._subsample_group.addButton(self.radioButton_kent_outlier)

        # (o) Carmichael — iterative multi-outlier removal
        self.radioButton_carmichael = QtWidgets.QRadioButton("Carmichael")
        self.lineEdit_grid_size_carm = QtWidgets.QLineEdit("5000")
        self.lineEdit_grid_size_carm.setFixedWidth(75)
        row_carm = QtWidgets.QHBoxLayout()
        row_carm.addWidget(self.radioButton_carmichael)
        row_carm.addWidget(self._lbl("Grid Size (m)"))
        row_carm.addWidget(self.lineEdit_grid_size_carm)
        row_carm.addStretch()
        vbox.addLayout(row_carm)
        self._subsample_group.addButton(self.radioButton_carmichael)

        # Target N — shared by all the grid cell methods
        self.checkBox_target_n = QtWidgets.QCheckBox("Auto-fit to N pts")
        self.spinBox_target_n = QtWidgets.QSpinBox()
        self.spinBox_target_n.setRange(1, 100000)
        self.spinBox_target_n.setValue(100)
        self.spinBox_target_n.setFixedWidth(75)
        self.spinBox_target_n.setEnabled(False)
        self.checkBox_target_n.toggled.connect(self.spinBox_target_n.setEnabled)
        row_target = QtWidgets.QHBoxLayout()
        row_target.addSpacing(20)
        row_target.addWidget(self.checkBox_target_n)
        row_target.addWidget(self.spinBox_target_n)
        row_target.addWidget(self._lbl("points (±15%)", ALIGN_LEFT | ALIGN_VCENTER))
        row_target.addStretch()
        vbox.addLayout(row_target)

        # (o) 1o Sampling — fired by the shared Subsample Points button
        self.radioButton_1o = QtWidgets.QRadioButton("1o Sampling")
        self.lineEdit_1o_distance = QtWidgets.QLineEdit("1000")
        self.lineEdit_1o_distance.setFixedWidth(70)
        self.lineEdit_1o_angle = QtWidgets.QLineEdit("15")
        self.lineEdit_1o_angle.setFixedWidth(50)
        row_1o = QtWidgets.QHBoxLayout()
        row_1o.addWidget(self.radioButton_1o)
        row_1o.addWidget(self._lbl("Distance"))
        row_1o.addWidget(self.lineEdit_1o_distance)
        row_1o.addWidget(self._lbl("Angle"))
        row_1o.addWidget(self.lineEdit_1o_angle)
        row_1o.addStretch()
        vbox.addLayout(row_1o)
        self._subsample_group.addButton(self.radioButton_1o)

        layout.addWidget(grp)

    # ------------------------------------------------------------------
    # Fault Polylines group
    # ------------------------------------------------------------------

    def _build_fault_group(self, layout):
        grp = QtWidgets.QGroupBox("Fault Polylines")
        vbox = QtWidgets.QVBoxLayout(grp)

        # Merge Segments — kept as a direct action button
        self.pushButton_merge_segments = QtWidgets.QPushButton("Merge Segments")
        self.lineEdit_merge_tolerance    = QtWidgets.QLineEdit("10")
        self.lineEdit_merge_tolerance.setFixedWidth(50)
        self.lineEdit_merge_search_angle = QtWidgets.QLineEdit("30")
        self.lineEdit_merge_search_angle.setFixedWidth(50)
        self.lineEdit_merge_join_angle   = QtWidgets.QLineEdit("150")
        self.lineEdit_merge_join_angle.setFixedWidth(50)
        row_merge = QtWidgets.QHBoxLayout()
        row_merge.addWidget(self.pushButton_merge_segments)
        row_merge.addWidget(self._lbl("Dist.(m)"))
        row_merge.addWidget(self.lineEdit_merge_tolerance)
        row_merge.addWidget(self._lbl("Search Angle"))
        row_merge.addWidget(self.lineEdit_merge_search_angle)
        row_merge.addWidget(self._lbl("Join Angle"))
        row_merge.addWidget(self.lineEdit_merge_join_angle)
        vbox.addLayout(row_merge)

        vbox.addWidget(self._hline())

        # Radio group — fault attribute methods
        self._fault_group = QtWidgets.QButtonGroup(self)

        self.radioButton_fault_length = QtWidgets.QRadioButton("Length Only")
        self.radioButton_fault_length.setChecked(True)
        self._fault_group.addButton(self.radioButton_fault_length)
        vbox.addWidget(self.radioButton_fault_length)

        self.radioButton_fault_graph = QtWidgets.QRadioButton("Graph then Length")
        self._fault_group.addButton(self.radioButton_fault_graph)
        vbox.addWidget(self.radioButton_fault_graph)

        self.radioButton_fault_strat_offset = QtWidgets.QRadioButton("Strat Offset  then Length [Requires Geology Polygon Priorities to be defined]")
        self._fault_group.addButton(self.radioButton_fault_strat_offset)
        vbox.addWidget(self.radioButton_fault_strat_offset)

        self.radioButton_fault_clusters = QtWidgets.QRadioButton("Orientation Clusters then Length")
        self._fault_group.addButton(self.radioButton_fault_clusters)
        self.pushButton_process_faults = QtWidgets.QPushButton("Process Faults")
        row_fc = QtWidgets.QHBoxLayout()
        row_fc.addWidget(self.radioButton_fault_clusters)
        row_fc.addStretch()
        row_fc.addWidget(self.pushButton_process_faults)
        vbox.addLayout(row_fc)

        layout.addWidget(grp)

    # ------------------------------------------------------------------
    # Geology Polygons group
    # ------------------------------------------------------------------

    def _build_geology_group(self, layout):
        grp = QtWidgets.QGroupBox("Geology Polygons")
        vbox = QtWidgets.QVBoxLayout(grp)

        # Ignore (dyke) field
        self.mFieldComboBox_dyke = QgsFieldComboBox()
        row_ignore = QtWidgets.QHBoxLayout()
        row_ignore.addWidget(self._lbl("Ignore Field"))
        row_ignore.addWidget(self.mFieldComboBox_dyke)
        vbox.addLayout(row_ignore)

        # Dyke codes text box
        self.plainTextEdit_dyke_Codes = QtWidgets.QPlainTextEdit()
        self.plainTextEdit_dyke_Codes.setMaximumHeight(55)
        row_dyke = QtWidgets.QHBoxLayout()
        row_dyke.addWidget(self._lbl("Ignore Codes\n(comma sep)"))
        row_dyke.addWidget(self.plainTextEdit_dyke_Codes)
        vbox.addLayout(row_dyke)

        # Tolerance / diameter / series / increment row
        self.lineEdit_node_tolerance = QtWidgets.QLineEdit("1")
        self.lineEdit_node_tolerance.setFixedWidth(50)
        self.lineEdit_polygon_area = QtWidgets.QLineEdit("5")
        self.lineEdit_polygon_area.setFixedWidth(50)
        self.checkBox_series = QtWidgets.QCheckBox("Series")
        self.mQgsDoubleSpinBox_upinc = QgsDoubleSpinBox()
        self.mQgsDoubleSpinBox_upinc.setDecimals(1)
        self.mQgsDoubleSpinBox_upinc.setMinimum(0.1)
        self.mQgsDoubleSpinBox_upinc.setSingleStep(1.0)
        self.mQgsDoubleSpinBox_upinc.setValue(1.0)
        self.mQgsDoubleSpinBox_upinc.setFixedWidth(75)
        row_tol = QtWidgets.QHBoxLayout()
        row_tol.addWidget(self._lbl("Tolerance (m)"))
        row_tol.addWidget(self.lineEdit_node_tolerance)
        row_tol.addWidget(self._lbl("Diameter Threshold (km)"))
        row_tol.addWidget(self.lineEdit_polygon_area)
        row_tol.addWidget(self.checkBox_series)
        row_tol.addWidget(self._lbl("Increment (km)"))
        row_tol.addWidget(self.mQgsDoubleSpinBox_upinc)
        vbox.addLayout(row_tol)

        # Rescale Map button
        self.pushButton_minPolyArea = QtWidgets.QPushButton("Rescale Map")
        vbox.addWidget(self.pushButton_minPolyArea)

        layout.addWidget(grp)

    # ---- Fix Map group (topology clean-up of faults and geology polygons) --

    def _build_fixmap_group(self, layout):
        grp = QtWidgets.QGroupBox("Fix Map")
        vbox = QtWidgets.QVBoxLayout(grp)
        self.lineEdit_fixmap_tolerance = QtWidgets.QLineEdit("5")
        self.lineEdit_fixmap_tolerance.setFixedWidth(60)
        self.comboBox_fixmap_overlap = QtWidgets.QComboBox()
        self.comboBox_fixmap_overlap.addItems(["Keep smaller polygon", "Keep larger polygon", "Leave overlaps"])
        self.checkBox_fixmap_fillgaps = QtWidgets.QCheckBox("Fill enclosed gaps")
        self.checkBox_fixmap_fillgaps.setChecked(True)
        self.pushButton_repair_polygons = QtWidgets.QPushButton("Repair (buffer 0)")
        self.pushButton_fixmap = QtWidgets.QPushButton("Fix Map")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._lbl("Tolerance (m)"))
        row.addWidget(self.lineEdit_fixmap_tolerance)
        row.addStretch()
        row.addWidget(self.pushButton_repair_polygons)
        row.addWidget(self.pushButton_fixmap)
        vbox.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(self.checkBox_fixmap_fillgaps)
        row2.addWidget(self._lbl("Remaining overlaps"))
        row2.addWidget(self.comboBox_fixmap_overlap)
        row2.addStretch()
        vbox.addLayout(row2)
        layout.addWidget(grp)

    # ---- Densify group (faults and geology polygons) -----------------

    def _build_densify_group(self, layout):
        grp = QtWidgets.QGroupBox("Densify Faults and Geology")
        vbox = QtWidgets.QVBoxLayout(grp)
        self.lineEdit_densify_spacing = QtWidgets.QLineEdit("100")
        self.lineEdit_densify_spacing.setFixedWidth(60)
        self.checkBox_densify_crossings = QtWidgets.QCheckBox("Shared node at fault crossing")
        self.checkBox_densify_crossings.setChecked(True)
        self.pushButton_densify = QtWidgets.QPushButton("Densify Nodes")
        row_dens = QtWidgets.QHBoxLayout()
        row_dens.addWidget(self._lbl("Node Spacing (m)"))
        row_dens.addWidget(self.lineEdit_densify_spacing)
        row_dens.addWidget(self.checkBox_densify_crossings)
        row_dens.addStretch()
        row_dens.addWidget(self.pushButton_densify)
        vbox.addLayout(row_dens)
        layout.addWidget(grp)

    # ---- Thin group (remove close nodes, faults and geology) ------------

    def _build_thin_group(self, layout):
        grp = QtWidgets.QGroupBox("Thin Faults and Geology")
        vbox = QtWidgets.QVBoxLayout(grp)
        self.lineEdit_thin_distance = QtWidgets.QLineEdit("50")
        self.lineEdit_thin_distance.setFixedWidth(60)
        self.pushButton_thin = QtWidgets.QPushButton("Thin Nodes")
        row_thin = QtWidgets.QHBoxLayout()
        row_thin.addWidget(self._lbl("Min Node Spacing (m)"))
        row_thin.addWidget(self.lineEdit_thin_distance)
        row_thin.addStretch()
        row_thin.addWidget(self.pushButton_thin)
        vbox.addLayout(row_thin)
        layout.addWidget(grp)

    # ---- Simplify Map group --------------------------------------------

    def _build_simplify_group(self, layout):
        grp = QtWidgets.QGroupBox("Simplify Map")
        vbox = QtWidgets.QVBoxLayout(grp)
        self.lineEdit_area_tolerance = QtWidgets.QLineEdit("10000")
        self.lineEdit_area_tolerance.setFixedWidth(80)
        self.pushButton_simplifyMap = QtWidgets.QPushButton("Simplify Map")
        row_area = QtWidgets.QHBoxLayout()
        row_area.addWidget(self._lbl("Area Tolerance (m2)"))
        row_area.addWidget(self.lineEdit_area_tolerance)
        row_area.addWidget(self.pushButton_simplifyMap)
        row_area.addStretch()
        vbox.addLayout(row_area)
        layout.addWidget(grp)

    # ---- Töpfer & Pillewizer scaling group ---------------------------

    def _build_tp_group(self, layout):
        grp = QtWidgets.QGroupBox("Töpfer && Pillewizer Scaling")
        vbox = QtWidgets.QVBoxLayout(grp)

        # Equation reminder label
        eq_lbl = QtWidgets.QLabel("TN = ON × (OS/TS)^(x/2)   x: 1=points  2=lines  3=polygons")
        eq_lbl.setAlignment(ALIGN_CENTER)
        f = eq_lbl.font()
        f.setItalic(True)
        eq_lbl.setFont(f)
        eq_lbl.setToolTip(
            "Töpfer & Pillewizer (1966) selection equation\n\n"
            "TN  — target number of objects after generalisation\n"
            "ON  — original number of objects before generalisation\n"
            "OS  — original map scale denominator (e.g. 50 000)\n"
            "TS  — target map scale denominator (e.g. 500 000)\n"
            "x   — object type: 1 = points, 2 = lines, 3 = polygons\n\n"
            "Example: going from 1:50 000 to 1:500 000\n"
            "  OS/TS = 0.1,  points kept = ON × 0.1^0.5 ≈ 32 %\n"
            "                lines  kept = ON × 0.1^1.0  = 10 %\n"
            "                polys  kept = ON × 0.1^1.5  ≈  3 %"
        )
        vbox.addWidget(eq_lbl)

        vbox.addWidget(self._hline())

        # OS/TS ratio + increment row
        self.lineEdit_tp_ratio = QtWidgets.QLineEdit("0.1")
        self.lineEdit_tp_ratio.setFixedWidth(70)
        self.lineEdit_tp_increment = QtWidgets.QLineEdit("0.1")
        self.lineEdit_tp_increment.setFixedWidth(70)
        self.label_tp_steps = QtWidgets.QLabel("")
        self.pushButton_tp_run = QtWidgets.QPushButton("Run T&&P Scaling")
        row_params = QtWidgets.QHBoxLayout()
        row_params.addWidget(self._lbl("OS/TS ratio"))
        row_params.addWidget(self.lineEdit_tp_ratio)
        row_params.addWidget(self._lbl("Increment"))
        row_params.addWidget(self.lineEdit_tp_increment)
        row_params.addWidget(self.label_tp_steps)
        row_params.addStretch()
        row_params.addWidget(self.pushButton_tp_run)
        vbox.addLayout(row_params)

        self.lineEdit_tp_ratio.textChanged.connect(self._update_tp_steps_label)
        self.lineEdit_tp_increment.textChanged.connect(self._update_tp_steps_label)
        self._update_tp_steps_label()

        layout.addWidget(grp)

    # ------------------------------------------------------------------

    def _update_tp_steps_label(self, _=None):
        try:
            ratio     = float(self.lineEdit_tp_ratio.text())
            increment = float(self.lineEdit_tp_increment.text())
        except ValueError:
            self.label_tp_steps.setText("")
            return

        if increment <= 0:
            n = 1
        else:
            n, current = 0, round(1.0 - increment, 12)
            while current > ratio:
                n += 1
                current = round(current - increment, 12)
            n += 1  # the final step that reaches target_ratio

        self.label_tp_steps.setText(f"({n} step{'s' if n != 1 else ''})")

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()