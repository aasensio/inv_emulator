"""Interactive viewer of an inversion result (mlp_bspline.inversion.save_result).

    python view_inversion.py inversion_result_si10827_he10830.h5 [--telluric 10832.0]

Works for any spectral region (e.g. Fe I 6301/6302 or Si I + He I 10830).

Top: maps of a photospheric variable at the log(tau) chosen with the
slider, and of a per-pixel variable (He I slab, telluric line, chi2).
Hovering over either map selects the pixel shown below: the depth
stratification of every photospheric variable (line) with its node values
(dots), and the observed (grey) and synthetic (red) Stokes profiles. Click on
a map to freeze/unfreeze the pixel; zoom and pan of both maps are linked.
"""
import argparse
import sys

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

STOKES = ("I", "Q", "U", "V")
UNITS = {
    "T": "K", "vz": "km/s", "vmic": "km/s", "Bx": "G", "By": "G", "Bz": "G", "v_ref": "km/s",
    "He_Bx": "G", "He_By": "G", "He_Bz": "G", "He_v": "km/s", "He_deltav": "km/s", "He_height": "arcsec",
    "Tel_shift": "A", "Tel_width": "A",
}
# signed quantities: diverging colormap, levels symmetric about 0
SIGNED = {"vz", "Bx", "By", "Bz", "v_ref", "He_Bx", "He_By", "He_Bz", "He_v", "Tel_shift"}
# rest wavelengths (A); only those inside the observed range are marked
LINES = {"Fe I 6301": 6301.5012, "Fe I 6302": 6302.4936, "Si I": 10827.089, "He I": 10830.30}


def label(name):
    return f"{name} [{UNITS[name]}]" if name in UNITS else name


class Result:
    """Contents of the HDF5 file; depth variables are those with a
    stratification, all other node variables have one value per pixel."""

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.logtau = f["logtau"][:]
            self.strat = {k: f["stratification"][k][:] for k in f["stratification"]}
            self.nodes = {k: f["nodes"][k][:] for k in f["nodes"]}
            self.node_pos = {k: f["node_positions"][k][:] for k in f["node_positions"]}
            self.wave = f["wavelength"][:]
            self.syn = f["synthetic"][:]
            if "synthetic_nopsf" in f:
                self.syn_nopsf = f["synthetic_nopsf"][:]
            else:
                self.syn_nopsf = None            
            self.obs = f["observations"][:] if "observations" in f else None
            self.chi2 = f["chi2"][:]
            self.v_ref = f["v_ref"][:]
        order = ["T", "vz", "vmic", "Bx", "By", "Bz"]
        self.depth_vars = [k for k in order if k in self.strat] + [k for k in self.strat if k not in order]
        self.slab_vars = [k for k in self.nodes if k not in self.strat]
        self.nx, self.ny = self.chi2.shape
        self.slab_maps = {k: self.nodes[k][..., 0] for k in self.slab_vars}
        self.slab_maps["chi2"] = self.chi2

    def depth_map(self, name, k):
        if name == "chi2":
            return self.chi2
        if name == "v_ref":
            return self.v_ref
        return self.strat[name][:, :, k]


class MapPanel(QtWidgets.QWidget):
    """Variable selector + image with colorbar/histogram, a pixel marker and
    the value at that pixel as a horizontal line in the colorbar."""

    def __init__(self, title, names):
        super().__init__()
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(names)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"))
        head.addWidget(self.combo, 1)
        self.plot = pg.PlotItem()
        self.plot.setLabel("bottom", "x [pixel]")
        self.plot.setLabel("left", "y [pixel]")
        self.imv = pg.ImageView(view=self.plot)
        self.plot.invertY(False)  # y up, as solar maps
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()
        self.marker = pg.ScatterPlotItem(size=14, symbol="+", pen=pg.mkPen("w", width=2), brush=None)
        self.plot.addItem(self.marker)
        # the histogram's vertical axis is the data value (same as the colorbar)
        # (cyan: not present in the viridis or CET-D1 colormaps)
        self.value_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#00e5ff", width=2),
                                          label="{value:.4g}", labelOpts={"position": 0.5, "color": "w",
                                                                         "fill": (0, 0, 0, 200)})
        self.value_line.setZValue(100)
        self.imv.getHistogramWidget().item.vb.addItem(self.value_line)
        self.data = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(head)
        lay.addWidget(self.imv)
        self._first = True

    def show_map(self, name, data):
        data = np.asarray(data, dtype=np.float32)
        self.data = data
        finite = data[np.isfinite(data)]
        lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
        if name in SIGNED:
            m = max(abs(lo), abs(hi), 1e-12)
            lo, hi = -m, m
            cmap = "CET-D1"
        else:
            cmap = "viridis"
        if hi <= lo:
            hi = lo + 1e-6
        self.imv.setImage(data, autoLevels=False, levels=(lo, hi), autoRange=self._first,
                          autoHistogramRange=True)
        self.imv.setColorMap(pg.colormap.get(cmap))
        self.plot.setTitle(label(name))
        self._first = False

    def set_marker(self, i, j):
        self.marker.setData([i + 0.5], [j + 0.5])
        self.show_value(i, j)

    def show_value(self, i, j):
        v = None if self.data is None else float(self.data[i, j])
        self.value_line.setVisible(bool(v is not None and np.isfinite(v)))
        if self.value_line.isVisible():
            self.value_line.setValue(v)


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, res: Result, telluric=None):
        super().__init__()
        self.res = res
        self.telluric = telluric
        self.pixel = (res.nx // 2, res.ny // 2)
        self.frozen = False
        self.setWindowTitle("Inversion viewer")

        # ---- top: maps ----
        self.si = MapPanel("Photosphere", res.depth_vars + ["v_ref", "chi2"])
        self.he = MapPanel("Per-pixel (slab / telluric)", res.slab_vars + ["chi2"])
        self.he.plot.setXLink(self.si.plot)
        self.he.plot.setYLink(self.si.plot)
        self.depth_order = np.argsort(res.logtau)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(res.logtau) - 1)
        self.slider.setValue(int(np.argmin(np.abs(res.logtau[self.depth_order] + 1.0))))
        self.slider_label = QtWidgets.QLabel()
        self.slider_label.setMinimumWidth(110)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("log τ"))
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.slider_label)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.si, 1)
        left.addLayout(srow)
        top = QtWidgets.QWidget()
        tl = QtWidgets.QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addLayout(left, 1)
        tl.addWidget(self.he, 1)

        # ---- middle: stratifications ----
        self.strat_win = pg.GraphicsLayoutWidget()
        self.strat_curves, self.strat_nodes, self.strat_vlines = {}, {}, []
        for k, name in enumerate(res.depth_vars):
            p = self.strat_win.addPlot(row=0, col=k)
            p.setLabel("bottom", "log τ")
            p.setTitle(label(name), size="9pt")
            p.showGrid(x=True, y=True, alpha=0.3)
            self.strat_curves[name] = p.plot(pen=pg.mkPen("#2a78d6", width=2))
            self.strat_nodes[name] = p.plot(pen=None, symbol="o", symbolSize=7, symbolBrush="#eb6834")
            vl = pg.InfiniteLine(angle=90, pen=pg.mkPen("#888", style=QtCore.Qt.PenStyle.DashLine))
            p.addItem(vl)
            self.strat_vlines.append(vl)

        # ---- bottom: Stokes profiles ----
        self.stokes_win = pg.GraphicsLayoutWidget()
        self.obs_curves, self.syn_curves, self.syn_nopsf_curves, self.stokes_plots = [], [], [], []
        for k, s in enumerate(STOKES):
            p = self.stokes_win.addPlot(row=0, col=k)
            p.setLabel("bottom", "λ [Å]")
            title = s if k == 0 else f"{s} / I<sub>c</sub>"
            if k == 0:
                title += "  (<span style='color:#999'>observed</span>, <span style='color:#d6402a'>synthetic</span>)"
            p.setTitle(title, size="9pt")
            p.showGrid(x=True, y=True, alpha=0.3)
            self.obs_curves.append(p.plot(pen=pg.mkPen("#999", width=1), symbol="o", symbolSize=3,
                                          symbolPen=None, symbolBrush="#999",
                                          connect="finite"))
            self.syn_curves.append(p.plot(pen=pg.mkPen("#d6402a", width=2), connect="finite"))
            if res.syn_nopsf is not None:
                self.syn_nopsf_curves.append(p.plot(pen=pg.mkPen("#5a6a8a", width=2), connect="finite"))
            for l0 in LINES.values():
                if not res.wave.min() <= l0 <= res.wave.max():
                    continue  # an InfiniteLine outside the data would stretch the axis
                p.addItem(pg.InfiniteLine(l0, angle=90, pen=pg.mkPen("#ccc", style=QtCore.Qt.PenStyle.DotLine)))
            self.stokes_plots.append(p)
        if self.telluric is not None and "Tel_shift" in res.nodes:
            self.tel_lines = [pg.InfiniteLine(angle=90, pen=pg.mkPen("#3a9a3a", style=QtCore.Qt.PenStyle.DotLine))
                              for _ in STOKES]
            for p, l in zip(self.stokes_plots, self.tel_lines):
                p.addItem(l)
        else:
            self.tel_lines = []

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        for w in (top, self.strat_win, self.stokes_win):
            split.addWidget(w)
        split.setSizes([520, 230, 250])
        self.setCentralWidget(split)
        self.status = self.statusBar()

        # ---- signals ----
        self.si.combo.currentTextChanged.connect(self.update_si_map)
        self.he.combo.currentTextChanged.connect(self.update_he_map)
        self.slider.valueChanged.connect(self.update_si_map)
        for panel in (self.si, self.he):
            panel.imv.scene.sigMouseMoved.connect(lambda pos, pl=panel: self.on_move(pl, pos))
            panel.imv.scene.sigMouseClicked.connect(lambda ev, pl=panel: self.on_click(pl, ev))

        self.update_si_map()
        self.update_he_map()
        self.set_pixel(*self.pixel)

    # -- maps ----------------------------------------------------------------
    def depth_index(self):
        return int(self.depth_order[self.slider.value()])

    def update_si_map(self, *_):
        k = self.depth_index()
        lt = self.res.logtau[k]
        self.slider_label.setText(f"{lt:+.2f}")
        name = self.si.combo.currentText()
        self.si.show_map(name, self.res.depth_map(name, k))
        if name in self.res.strat:
            self.si.plot.setTitle(f"{label(name)} at log τ = {lt:+.2f}")
        for vl in self.strat_vlines:
            vl.setValue(lt)
        self.si.show_value(*self.pixel)

    def update_he_map(self, *_):
        name = self.he.combo.currentText()
        self.he.show_map(name, self.res.slab_maps[name])
        self.he.show_value(*self.pixel)

    # -- pixel selection -----------------------------------------------------
    def _pixel_at(self, panel, scene_pos):
        vb = panel.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return None
        p = vb.mapSceneToView(scene_pos)
        i, j = int(np.floor(p.x())), int(np.floor(p.y()))
        if 0 <= i < self.res.nx and 0 <= j < self.res.ny:
            return i, j
        return None

    def on_move(self, panel, pos):
        if self.frozen:
            return
        ij = self._pixel_at(panel, pos)
        if ij is not None and ij != self.pixel:
            self.set_pixel(*ij)

    def on_click(self, panel, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ij = self._pixel_at(panel, ev.scenePos())
        if ij is None:
            return
        self.frozen = not self.frozen
        self.set_pixel(*ij)

    def set_pixel(self, i, j):
        r = self.res
        self.pixel = (i, j)
        for panel in (self.si, self.he):
            panel.set_marker(i, j)
            panel.marker.setPen(pg.mkPen("r" if self.frozen else "w", width=2))
        for name in r.depth_vars:
            self.strat_curves[name].setData(r.logtau, r.strat[name][i, j])
            self.strat_nodes[name].setData(r.node_pos[name], r.nodes[name][i, j])
        for s in range(4):
            if r.obs is not None:
                self.obs_curves[s].setData(r.wave, r.obs[i, j, s])
            self.syn_curves[s].setData(r.wave, r.syn[i, j, s])
            if r.syn_nopsf is not None:
                self.syn_nopsf_curves[s].setData(r.wave, r.syn_nopsf[i, j, s])
        if self.tel_lines:
            for l in self.tel_lines:
                l.setValue(self.telluric + float(r.nodes["Tel_shift"][i, j, 0]))
        slab = "  ".join(f"{k}={r.slab_maps[k][i, j]:.4g}" for k in r.slab_vars)
        state = "FROZEN (click to release)" if self.frozen else "hover (click to freeze)"
        self.status.showMessage(f"pixel ({i}, {j})  chi2={r.chi2[i, j]:.3g}  v_ref={r.v_ref[i, j]:.3g} km/s  "
                                f"{slab}   [{state}]")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", nargs="?", default="inversion_result_si10827_he10830.h5")
    ap.add_argument("--telluric", type=float, default=None,
                    help="reference wavelength (A) of the telluric line, to mark lambda0 = ref + Tel_shift")
    ap.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    pg.setConfigOptions(antialias=True, imageAxisOrder="col-major")
    app = pg.mkQApp("Inversion viewer")
    win = Viewer(Result(args.result), args.telluric)
    win.resize(1500, 1000)
    win.show()
    if args.screenshot:
        app.processEvents()
        win.grab().save(args.screenshot)
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
