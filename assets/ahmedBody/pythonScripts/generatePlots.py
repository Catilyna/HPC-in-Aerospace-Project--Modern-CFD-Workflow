"""Paper-style figures for the single Ahmed-body OpenFOAM case.

Reproduces the figures of Tunay, Sahin & Ozbolat, "Effects of rear slant angles
on the flow characteristics of Ahmed body", Exp. Therm. Fluid Sci. 57 (2014)
165-176, for ONE geometry (the paper shows three slant angles side by side):

    fig04_streamlines          Fig. 4   mean streamlines + critical points
    fig05_critical_points      Fig. 5   locations of F1, F2, S
    fig06_vorticity            Fig. 6   mean vorticity contours
    fig07_velocity_vectors     Fig. 7   mean velocity vectors, <u>=0 line
    fig08_vorticity_transverse Fig. 8   mean vorticity, transverse planes
    fig09_u_profiles           Fig. 9   <u>/U profiles, slant + wake
    fig10_v_profiles           Fig. 10  <v>/U profiles, slant + wake
    fig11_tke                  Fig. 11  mean TKE
    fig12_vorticity_instant    Fig. 12  instantaneous vorticity, 3 times
    velocity_psd               Fig. 13  probe map + spectra of u and v
    velocity_profiles          (extra)  line-sampled profiles from 'sets'
    force_coefficients         (extra) drag and viscous-drag histories

Run AFTER the OpenFOAM post-processing (wakePlanes, bodySurfaces, probes1,
velocityProfiles, forces).  Tables strouhal.tex, critical_points.tex and
force_coefficients.tex are written to report/ and are \\input by report.tex.
"""
import csv
import re
import sys
import traceback
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import vtk
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, Rectangle
from scipy.signal import welch
from scipy.spatial import ConvexHull
from vtk.util.numpy_support import vtk_to_numpy

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
CASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = CASE_DIR / "report"
IMAGE_DIR = REPORT_DIR / "images"
POST_DIR = CASE_DIR / "postProcessing"

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
# Component index of the streamwise, vertical and spanwise directions
# (paper convention: x streamwise, y vertical, z spanwise, symmetry plane z=0).
IX, IV, IS = 0, 2, 1     # this case: x streamwise, z VERTICAL, y spanwise (symmetry plane y = const)

# Reference quantities.  None = detect from the case (body.vtk, 0/U); the
# FALLBACK values are used (with a warning) when detection is impossible.
MODEL_H = None      # model height H [m]
X_REAR = None       # x of the rear vertical face  (x/H = 0)
Y_BOTTOM = None     # y of the underside of the body (y/H = 0)
U_INF = None        # free-stream speed [m/s]
FALLBACK = dict(H=0.288, X_REAR=1.044, Y_BOTTOM=0.05, U_INF=1.0)  # standard Ahmed

STATS_START_TIME = 75.0      # discard probe data before this time (= fieldAverage timeStart) TODO: Automate this number decision (when is stationary state reached?).
PSD_SEGMENT = 1024          # Welch segment length [samples]

XLIM = (-0.7, 1.5)          # x/H window of the symmetry-plane figures
SLANT_STEP = 0.069          # spacing of the profiles on the slant  [x/H]  (paper)
WAKE_STEP = 0.139           # spacing of the profiles in the wake   [x/H]  (paper)
WAKE_MAX = 1.45             # last wake profile [x/H]
PROFILE_SCALE_U = 0.07      # x/H length of 1 unit of <u>/U in the profile plots
PROFILE_SCALE_V = 0.21      # x/H length of 1 unit of <v>/U in the profile plots
BLUE = "#2b2bff"            # body colour used by the paper
RHO_INF = 1.0               # must match system/postProcessing (incompressible)
A_REF = 0.0835              # frontal reference area [m2], must match Aref

plt.rcParams.update({
    "font.size": 9,
    "axes.linewidth": 0.6,
    "contour.negative_linestyle": "dashed",
    "mathtext.default": "regular",
})


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def time_dirs(base: Path):
    """Sorted list of (time, path) for the numeric sub-directories of base."""
    found = []
    if base.is_dir():
        for p in base.iterdir():
            if p.is_dir():
                try:
                    found.append((float(p.name), p))
                except ValueError:
                    pass
    return sorted(found, key=lambda item: item[0])


def latest_time_dir(base: Path):
    dirs = time_dirs(base)
    return dirs[-1][1] if dirs else None


def nice_step(value: float) -> float:
    """Round value up to 1-2-2.5-5-10 x 10^n."""
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / 10 ** exponent
    for m in (1, 2, 2.5, 5, 10):
        if fraction <= m * 1.0000001:
            return m * 10 ** exponent
    return 10 ** (exponent + 1)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


# ----------------------------------------------------------------------------
# VTK surface reading (VTK is only used as a file reader here)
# ----------------------------------------------------------------------------
class Surface:
    """Triangulated surface with point data as numpy arrays."""

    def __init__(self, points, tri, data):
        self.points = points      # (N, 3)
        self.tri = tri            # (M, 3) int
        self.data = data          # name -> (N,) or (N, k)

    def has(self, name):
        return name in self.data

    @property
    def normal_axis(self):
        spread = self.points.max(axis=0) - self.points.min(axis=0)
        return int(np.argmin(spread))


def read_surface(path: Path) -> Surface:
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(str(path))
    reader.Update()
    dataset = reader.GetOutput()
    if dataset is None or dataset.GetNumberOfPoints() == 0:
        raise RuntimeError(f"cannot read any points from {path}")

    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputData(dataset)
    clean = vtk.vtkCleanPolyData()          # merge duplicated points
    clean.SetInputConnection(geometry.GetOutputPort())
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(clean.GetOutputPort())
    c2p = vtk.vtkCellDataToPointData()      # OpenFOAM writes face (cell) data
    c2p.SetInputConnection(triangles.GetOutputPort())
    c2p.PassCellDataOff()
    c2p.Update()
    poly = c2p.GetOutput()

    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    polys = poly.GetPolys()
    if hasattr(polys, "GetConnectivityArray"):          # VTK >= 9 (all cells are triangles here)
        tri = vtk_to_numpy(polys.GetConnectivityArray()).reshape(-1, 3).astype(int)
    else:
        tri = vtk_to_numpy(polys.GetData()).reshape(-1, 4)[:, 1:4].astype(int)
    data = {}
    pdata = poly.GetPointData()
    for i in range(pdata.GetNumberOfArrays()):
        name = pdata.GetArrayName(i)
        if name:
            data[name] = vtk_to_numpy(pdata.GetArray(i)).astype(float)
    return Surface(points, tri, data)


# ----------------------------------------------------------------------------
# Case description: H, x/H origin, free-stream speed, body outline
# ----------------------------------------------------------------------------
class Case:
    pass


def detect_u_inf():
    for rel in ("0/U", "0.orig/U"):
        f = CASE_DIR / rel
        if f.exists():
            text = f.read_text(errors="ignore")
            number = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            for m in re.finditer(r"uniform\s*\(\s*" + number + r"\s+" + number + r"\s+" + number + r"\s*\)", text):
                v = np.array([float(g) for g in m.groups()])
                if np.linalg.norm(v) > 0:
                    return float(np.linalg.norm(v))
    return None


def detect_case() -> Case:
    case = Case()
    case.hull = None
    case.body_z = None
    case.span_center = 0.0
    case.ground = None                      # z of the ground: set from the symmetry plane in main()
    points = None
    body_dir = latest_time_dir(POST_DIR / "bodySurfaces")
    if body_dir and (body_dir / "body.vtk").exists():
        points = read_surface(body_dir / "body.vtk").points

    if points is not None:
        y_min, y_max = points[:, IV].min(), points[:, IV].max()
        height = y_max - y_min
        x_rear = points[:, IX].max()
        y_bottom = y_min
        flat_roof = points[points[:, IV] >= y_max - 1e-3 * height, IX]
        x_slant = flat_roof.max()
        case.body_z = (points[:, IS].min(), points[:, IS].max())      # spanwise extent of the body
        case.span_center = 0.5 * (case.body_z[0] + case.body_z[1])    # symmetry-plane position
    else:
        warn("bodySurfaces/body.vtk not found: using standard Ahmed-body dimensions")
        height, x_rear, y_bottom = FALLBACK["H"], FALLBACK["X_REAR"], FALLBACK["Y_BOTTOM"]
        x_slant = x_rear - 0.77 * height

    case.H = MODEL_H or height
    case.x_rear = X_REAR if X_REAR is not None else x_rear
    case.y_bottom = Y_BOTTOM if Y_BOTTOM is not None else y_bottom
    if case.ground is None:
        case.ground = case.y_bottom - 0.17 * case.H      # fallback: standard ground clearance
    case.slant_start = (x_slant - case.x_rear) / case.H          # x/H of the slant leading edge (<0)
    u_inf = U_INF or detect_u_inf()
    if u_inf is None:
        warn(f"free-stream speed not found in 0/U: assuming U = {FALLBACK['U_INF']} m/s (set U_INF)")
        u_inf = FALLBACK["U_INF"]
    case.u_inf = u_inf

    if points is not None:
        pts = np.c_[(points[:, IX] - case.x_rear) / case.H, (points[:, IV] - case.y_bottom) / case.H]
        case.hull = pts[ConvexHull(pts).vertices]      # side silhouette of the body
    print(f"Case: H = {case.H:.4f} m, x_rear = {case.x_rear:.4f}, y_bottom = {case.y_bottom:.4f}, "
          f"U = {case.u_inf:.4g} m/s, slant starts at x/H = {case.slant_start:.3f}")
    return case


# ----------------------------------------------------------------------------
# Field processing on triangulated planes
# ----------------------------------------------------------------------------
def tri_gradient(x, y, tri, f):
    """Constant gradient of the linear interpolant of f on each triangle."""
    x1, x2, x3 = x[tri].T
    y1, y2, y3 = y[tri].T
    f1, f2, f3 = f[tri].T
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dfdx = (f1 * (y2 - y3) + f2 * (y3 - y1) + f3 * (y1 - y2)) / det
        dfdy = (f1 * (x3 - x2) + f2 * (x1 - x3) + f3 * (x2 - x1)) / det
    area = 0.5 * np.abs(det)
    bad = ~np.isfinite(dfdx) | ~np.isfinite(dfdy)
    dfdx[bad] = 0.0
    dfdy[bad] = 0.0
    area[bad] = 0.0
    return dfdx, dfdy, area


def point_gradient(x, y, tri, f):
    """Area-weighted nodal gradient of f."""
    dfdx, dfdy, area = tri_gradient(x, y, tri, f)
    gx = np.zeros(len(x))
    gy = np.zeros(len(x))
    wsum = np.zeros(len(x))
    for k in range(3):
        np.add.at(gx, tri[:, k], area * dfdx)
        np.add.at(gy, tri[:, k], area * dfdy)
        np.add.at(wsum, tri[:, k], area)
    wsum[wsum == 0] = 1.0
    return gx / wsum, gy / wsum


def velocity_name(surf: Surface, mean=True) -> str:
    if mean and surf.has("UMean"):
        return "UMean"
    if mean:
        warn("UMean not found on the surface: using the instantaneous U instead")
    return "U"


def plane_vorticity(surf: Surface, name: str) -> np.ndarray:
    """Vorticity component normal to the plane: d(u_b)/da - d(u_a)/db."""
    n = surf.normal_axis
    a, b = (n + 1) % 3, (n + 2) % 3
    xa, xb = surf.points[:, a], surf.points[:, b]
    u = surf.data[name]
    dub_da, _ = point_gradient(xa, xb, surf.tri, u[:, b])
    _, dua_db = point_gradient(xa, xb, surf.tri, u[:, a])
    return dub_da - dua_db


def sym_vorticity(surf: Surface, name: str) -> np.ndarray:
    """Vorticity in the symmetry plane, positive counter-clockwise AS DRAWN
    (x to the right, vertical up), as in the paper: d(u_vert)/dx - d(u_x)/d(vert).
    Independent of which Cartesian axis is vertical."""
    xa, xb = surf.points[:, IX], surf.points[:, IV]
    u = surf.data[name]
    dv_dx, _ = point_gradient(xa, xb, surf.tri, u[:, IV])
    _, du_dv = point_gradient(xa, xb, surf.tri, u[:, IX])
    return dv_dx - du_dv


def levi(i: int, j: int, k: int) -> int:
    """Levi-Civita symbol for axis indices 0..2."""
    return (i - j) * (j - k) * (k - i) // 2


def sym_triangulation(surf: Surface, case: Case):
    """Triangulation of the symmetry plane in (x/H, y/H)."""
    xs = (surf.points[:, IX] - case.x_rear) / case.H
    ys = (surf.points[:, IV] - case.y_bottom) / case.H
    return mtri.Triangulation(xs, ys, surf.tri)


def grid_sample(triang, fields, xlim, ylim, nx):
    """Interpolate nodal fields on a regular grid (masked outside the mesh)."""
    ny = int(round(nx * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])))
    xg = np.linspace(xlim[0], xlim[1], nx)
    yg = np.linspace(ylim[0], ylim[1], max(ny, 2))
    X, Y = np.meshgrid(xg, yg)
    finder = triang.get_trifinder()
    out = [mtri.LinearTriInterpolator(triang, f, trifinder=finder)(X, Y) for f in fields]
    return xg, yg, X, Y, out


def window_mask(triang, xlim, ylim):
    return ((triang.x >= xlim[0]) & (triang.x <= xlim[1]) &
            (triang.y >= ylim[0]) & (triang.y <= ylim[1]))


def default_ylim(case: Case):
    return ((case.ground - case.y_bottom) / case.H - 0.03, 1.45)


def symmetric_levels(fields, n=10):
    """Levels +-k*step (k=1..n); step from the 99th percentile of |field|."""
    vals = np.concatenate([np.abs(f[np.isfinite(f)]) for f in fields])
    peak = np.percentile(vals, 99) if vals.size else 1.0
    step = nice_step(peak / n)
    levels = step * np.arange(-n, n + 1)
    return levels[levels != 0], step


def contour_lines(ax, triang, values, levels, lw=0.6):
    styles = ["dashed" if lv < 0 else "solid" for lv in levels]
    return ax.tricontour(triang, values, levels=levels, colors="k", linewidths=lw, linestyles=styles)


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def draw_body(ax, case: Case):
    if case.hull is not None:
        ax.add_patch(Polygon(case.hull, closed=True, facecolor=BLUE, edgecolor="k",
                             linewidth=0.5, zorder=5))


def style_sym_axes(ax, case: Case, xlim=XLIM, ylim=None, xlabel=True, ylabel=True):
    ylim = ylim or default_ylim(case)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([t for t in (-0.7, 0, 0.7, 1.4) if xlim[0] <= t <= xlim[1]])
    ax.set_yticks([0, 0.7, 1.4])
    if xlabel:
        ax.set_xlabel("x/H")
    if ylabel:
        ax.set_ylabel("y/H")
    draw_body(ax, case)


def corner_label(ax, text, loc="upper right"):
    pos = {"upper right": (0.985, 0.97, "right", "top"), "upper left": (0.015, 0.97, "left", "top"),
           "lower right": (0.985, 0.03, "right", "bottom")}[loc]
    ax.text(pos[0], pos[1], text, transform=ax.transAxes, ha=pos[2], va=pos[3], fontsize=8,
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="0.5", lw=0.4), zorder=10)


def save(fig, name: str) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png")


# ----------------------------------------------------------------------------
# Aerodynamic coefficients
# ----------------------------------------------------------------------------
def read_force_history(path: Path):
    """Read OpenFOAM forces.dat and return time, total Fx and viscous Fx."""
    number = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(value) for value in number.findall(line)]
        # time, pressure force (3), viscous force (3), then moments/other terms
        if len(values) >= 7:
            rows.append((values[0], values[1] + values[4], values[4]))
    if not rows:
        raise ValueError(f"no force samples found in {path}")
    return np.asarray(rows)


def plot_force_coefficients(case):
    force_dirs = time_dirs(POST_DIR / "forces")
    path = force_dirs[-1][1] / "forces.dat" if force_dirs else None
    if path is None or not path.exists():
        raise FileNotFoundError("postProcessing/forces/<time>/forces.dat not found")

    history = read_force_history(path)
    time = history[:, 0]
    scale = -2.0 / (RHO_INF * case.u_inf**2 * A_REF)
    cd = scale * history[:, 1]
    cf = scale * history[:, 2]
    mask = time >= STATS_START_TIME
    if mask.sum() < 2:
        warn(f"fewer than two force samples after t = {STATS_START_TIME}: using all samples")
        mask = np.ones_like(time, dtype=bool)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(time, cd, label=r"$C_D$", lw=0.9)
    ax.plot(time, cf, label=r"$C_F$", lw=0.9)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("coefficient")
    ax.grid(True, lw=0.3)
    ax.legend()
    save(fig, "force_coefficients")

    rows = []
    for name, values in (("$C_D$", cd[mask]), ("$C_F$", cf[mask])):
        rows.append(f"{name} & {values.mean():.5f} & {values.std(ddof=1):.5f} & {len(values)} \\\\\n")
    (REPORT_DIR / "force_coefficients.tex").write_text(
        "\\begin{tabular}{lrrr}\n"
        "\\hline\nCoefficient & Mean & Standard deviation & Samples \\\\\n"
        "\\hline\n" + "".join(rows) +
        "\\hline\n\\end{tabular}\n")


# ----------------------------------------------------------------------------
# Critical points (Figs 4 and 5)
# ----------------------------------------------------------------------------
def find_critical_points(triang, u, v, case: Case):
    """Zeros of the in-plane mean velocity, classified as focus / saddle / node.

    The velocity is linear on each triangle, so its zero is found exactly and
    classified from the eigenvalues of the velocity-gradient matrix.
    Only the wake (x/H >= 0, above the ground) is searched.
    """
    x, y, tri = triang.x, triang.y, triang.triangles
    ux, uy, _ = tri_gradient(x, y, tri, u)
    vx, vy, _ = tri_gradient(x, y, tri, v)
    xc, yc = x[tri].mean(axis=1), y[tri].mean(axis=1)
    uc, vc = u[tri].mean(axis=1), v[tri].mean(axis=1)
    det = ux * vy - uy * vx
    with np.errstate(divide="ignore", invalid="ignore"):
        px = xc + (-vy * uc + uy * vc) / det
        py = yc + (vx * uc - ux * vc) / det
        x1, x2, x3 = x[tri].T
        y1, y2, y3 = y[tri].T
        a, b, c, d = x1 - x3, x2 - x3, y1 - y3, y2 - y3
        dt = a * d - b * c
        l1 = (d * (px - x3) - b * (py - y3)) / dt
        l2 = (-c * (px - x3) + a * (py - y3)) / dt
    l3 = 1 - l1 - l2
    tol = 1e-9
    inside = np.isfinite(px) & np.isfinite(py) & (l1 >= -tol) & (l2 >= -tol) & (l3 >= -tol)
    ground = (case.ground - case.y_bottom) / case.H
    inside &= (px >= 0.0) & (px <= XLIM[1]) & (py >= ground + 0.01) & (py <= 1.3)

    found = []
    for k in np.flatnonzero(inside):
        trace = ux[k] + vy[k]
        disc = trace ** 2 - 4 * det[k]
        if det[k] < 0:
            kind = "saddle"
        elif disc < 0:
            kind = "focus"
        else:
            kind = "node"
        found.append(dict(kind=kind, x=float(px[k]), y=float(py[k])))

    merged = []                                   # merge duplicates on shared edges
    for cp in found:
        for m in merged:
            if m["kind"] == cp["kind"] and np.hypot(m["x"] - cp["x"], m["y"] - cp["y"]) < 0.04:
                break
        else:
            merged.append(cp)
    return merged


def label_critical_points(points):
    """Heuristic labels: F1 = upper focus, F2 = lower focus, S = saddle nearest F2."""
    labelled = []
    foci = sorted([p for p in points if p["kind"] == "focus"], key=lambda p: -p["y"])
    saddles = [p for p in points if p["kind"] == "saddle"]
    if foci:
        labelled.append(dict(foci[0], label="F1"))
    f2 = None
    if len(foci) > 1:
        f2 = dict(foci[-1], label="F2")
        labelled.append(f2)
    if saddles:
        ref = f2 or (labelled[0] if labelled else None)
        if ref is not None:
            saddles.sort(key=lambda p: np.hypot(p["x"] - ref["x"], p["y"] - ref["y"]))
        labelled.append(dict(saddles[0], label="S"))
    return labelled


def write_critical_point_tables(all_points, labelled):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(IMAGE_DIR / "critical_points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "type", "x/H", "y/H"])
        names = {(round(p["x"], 6), round(p["y"], 6)): p["label"] for p in labelled}
        for p in sorted(all_points, key=lambda p: (p["kind"], -p["y"])):
            w.writerow([names.get((round(p["x"], 6), round(p["y"], 6)), ""), p["kind"],
                        f"{p['x']:.4f}", f"{p['y']:.4f}"])
    rows = "".join(f"{p['label']} & {p['kind']} & {p['x']:.3f} & {p['y']:.3f} \\\\\n" for p in labelled)
    (REPORT_DIR / "critical_points.tex").write_text(
        "\\begin{tabular}{llcc}\n\\hline\nPoint & Type & $x/H$ & $y/H$ \\\\\n\\hline\n"
        + rows + "\\hline\n\\end{tabular}\n")


# ----------------------------------------------------------------------------
# Figures from the symmetry plane
# ----------------------------------------------------------------------------
def fig_streamlines_and_critical_points(sym, case):
    name = velocity_name(sym)
    triang = sym_triangulation(sym, case)
    u = sym.data[name][:, IX] / case.u_inf
    v = sym.data[name][:, IV] / case.u_inf
    ylim = default_ylim(case)
    xg, yg, X, Y, (ug, vg) = grid_sample(triang, [u, v], XLIM, ylim, nx=600)

    cps = find_critical_points(triang, u, v, case)
    labelled = label_critical_points(cps)
    print("  critical points found:", [(p["kind"], round(p["x"], 3), round(p["y"], 3)) for p in cps])
    write_critical_point_tables(cps, labelled)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.streamplot(xg, yg, ug, vg, density=(6, 4), color="k", linewidth=0.45, arrowsize=0.4)
    style_sym_axes(ax, case)
    for p in labelled:
        ax.plot(p["x"], p["y"], marker="o", mfc="none", mec="k", ms=4, zorder=7)
        ax.annotate(p["label"], (p["x"], p["y"]), xytext=(8, 8), textcoords="offset points", fontsize=8,
                    zorder=8, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="k", lw=0.5))
    corner_label(ax, r"$\langle\psi\rangle$")
    save(fig, "fig04_streamlines")
    return labelled


def fig_critical_points(labelled, case):
    if not labelled:
        warn("no critical points were detected: fig05 skipped")
        return
    markers = {"F1": "s", "F2": "^", "S": "o"}
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for p in labelled:
        ax.plot(p["x"], p["y"], marker=markers[p["label"]], mfc="none", mec="k", color="k", ls="",
                label=p["label"], ms=7)
        ax.annotate(p["label"], (p["x"], p["y"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("x/H")
    ax.set_ylabel("y/H")
    ax.margins(0.25)
    ax.grid(False)
    ax.legend(frameon=True, edgecolor="k", fancybox=False, loc="best")
    save(fig, "fig05_critical_points")


def fig_vorticity(sym, case, probes=None, name="fig06_vorticity"):
    vel = velocity_name(sym)
    triang = sym_triangulation(sym, case)
    w = sym_vorticity(sym, vel) * case.H / case.u_inf
    ylim = default_ylim(case)
    levels, step = symmetric_levels([w[window_mask(triang, XLIM, ylim)]])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    contour_lines(ax, triang, w, levels)
    style_sym_axes(ax, case)
    corner_label(ax, r"$\langle\omega\rangle$")
    corner_label(ax, rf"$\Delta\langle\omega\rangle H/U = {step:g}$", "lower right")
    save(fig, name)


def fig_velocity_vectors(sym, case):
    vel = velocity_name(sym)
    triang = sym_triangulation(sym, case)
    u = sym.data[vel][:, IX] / case.u_inf
    v = sym.data[vel][:, IV] / case.u_inf
    ylim = default_ylim(case)
    vector_nx = 30
    _, _, Xq, Yq, (uq, vq) = grid_sample(triang, [u, v], XLIM, ylim, nx=vector_nx)
    _, _, Xf, Yf, (uf,) = grid_sample(triang, [u], XLIM, ylim, nx=70)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    spacing = (XLIM[1] - XLIM[0]) / vector_nx
    ax.quiver(Xq, Yq, uq, vq, angles="xy", scale_units="xy", scale=2.9 / spacing,
              pivot="tail", width=0.0016, headwidth=3.5, headlength=4, color="k")
    if np.ma.min(uf) < 0 < np.ma.max(uf):
        ax.contour(Xf, Yf, uf, levels=[0.0], colors="k", linewidths=1.8, zorder=4)
    style_sym_axes(ax, case)
    corner_label(ax, r"$\langle V\rangle$")
    save(fig, "fig07_velocity_vectors")


def fig_tke(sym, case):
    triang = sym_triangulation(sym, case)
    tke, source = None, ""
    if sym.has("UPrime2Mean"):
        tensor = sym.data["UPrime2Mean"]
        idx = {6: (0, 3, 5), 9: (0, 4, 8)}.get(tensor.shape[1])
        if idx:
            tke = 0.5 * tensor[:, list(idx)].sum(axis=1)
            source = "resolved"
            if sym.has("kMean"):
                tke = tke + sym.data["kMean"]
                source = "resolved + modelled"
    if tke is None:
        for fallback in ("kMean", "k"):
            if sym.has(fallback):
                tke, source = sym.data[fallback], f"modelled ({fallback} only)"
                warn(f"UPrime2Mean not found: fig11 uses {fallback} only")
                break
    if tke is None:
        raise RuntimeError("no UPrime2Mean / kMean / k field on the symmetry plane")
    tke = tke / case.u_inf ** 2
    ylim = default_ylim(case)
    peak = np.nanmax(tke[window_mask(triang, XLIM, ylim)])
    step = nice_step(peak / 12)
    levels = step * np.arange(0, int(np.ceil(peak / step)) + 2)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    cf = ax.tricontourf(triang, tke, levels=levels, cmap="turbo", extend="max")
    ax.tricontour(triang, tke, levels=levels, colors="k", linewidths=0.3)
    style_sym_axes(ax, case)
    corner_label(ax, r"$\langle TKE\rangle/U^2$")
    corner_label(ax, f"increment {step:g}, {source}", "lower right")
    fig.colorbar(cf, ax=ax, fraction=0.025, pad=0.02)
    save(fig, "fig11_tke")


def fig_profiles(sym, case, component, name, label, ref):
    """Stacked profiles of <u>/U or <v>/U along the slant and in the wake."""
    vel = velocity_name(sym)
    triang = sym_triangulation(sym, case)
    values = sym.data[vel][:, component] / case.u_inf
    interp = mtri.LinearTriInterpolator(triang, values, trifinder=triang.get_trifinder())
    scale = PROFILE_SCALE_U if component == IX else PROFILE_SCALE_V
    ylim = default_ylim(case)
    slant = np.arange(case.slant_start, 1e-9, SLANT_STEP)
    wake = np.arange(WAKE_STEP, WAKE_MAX + 1e-9, WAKE_STEP)
    yy = np.linspace(ylim[0], ylim[1], 500)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for xs in np.concatenate([slant, wake]):
        ax.plot([xs, xs], ylim, color="0.65", lw=0.3, zorder=1)
        prof = interp(np.full_like(yy, xs), yy)
        ax.plot(xs + scale * prof, yy, "k-", lw=0.8, zorder=3)
    style_sym_axes(ax, case, xlim=(XLIM[0], WAKE_MAX + 0.15))
    bar_y = ylim[1] - 0.08
    x_end = WAKE_MAX + 0.1
    ax.plot([x_end - scale * ref, x_end], [bar_y, bar_y], "k-", lw=1.0, zorder=6)
    corner_label(ax, rf"$\langle {label}\rangle/U = {ref:g}$ (bar length)")
    save(fig, name)


# ----------------------------------------------------------------------------
# Transverse planes (Fig. 8)
# ----------------------------------------------------------------------------
def fig_transverse_vorticity(wake_dir, case):
    files = sorted(wake_dir.glob("transverse_xH*.vtk"))
    if not files:
        files = sorted(wake_dir.glob("transverse_*.vtk"))
    if not files:
        raise FileNotFoundError(f"no transverse_*.vtk files in {wake_dir}")
    planes = []
    for f in files:
        s = read_surface(f)
        if s.normal_axis != IX:
            warn(f"{f.name} is not normal to the streamwise axis: skipped")
            continue
        planes.append((float(s.points[:, IX].mean() - case.x_rear) / case.H, s))
    planes.sort(key=lambda item: item[0])

    ncols = 2 if len(planes) > 3 else 1
    nrows = int(np.ceil(len(planes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows), squeeze=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    zlim = (-1.0, 1.0)
    ylim = default_ylim(case)
    # viewed from downstream with the vertical axis up, the spanwise axis points left iff this is -1
    reverse = levi(IV, IX, IS) == -1
    for ax in axes.flat[len(planes):]:
        ax.set_visible(False)
    for ax, (xh, s) in zip(axes.flat, planes):
        vel = velocity_name(s)
        w = plane_vorticity(s, vel) * case.H / case.u_inf          # streamwise vorticity
        ys = (s.points[:, IV] - case.y_bottom) / case.H             # vertical
        zs = (s.points[:, IS] - case.span_center) / case.H          # spanwise, 0 on the symmetry plane
        triang = mtri.Triangulation(zs, ys, s.tri)
        levels, step = symmetric_levels([w[window_mask(triang, zlim, ylim)]], n=8)
        contour_lines(ax, triang, w, levels, lw=0.55)
        if case.body_z is not None:
            z0, z1 = (case.body_z[0] - case.span_center) / case.H, (case.body_z[1] - case.span_center) / case.H
            ax.add_patch(Rectangle((z0, 0), z1 - z0, 1.0, fill=False, ec="0.5", lw=0.5, zorder=1))
        if reverse:                                                 # seen from downstream, as in the paper
            ax.set_xlim(zlim[1], zlim[0])
            ax.set_xticks([0.7, 0, -0.7])
        else:
            ax.set_xlim(zlim[0], zlim[1])
            ax.set_xticks([-0.7, 0, 0.7])
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_yticks([0, 0.7, 1.4])
        ax.set_xlabel("z/H")
        ax.set_ylabel("y/H")
        corner_label(ax, f"x/H={xh:.2f}")
        corner_label(ax, rf"$\Delta\omega H/U={step:g}$", "lower right")
    save(fig, "fig08_vorticity_transverse")


# ----------------------------------------------------------------------------
# Instantaneous vorticity (Fig. 12)
# ----------------------------------------------------------------------------
def fig_instantaneous_vorticity(case, n_snapshots=3):
    snaps = [(t, d / "symmetryPlane.vtk") for t, d in time_dirs(POST_DIR / "wakePlanes")
             if (d / "symmetryPlane.vtk").exists()][-n_snapshots:]
    if not snaps:
        raise FileNotFoundError("no symmetryPlane.vtk snapshots")
    ylim = default_ylim(case)
    items = []
    for t, f in snaps:
        s = read_surface(f)
        tri = sym_triangulation(s, case)
        w = sym_vorticity(s, "U") * case.H / case.u_inf
        items.append((t, tri, w))
    levels, step = symmetric_levels([w[window_mask(tr, XLIM, ylim)] for _, tr, w in items])
    fig, axes = plt.subplots(len(items), 1, figsize=(7, 3.0 * len(items)), squeeze=False)
    for k, (ax, (t, tri, w)) in enumerate(zip(axes[:, 0], items)):
        contour_lines(ax, tri, w, levels, lw=0.5)
        style_sym_axes(ax, case, xlabel=(k == len(items) - 1))
        corner_label(ax, rf"$t^*={t * case.u_inf / case.H:.1f}$")
        corner_label(ax, rf"$\Delta\omega H/U={step:g}$", "lower right")
    save(fig, "fig12_vorticity_instant")
    if len(items) < n_snapshots:
        warn(f"only {len(items)} snapshot(s) available for fig12 (paper shows {n_snapshots})")


# ----------------------------------------------------------------------------
# Probes and spectra (Fig. 13)
# ----------------------------------------------------------------------------
def read_probes(path: Path):
    text = path.read_text()
    positions = [tuple(float(g) for g in m.groups())
                 for m in re.finditer(r"#\s*Probe\s+\d+\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)", text)]
    data = np.loadtxt(StringIO(re.sub(r"[()]", " ", text)), comments="#", ndmin=2)
    return positions, data


def spectrum(t, signal):
    """Welch spectrum of a (possibly non-uniformly sampled) signal."""
    dt = np.median(np.diff(t))
    if not dt > 0:
        raise ValueError("probe times are not increasing")
    tu = np.arange(t[0], t[-1], dt)
    su = np.interp(tu, t, signal)
    nper = min(PSD_SEGMENT, len(su))
    f, s = welch(su - su.mean(), fs=1.0 / dt, nperseg=nper)
    return f, s


def fig_spectra(case, sym):
    probe_dir = latest_time_dir(POST_DIR / "probes1")
    path = probe_dir / "U" if probe_dir else None
    if not path or not path.exists():
        raise FileNotFoundError("postProcessing/probes1/<time>/U not found")
    positions, data = read_probes(path)
    t = data[:, 0]
    if (t >= STATS_START_TIME).sum() > 64:
        data = data[t >= STATS_START_TIME]
        t = data[:, 0]
    else:
        warn(f"fewer than 64 probe samples after t = {STATS_START_TIME}: using all samples")
    n = (data.shape[1] - 1) // 3
    if len(positions) != n:
        positions = [(np.nan, np.nan, np.nan)] * n

    fig = plt.figure(figsize=(2.6 * n, 9.0))
    gs = GridSpec(3, n, height_ratios=[1.7, 1, 1], hspace=0.45, wspace=0.55)
    rows = []
    # (a) mean vorticity map with numbered probe points
    ax_map = fig.add_subplot(gs[0, :])
    triang = sym_triangulation(sym, case)
    w = sym_vorticity(sym, velocity_name(sym)) * case.H / case.u_inf
    ylim = default_ylim(case)
    levels, _ = symmetric_levels([w[window_mask(triang, XLIM, ylim)]])
    contour_lines(ax_map, triang, w, levels)
    style_sym_axes(ax_map, case)
    corner_label(ax_map, r"$\langle\omega\rangle$")
    # (b) spectra of u (row 1) and v (row 2)
    for i in range(n):
        px, py = (positions[i][IX] - case.x_rear) / case.H, (positions[i][IV] - case.y_bottom) / case.H
        if np.isfinite(px):
            ax_map.plot(px, py, "o", color="red", ms=4, zorder=8)
            ax_map.annotate(str(i + 1), (px, py), xytext=(5, 5), textcoords="offset points", fontsize=8,
                            zorder=9, bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="k", lw=0.5))
        entry = dict(n=i + 1, x=px, y=py)
        for row, (comp, sym_name) in enumerate(((IX, "u"), (IV, "v")), start=1):
            f, s = spectrum(t, data[:, 1 + 3 * i + comp])
            peak = 1 + int(np.argmax(s[1:])) if len(s) > 1 else 0
            ax = fig.add_subplot(gs[row, i])
            ax.plot(f, s, "k-", lw=0.8)
            ax.plot(f[peak], s[peak], "o", mfc="none", mec="red", ms=5)
            ax.set_title(f"({i + 1})", fontsize=9)
            ax.set_xlabel("f (Hz)")
            ax.set_ylabel(rf"$S_{sym_name}$")
            ax.annotate(f"f={f[peak]:.3g}", (f[peak], s[peak]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, va="bottom")
            ax.set_xlim(0, min(f[-1], max(10 * f[peak], 1e-9)))
            entry[sym_name] = (f[peak], f[peak] * case.H / case.u_inf)
        rows.append(entry)
    save(fig, "velocity_psd")

    body = ""
    for e in rows:
        body += (f"{e['n']} & {e['x']:.2f} & {e['y']:.2f} & {e['u'][0]:.3g} & {e['u'][1]:.3f} & "
                 f"{e['v'][0]:.3g} & {e['v'][1]:.3f} \\\\\n")
    (REPORT_DIR / "strouhal.tex").write_text(
        "\\begin{tabular}{ccccccc}\n"
        "\\hline\nProbe & $x/H$ & $y/H$ & $f_u$ [Hz] & $St_{H,u}$ & $f_v$ [Hz] & $St_{H,v}$ \\\\\n\\hline\n"
        + body + "\\hline\n\\end{tabular}\n")


# ----------------------------------------------------------------------------
# Line-sampled profiles (extra figure, from the 'sets' function object)
# ----------------------------------------------------------------------------
def plot_line_profiles(case):
    out = POST_DIR / "velocityProfiles"
    latest = latest_time_dir(out)
    files = sorted(latest.glob("*.xy")) if latest else []
    if not files:
        raise FileNotFoundError("no velocityProfiles/*.xy files")
    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(8, 3.8))
    for filepath in files:
        data = np.loadtxt(filepath, comments="#", ndmin=2)
        if data.shape[1] < 4:
            warn(f"skipping {filepath}: expected y Ux Uy Uz columns")
            continue
        off = 4 if data.shape[1] >= 7 else 1                       # UMean columns when sampled
        cu, cv = off + IX, off + IV
        m = re.match(r"x_H_(\d+)_(\d+)", filepath.stem)
        label = f"x/H = {m.group(1)}.{m.group(2)}" if m else filepath.stem
        y = (data[:, 0] - case.y_bottom) / case.H
        ax_u.plot(data[:, cu] / case.u_inf, y, label=label)
        ax_v.plot(data[:, cv] / case.u_inf, y, label=label)
    for ax, title, xl in ((ax_u, "Mean streamwise velocity", r"$\langle u\rangle/U$"),
                          (ax_v, "Mean vertical velocity", r"$\langle v\rangle/U$")):
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel("y/H")
        ax.grid(True, lw=0.3)
        ax.legend(fontsize=8)
    save(fig, "velocity_profiles")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    case = detect_case()
    wake_dir = latest_time_dir(POST_DIR / "wakePlanes")
    sym = None
    if wake_dir and (wake_dir / "symmetryPlane.vtk").exists():
        sym = read_surface(wake_dir / "symmetryPlane.vtk")
        if sym.normal_axis != IS:
            warn("symmetryPlane.vtk is not normal to the spanwise axis (z): check the cuttingPlane in "
                 "postProcessing and the IX/IV/IS settings")
        case.ground = float(sym.points[:, IV].min())      # the ground is the bottom of the symmetry plane
        print(f"Ground at {case.ground:.4f} (clearance {case.y_bottom - case.ground:.4f} m); "
              f"arrays on the symmetry plane:", sorted(sym.data))
    else:
        warn("postProcessing/wakePlanes/<time>/symmetryPlane.vtk not found")

    failures = []

    def run(label, func, *args, needs_sym=True):
        if needs_sym and sym is None:
            failures.append(label)
            print(f"- {label}: SKIPPED (no symmetry plane)")
            return None
        print(f"- {label}")
        try:
            return func(*args)
        except Exception as exc:                       # keep going: one missing file must not stop the rest
            failures.append(label)
            warn(f"{label} failed: {exc}")
            traceback.print_exc()
            return None

    labelled = run("Fig. 4 streamlines + critical points", fig_streamlines_and_critical_points, sym, case)
    run("Fig. 5 critical points", fig_critical_points, labelled or [], case, needs_sym=False)
    run("Fig. 6 mean vorticity", fig_vorticity, sym, case)
    run("Fig. 7 velocity vectors", fig_velocity_vectors, sym, case)
    run("Fig. 8 transverse vorticity", fig_transverse_vorticity, wake_dir, case, needs_sym=False)
    run("Fig. 9 <u>/U profiles", fig_profiles, sym, case, IX, "fig09_u_profiles", "u", 2.0)
    run("Fig. 10 <v>/U profiles", fig_profiles, sym, case, IV, "fig10_v_profiles", "v", 0.66)
    run("Fig. 11 TKE", fig_tke, sym, case)
    run("Fig. 12 instantaneous vorticity", fig_instantaneous_vorticity, case, needs_sym=False)
    run("Fig. 13 probes and spectra", fig_spectra, case, sym)
    run("extra: line-sampled profiles", plot_line_profiles, case, needs_sym=False)
    run("extra: force coefficients", plot_force_coefficients, case, needs_sym=False)

    if failures:
        print("\nNot generated:", ", ".join(failures))
        return 1
    print(f"\nAll figures saved to {IMAGE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
