from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
CASE_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = CASE_DIR / "report" / "images"
POST_DIR = CASE_DIR / "postProcessing"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
IMG_W, IMG_H = 1920, 1080
VIEW_CENTER = (0.5, 0.17, 0.0)   # focal point of the side view (x, y, z)
VIEW_SCALE = 0.6                 # parallel scale = half of the visible height
PLANE_NORMAL_AXIS = 1            # symmetry plane normal is y (index 1)
X_PLANES = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 1.0, 1.2, 1.50]
TARGET_GLYPHS = 1500             # approx. number of arrows in the vector plot
GLYPH_SIZE = 0.03                # constant arrow length [m]
TKE_LINE_WIDTH = 2.0
GREY = (0.85, 0.85, 0.85)


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def get_latest_time_dir(base_path: Path) -> Path | None:
    """Return the latest numerical time directory inside postProcessing subfolders."""
    if not base_path.exists():
        return None
    dirs = []
    for p in base_path.iterdir():
        if p.is_dir():
            try:
                dirs.append((float(p.name), p))
            except ValueError:
                pass
    return sorted(dirs, key=lambda item: item[0])[-1][1] if dirs else None


def read_vtk_surface(filepath: Path) -> vtk.vtkPolyData:
    """Read a legacy VTK surface file (polydata or unstructured grid) as polydata."""
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(str(filepath))
    reader.Update()
    out = reader.GetOutput()
    if not out.IsA("vtkPolyData"):
        geo = vtk.vtkGeometryFilter()
        geo.SetInputData(out)
        geo.Update()
        out = geo.GetOutput()
    result = vtk.vtkPolyData()
    result.ShallowCopy(out)
    if result.GetNumberOfPoints() == 0:
        raise RuntimeError(f"No points read from {filepath}")
    return result


def cell_to_point(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Interpolate cell data to point data (same as ParaView's CellDatatoPointData).

    OpenFOAM surfaces are often written as CELL data. Streamlines, glyphs,
    contours and point-based colouring all need POINT data, which is why the
    original script produced nothing for them.
    """
    f = vtk.vtkCellDataToPointData()
    f.SetInputData(polydata)
    f.PassCellDataOn()
    f.Update()
    out = vtk.vtkPolyData()
    out.ShallowCopy(f.GetOutput())
    return out


def compute_vector_magnitude(polydata: vtk.vtkPolyData, field_name: str = "U"):
    """Attach '{field_name}_Magnitude' as a point-data array (does not change active scalars)."""
    pdata = polydata.GetPointData()
    vec = pdata.GetArray(field_name)
    if vec is None or vec.GetNumberOfComponents() != 3:
        return None
    mag = np.linalg.norm(vtk_to_numpy(vec), axis=1)
    arr = numpy_to_vtk(mag, deep=True)
    arr.SetName(f"{field_name}_Magnitude")
    pdata.AddArray(arr)
    return arr


def load_surface(filepath: Path) -> vtk.vtkPolyData:
    pd = cell_to_point(read_vtk_surface(filepath))
    compute_vector_magnitude(pd, "U")
    return pd


def array_range(polydata: vtk.vtkPolyData, name: str):
    arr = polydata.GetPointData().GetArray(name)
    return arr.GetRange() if arr is not None else None


# ----------------------------------------------------------------------------
# Colour map
# ----------------------------------------------------------------------------
def create_turbo_lut(scalar_range: tuple[float, float]) -> vtk.vtkColorTransferFunction:
    """Turbo colour map (Google polynomial approximation) over scalar_range."""
    s_min, s_max = scalar_range
    if s_max - s_min < 1e-12:
        s_max = s_min + 1e-12
    lut = vtk.vtkColorTransferFunction()
    lut.SetColorSpaceToRGB()
    n = 64
    for i in range(n):
        t = i / (n - 1)
        r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))
        g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))
        b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))
        lut.AddRGBPoint(s_min + t * (s_max - s_min),
                        float(np.clip(r, 0, 1)), float(np.clip(g, 0, 1)), float(np.clip(b, 0, 1)))
    return lut


def color_by_point_array(mapper: vtk.vtkMapper, array_name: str, rng) -> None:
    """Colour a mapper by a named POINT array using the Turbo map."""
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray(array_name)
    mapper.SetColorModeToMapScalars()
    mapper.SetLookupTable(create_turbo_lut(rng))
    mapper.SetScalarRange(rng[0], rng[1])


# ----------------------------------------------------------------------------
# Actor / rendering helpers
# ----------------------------------------------------------------------------
def grey_actor(polydata, color=GREY, edges=False) -> vtk.vtkActor:
    """Uniformly coloured surface. Scalars are explicitly disabled, otherwise VTK
    colours it with the default blue-red map when point arrays are present."""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()
    mapper.SetResolveCoincidentTopologyToPolygonOffset()  # avoid z-fighting with overlays
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    if edges:
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.2)
    return actor


def field_actor(polydata, array_name: str, rng) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.InterpolateScalarsBeforeMappingOn()
    color_by_point_array(mapper, array_name, rng)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


def save_screenshot(ren_win: vtk.vtkRenderWindow, filepath: Path) -> None:
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(ren_win)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    w2if.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(filepath))
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()


def setup_renderer() -> tuple[vtk.vtkRenderer, vtk.vtkRenderWindow]:
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    ren_win = vtk.vtkRenderWindow()
    ren_win.SetOffScreenRendering(1)
    ren_win.AddRenderer(renderer)
    ren_win.SetSize(IMG_W, IMG_H)
    return renderer, ren_win


def set_side_camera(renderer: vtk.vtkRenderer) -> None:
    cam = renderer.GetActiveCamera()
    cam.SetParallelProjection(1)
    cam.SetPosition(VIEW_CENTER[0], 5.0, VIEW_CENTER[2])
    cam.SetFocalPoint(*VIEW_CENTER)
    cam.SetViewUp(0.0, 0.0, 1.0)
    cam.SetParallelScale(VIEW_SCALE)


def shoot(renderer, ren_win, actors, filename: str) -> None:
    """Add actors, render, save PNG, remove actors."""
    for a in actors:
        renderer.AddActor(a)
    renderer.ResetCameraClippingRange()
    ren_win.Render()
    save_screenshot(ren_win, IMG_DIR / filename)
    for a in actors:
        renderer.RemoveActor(a)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    renderer, ren_win = setup_renderer()
    wake_dir = get_latest_time_dir(POST_DIR / "wakePlanes")
    sym_file = wake_dir / "symmetryPlane.vtk" if wake_dir else None

    if sym_file and sym_file.exists():
        sym_data = load_surface(sym_file)
        print("Arrays on symmetry plane:",
              [sym_data.GetPointData().GetArrayName(i) for i in range(sym_data.GetPointData().GetNumberOfArrays())])
        sym_data.GetPointData().SetActiveVectors("U")
        set_side_camera(renderer)

        u_range = array_range(sym_data, "U_Magnitude") or (0.0, 1.0)

        # 1. Domain slice (mesh edges)
        shoot(renderer, ren_win, [grey_actor(sym_data, (0.7, 0.7, 0.7), edges=True)], "domain_slice.png")

        # 2. Velocity magnitude
        shoot(renderer, ren_win, [field_actor(sym_data, "U_Magnitude", u_range)], "velocity_field.png")

        # 3. Pressure
        p_range = array_range(sym_data, "p")
        if p_range:
            shoot(renderer, ren_win, [field_actor(sym_data, "p", p_range)], "pressure_field.png")
        else:
            print("WARNING: field 'p' not found, pressure_field.png skipped")

        # 4. Streamlines
        bounds = sym_data.GetBounds()
        y_plane = 0.5 * (bounds[2] + bounds[3])
        x_seed = max(-0.5, bounds[0] + 0.01 * (bounds[1] - bounds[0]))
        z_floor = bounds[4]
        z_top = min(0.7 * bounds[5], VIEW_CENTER[2] + VIEW_SCALE)

        # Remove the out-of-plane component so integration stays on the plane
        u_vec = vtk_to_numpy(sym_data.GetPointData().GetArray("U")).copy()
        u_vec[:, PLANE_NORMAL_AXIS] = 0.0
        u_plane = numpy_to_vtk(u_vec, deep=True)
        u_plane.SetName("U_inplane")
        sym_data.GetPointData().AddArray(u_plane)

        line_seed = vtk.vtkLineSource()
        line_seed.SetPoint1(x_seed, y_plane, z_floor + 0.001)
        line_seed.SetPoint2(x_seed, y_plane, z_top)
        line_seed.SetResolution(60)

        streamer = vtk.vtkStreamTracer()
        streamer.SetInputData(sym_data)
        streamer.SetSourceConnection(line_seed.GetOutputPort())
        streamer.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U_inplane")
        streamer.SetIntegrationDirectionToBoth()
        streamer.SetIntegratorTypeToRungeKutta45()
        streamer.SetMaximumPropagation(10.0)       # default is only 1.0 m
        streamer.SetMaximumNumberOfSteps(20000)
        streamer.Update()
        print("Streamlines generated:", streamer.GetOutput().GetNumberOfLines())

        tubes = vtk.vtkTubeFilter()
        tubes.SetInputConnection(streamer.GetOutputPort())
        tubes.SetRadius(0.003)
        tubes.SetNumberOfSides(8)
        tubes.Update()

        st_mapper = vtk.vtkPolyDataMapper()
        st_mapper.SetInputConnection(tubes.GetOutputPort())
        color_by_point_array(st_mapper, "U_Magnitude", u_range)
        st_actor = vtk.vtkActor()
        st_actor.SetMapper(st_mapper)
        shoot(renderer, ren_win, [grey_actor(sym_data, (0.85, 0.85, 0.85)), st_actor], "streamlines.png")

        # 5. Velocity vectors (glyphs): crop to the visible region, subsample, constant size
        half_h = VIEW_SCALE
        half_w = VIEW_SCALE * IMG_W / IMG_H
        box = vtk.vtkBox()
        box.SetBounds(VIEW_CENTER[0] - half_w, VIEW_CENTER[0] + half_w,
                      bounds[2] - 1.0, bounds[3] + 1.0,
                      VIEW_CENTER[2] - half_h, VIEW_CENTER[2] + half_h)
        extract = vtk.vtkExtractPolyDataGeometry()
        extract.SetInputData(sym_data)
        extract.SetImplicitFunction(box)
        extract.ExtractInsideOn()
        extract.ExtractBoundaryCellsOn()
        extract.Update()
        n_pts = max(extract.GetOutput().GetNumberOfPoints(), 1)

        mask = vtk.vtkMaskPoints()
        mask.SetInputConnection(extract.GetOutputPort())
        mask.SetOnRatio(max(1, n_pts // TARGET_GLYPHS))
        mask.RandomModeOff()
        mask.GenerateVerticesOn()
        mask.SingleVertexPerCellOn()

        arrow = vtk.vtkArrowSource()
        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetInputConnection(mask.GetOutputPort())
        glyph.SetInputArrayToProcess(1, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U")  # 1 = vectors
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToDataScalingOff()      # constant arrow length, orientation only
        glyph.SetScaleFactor(GLYPH_SIZE)
        glyph.SetColorModeToColorByVector()       # scalars = |U|
        glyph.OrientOn()
        glyph.Update()

        gl_mapper = vtk.vtkPolyDataMapper()
        gl_mapper.SetInputConnection(glyph.GetOutputPort())
        gl_mapper.ScalarVisibilityOn()
        gl_mapper.SetLookupTable(create_turbo_lut(u_range))
        gl_mapper.SetScalarRange(u_range[0], u_range[1])
        gl_actor = vtk.vtkActor()
        gl_actor.SetMapper(gl_mapper)
        shoot(renderer, ren_win, [grey_actor(sym_data, (0.85, 0.85, 0.85)), gl_actor], "velocity_vectors.png")

        # 6. Turbulent kinetic energy contours (converted from the ParaView script)
        #    CellDatatoPointData is already applied in load_surface().
        k_range = array_range(sym_data, "k")
        if k_range:
            k_min, k_max = k_range
            levels = [k_min + (k_max - k_min) * (i / 12.0) for i in range(1, 12)]

            tke_contour = vtk.vtkContourFilter()
            tke_contour.SetInputData(sym_data)
            tke_contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "k")
            tke_contour.ComputeScalarsOn()
            tke_contour.SetNumberOfContours(len(levels))
            for i, v in enumerate(levels):
                tke_contour.SetValue(i, v)
            tke_contour.Update()

            tke_mapper = vtk.vtkPolyDataMapper()
            tke_mapper.SetInputConnection(tke_contour.GetOutputPort())
            color_by_point_array(tke_mapper, "k", (k_min, k_max))   # ColorBy k + Turbo
            tke_actor = vtk.vtkActor()
            tke_actor.SetMapper(tke_mapper)
            tke_actor.GetProperty().SetLineWidth(TKE_LINE_WIDTH)    # LineWidth = 2.0
            tke_actor.GetProperty().LightingOff()

            bg_actor = grey_actor(sym_data, GREY)                   # Surface, DiffuseColor 0.85
            shoot(renderer, ren_win, [bg_actor, tke_actor], "tke.png")
        else:
            print("WARNING: field 'k' not found, tke.png skipped")
    else:
        print("WARNING: symmetryPlane.vtk not found, symmetry-plane images skipped")

    # 7. Transverse planes
    if wake_dir:
        for x_val in X_PLANES:
            tag = f"{round(x_val * 100):03d}"
            matches = sorted(f for f in wake_dir.glob("transverse_*.vtk") if tag in f.name)
            if not matches:
                print(f"WARNING: no transverse plane file for x={x_val} (tag {tag})")
                continue
            plane_data = load_surface(matches[0])
            prange = array_range(plane_data, "U_Magnitude") or (0.0, 1.0)

            cam = renderer.GetActiveCamera()
            cam.SetParallelProjection(1)
            cam.SetPosition(x_val + 3.0, 0.17, 0.15)
            cam.SetFocalPoint(x_val, 0.17, 0.15)
            cam.SetViewUp(0.0, 0.0, 1.0)
            cam.SetParallelScale(0.35)

            shoot(renderer, ren_win, [field_actor(plane_data, "U_Magnitude", prange)],
                  f"velocity_plane_x_{x_val:.5f}.png")

    # 8. 3D body and legs
    body_dir = get_latest_time_dir(POST_DIR / "bodySurfaces")
    actors = []
    if body_dir:
        for name, color in (("body.vtk", (0.85, 0.85, 0.85)), ("legs.vtk", (0.2, 0.2, 0.2))):
            f = body_dir / name
            if f.exists():
                actors.append(grey_actor(read_vtk_surface(f), color))

    if actors:
        cam = renderer.GetActiveCamera()
        cam.SetParallelProjection(0)
        x_c, y_c, z_c = 0.4, 0.17, 0.15
        cam.SetPosition(x_c + 1.8, y_c - 1.5, z_c + 0.8)
        cam.SetFocalPoint(x_c, y_c, z_c)
        cam.SetViewUp(0.0, 0.0, 1.0)
        shoot(renderer, ren_win, actors, "ahmed_body_and_legs.png")

    print("VTK offscreen rendering completed. Images saved to report/images/")


if __name__ == "__main__":
    main()