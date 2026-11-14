"""VTK off-screen renderings (Turbo colour maps, 3-D geometry) of the Ahmed-body case.

Coordinate convention of this case: x streamwise, z vertical, y spanwise, symmetry
plane y = const (set IX, IV, IS below for another layout).  The symmetry plane is
viewed from the side that keeps x pointing to the right, vertical axis up.

Images written to report/images:
    ahmed_body_and_legs.png   3-D view of the body and the legs
    domain_slice.png          mesh of the symmetry-plane slice
    velocity_field.png        |U| (mean if available) on the symmetry plane
    pressure_field.png        p on the symmetry plane
    streamlines.png           streamlines on the symmetry plane
    velocity_vectors.png      velocity glyphs on the symmetry plane
    tke.png                   TKE contours (k), coloured with Turbo
    velocity_plane_<plane>.png  |U| on every transverse plane
The paper-style (black and white) figures are made by generatePlots.py.
"""
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

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
IMG_W, IMG_H = 1920, 1080        # Full HD, 16:9 aspect ratio
IX, IV, IS = 0, 2, 1             # this case: x streamwise, z VERTICAL, y spanwise (symmetry plane y = const)
FALLBACK_H, FALLBACK_X_REAR, FALLBACK_Y_BOTTOM = 0.288, 1.044, 0.05   # used if body.vtk is missing
WINDOW_UPSTREAM_H = 0.6          # view window: this many H upstream of the nose ...
WINDOW_DOWNSTREAM_H = 4.0        # ... and downstream of the rear face
WINDOW_TOP_H = 1.6               # view window top, in H above the body underside
TARGET_GLYPHS = 1500             # approx. number of arrows in the vector plot
GLYPH_SIZE_H = 0.10              # arrow length in model heights
TKE_LINE_WIDTH = 2.0             # line width of the TKE contours
GREY = (0.85, 0.85, 0.85)        # uniform surface colour


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def get_latest_time_dir(base_path: Path) -> Path | None:
    """Return the latest numerical time directory inside a postProcessing subfolder."""
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
    """Read a legacy VTK surface (polydata or unstructured grid) as polydata."""
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
    """Cell data -> point data (ParaView's CellDatatoPointData).

    OpenFOAM writes face (cell) data; streamlines, glyphs, contours and
    point-based colouring need point data.
    """
    f = vtk.vtkCellDataToPointData()
    f.SetInputData(polydata)
    f.PassCellDataOn()
    f.Update()
    out = vtk.vtkPolyData()
    out.ShallowCopy(f.GetOutput())
    return out


def add_magnitude(polydata: vtk.vtkPolyData, field_name: str):
    """Add '{field_name}_Magnitude' as a point array."""
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
    for name in ("U", "UMean"):
        add_magnitude(pd, name)
    return pd


def array_range(polydata: vtk.vtkPolyData, name: str):
    arr = polydata.GetPointData().GetArray(name)
    return arr.GetRange() if arr is not None else None


def pick_velocity(polydata: vtk.vtkPolyData) -> str:
    """Mean velocity if sampled, otherwise the instantaneous one."""
    return "UMean" if polydata.GetPointData().GetArray("UMean") is not None else "U"


# ----------------------------------------------------------------------------
# Colour map (Turbo)
# ----------------------------------------------------------------------------
def create_turbo_lut(scalar_range) -> vtk.vtkColorTransferFunction:
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
    """Colour a mapper by a named POINT array with the Turbo map."""
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray(array_name)
    mapper.SetColorModeToMapScalars()
    mapper.SetLookupTable(create_turbo_lut(rng))
    mapper.SetScalarRange(rng[0], rng[1])


# ----------------------------------------------------------------------------
# Actors and rendering helpers
# Actors are the objects that are rendered in the scene. 
# They are responsible for representing the geometry and properties of the 
# objects being visualized.
# ----------------------------------------------------------------------------

# Draw a polydata with a uniform colour (scalars explicitly disabled).  Optionally show mesh edges.
def grey_actor(polydata, color=GREY, edges=False) -> vtk.vtkActor:
    """Uniformly coloured surface (scalars explicitly disabled)."""
    mapper = vtk.vtkPolyDataMapper() # Create a mapper for the polydata
    mapper.SetInputData(polydata) # Set the input data for the mapper
    mapper.ScalarVisibilityOff() # Disable scalar visibility to use uniform color
    mapper.SetResolveCoincidentTopologyToPolygonOffset() # Resolve coincident topology issues
    actor = vtk.vtkActor() # Create an actor to represent the polydata
    actor.SetMapper(mapper) # Set the mapper for the actor (a mapper is responsible for converting data into graphical primitives)
    actor.GetProperty().SetColor(*color) # Set the colour of the actor's property
    if edges: # If edges are to be shown, enable edge visibility and set the edge color
        actor.GetProperty().EdgeVisibilityOn() # Enable edge visibility for the actor's property (we are basically showing the cells in the generated mesh)
        actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.2) # Make edges dark grey, compared to the light grey of the surface
    return actor

# Draw a polydata with a field (scalars) mapped to a color.
def field_actor(polydata, array_name: str, rng) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.InterpolateScalarsBeforeMappingOn() # Interpolate the scalars before mapping to colors
    color_by_point_array(mapper, array_name, rng) # Colour according to the turbo map previously defined
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor

# Save a screenshot of the render window to a file.
def save_screenshot(ren_win: vtk.vtkRenderWindow, filepath: Path) -> None:
    w2if = vtk.vtkWindowToImageFilter() # Create a filter to convert the render window to an image
    w2if.SetInput(ren_win) # Set the input of the filter to the render window
    w2if.SetInputBufferTypeToRGB() # Set the input buffer type to RGB (we want a colour image)
    w2if.ReadFrontBufferOff() # Read from the back buffer (we want the rendered image, not the front buffer)
    w2if.Update() # I imagine this step just requires the filter to update its internal state and prepare for the next step
    writer = vtk.vtkPNGWriter() # Create a PNG writer to write the image to a file
    writer.SetFileName(str(filepath)) # Set the output file name for the writer
    writer.SetInputConnection(w2if.GetOutputPort()) # Connect the output of the window-to-image filter to the input of the PNG writer
    writer.Write() # Finally write the image to the file

# Setup the rendered and the render window
# The renderer renders the scen (guess what...)
# The render window is the window that displays the rendered scene.
def setup_renderer():
    renderer = vtk.vtkRenderer() # Create a renderer to render the scene
    renderer.SetBackground(1.0, 1.0, 1.0) # Set the background colour of the renderer to white
    ren_win = vtk.vtkRenderWindow() # Create a render window to display the rendered scene
    ren_win.SetOffScreenRendering(1) # Enable off-screen rendering (we don't want to display the window on the screen, which we don't even have on Cineca)
    ren_win.AddRenderer(renderer) # Add the renderer to the render window
    ren_win.SetSize(IMG_W, IMG_H) # Set the size of the render window to the specified width and height (fullHD, as previously defined)
    return renderer, ren_win # Return the renderer and the render window for further use


def unit(axis: int, sign: float = 1.0):
    v = [0.0, 0.0, 0.0]
    v[axis] = sign
    return v

# Levi-Civita tensor to determine the camera side that keeps x to the right (for a given up axis).
def levi(i, j, k):
    return (i - j) * (j - k) * (k - i) // 2

# Set a parallel-projection camera looking along -look_axis, with up_axis pointing up.
def set_parallel_view(renderer, center, look_axis, half_w, half_h, up_axis=IV, distance=10.0, side=1.0):
    """Parallel-projection camera looking along -look_axis, `up_axis` pointing up.

    The scale is chosen so that a window of half_w x half_h around `center`
    fits in the image.
    """
    cam = renderer.GetActiveCamera() # Get the active camera from the renderer
    cam.SetParallelProjection(1) # Enable parallel projection for the camera (there is a perspective projection as well, but I hate it with a passion)
    pos = list(center) # Create a list of the center coordinates (the camera will be positioned at the center of the scene)
    pos[look_axis] += side * distance # Move the camera along the look_axis by a distance (the side parameter determines which side of the scene the camera is on)
    cam.SetPosition(*pos) # Set the position of the camera to the calculated position
    cam.SetFocalPoint(*center) # Set the focal point of the camera to the center of the scene (the camera will look at the center)
    cam.SetViewUp(*unit(up_axis)) # Set the view up vector of the camera to the specified up_axis (the camera will be oriented so that the up_axis points up in the image)
    aspect = IMG_W / IMG_H # Calculate the aspect ratio of the image (width / height), 16:9 in this case
    cam.SetParallelScale(max(half_h, half_w / aspect)) # Set the parallel scale of the camera to the maximum of half_h and half_w / aspect (this ensures that the entire scene fits in the image)

# Say cheese and shoot the scene, saving the image to a file.  The actors are added to the renderer for this shot, then removed.
def shoot(renderer, ren_win, actors, filename: str) -> None:
    for a in actors:
        renderer.AddActor(a) # Add each actor to the renderer (the actors are the objects that will be rendered in the scene)
    renderer.ResetCameraClippingRange() # Reset the camera clipping range (this ensures that the entire scene is visible in the image)
    ren_win.Render() # Render the scene (this generates the image of the scene)
    save_screenshot(ren_win, IMG_DIR / filename) # Save the rendered image to a file (the image is saved in the specified directory with the specified filename)
    for a in actors:
        renderer.RemoveActor(a) # Remove each actor from the renderer (this ensures that the actors are not rendered in subsequent shots)
    print(f"  wrote {filename}")


# ----------------------------------------------------------------------------
# Geometry (taken from body.vtk when available)
# ----------------------------------------------------------------------------

# Return the height, rear x-coordinate and bottom y-coordinate of the Ahmed body.
def body_geometry(body_data):
    """Return (H, x_rear, y_bottom)."""
    if body_data is None:
        print("WARNING: body.vtk not found, using standard Ahmed-body dimensions")
        return FALLBACK_H, FALLBACK_X_REAR, FALLBACK_Y_BOTTOM
    b = body_data.GetBounds()
    return b[2 * IV + 1] - b[2 * IV], b[2 * IX + 1], b[2 * IV]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    renderer, ren_win = setup_renderer() # Create the renderer and render window for off-screen rendering
    wake_dir = get_latest_time_dir(POST_DIR / "wakePlanes") # Get latest time dir for wake planes
    body_dir = get_latest_time_dir(POST_DIR / "bodySurfaces")  # Get latest time dir for body surfaces

    body_data = legs_data = None
    # Read the body and legs geometry from VTK files if they exist
    if body_dir and (body_dir / "body.vtk").exists():
        body_data = read_vtk_surface(body_dir / "body.vtk")
    if body_dir and (body_dir / "legs.vtk").exists():
        legs_data = read_vtk_surface(body_dir / "legs.vtk")
    H, x_rear, y_bottom = body_geometry(body_data) # Get the height, rear x-coordinate and bottom y-coordinate of the Ahmed body
    print(f"Geometry: H = {H:.4f}, x_rear = {x_rear:.4f}, y_bottom = {y_bottom:.4f}")

    # View window on the symmetry plane
    x_nose = body_data.GetBounds()[2 * IX] if body_data is not None else x_rear - 3.6 * H # Get the x-coordinate of the nose of the Ahmed body (the front of the body)
    x0, x1 = x_nose - WINDOW_UPSTREAM_H * H, x_rear + WINDOW_DOWNSTREAM_H * H # Set the x-coordinates of the view window (the view window is defined by the x-coordinates of the nose and rear of the body, plus some padding)
    y0, y1 = 0.0, y_bottom + WINDOW_TOP_H * H # Set the y-coordinates of the view window (the view window is defined by the bottom of the body plus some padding)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1) # Calculate the center of the view window (the center is the average of the x and y coordinates of the view window)
    half_w, half_h = 0.5 * (x1 - x0), 0.5 * (y1 - y0) # Calculate the half-width and half-height of the view window (the half-width and half-height are used to set the parallel scale of the camera)

    sym_file = wake_dir / "symmetryPlane.vtk" if wake_dir else None # Get the symmetry plane VTK file if it exists
    if sym_file and sym_file.exists(): 
        sym = load_surface(sym_file) # Load the symmetry plane surface from the VTK file
        names = [sym.GetPointData().GetArrayName(i) for i in range(sym.GetPointData().GetNumberOfArrays())] # Extract the names of the point data
        print("Arrays on the symmetry plane:", names)
        b = sym.GetBounds() # Get the bounds of the symmetry plane surface
        z_plane = 0.5 * (b[2 * IS] + b[2 * IS + 1]) # Calculate the z-coordinate of the symmetry plane
        if b[2 * IS + 1] - b[2 * IS] > 1e-6 * max(1.0, H):
            print("WARNING: symmetryPlane.vtk is not normal to the spanwise axis: check the cuttingPlane in postProcessing")
        center = [0.0, 0.0, z_plane] # Set the center of the symmetry plane (the center is defined by the z-coordinate of the symmetry plane)
        center[IX], center[IV] = cx, cy # Set the x and y coordinates of the center of the symmetry plane (the center is defined by the x and y coordinates of the view window)

        vel = pick_velocity(sym)                      # UMean when sampled
        mag_name = f"{vel}_Magnitude" # name of the magnitude array
        u_range = array_range(sym, mag_name) or (0.0, 1.0) # range of the magnitude array (if the array does not exist, set the range to (0.0, 1.0))
        sym.GetPointData().SetActiveVectors(vel) # set the active vector array to the velocity array (this is used for streamlines and glyphs)

        side = -float(levi(IS, IV, IX))             # camera side that keeps x to the right
        set_parallel_view(renderer, center, IS, half_w, half_h, side=side) # set the parallel view of the renderer (the camera is positioned to look along the symmetry plane, with the up axis pointing up)

        # 1. Domain slice (mesh edges) - zoomed on the body
        DOMAIN_ZOOM = 1         # <1 = closer, >1 = further away
        zoom_center = [0.0, 0.0, z_plane] # Set the zoom centre of the view
        zoom_center[IX] = 0.5 * (x_nose + x_rear)             # middle of the body
        zoom_center[IV] = y_bottom + 0.5 * H                  # mid-height of the body
        set_parallel_view(renderer, zoom_center, IS,
                          DOMAIN_ZOOM * half_w, DOMAIN_ZOOM * half_h, side=side)
        shoot(renderer, ren_win, [grey_actor(sym, (0.7, 0.7, 0.7), edges=True)], "domain_slice.png") # shoot the domain slice image
        set_parallel_view(renderer, center, IS, half_w, half_h, side=side) # reset the parallel view of the renderer to the original view

        # 2. Velocity magnitude
        shoot(renderer, ren_win, [field_actor(sym, mag_name, u_range)], "velocity_field.png") # shoot the velocity field image

        # 3. Pressure
        p_range = array_range(sym, "p") 
        if p_range:
            shoot(renderer, ren_win, [field_actor(sym, "p", p_range)], "pressure_field.png") # shoot the pressure field image
        else:
            print("WARNING: field 'p' not found, pressure_field.png skipped")

        # 4. Streamlines, seeded on a vertical line upstream of the body
        u_vec = vtk_to_numpy(sym.GetPointData().GetArray(vel)).copy() # Get the velocity vector array from the symmetry plane surface and convert it to a numpy array
        u_vec[:, IS] = 0.0                            # keep the integration in the plane
        u_plane = numpy_to_vtk(u_vec, deep=True) # Convert the numpy array back to a VTK array, DEEP COPY, so that the original array is not modified
        u_plane.SetName("U_inplane") # Set the name of the new VTK array to "U_inplane"
        sym.GetPointData().AddArray(u_plane) # Add the new VTK array to the point data of the symmetry plane surface

        x_seed = max(b[2 * IX] + 1e-3, x_nose - 0.4 * H)      # upstream of the nose
        p1, p2 = [0.0] * 3, [0.0] * 3 # Create two points for the seed line (the seed line is a vertical line upstream of the body)
        p1[IX] = p2[IX] = x_seed # Set the x-coordinate of the seed line to the upstream position
        p1[IS] = p2[IS] = z_plane # Set the z-coordinate of the seed line to the symmetry plane
        p1[IV] = b[2 * IV] + 0.001 # Set the y-coordinate of the seed line to the bottom of the symmetry plane (the seed line is a vertical line that spans the entire height of the symmetry plane)
        visible_top = cy + max(half_h, half_w * IMG_H / IMG_W)   # top edge of the rendered image
        p2[IV] = min(b[2 * IV + 1], visible_top) * 0.98 # Set the y-coordinate of the top point of the seed line to the minimum of the top of the symmetry plane and the top edge of the rendered image 
        seed = vtk.vtkLineSource() # Create a line source for the seed line
        seed.SetPoint1(*p1) # Set the first point of the seed line
        seed.SetPoint2(*p2) # Set the second point of the seed line
        seed.SetResolution(80) # Set the resolution of the seed line (the number of points along the line)

        streamer = vtk.vtkStreamTracer() # Create a stream tracer to generate streamlines from the seed line (this is the exact same filter Paraview employs)
        streamer.SetInputData(sym) # Set the input data for the stream tracer to the symmetry plane surface
        streamer.SetSourceConnection(seed.GetOutputPort()) # Set the source of the stream tracer to the seed line
        streamer.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U_inplane") # Array to integrate (the stream tracer will integrate the velocity field along the seed line)
        streamer.SetIntegrationDirectionToBoth() # Integrate in both directions (upstream and downstream) from the seed line
        streamer.SetIntegratorTypeToRungeKutta45() # Use the Runge-Kutta 4-5 integrator
        streamer.SetMaximumPropagation(10.0 * (b[1] - b[0])) # Set the maximum propagation distance for the streamlines (the streamlines will be integrated for a maximum distance of 10 times the width of the symmetry plane)
        streamer.SetMaximumNumberOfSteps(50000) # Set the maximum number of steps for the integrator (the integrator will take a maximum of 50000 steps to integrate the streamlines)
        streamer.Update() # Update the stream tracer to generate the streamlines
        print("  streamlines generated:", streamer.GetOutput().GetNumberOfLines())

        tubes = vtk.vtkTubeFilter() # Create a tube filter to convert the streamlines into tubes (this makes the streamlines more visible in the rendered image)
        tubes.SetInputConnection(streamer.GetOutputPort())
        tubes.SetRadius(0.01 * H)
        tubes.SetNumberOfSides(8)
        tubes.Update()
        st_mapper = vtk.vtkPolyDataMapper()
        st_mapper.SetInputConnection(tubes.GetOutputPort())
        color_by_point_array(st_mapper, mag_name, u_range)
        st_actor = vtk.vtkActor()
        st_actor.SetMapper(st_mapper)
        shoot(renderer, ren_win, [grey_actor(sym, GREY), st_actor], "streamlines.png")

        # 5. Velocity vectors: crop to the view window, subsample, constant length
        box = vtk.vtkBox()
        lo, hi = [0.0] * 3, [0.0] * 3
        lo[IX], hi[IX] = x0, x1
        lo[IV], hi[IV] = y0, y1
        lo[IS], hi[IS] = b[2 * IS] - 1.0, b[2 * IS + 1] + 1.0
        box.SetBounds(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
        extract = vtk.vtkExtractPolyDataGeometry()
        extract.SetInputData(sym)
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
        glyph.SetInputArrayToProcess(1, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U_inplane")
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToDataScalingOff()
        glyph.SetScaleFactor(GLYPH_SIZE_H * H)
        glyph.OrientOn()
        glyph.Update()

        gl_mapper = vtk.vtkPolyDataMapper()
        gl_mapper.SetInputConnection(glyph.GetOutputPort())
        # colour each arrow by |U| of its seed point (glyph copies the point scalars)
        extract_scalars = glyph.GetOutput().GetPointData().GetArray(mag_name)
        if extract_scalars is not None:
            color_by_point_array(gl_mapper, mag_name, u_range)
        else:
            gl_mapper.ScalarVisibilityOff()
        gl_actor = vtk.vtkActor()
        gl_actor.SetMapper(gl_mapper)
        shoot(renderer, ren_win, [grey_actor(sym, GREY), gl_actor], "velocity_vectors.png")

        # 6. TKE contours from the actual k field (ParaView script converted to VTK)
        #    CellDatatoPointData is already applied in load_surface().
        k_range = array_range(sym, "k")
        if k_range:
            k_min, k_max = k_range
            levels = [k_min + (k_max - k_min) * (i / 12.0) for i in range(1, 12)]   # 11 isolines
            tke_contour = vtk.vtkContourFilter()
            tke_contour.SetInputData(sym)
            tke_contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "k")
            tke_contour.ComputeScalarsOn()
            tke_contour.SetNumberOfContours(len(levels))
            for i, v in enumerate(levels):
                tke_contour.SetValue(i, v)
            tke_contour.Update()

            tke_mapper = vtk.vtkPolyDataMapper()
            tke_mapper.SetInputConnection(tke_contour.GetOutputPort())
            color_by_point_array(tke_mapper, "k", (k_min, k_max))
            tke_actor = vtk.vtkActor()
            tke_actor.SetMapper(tke_mapper)
            tke_actor.GetProperty().SetLineWidth(TKE_LINE_WIDTH)
            tke_actor.GetProperty().LightingOff()
            shoot(renderer, ren_win, [grey_actor(sym, GREY), tke_actor], "tke.png")
        else:
            print("WARNING: field 'k' not found, tke.png skipped")
    else:
        print("WARNING: symmetryPlane.vtk not found, symmetry-plane images skipped")

    # 7. Transverse planes, viewed from downstream (looking upstream), y up
    if wake_dir:
        for plane_file in sorted(wake_dir.glob("transverse_*.vtk")):
            plane = load_surface(plane_file)
            vel = pick_velocity(plane)
            mag_name = f"{vel}_Magnitude"
            rng = array_range(plane, mag_name) or (0.0, 1.0)
            pb = plane.GetBounds()
            c = [0.5 * (pb[2 * i] + pb[2 * i + 1]) for i in range(3)]
            set_parallel_view(renderer, c, IX, 0.5 * (pb[2 * IS + 1] - pb[2 * IS]) * 1.1,
                              0.5 * (pb[2 * IV + 1] - pb[2 * IV]) * 1.1, up_axis=IV)
            shoot(renderer, ren_win, [field_actor(plane, mag_name, rng)],
                  f"velocity_plane_{plane_file.stem.replace('transverse_', '')}.png")

    # 8. 3-D body and legs, seen from the rear/side/above
    actors = []
    if body_data is not None:
        actors.append(grey_actor(body_data, (0.85, 0.85, 0.85)))
    if legs_data is not None:
        actors.append(grey_actor(legs_data, (0.2, 0.2, 0.2)))
    if actors:
        # camera direction (in x, y, z): downstream, above, and to the side
        d = [0.0, 0.0, 0.0]
        d[IX], d[IV], d[IS] = 1.0, 0.25, 0.8
        d = np.array(d) / np.linalg.norm(d)
        bb = body_data.GetBounds() if body_data is not None else (0,) * 6
        cen = np.array([0.5 * (bb[2 * i] + bb[2 * i + 1]) for i in range(3)])
        size = np.linalg.norm([bb[2 * i + 1] - bb[2 * i] for i in range(3)])
        cam = renderer.GetActiveCamera()
        cam.SetParallelProjection(0)
        cam.SetFocalPoint(*cen)
        cam.SetPosition(*(cen + d * 1.6 * size))
        cam.SetViewUp(*unit(IV))
        renderer.ResetCameraClippingRange()
        shoot(renderer, ren_win, actors, "ahmed_body_and_legs.png")

    print("VTK offscreen rendering completed. Images saved to", IMG_DIR)


if __name__ == "__main__":
    main()
