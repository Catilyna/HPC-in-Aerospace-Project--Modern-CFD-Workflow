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
from scipy.spatial import cKDTree

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
        # the ground is the bottom of the symmetry plane (NOT necessarily z = 0): rebuild the vertical view window
        y0, y1 = b[2 * IV], y_bottom + WINDOW_TOP_H * H
        cy, half_h = 0.5 * (y0 + y1), 0.5 * (y1 - y0)
        scale_v = max(half_h, half_w * IMG_H / IMG_W)   # half-height actually shown (the window is wider than tall)
        cy, half_h = y0 + scale_v, scale_v               # anchor the bottom edge of the image on the ground
        print(f"Ground at {y0:.4f} (clearance {y_bottom - y0:.4f} m)")
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

        # 4. Streamlines, evenly spaced in space: candidate seeds on a regular grid; a streamline is traced from a
        #    candidate only if the candidate is farther than STREAM_SPACING from every streamline traced so far
        STREAM_SPACING_H = 0.07      # NEW: target distance between neighbouring streamlines, in model heights (bigger = fewer lines)
        u_vec = vtk_to_numpy(sym.GetPointData().GetArray(vel)).copy() # Get the velocity vector array from the symmetry plane surface and convert it to a numpy array
        u_vec[:, IS] = 0.0                            # keep the integration in the plane
        u_plane = numpy_to_vtk(u_vec, deep=True) # Convert the numpy array back to a VTK array, DEEP COPY, so that the original array is not modified
        u_plane.SetName("U_inplane") # Set the name of the new VTK array to "U_inplane"
        sym.GetPointData().AddArray(u_plane) # Add the new VTK array to the point data of the symmetry plane surface

        visible_top = cy + max(half_h, half_w * IMG_H / IMG_W)   # top edge of the rendered image
        v_min = b[2 * IV] + 0.001 # Lowest seed height: the bottom of the symmetry plane
        v_max = min(b[2 * IV + 1], visible_top) * 0.98 # Highest seed height: the minimum of the top of the symmetry plane and the top edge of the rendered image
        spacing = STREAM_SPACING_H * H # NEW: spacing in metres
        cand_x = np.arange(max(b[2 * IX] + 1e-3, x0), x1, spacing) # NEW: candidate seed positions, streamwise (whole view window, from upstream of the nose)
        cand_v = np.arange(v_min, v_max, spacing) # NEW: candidate seed positions, vertical
        candidates = [(x, v) for x in cand_x for v in cand_v] # NEW: ordered from upstream to downstream, so the free stream is seeded first

        streamer = vtk.vtkStreamTracer() # Create a stream tracer to generate streamlines from the seeds (this is the exact same filter Paraview employs)
        streamer.SetInputData(sym) # Set the input data for the stream tracer to the symmetry plane surface
        streamer.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U_inplane") # Array to integrate (the stream tracer will integrate the velocity field along the seed line)
        streamer.SetIntegrationDirectionToBoth() # Integrate in both directions (upstream and downstream) from the seed
        streamer.SetIntegratorTypeToRungeKutta45() # Use the Runge-Kutta 4-5 integrator
        streamer.SetMaximumPropagation(10.0 * (b[1] - b[0])) # Set the maximum propagation distance for the streamlines (the streamlines will be integrated for a maximum distance of 10 times the width of the symmetry plane)
        streamer.SetMaximumNumberOfSteps(20000) # CHANGED (was 50000): lines spiralling into a focus never end, so keep the cap per line moderate

        traced = vtk.vtkAppendPolyData() # NEW: collects the accepted streamlines
        line_pts = np.empty((0, 2)) # NEW: in-plane (x, vertical) points of all accepted streamlines, used for the distance test
        tree = None # NEW: KD-tree of line_pts
        n_lines = 0 # NEW
        for (cx_s, cv_s) in candidates: # NEW: loop over the candidate seeds
            if tree is not None and tree.query([cx_s, cv_s])[0] < 0.9 * spacing: # NEW: too close to an existing streamline
                continue
            seed_xyz = [0.0] * 3 # NEW: 3-D position of the seed point
            seed_xyz[IX], seed_xyz[IV], seed_xyz[IS] = cx_s, cv_s, z_plane # NEW
            seed_pts = vtk.vtkPoints() # NEW: one-point seed source
            seed_pts.InsertNextPoint(*seed_xyz) # NEW
            seed_pd = vtk.vtkPolyData() # NEW
            seed_pd.SetPoints(seed_pts) # NEW
            seed_cells = vtk.vtkCellArray() # NEW
            seed_cells.InsertNextCell(1) # NEW
            seed_cells.InsertCellPoint(0) # NEW
            seed_pd.SetVerts(seed_cells) # NEW
            streamer.SetSourceData(seed_pd) # NEW: trace from this seed only
            streamer.Update() # Update the stream tracer to generate the streamline
            line = vtk.vtkPolyData() # NEW
            line.DeepCopy(streamer.GetOutput()) # NEW: copy, because the filter output is overwritten at the next seed
            if line.GetNumberOfPoints() < 2: # NEW: seed outside the mesh (e.g. inside the body) or stagnant
                continue
            traced.AddInputData(line) # NEW
            n_lines += 1 # NEW
            xyz = vtk_to_numpy(line.GetPoints().GetData()) # NEW: points of the new line
            new_pts = [] # NEW: densify the polyline so that the distance test has no gaps between points
            for ic in range(line.GetNumberOfCells()): # NEW
                ids = line.GetCell(ic).GetPointIds() # NEW
                seg = np.array([[xyz[ids.GetId(k), IX], xyz[ids.GetId(k), IV]] for k in range(ids.GetNumberOfIds())]) # NEW
                if len(seg) < 2: # NEW
                    continue
                for p, q in zip(seg[:-1], seg[1:]): # NEW
                    n_sub = max(1, int(np.ceil(np.hypot(*(q - p)) / (0.3 * spacing)))) # NEW: sub-points at <= 0.3 spacing
                    new_pts.extend(p + (q - p) * t for t in np.linspace(0.0, 1.0, n_sub, endpoint=False)) # NEW
                new_pts.append(seg[-1]) # NEW
            if new_pts: # NEW
                line_pts = np.vstack([line_pts, np.array(new_pts)]) # NEW
                tree = cKDTree(line_pts) # NEW: rebuild the tree with the new line
        traced.Update() # NEW
        print("  streamlines generated:", n_lines)

        tubes = vtk.vtkTubeFilter() # Create a tube filter to convert the streamlines into tubes (this makes the streamlines more visible in the rendered image)
        tubes.SetInputConnection(traced.GetOutputPort()) # CHANGED: the accepted streamlines instead of a single tracer output
        tubes.SetRadius(0.01 * H) # Set the radius of the tubes to 1% of the height of the Ahmed body
        tubes.SetNumberOfSides(8) # Set the number of sides for the tubes (the tubes will be octagonal)
        tubes.Update()
        st_mapper = vtk.vtkPolyDataMapper()
        st_mapper.SetInputConnection(tubes.GetOutputPort())
        color_by_point_array(st_mapper, mag_name, u_range)
        st_actor = vtk.vtkActor()
        st_actor.SetMapper(st_mapper)
        shoot(renderer, ren_win, [grey_actor(sym, GREY), st_actor], "streamlines.png")

        # 5. Velocity vectors: arrows on a REGULAR grid (uniform in space, independent of the mesh density),
        #    velocity interpolated from the symmetry plane, constant length
        GLYPH_SPACING_H = 0.12       # NEW: distance between arrows, in model heights (bigger = fewer arrows)
        spacing_g = GLYPH_SPACING_H * H # NEW: spacing in metres
        gx = np.arange(x0 + 0.5 * spacing_g, x1, spacing_g) # NEW: grid positions, streamwise (whole view window)
        v_ground = b[2 * IV] + 0.001         # bottom of the symmetry plane (the real ground), not z = 0
        gv = np.arange(v_ground + 0.5 * spacing_g, min(b[2 * IV + 1], visible_top) * 0.98, spacing_g) # grid positions, vertical (from the ground to the top of the image)
        GX, GV = np.meshgrid(gx, gv) # NEW
        grid_xyz = np.zeros((GX.size, 3)) # NEW: 3-D coordinates of the grid points
        grid_xyz[:, IX] = GX.ravel() # NEW
        grid_xyz[:, IV] = GV.ravel() # NEW
        grid_xyz[:, IS] = z_plane # NEW: on the symmetry plane
        grid_pts = vtk.vtkPoints() # NEW
        grid_pts.SetData(numpy_to_vtk(grid_xyz, deep=True)) # NEW
        grid_pd = vtk.vtkPolyData() # NEW
        grid_pd.SetPoints(grid_pts) # NEW
        grid_vertices = vtk.vtkVertexGlyphFilter() # NEW: turns the points into vertex cells
        grid_vertices.SetInputData(grid_pd) # NEW
        grid_vertices.Update() # NEW

        probe = vtk.vtkProbeFilter() # NEW: interpolates the fields of the symmetry plane onto the grid points
        probe.SetInputConnection(grid_vertices.GetOutputPort()) # NEW: where to sample
        probe.SetSourceData(sym) # NEW: what to sample
        probe.Update() # NEW

        inside = vtk.vtkThresholdPoints() # NEW: keep only the grid points that fall on the mesh (not in the body hole)
        inside.SetInputConnection(probe.GetOutputPort()) # NEW
        inside.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "vtkValidPointMask") # NEW
        inside.ThresholdByUpper(0.5) # NEW: mask is 1 for valid points
        inside.Update() # NEW
        print("  velocity arrows:", inside.GetOutput().GetNumberOfPoints()) # NEW

        arrow = vtk.vtkArrowSource()
        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetInputConnection(inside.GetOutputPort()) # CHANGED: the regular grid instead of the masked mesh points
        glyph.SetInputArrayToProcess(1, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U_inplane")
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToDataScalingOff()
        glyph.SetScaleFactor(0.85 * spacing_g) # CHANGED: arrow length slightly below the grid spacing, so arrows never overlap
        glyph.OrientOn()
        glyph.Update()

        gl_mapper = vtk.vtkPolyDataMapper()
        gl_mapper.SetInputConnection(glyph.GetOutputPort())
        # colour each arrow by |U| of its grid point (glyph copies the point scalars)
        if glyph.GetOutput().GetPointData().GetArray(mag_name) is not None:
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
            k_min, k_max = k_range # Get the range of the turbulent kinetic energy (TKE) field
            levels = [k_min + (k_max - k_min) * (i / 12.0) for i in range(1, 12)]   # 11 isolines
            tke_contour = vtk.vtkContourFilter() # Create a contour filter to generate contours of the TKE field
            tke_contour.SetInputData(sym) # Set the input data for the contour filter to the symmetry plane surface
            tke_contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "k") # Set the input array to process for the contour filter to the TKE field
            tke_contour.ComputeScalarsOn() # Enable scalar computation for the contour filter (this will compute the scalar values at the contour lines)
            tke_contour.SetNumberOfContours(len(levels)) # Set the number of contours to generate for the contour filter (this will generate 11 contour lines)
            for i, v in enumerate(levels): 
                tke_contour.SetValue(i, v) # Set the value of each contour line to the corresponding level (this will generate contour lines at the specified levels)
            tke_contour.Update()

            tke_mapper = vtk.vtkPolyDataMapper() # Create a mapper for the contour lines
            tke_mapper.SetInputConnection(tke_contour.GetOutputPort()) # Set the input connection for the mapper to the output port of the contour filter (this will map the contour lines to graphical primitives)
            color_by_point_array(tke_mapper, "k", (k_min, k_max)) # Colour the contour lines by the TKE field using the Turbo colour map
            tke_actor = vtk.vtkActor() # Create an actor to represent the contour lines
            tke_actor.SetMapper(tke_mapper) # Set the mapper for the actor to the contour lines
            tke_actor.GetProperty().SetLineWidth(TKE_LINE_WIDTH) # Set the line width of the contour lines
            tke_actor.GetProperty().LightingOff() # Disable lighting for the contour lines (this will make the contour lines appear flat and not affected by lighting)
            shoot(renderer, ren_win, [grey_actor(sym, GREY), tke_actor], "tke.png") # Save
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
