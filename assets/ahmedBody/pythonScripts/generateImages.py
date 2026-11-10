# pythonScripts/generate_images.py
import os
from paraview.simple import *

case_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Case directory
img_dir = os.path.join(case_dir, 'report', 'images') # Output directory for images
os.makedirs(img_dir, exist_ok=True) # Create the output directory if it doesn't exist

case_foam = OpenFOAMReader(FileName=os.path.join(case_dir, 'case.foam')) # Open the OpenFOAM case
case_foam.MeshRegions = ['internalMesh']
case_foam.CellArrays = ['U', 'k', 'p'] # Select the fields to visualize
case_foam.UpdatePipeline()

anim = GetAnimationScene() # Get the animation scene
anim.UpdateAnimationUsingDataTimeSteps() # Update the animation scene to use the time steps from the data

view = CreateRenderView() # Create a new fullHD render view
view.ViewSize = [1920, 1080]
view.UseColorPaletteForBackground = 0 # Disables the global palette override
view.Background = [1.0, 1.0, 1.0]

# Align camera to look at the XZ plane
view.CameraPosition = [2.0, 5.0, 0.0]
view.CameraFocalPoint = [0.0, 0.17, 0.0]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.CameraParallelProjection = 1 # Orthographic projection
view.CameraParallelScale = 0.6 # Set the parallel scale to control the zoom level of the orthographic projection

# The STL touches yMin, so this half-domain uses y = yMin as its symmetry plane.
slice_z = Slice(Input=case_foam) # Create a slice filter for the symmetry plane
slice_z.SliceType = 'Plane'
slice_z.SliceType.Origin = [0.0, 0.17, 0.0]
slice_z.SliceType.Normal = [0.0, 1.0, 0.0]

# Domain slice
slice_disp = Show(slice_z, view)
ColorBy(slice_disp, None) # Turn off field coloring to show uniform grey
slice_disp.Representation = 'Surface With Edges' # Show edges
slice_disp.DiffuseColor = [0.7, 0.7, 0.7]
SaveScreenshot(os.path.join(img_dir, 'domain_slice.png'), view)
Hide(slice_z, view)

# Velocity magnitude field on the domain slice
velocity_disp = Show(slice_z, view)
velocity_disp.Representation = 'Surface' 
ColorBy(velocity_disp, ('CELLS', 'U', 'Magnitude'))
GetColorTransferFunction('U').ApplyPreset('Turbo', True)
SaveScreenshot(os.path.join(img_dir, 'velocity_field.png'), view)
Hide(slice_z, view)

# Pressure field on the domain slice
pressure_disp = Show(slice_z, view)
pressure_disp.Representation = 'Surface' 
ColorBy(pressure_disp, ('CELLS', 'p'))
GetColorTransferFunction('p').ApplyPreset('Turbo', True)
SaveScreenshot(os.path.join(img_dir, 'pressure_field.png'), view)
Hide(slice_z, view)

# Streamlines for the single STL geometry
# streamer = StreamTracer(Input=slice_z, SeedType='Line') # Create a stream tracer filter for the symmetry plane
# streamer.Vectors = ['CELLS', 'U'] # Set the vector field to be used for streamlines (velocity field)
# streamer.SeedType.Point1 = [-0.05, 0.0002, -0.15] # Set the starting point of the seed line for streamlines
# streamer.SeedType.Point2 = [-0.05, 0.0002, 0.14] # Set the ending point of the seed line for streamlines
# streamer.SeedType.Resolution = 100 # Generate 100 seed points along the line for streamlines
# stream_disp = Show(streamer, view) # Show the streamlines in the render view
# ColorBy(stream_disp, ('POINTS', 'U', 'Magnitude')) # Color the streamlines by the magnitude of the velocity field
# SaveScreenshot(os.path.join(img_dir, 'streamlines.png'), view) # Save the screenshot of the streamlines to the output directory
# Hide(streamer, view)
# Convert cell data to point data for smooth streamline integration
c2p_slice = CellDatatoPointData(Input=slice_z)
c2p_slice.UpdatePipeline()
# Create a 2D vector field on the plane by setting Uy = 0
calc = Calculator(Input=c2p_slice)
calc.ResultArrayName = 'U_2D'
calc.Function = 'U_X*iHat + 0*jHat + U_Z*kHat'
calc.UpdatePipeline()
bg_disp = Show(slice_z, view)
ColorBy(bg_disp, None)
bg_disp.Representation = 'Surface'
bg_disp.DiffuseColor = [0.85, 0.85, 0.85]
# Configure 2D streamlines
streamer = StreamTracer(Input=calc, SeedType='Line')
streamer.Vectors = ['POINTS', 'U_2D']
# Automatically retrieve slice boundaries [xmin, xmax, ymin, ymax, zmin, zmax]
bounds = slice_z.GetDataInformation().GetBounds()
z_floor = bounds[4]
z_top = bounds[5]
# Set Seed Line from just above the floor (z_floor + offset) to the desired top height
streamer.SeedType.Point1 = [-0.5, 0.17, z_floor + 0.001]  # Micro-offset prevents getting stuck on U=0 wall
streamer.SeedType.Point2 = [-0.5, 0.17, z_top * 0.7]       # Adjust upper bound height as needed
streamer.SeedType.Resolution = 60
streamer.IntegrationDirection = 'BOTH'
# Render thicker colored lines over the geometry
stream_disp = Show(streamer, view)
stream_disp.LineWidth = 3.0  # Increases visibility of streamlines
ColorBy(stream_disp, ('POINTS', 'U_2D', 'Magnitude'))
GetColorTransferFunction('U_2D').ApplyPreset('Turbo', True)
SaveScreenshot(os.path.join(img_dir, 'streamlines.png'), view)
Hide(streamer, view)
Hide(slice_z, view)

# Velocity vectors on the symmetry plane
glyph = Glyph(Input=slice_z, GlyphType='Arrow') # Create a glyph filter for the symmetry plane to visualize velocity vectors
glyph.OrientationArray = ['CELLS', 'U'] # Set the orientation of the glyphs to be based on the velocity field
glyph.ScaleArray = ['CELLS', 'U'] # Set the scale of the glyphs to be based on the magnitude of the velocity field 
glyph.ScaleFactor = 0.005
glyph_disp = Show(glyph, view) # Show the glyphs in the render view
SaveScreenshot(os.path.join(img_dir, 'velocity_vectors.png'), view) # Save the screenshot of the velocity vectors to the output directory
Hide(glyph, view)

# Turbulent kinetic energy contours from the actual k field
c2p = CellDatatoPointData(Input=slice_z)
c2p.UpdatePipeline()
# Automatically fetch min and max k values from the slice data
k_array = c2p.PointData.GetArray('k')
k_min, k_max = k_array.GetRange()
tke_contour = Contour(Input=c2p)
tke_contour.ContourBy = ['POINTS', 'k']
# Divide the range dynamically into 12 levels
tke_contour.Isosurfaces = [k_min + (k_max - k_min) * (i / 12.0) for i in range(1, 12)]
tke_disp = Show(tke_contour, view)
tke_disp.LineWidth = 2.0
ColorBy(tke_disp, ('POINTS', 'k'))
GetColorTransferFunction('k').ApplyPreset('Turbo', True)
SaveScreenshot(os.path.join(img_dir, 'tke.png'), view)
Hide(tke_contour, view)

# Transverse planes are different views of the same STL, not different geometries.
x_planes = [1.20, 1.50, 2.00] # Define the x-coordinates of the transverse planes to be visualized
for x_val in x_planes:
    slice_x = Slice(Input=case_foam)
    slice_x.SliceType = 'Plane'
    slice_x.SliceType.Origin = [x_val, 0.0, 0.0]
    slice_x.SliceType.Normal = [1.0, 0.0, 0.0]
    
    slice_disp = Show(slice_x, view)
    ColorBy(slice_disp, ('CELLS', 'U', 'Magnitude'))
    SaveScreenshot(os.path.join(img_dir, f'velocity_plane_x_{x_val:.5f}.png'), view)
    Hide(slice_x, view)
    Delete(slice_x)

print("ParaView rendering saved to report/images/")