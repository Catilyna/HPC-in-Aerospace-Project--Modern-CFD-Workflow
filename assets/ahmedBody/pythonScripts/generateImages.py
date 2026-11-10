# pythonScripts/generate_images.py
import os
from paraview.simple import *

case_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Case directory
img_dir = os.path.join(case_dir, 'report', 'images') # Output directory for images
os.makedirs(img_dir, exist_ok=True) # Create the output directory if it doesn't exist

case_foam = OpenFOAMReader(FileName=os.path.join(case_dir, 'case.foam')) # Open the OpenFOAM case
case_foam.MeshRegions = ['internalMesh']
case_foam.CellArrays = ['U', 'k'] # Select the fields to visualize (velocity and turbulent kinetic energy)
case_foam.UpdatePipeline()

anim = GetAnimationScene() # Get the animation scene
anim.UpdateAnimationUsingDataTimeSteps() # Update the animation scene to use the time steps from the data

view = CreateRenderView() # Create a new fullHD render view
view.ViewSize = [1920, 1080]
view.Background = [1.0, 1.0, 1.0]

# Symmetry Plane (z = 0)
slice_z = Slice(Input=case_foam) # Create a slice filter for the symmetry plane
slice_z.SliceType = 'Plane'
slice_z.SliceType.Origin = [0.0, 0.0, 0.0] # Set the origin of the slice to the symmetry plane
slice_z.SliceType.Normal = [0.0, 0.0, 1.0] # Set the normal of the slice to be perpendicular to the symmetry plane

# Streamlines for the single STL geometry
streamer = StreamTracer(Input=slice_z, SeedType='Line') # Create a stream tracer filter for the symmetry plane
streamer.Vectors = ['POINTS', 'U'] # Set the vector field to be used for streamlines (velocity field)
streamer.SeedType.Point1 = [-0.05, 0.0, 0.0] # Set the starting point of the seed line for streamlines
streamer.SeedType.Point2 = [-0.05, 0.072, 0.0] # Set the ending point of the seed line for streamlines
streamer.SeedType.Resolution = 100 # Generate 100 seed points along the line for streamlines
stream_disp = Show(streamer, view) # Show the streamlines in the render view
ColorBy(stream_disp, ('POINTS', 'U', 'Magnitude')) # Color the streamlines by the magnitude of the velocity field
SaveScreenshot(os.path.join(img_dir, 'streamlines.png'), view) # Save the screenshot of the streamlines to the output directory
Hide(streamer, view)

# Velocity vectors on the symmetry plane
glyph = Glyph(Input=slice_z, GlyphType='Arrow') # Create a glyph filter for the symmetry plane to visualize velocity vectors
glyph.OrientationArray = ['POINTS', 'U'] # Set the orientation of the glyphs to be based on the velocity field
glyph.ScaleArray = ['POINTS', 'U'] # Set the scale of the glyphs to be based on the magnitude of the velocity field 
glyph_disp = Show(glyph, view) # Show the glyphs in the render view
SaveScreenshot(os.path.join(img_dir, 'velocity_vectors.png'), view) # Save the screenshot of the velocity vectors to the output directory
Hide(glyph, view)

# Turbulent kinetic energy contours from the actual k field
tke_contour = Contour(Input=slice_z) # Create a contour filter for the symmetry plane to visualize turbulent kinetic energy
tke_contour.ContourBy = ['POINTS', 'k'] # Set the field to be contoured to be the turbulent kinetic energy field
tke_contour.Isosurfaces = [0.02 * i for i in range(1, 15)]
tke_disp = Show(tke_contour, view)
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
    ColorBy(slice_disp, ('POINTS', 'U', 'Magnitude'))
    SaveScreenshot(os.path.join(img_dir, f'velocity_plane_x_{x_val:.5f}.png'), view)
    Hide(slice_x, view)
    Delete(slice_x)

print("ParaView rendering saved to report/images/")