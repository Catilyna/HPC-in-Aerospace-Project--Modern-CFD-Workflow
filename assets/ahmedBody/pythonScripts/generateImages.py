# pythonScripts/generate_images.py
import os
from paraview.simple import *

case_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_dir = os.path.join(case_dir, 'report', 'images')
os.makedirs(img_dir, exist_ok=True)

case_foam = OpenFOAMReader(FileName=os.path.join(case_dir, 'case.foam'))
case_foam.MeshRegions = ['internalMesh']
case_foam.CellArrays = ['U', 'k']
case_foam.UpdatePipeline()

anim = GetAnimationScene()
anim.UpdateAnimationUsingDataTimeSteps()

view = CreateRenderView()
view.ViewSize = [1920, 1080]
view.Background = [1.0, 1.0, 1.0]

# Symmetry Plane (z = 0)
slice_z = Slice(Input=case_foam)
slice_z.SliceType = 'Plane'
slice_z.SliceType.Origin = [0.0, 0.0, 0.0]
slice_z.SliceType.Normal = [0.0, 0.0, 1.0]

# Streamlines for the single STL geometry
streamer = StreamTracer(Input=slice_z, SeedType='High Resolution Line Source')
streamer.Vectors = ['POINTS', 'U']
streamer.SeedType.Point1 = [-0.05, 0.0, 0.0]
streamer.SeedType.Point2 = [-0.05, 0.072, 0.0]
streamer.SeedType.Resolution = 100
stream_disp = Show(streamer, view)
ColorBy(stream_disp, None)
SaveScreenshot(os.path.join(img_dir, 'streamlines.png'), view)
Hide(streamer, view)

# Velocity vectors on the symmetry plane
glyph = Glyph(Input=slice_z, GlyphType='Arrow')
glyph.OrientationArray = ['POINTS', 'U']
glyph.ScaleArray = ['POINTS', 'U']
glyph_disp = Show(glyph, view)
SaveScreenshot(os.path.join(img_dir, 'velocity_vectors.png'), view)
Hide(glyph, view)

# Turbulent kinetic energy contours from the actual k field
tke_contour = Contour(Input=slice_z)
tke_contour.SelectInputScalars = ['POINTS', 'k']
tke_contour.Isosurfaces = [0.02 * i for i in range(1, 15)]
tke_disp = Show(tke_contour, view)
SaveScreenshot(os.path.join(img_dir, 'tke.png'), view)
Hide(tke_contour, view)

# Transverse planes are different views of the same STL, not different geometries.
x_planes = [1.20, 1.50, 2.00]
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