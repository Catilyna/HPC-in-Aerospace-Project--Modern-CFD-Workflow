#include "OFstream.H"
#include "argList.H"
#include "boundBox.H"
#include "triSurface.H"
#include <cmath>
#include <string>

using namespace Foam;

struct Domain {
  double xMin, xMax;
  double yMin, yMax;
  double zMin, zMax;
};


Domain computeDomain(double Lx, double Ly, double Lz, double x0, double y0,
                     double z0, double kFront = 5, double kBack = 10,
                     double kSide = 5, double kTop = 5) {
  Domain d;

  d.xMin = x0 - kFront * Lx;
  d.xMax = x0 + Lx + kBack * Lx;

  d.yMin = y0;
  d.yMax = y0 + Ly + kSide * Ly;

  d.zMin = z0;
  d.zMax = z0 + Lz + kTop * Lz;

  return d;
}

scalar computeSurfaceArea(const triSurface& surface) {
    scalar area = 0.0;

    forAll(surface, facei) {
        const labelledTri& tri = surface[facei];

        const point& a = surface.points()[tri[0]];
        const point& b = surface.points()[tri[1]];
        const point& c = surface.points()[tri[2]];

        area += 0.5 * mag((b - a) ^ (c - a));
    }

    return area;
}

int main(int argc, char *argv[]) {
  argList::noParallel(); // Avoid parallelisation
  argList::validArgs.append(
      "stlFile"); // Add the expected argument for the STL file
  argList::validArgs.append("targetSize");
  argList args(argc,
               argv); // Create an argument list from the command line arguments
  
  if (argc != 3) {
    FatalErrorIn(args.executable())
        << "Wrong number of arguments. Expected: " << args.executable()
        << " <stlFile> <targetSize>" << nl << "Usage: "
        << args.executable() << " <stlFile> <targetSize>" << exit(FatalError);
  }

  Info << "Reading STL file: " << args[1] << nl << endl;
  // Correct class for loading STL files natively
  triSurface surface(args[1]);

  const scalar surfaceArea = computeSurfaceArea(surface);

  std::cout << "STL surface area: " << surfaceArea << nl;

  const scalar targetSize = std::stod(args[2]);
  if (targetSize <= 0) {
    FatalErrorIn(args.executable())
        << "Target size must be positive. Received targetSize = " << targetSize
        << exit(FatalError);
  }

  // Calculate bounding box natively in memory
  boundBox bb(surface.points(), true);

  Domain d = computeDomain(bb.span().x(), bb.span().y(), bb.span().z(),
                           bb.min().x(), bb.min().y(), bb.min().z());

  double Lx = bb.span().x();
  double Ly = bb.span().y();
  double Lz = bb.span().z();

  std::cout << " stl dimensions are: " << Lx << "\t" << Ly << "\t" << Lz << "\n";

  // compute cell number in each direction based on the target size
  double xLength = d.xMax - d.xMin;
  double yLength = d.yMax - d.yMin;
  double zLength = d.zMax - d.zMin;
  int xNumCell = std::ceil(xLength / targetSize);
  int yNumCell = std::ceil(yLength / targetSize);
  int zNumCell = std::ceil(zLength / targetSize);

  // compute offsets to exit from the body but still be inside the domain
  const double xOffset = std::max(2.5 * targetSize, 0.40 * Lx);
  const double yOffset = std::max(1.5 * targetSize, 0.25 * Ly);
  const double zOffset = std::max(2.0 * targetSize, 0.35 * Lz);

  const double locX = bb.min().x() - xOffset;
  const double locY = bb.max().y() + yOffset;
  const double locZ = bb.max().z() + zOffset;

  // =========================================================================
  // REFINEMENT BOX CALCULATION
  // =========================================================================

  // X: upstream margin (20% Lx) and extended downstream wake region (1.80 Lx)
  const double refBoxXMin = std::max(d.xMin, bb.min().x() - 0.20 * Lx);
  const double refBoxXMax = std::min(d.xMax, bb.max().x() + 0.20 * Lx);

  // Y: symmetric lateral expansion clamped to the computational domain
  // boundaries (natively respects symmetry planes)
  const double refBoxYMin = std::max(d.yMin, bb.min().y() - 0.30 * Ly);
  const double refBoxYMax = std::min(d.yMax, bb.max().y() + 0.30 * Ly);

  // Z: bounded from ground/road level up to 35% vehicle height above the roof
  const double refBoxZMin = std::max(d.zMin, bb.min().z());
  const double refBoxZMax = std::min(d.zMax, bb.max().z() + 0.35 * Lz);

  OFstream os("system/domainParams");
  os << "xMin " << d.xMin << ";" << nl << "xMax " << d.xMax << ";" << nl
     << "yMin " << d.yMin << ";" << nl << "yMax " << d.yMax << ";" << nl
     << "zMin " << d.zMin << ";" << nl << "zMax " << d.zMax << ";" << nl
     << "nCells (" << xNumCell << " " << yNumCell << " " << zNumCell << ");"
     << endl;

  os << "// --- snappyHexMesh seed point ---" << nl << "locInMeshX " << locX
     << ";" << nl << "locInMeshY " << locY << ";" << nl << "locInMeshZ " << locZ
     << ";" << endl;

  // Export refinementBox vector and scalar definitions for snappyHexMeshDict
  os << "// --- snappyHexMesh refinementBox ---" << nl << "refBoxMin ("
     << refBoxXMin << " " << refBoxYMin << " " << refBoxZMin << ");" << nl
     << "refBoxMax (" << refBoxXMax << " " << refBoxYMax << " " << refBoxZMax
     << ");" << nl << "refBoxXMin " << refBoxXMin << ";" << nl << "refBoxXMax "
     << refBoxXMax << ";" << nl << "refBoxYMin " << refBoxYMin << ";" << nl
     << "refBoxYMax " << refBoxYMax << ";" << nl << "refBoxZMin " << refBoxZMin
     << ";" << nl << "refBoxZMax " << refBoxZMax << ";" << endl;

  Info << "Writing domain parameters to system/domainParams" << nl << endl;
  Info << "\nEnd\n" << endl;
  
  return 0;
}
