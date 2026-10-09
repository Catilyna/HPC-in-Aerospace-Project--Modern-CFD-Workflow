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

int main(int argc, char *argv[]) {
  argList::noParallel(); // Avoid parallelisation
  argList::validArgs.append(
      "stlFile"); // Add the expected argument for the STL file
  argList::validArgs.append("targetSize");
  argList args(argc,
               argv); // Create an argument list from the command line arguments

  // Correct class for loading STL files natively
  triSurface surface(args[1]);

  const scalar targetSize = std::stod(args[2]);
  // Calculate bounding box natively in memory
  boundBox bb(surface.points(), true);

  Domain d = computeDomain(bb.span().x(), bb.span().y(), bb.span().z(),
                           bb.min().x(), bb.min().y(), bb.min().z());

  // compute initial cell number

  double xLength = d.xMax - d.xMin;
  double yLength = d.yMax - d.yMin;
  double zLength = d.zMax - d.zMin;

  int xNumCell = std::ceil(xLength / targetSize);
  int yNumCell = std::ceil(yLength / targetSize);
  int zNumCell = std::ceil(zLength / targetSize);

  Info << "X_size: " << bb.span().x() << nl << "Y_size: " << bb.span().y() << nl
       << "Z_size: " << bb.span().z() << nl << "Min_x: " << bb.min().x() << nl
       << "Min_y: " << bb.min().y() << nl << "Min_z: " << bb.min().z() << endl;

  OFstream os("system/domainParams");
  os << "xMin " << d.xMin << ";" << nl << "xMax " << d.xMax << ";" << nl
     << "yMin " << d.yMin << ";" << nl << "yMax " << d.yMax << ";" << nl
     << "zMin " << d.zMin << ";" << nl << "zMax " << d.zMax << ";" << nl
     << "nCells (" << xNumCell << " " << yNumCell << " " << zNumCell << ");"
     << endl;

  Info << "mesh_x_Min " << d.xMin << nl << "mesh_x_Max " << d.xMax << nl
       << "mesh_y_Min " << d.yMin << nl << "mesh_y_Max " << d.yMax << nl
       << "mesh_z_Min " << d.zMin << nl << "mesh_z_Max " << d.zMax << endl;

  return 0;
}
