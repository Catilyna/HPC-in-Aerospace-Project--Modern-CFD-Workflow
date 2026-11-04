#include "OFstream.H"
#include "argList.H"
#include "error.H"

#include <cmath>
#include <string>

using namespace Foam;

// Checking if arguments are positive, otherwise throwing a fatal error
scalar readPositive(const argList& args, const label index, const char* name)
{
    const scalar value = std::stod(args[index]);

    if (value <= 0)
    {
        FatalErrorInFunction
            << name << " must be positive; received " << value
            << exit(FatalError);
    }

    return value;
}

int main(int argc, char* argv[])
{
    argList::noParallel(); // No parallelisation

    argList::validArgs.append("targetYPlus"); 
    argList::validArgs.append("referenceVelocity");
    argList::validArgs.append("kinematicViscosity");
    argList::validArgs.append("skinFrictionCoefficient");
    argList::validArgs.append("nSurfaceLayers");
    argList::validArgs.append("expansionRatio");

    argList args(argc, argv); // As in the other utilities, create an argument list from the command line arguments

    if (argc != 7) // throw a fatal error if the number of arguments is not 7 (the executable name + 6 parameters)
    {
        FatalErrorIn(args.executable())
            << "Wrong number of arguments. Expected: " << args.executable()
            << " <targetYPlus> <referenceVelocity> <nu> <Cf>"
            << " <nSurfaceLayers> <expansionRatio>" << nl
            << "Usage: " << args.executable()
            << " <targetYPlus> <referenceVelocity> <nu> <Cf>"
            << " <nSurfaceLayers> <expansionRatio>"
            << exit(FatalError);
    }

    Info << "Calculating wall-layer parameters" << nl << endl;

    const scalar targetYPlus = readPositive(args, 1, "targetYPlus");
    const scalar referenceVelocity =
        readPositive(args, 2, "referenceVelocity");
    const scalar nu = readPositive(args, 3, "kinematicViscosity");
    const scalar skinFrictionCoefficient =
        readPositive(args, 4, "skinFrictionCoefficient");
    const scalar nSurfaceLayersInput =
        readPositive(args, 5, "nSurfaceLayers");
    const scalar expansionRatio = readPositive(args, 6, "expansionRatio");

    // More erros if stuff is not correct, e.g. nSurfaceLayers must be an integer and expansionRatio must be greater than 1
    if (std::floor(nSurfaceLayersInput) != nSurfaceLayersInput)
    {
        FatalErrorIn(args.executable())
            << "nSurfaceLayers must be an integer. Received nSurfaceLayers = "
            << nSurfaceLayersInput
            << exit(FatalError);
    }

    const label nSurfaceLayers =
        static_cast<label>(nSurfaceLayersInput);

    if (expansionRatio <= 1)
    {
        FatalErrorIn(args.executable())
            << "expansionRatio must be greater than 1. Received expansionRatio = "
            << expansionRatio
            << exit(FatalError);
    }


    const scalar frictionVelocity =
        referenceVelocity*std::sqrt(skinFrictionCoefficient/2); // Cf = 2*(u_tau/U)^2 => 
        //u_tau = U*sqrt(Cf/2) = sqrt(tau_w/rho), which is the formula in the slides 
        // rearranges starting from more common parameters
    const scalar firstCellCentre = targetYPlus*nu/frictionVelocity; // y+ = u_tau*y/nu => y = y+*nu/u_tau, exactly the formula in the slides
    const scalar firstLayerThickness = 2*firstCellCentre; // twice the distance from the wall, assuming orthogonal cells
    const scalar totalLayerThickness = firstLayerThickness*(std::pow(expansionRatio, nSurfaceLayers) - 1)/(expansionRatio - 1); // just a convenient geometric series to obtain the value

    OFstream os("system/layerParams");

    os << "// --- snappyHexMesh wall-layer parameters ---" << nl
       << "nSurfaceLayers " << nSurfaceLayers << ";" << nl
       << "expansionRatio " << expansionRatio << ";" << nl
       << "firstLayerThickness " << firstLayerThickness << ";" << nl
       << "totalLayerThickness " << totalLayerThickness << ";" << endl;

    Info << "Estimated friction velocity: " << frictionVelocity << nl
         << "Estimated first-cell centre distance: " << firstCellCentre << nl
         << "First-layer thickness: " << firstLayerThickness << nl
         << "Total layer thickness: " << totalLayerThickness << nl
         << "Writing layer parameters to system/layerParams" << nl << endl;

    Info << "\nEnd\n" << endl;

    return 0;
}