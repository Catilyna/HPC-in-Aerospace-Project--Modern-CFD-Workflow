#include "argList.H"
#include "triSurface.H"
#include "boundBox.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::noParallel(); // Avoid parallelisation 
    argList::validArgs.append("stlFile"); // Add the expected argument for the STL file
    argList args(argc, argv); // Create an argument list from the command line arguments

    // Correct class for loading STL files natively
    triSurface surface(args[1]);

    // Calculate bounding box natively in memory
    boundBox bb(surface.points(), true);

    Info<< "X_size: " << bb.span().x() << nl
        << "Y_size: " << bb.span().y() << nl
        << "Z_size: " << bb.span().z() << endl;

    return 0;
}