# HPC-in-Aerospace-Project--Modern-CFD-Workflow

High-performance CFD workflow built on OpenFOAM, optimized for scalable execution and resource efficiency on HPC clusters.

## Repository structure

```text
HPC-in-Aerospace-Project--Modern-CFD-Workflow/
├── LICENSE                         # Project license
├── README.md                      # Project overview and usage notes
├── .gitignore                     # Git ignore rules
├── assets/                        # Case assets and example simulation files
│   └── ahmedBody/                 # Ahmed body OpenFOAM case used for CFD setup
│       ├── 0.orig/                # Original boundary and initial conditions
│       │   ├── k                  # Turbulence kinetic energy field
│       │   ├── nut                # Turbulent viscosity field
│       │   ├── omega              # Specific dissipation rate field
│       │   ├── p                  # Pressure field
│       │   └── U.orig             # Velocity field template
│       ├── Allclean               # Removes generated case files
│       ├── Allrun                 # Main execution script for the case
│       ├── Allrun.pre             # Pre-processing steps before running the solver
│       ├── constant/              # Permanent flow properties and geometry
│       │   ├── geometry/          # Mesh geometry input, including the Ahmed body STL/mesh
│       │   ├── momentumTransport # Turbulence model settings
│       │   └── physicalProperties # Fluid properties and material model data
│       └── system/                # Solver, discretization, and mesh configuration
│           ├── blockMeshDict      # Mesh generation settings
│           ├── controlDict        # Solver control parameters and runtime settings
│           ├── decomposeParDict   # Domain decomposition configuration for parallel runs
│           ├── fvSchemes          # Discretization schemes
│           ├── fvSolution         # Linear solver and algorithm settings
│           ├── meshQualityDict    # Mesh quality checks
│           ├── refineMeshDict     # Mesh refinement configuration
│           ├── snappyHexMeshDict  # SnappyHexMesh setup for geometry refinement
│           ├── surfaceFeaturesDict# Surface feature extraction settings
│           └── ...
├── src/                           # Source code folder for workflow components
│   └── (currently empty)          # Intended for future scripts or solver integrations
└── utilities/                     # Helper utilities and supporting tools
    └── STL_dim_extractor/         # Utility to extract STL dimensions from geometry files
        ├── get_stl_dim.C         # C++ source for dimension extraction
        └── Make/                  # Build files for compiling the utility
            ├── files
            ├── options
            └── linux64GccDPInt32Opt/
``` 

This repository is organized around a reusable OpenFOAM-based CFD workflow, with the simulation case under the `assets/ahmedBody` folder and supporting utilities under `utilities/` for preprocessing and geometry analysis.

## Utilities

### Building the utilities

A Paraview utility can be build as follows:

1) first, load the OpenFoam enviroment variables;
2) navigate the the main folder of the utility (where the source code is located);
3) clean the building enviroment with:

```bash
wclean
```

4) build the utility with the following command:

```bash
wbuild
```

Once the building process is terminanted, the executable can be invoked as a native OpenFoam terminal utility.

### Utilities created so far

-  `get_stl_dim`: returns SLT dimentions along $x$, $y$, $z$ axis. Requires an STL file to process.