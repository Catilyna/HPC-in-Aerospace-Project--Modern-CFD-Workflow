# HPC-in-Aerospace-Project--Modern-CFD-Workflow

High-performance CFD workflow built on OpenFOAM, optimized for scalable execution and resource efficiency on HPC clusters.

## Repository structure

```text
HPC-in-Aerospace-Project--Modern-CFD-Workflow/
├── LICENSE                        # Project license
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
│       │   ├── momentumTransport  # Turbulence model settings
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
    ├── getStlDim/                 # Utility to extract STL and domain dimensions
    │   ├── getStlDim.C            # C++ source for dimension and mesh parameters
    │   └── Make/                  # OpenFOAM build files
    │       ├── files
    │       └── options
    └── calculateWallLayers/       # Utility to estimate near-wall layer parameters
        ├── calculateWallLayers.C  # C++ source for y+-based layer sizing
        └── Make/                  # OpenFOAM build files
            ├── files
            └── options
```

This repository is organized around a reusable OpenFOAM-based CFD workflow, with the simulation case under the `assets/ahmedBody` folder and supporting utilities under `utilities/` for preprocessing and geometry analysis.

## Utilities

### Building the utilities

A utility can be built as follows:

1. Load the OpenFOAM environment variables.
2. Navigate to the utility directory.
3. Clean the build environment:

```bash
wclean
```

4. Build the utility:

```bash
wmake
```

Once the build is complete, the executable can be invoked as a native OpenFOAM utility.

### Utilities created so far

- `getStlDim`: returns STL dimensions along $x$, $y$, $z$ axes and the dimensions of the refinement box. Requires an STL file to process and a target cell size.
- `calculateWallLayers`: estimates the first wall-layer thickness from a target $y^+$ and writes the result to `system/layerParams`.

#### `calculateWallLayers`

`calculateWallLayers` provides an initial estimate for the prism-layer mesh required by
`snappyHexMesh`. It is located in
`utilities/calculateWallLayers/` and writes the generated parameters to
`system/layerParams` relative to the current OpenFOAM case directory.

The utility requires the following parameters:

1. `targetYPlus`: desired dimensionless wall distance.
2. `referenceVelocity`: representative external-flow velocity $U$ in m/s.
3. `kinematicViscosity`: fluid kinematic viscosity $\nu$ in m²/s.
4. `skinFrictionCoefficient`: representative skin-friction coefficient $C_f$.
5. `nSurfaceLayers`: number of prism layers.
6. `expansionRatio`: ratio between successive layer thicknesses. It must be greater than one.

The friction velocity is estimated from the reference velocity and skin-friction
coefficient:

$$
u_{\tau} = U\sqrt{\frac{C_f}{2}}
$$

This is equivalent to the wall-stress definition:

$$
u_{\tau} = \sqrt{\frac{\tau_w}{\rho}}
$$

The distance from the wall to the centre of the first cell is then estimated from:

$$
y = \frac{y^+\nu}{u_{\tau}}
$$

Assuming that the first cell centre is approximately halfway through the first
layer, the required first-layer thickness is:

$$
t_1 \approx 2y = \frac{2y^+\nu}{u_{\tau}}
$$

For $N$ layers and expansion ratio $r$, the estimated total layer thickness is:

$$
T = t_1\frac{r^N-1}{r-1}
$$

Example:

```bash
cd assets/ahmedBody
./Allrun.pre
```

The workflow parameters are provided in the single user-editable file
`assets/ahmedBody/system/workflowParams`. Edit this file before running the
workflow; it contains the background mesh, parallel decomposition, and
wall-layer inputs. `Allrun.pre` parses the entries and invokes
`calculateWallLayers` automatically. The utility writes a file containing
entries that are included in `system/snappyHexMeshDict`:

```foam
#include "${FOAM_CASE}/system/layerParams"
```

The generated file contains `nSurfaceLayers`, `expansionRatio`,
`firstLayerThickness`, and `totalLayerThickness`.

Note that the result of this utility is just an initial estimate. The local wall shear stress varies over the Ahmed body, and wherefore one fixed first-layer thickness cannot produce the target $y^+$ everywhere. After running the simulation, it is possible to calculate and inspect the actual wall-distance field with:

```bash
postProcess -func yPlus
```

The mesh should then be adjusted if the resulting $y^+$ distribution is outside
the range required by the selected wall treatment.
