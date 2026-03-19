# MuJoCoEnvironments Examples

This directory contains examples demonstrating how to use the MuJoCoEnvironments package.

## Examples Overview

### 1. `basic_usage.jl`
Basic introduction to using MuJoCo environments with the core interface.

**What it covers:**
- Creating environments (HalfCheetah, Hopper, Walker2d)
- Resetting environments
- Taking steps and getting observations
- Understanding rewards and termination
- Basic rendering instructions

**Run with:**
```bash
julia examples/basic_usage.jl
```

### 2. `dril_interface_example.jl`
Using MuJoCo environments with the Drill.jl interface.

**What it covers:**
- Creating environments with Drill interface
- Using Drill's standard methods (`reset!`, `act!`, `observe`, etc.)
- Working with Drill spaces
- Seeding for reproducibility
- Wrapper pattern for existing environments

**Prerequisites:**
```julia
using Pkg
Pkg.add("Drill")
```

**Run with:**
```bash
julia examples/dril_interface_example.jl
```

### 3. `all_environments_showcase.jl`
Comprehensive showcase of all available environments.

**What it covers:**
- All six MuJoCo environments:
  - HalfCheetah
  - Hopper
  - Walker2d
  - Ant
  - Humanoid
  - Swimmer
- Environment specifications (observation/action dimensions)
- Basic episode execution for each
- Summary comparison table

**Run with:**
```bash
julia examples/all_environments_showcase.jl
```

## Quick Start

To run any example:

1. Make sure MuJoCoEnvironments is installed:
```julia
using Pkg
Pkg.develop(path="path/to/MuJoCoEnvironments")
```

2. Run an example:
```bash
julia examples/basic_usage.jl
```

## Notes

- **Model Files**: The package attempts to find MuJoCo model files from the MuJoCo.jl installation. If models are not found, it will fall back to sample models with a warning.

- **Rendering**: To render environments visually, you need to initialize the MuJoCo visualizer first:
  ```julia
  using MuJoCo
  MuJoCo.init_visualiser()
  # Then call render(env) to display
  ```

- **Performance**: The environments use Float64 for internal calculations but convert to Float32 for interface compatibility where needed.

## Environment Descriptions

- **HalfCheetah**: 2D robot that learns to run forward (6 actuators)
- **Hopper**: 2D one-legged robot that learns to hop (3 actuators)
- **Walker2d**: 2D two-legged robot that learns to walk (6 actuators)
- **Ant**: 3D four-legged robot that learns to walk (8 actuators)
- **Humanoid**: 3D bipedal robot that learns to walk (17 actuators)
- **Swimmer**: 3-link swimming robot (2 actuators)

## Troubleshooting

If you encounter issues:

1. **Model not found warnings**: This is expected if MuJoCo model files aren't in the standard location. The environment will use a fallback sample model.

2. **Drill not loading**: Make sure Drill is installed before running the Drill interface example.

3. **Rendering issues**: Ensure you have the MuJoCo visualizer properly installed and initialized.

## Further Reading

- Main package documentation: See the main README.md
- MuJoCo.jl documentation: https://github.com/JamieMair/MuJoCo.jl
- Drill.jl: https://github.com/KristianHolme/Drill.jl

