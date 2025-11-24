# MuJoCoEnvironments.jl

[![Build Status](https://github.com/KristianHolme/MuJoCoEnvironments.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/KristianHolme/MuJoCoEnvironments.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/KristianHolme/MuJoCoEnvironments.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/KristianHolme/MuJoCoEnvironments.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

MuJoCoEnvironments.jl provides a collection of reinforcement learning environments based on [MuJoCo.jl](https://github.com/JamieMair/MuJoCo.jl), compatible with multiple RL interfaces including [DRiL.jl](https://github.com/KristianHolme/DRiL.jl) and [CommonRLInterface.jl](https://github.com/JuliaReinforcementLearning/CommonRLInterface.jl).

## Features

- **Multiple Environments**: HalfCheetah, Hopper, Walker2d, Ant, Humanoid, Swimmer
- **Flexible Interface**: Core internal interface with extensions for DRiL.jl and CommonRLInterface.jl
- **MuJoCo.jl Integration**: Uses native Julia bindings for MuJoCo physics simulation
- **No XML Files Required**: Automatically finds and uses MuJoCo's built-in models
- **Visualization Support**: Native MuJoCo rendering through MuJoCo.jl

## Installation

```julia
using Pkg
Pkg.add("MuJoCoEnvironments")
```

For development:
```julia
Pkg.develop(path="path/to/MuJoCoEnvironments")
```

## Quick Start

### Basic Usage

```julia
using MuJoCoEnvironments

# Create an environment
env = HalfCheetahEnv(max_episode_steps=1000)

# Reset to initial state
obs = reset_env!(env)

# Take a step
action = randn(env.model.nu)  # Random action
obs, reward, terminated, truncated, info = step_env!(env, action)
```

### Using with DRiL.jl

```julia
using MuJoCoEnvironments
using DRiL  # Automatically loads the extension

# Create environment with DRiL interface
env = DRiLHalfCheetahEnv(max_episode_steps=1000)

# Use DRiL interface
DRiL.reset!(env)
obs = DRiL.observe(env)
action = rand(DRiL.action_space(env))
reward = DRiL.act!(env, action)
```

### Using with CommonRLInterface.jl

```julia
using MuJoCoEnvironments
using CommonRLInterface

# Create core environment and wrap it
core_env = HopperEnv(max_episode_steps=1000)
env = CommonRLMuJoCoEnv(core_env)

# Use CommonRLInterface
CommonRLInterface.reset!(env)
obs = CommonRLInterface.observe(env)
CommonRLInterface.act!(env, action)
```

## Available Environments

| Environment | Action Dim | Observation Dim | Description |
|-------------|-----------|-----------------|-------------|
| HalfCheetah | 6 | 17 | 2D robot learning to run forward |
| Hopper | 3 | 11 | One-legged robot learning to hop |
| Walker2d | 6 | 17 | Two-legged robot learning to walk |
| Ant | 8 | 27 | Four-legged robot learning to walk |
| Humanoid | 17 | ~376 | Bipedal humanoid learning to walk |
| Swimmer | 2 | 8 | Three-link robot learning to swim |

## Examples

See the [`examples/`](examples/) directory for detailed usage examples:

- `basic_usage.jl` - Core interface usage
- `dril_interface_example.jl` - Using with DRiL.jl
- `commonrl_interface_example.jl` - Using with CommonRLInterface.jl
- `all_environments_showcase.jl` - Overview of all environments

## Rendering

To visualize environments:

```julia
using MuJoCo

# Initialize visualizer (once per session)
MuJoCo.init_visualiser()

# Render environment
render(env)
```

## Architecture

The package uses a **minimal internal interface** (`AbstractMuJoCoEnv`) that is separate from specific RL frameworks. **Extensions** provide compatibility with different RL interfaces:

- `MuJoCoEnvironmentsDRiLExt` - DRiL.jl interface
- `MuJoCoEnvironmentsCommonRLInterfaceExt` - CommonRLInterface.jl interface

This design prevents method conflicts when multiple interface packages are loaded, as each extension defines its own wrapper type (`DRiLMuJoCoEnv`, `CommonRLMuJoCoEnv`).

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See LICENSE file for details.

## Acknowledgments

- [MuJoCo.jl](https://github.com/JamieMair/MuJoCo.jl) for Julia bindings to MuJoCo
- [OpenAI Gym](https://github.com/openai/gym) for environment specifications
- [DeepMind Control Suite](https://github.com/deepmind/dm_control) for reference implementations
