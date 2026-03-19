# MuJoCoEnvironments.jl

[![Build Status](https://github.com/KristianHolme/MuJoCoEnvironments.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/KristianHolme/MuJoCoEnvironments.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/KristianHolme/MuJoCoEnvironments.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/KristianHolme/MuJoCoEnvironments.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

MuJoCoEnvironments.jl provides a collection of reinforcement learning environments based on [MuJoCo.jl](https://github.com/JamieMair/MuJoCo.jl), for use with [Drill.jl](https://github.com/KristianHolme/Drill.jl).

## Features

- **Multiple Environments**: HalfCheetah, Hopper, Walker2d, Ant, Humanoid, Swimmer, InvertedPendulum, InvertedDoublePendulum, Reacher, Pusher, HumanoidStandup
- **Drill.jl**: Environments use the [DrillInterface.jl](https://github.com/KristianHolme/Drill.jl/tree/main/DrillInterface) RL interface, for usage with [Drill.jl](https://github.com/KristianHolme/Drill.jl)
- **MuJoCo.jl Integration**: Uses native Julia bindings for MuJoCo physics simulation
- **Built-in Models**: Includes MuJoCo XML model files from Gymnasium
- **Visualization Support**: Native MuJoCo rendering through MuJoCo.jl

## Installation

```julia
using Pkg
Pkg.add("MuJoCoEnvironments")
```

## Quick Start

```julia
using MuJoCoEnvironments
using Drill

# Create environment
env = HalfCheetahEnv(max_episode_steps=1000)

# Use Drill interface
reset!(env)
obs = observe(env)
action = rand(action_space(env))
reward = act!(env, action)
```

## Available Environments

| Environment            | Action Dim | Observation Dim | Description                        |
| ---------------------- | ---------- | --------------- | ---------------------------------- |
| HalfCheetah            | 6          | 17              | 2D robot learning to run forward   |
| Hopper                 | 3          | 11              | One-legged robot learning to hop   |
| Walker2d               | 6          | 17              | Two-legged robot learning to walk  |
| Ant                    | 8          | ~105            | Four-legged robot learning to walk |
| Humanoid               | 17         | ~376            | Bipedal humanoid learning to walk  |
| Swimmer                | 2          | 8               | Three-link robot learning to swim  |
| InvertedPendulum       | 1          | 4               | Cart-pole balance                  |
| InvertedDoublePendulum | 1          | 9               | Double cart-pole balance           |
| Reacher                | 2          | 10              | Two-jointed arm reaching           |
| Pusher                 | 7          | 23              | Arm pushing object to goal         |
| HumanoidStandup        | 17         | ~346            | Bipedal robot standing up          |

## Examples

See the [`examples/`](examples/) directory for detailed usage examples:

- `basic_usage.jl` - Core interface usage
- `dril_interface_example.jl` - Using with Drill.jl
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

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See LICENSE file for details.

## Acknowledgments

- [MuJoCo.jl](https://github.com/JamieMair/MuJoCo.jl) for Julia bindings to MuJoCo
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for MuJoCo model files (Copyright (c) 2016 OpenAI, Copyright (c) 2022 Farama Foundation, MIT License)
- [OpenAI Gym](https://github.com/openai/gym) for environment specifications
- [DeepMind Control Suite](https://github.com/deepmind/dm_control) for reference implementations
