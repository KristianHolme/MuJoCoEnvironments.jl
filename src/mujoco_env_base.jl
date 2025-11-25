# ------------------------------------------------------------
# Base MuJoCo Environment Structure and Utilities
# ------------------------------------------------------------

using MuJoCo
using Random

"""
    MuJoCoEnvBase{T, R}

Base structure for MuJoCo environments containing common fields.

# Type Parameters
- `T`: Floating point type used by the environment
- `R`: Concrete type of the random number generator (default: Julia's default RNG)

# Fields
- `model::MuJoCo.Model`: MuJoCo model
- `data::MuJoCo.Data`: MuJoCo data (state)
- `timestep::Int`: Current timestep in the episode
- `max_episode_steps::Int`: Maximum number of steps per episode
- `frame_skip::Int`: Number of simulation steps per environment step
- `rng::R`: Random number generator
"""
mutable struct MuJoCoEnvBase{T, R} <: _AbstractMuJoCoEnv{T}
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    frame_skip::Int
    rng::R
end

"""
    load_model_from_path(model_path::String) -> (model, data)

Load a MuJoCo model and create initial data from an XML file path.

# Arguments
- `model_path::String`: Path to the MuJoCo XML file

# Returns
- `model`: MuJoCo model
- `data`: Initial MuJoCo data
"""
function load_model_from_path(model_path::String)
    model = MuJoCo.load_model(model_path)
    data = MuJoCo.init_data(model)
    return model, data
end

"""
    get_model_path(model_name::String) -> String

Get the path to a MuJoCo model XML file from the package's `assets/` directory.

# Arguments
- `model_name::String`: Name of the model (e.g., "humanoid", "half_cheetah", "ant", "hopper", "walker2d", "swimmer")

# Returns
- `String`: Path to the model XML file

# Throws
- `ErrorException`: If the model cannot be found in the assets directory
"""
function get_model_path(model_name::String)
    package_dir = @__DIR__
    package_root = dirname(package_dir)
    assets_path = joinpath(package_root, "assets", "$(model_name).xml")

    if isfile(assets_path)
        return assets_path
    end

    error("Could not find model: $model_name. Expected at: $assets_path")
end

"""
    apply_action!(env::_AbstractMuJoCoEnv, action::AbstractVector)

Apply an action to the environment by setting the control inputs.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment
- `action::AbstractVector`: Action vector to apply
"""
function apply_action!(env::_AbstractMuJoCoEnv{T}, action::AbstractVector) where {T}
    # Set the control inputs
    env.data.ctrl .= action
    return nothing
end

"""
    forward_step!(env::_AbstractMuJoCoEnv, nsteps::Int=1)

Step the simulation forward by the specified number of steps.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment
- `nsteps::Int`: Number of simulation steps (default: 1)
"""
function forward_step!(env::_AbstractMuJoCoEnv{T}, nsteps::Int = 1) where {T}
    for _ in 1:nsteps
        MuJoCo.step!(env.model, env.data)
    end
    return nothing
end

"""
    reset_mujoco_state!(env::_AbstractMuJoCoEnv; noise_scale::Real=0.0)

Reset the MuJoCo simulation state with optional noise.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment
- `noise_scale::Real`: Scale of noise to add to initial state (default: 0.0)
"""
function reset_mujoco_state!(env::_AbstractMuJoCoEnv{T}; noise_scale::Real = 0.0) where {T}
    # Reset to keyframe 0 (default initial state)
    MuJoCo.reset!(env.model, env.data)

    # Add noise if specified
    if noise_scale > 0.0
        nq = env.model.nq  # number of position coordinates
        nv = env.model.nv  # number of velocity coordinates

        # Add noise to positions (qpos)
        env.data.qpos .+= noise_scale .* randn(env.rng, nq)

        # Add noise to velocities (qvel)
        env.data.qvel .+= noise_scale .* randn(env.rng, nv)
    end

    # Forward the simulation to compute derived quantities
    MuJoCo.forward!(env.model, env.data)

    return nothing
end

"""
    is_truncated(env::_AbstractMuJoCoEnv) -> Bool

Check if the episode should be truncated (e.g., time limit reached).

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `Bool`: `true` if episode should be truncated, `false` otherwise
"""
function is_truncated(env::_AbstractMuJoCoEnv{T}) where {T}
    return env.timestep >= env.max_episode_steps
end

"""
    get_qpos(env::_AbstractMuJoCoEnv) -> Vector

Get the joint positions from the environment.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `Vector`: Joint positions
"""
function get_qpos(env::_AbstractMuJoCoEnv{T}) where {T}
    return copy(env.data.qpos)
end

"""
    get_qvel(env::_AbstractMuJoCoEnv) -> Vector

Get the joint velocities from the environment.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `Vector`: Joint velocities
"""
function get_qvel(env::_AbstractMuJoCoEnv{T}) where {T}
    return copy(env.data.qvel)
end

"""
    get_ctrl(env::_AbstractMuJoCoEnv) -> Vector

Get the current control inputs from the environment.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `Vector`: Control inputs
"""
function get_ctrl(env::_AbstractMuJoCoEnv{T}) where {T}
    return copy(env.data.ctrl)
end

"""
    clip_action(action::AbstractVector, low::AbstractVector, high::AbstractVector) -> Vector

Clip action to be within specified bounds.

# Arguments
- `action::AbstractVector`: Action to clip
- `low::AbstractVector`: Lower bounds
- `high::AbstractVector`: Upper bounds

# Returns
- `Vector`: Clipped action
"""
function clip_action(action::AbstractVector, low::AbstractVector, high::AbstractVector)
    return clamp.(action, low, high)
end

"""
    Random.seed!(env::_AbstractMuJoCoEnv, seed::Integer)

Seed the environment's random number generator.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment
- `seed::Integer`: Seed value for the random number generator

# Returns
- `Nothing`
"""
function Random.seed!(env::_AbstractMuJoCoEnv{T}, seed::Integer) where {T}
    Random.seed!(env.rng, seed)
    return nothing
end
