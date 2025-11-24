# ------------------------------------------------------------
# Base MuJoCo Environment Structure and Utilities
# ------------------------------------------------------------

using MuJoCo
using Random

"""
    MuJoCoEnvBase

Base structure for MuJoCo environments containing common fields.

# Fields
- `model::MuJoCo.Model`: MuJoCo model
- `data::MuJoCo.Data`: MuJoCo data (state)
- `timestep::Int`: Current timestep in the episode
- `max_episode_steps::Int`: Maximum number of steps per episode
- `rng::Random.AbstractRNG`: Random number generator
"""
mutable struct MuJoCoEnvBase <: AbstractMuJoCoEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    rng::Random.AbstractRNG
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

Get the path to a built-in MuJoCo model from the MuJoCo installation.

# Arguments
- `model_name::String`: Name of the model (e.g., "humanoid", "half_cheetah")

# Returns
- `String`: Path to the model XML file
"""
function get_model_path(model_name::String)
    # Try to find the model in the MuJoCo installation
    # MuJoCo models are typically in the mujoco package data directory
    mujoco_dir = dirname(dirname(pathof(MuJoCo)))

    # Common possible locations
    possible_paths = [
        joinpath(mujoco_dir, "model", model_name, "$(model_name).xml"),
        joinpath(mujoco_dir, "assets", "$(model_name).xml"),
        joinpath(mujoco_dir, "data", "model", model_name, "$(model_name).xml"),
    ]

    for path in possible_paths
        if isfile(path)
            return path
        end
    end

    error("Could not find model: $model_name. Tried paths: $possible_paths")
end

"""
    apply_action!(env::AbstractMuJoCoEnv, action::AbstractVector)

Apply an action to the environment by setting the control inputs.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment
- `action::AbstractVector`: Action vector to apply
"""
function apply_action!(env::AbstractMuJoCoEnv, action::AbstractVector)
    # Set the control inputs
    env.data.ctrl .= action
    return nothing
end

"""
    forward_step!(env::AbstractMuJoCoEnv, nsteps::Int=1)

Step the simulation forward by the specified number of steps.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment
- `nsteps::Int`: Number of simulation steps (default: 1)
"""
function forward_step!(env::AbstractMuJoCoEnv, nsteps::Int = 1)
    for _ in 1:nsteps
        MuJoCo.step!(env.model, env.data)
    end
    return nothing
end

"""
    reset_mujoco_state!(env::AbstractMuJoCoEnv; noise_scale::Float64=0.0)

Reset the MuJoCo simulation state with optional noise.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment
- `noise_scale::Float64`: Scale of noise to add to initial state (default: 0.0)
"""
function reset_mujoco_state!(env::AbstractMuJoCoEnv; noise_scale::Float64 = 0.0)
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
    is_truncated(env::AbstractMuJoCoEnv) -> Bool

Check if the episode should be truncated (e.g., time limit reached).

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `Bool`: `true` if episode should be truncated, `false` otherwise
"""
function is_truncated(env::AbstractMuJoCoEnv)
    return env.timestep >= env.max_episode_steps
end

"""
    get_qpos(env::AbstractMuJoCoEnv) -> Vector

Get the joint positions from the environment.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `Vector`: Joint positions
"""
function get_qpos(env::AbstractMuJoCoEnv)
    return copy(env.data.qpos)
end

"""
    get_qvel(env::AbstractMuJoCoEnv) -> Vector

Get the joint velocities from the environment.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `Vector`: Joint velocities
"""
function get_qvel(env::AbstractMuJoCoEnv)
    return copy(env.data.qvel)
end

"""
    get_ctrl(env::AbstractMuJoCoEnv) -> Vector

Get the current control inputs from the environment.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `Vector`: Control inputs
"""
function get_ctrl(env::AbstractMuJoCoEnv)
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
