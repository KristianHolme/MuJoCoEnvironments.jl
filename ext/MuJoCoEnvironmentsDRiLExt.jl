module MuJoCoEnvironmentsDRiLExt

using MuJoCoEnvironments
using DRiL
using Random

"""
    DRiLMuJoCoEnv{E <: AbstractMuJoCoEnv{T}, T} <: DRiL.AbstractEnv

Wrapper type that adapts a MuJoCo environment to the DRiL interface.

# Type Parameters
- `E`: The concrete environment type (must be a subtype of `AbstractMuJoCoEnv{T}`)
- `T`: The floating point type used by the environment

# Fields
- `core_env::E`: The underlying MuJoCo environment
- `observation_space::DRiL.Box`: The observation space definition
- `action_space::DRiL.Box`: The action space definition

# Accessing the RNG
The random number generator can be accessed via `env.core_env.rng`.
"""
struct DRiLMuJoCoEnv{E <: AbstractMuJoCoEnv{T}, T} <: DRiL.AbstractEnv
    core_env::E
    observation_space::DRiL.Box
    action_space::DRiL.Box

    function DRiLMuJoCoEnv(core_env::E) where {E <: AbstractMuJoCoEnv{T}} where {T}
        # Get bounds from the core environment
        obs_low, obs_high = MuJoCoEnvironments.get_observation_bounds(core_env)
        act_low, act_high = MuJoCoEnvironments.get_action_bounds(core_env)

        # Create DRiL Box spaces using the environment's type parameter T
        observation_space = DRiL.Box(T.(obs_low), T.(obs_high))
        action_space = DRiL.Box(T.(act_low), T.(act_high))

        return new{E, T}(core_env, observation_space, action_space)
    end
end

# Export the wrapper type
export DRiLMuJoCoEnv

"""
    Base.convert(::Type{DRiLMuJoCoEnv}, env::AbstractMuJoCoEnv{T}) -> DRiLMuJoCoEnv

Convert a core MuJoCo environment to a DRiL-compatible wrapper.

# Arguments
- `env::AbstractMuJoCoEnv{T}`: The core environment to wrap

# Returns
- `DRiLMuJoCoEnv`: A DRiL-compatible wrapper around the core environment
"""
function Base.convert(::Type{DRiLMuJoCoEnv}, env::AbstractMuJoCoEnv{T}) where {T}
    return DRiLMuJoCoEnv(env)
end

"""
    DRiL.reset!(env::DRiLMuJoCoEnv) -> Nothing

Reset the environment to its initial state.
"""
function DRiL.reset!(env::DRiLMuJoCoEnv)
    MuJoCoEnvironments.reset_env!(env.core_env)
    return nothing
end

"""
    DRiL.act!(env::DRiLMuJoCoEnv, action) -> reward

Take an action in the environment and return the reward.
"""
function DRiL.act!(env::DRiLMuJoCoEnv{E, T}, action::AbstractVector) where {E, T}
    # Convert to appropriate type if needed
    action_vec = convert(Vector{T}, action)

    # Step the environment
    obs, reward, terminated, truncated, info = MuJoCoEnvironments.step_env!(env.core_env, action_vec)

    # Store terminal observation in info if episode ended
    if terminated || truncated
        info["terminal_observation"] = obs
    end

    return convert(T, reward)
end

"""
    DRiL.observe(env::DRiLMuJoCoEnv) -> observation

Get the current observation from the environment.
"""
function DRiL.observe(env::DRiLMuJoCoEnv{E, T}) where {E, T}
    obs = MuJoCoEnvironments.get_observation(env.core_env)
    return convert(Vector{T}, obs)
end

"""
    DRiL.terminated(env::DRiLMuJoCoEnv) -> Bool

Check if the environment episode has terminated.
"""
function DRiL.terminated(env::DRiLMuJoCoEnv)
    return MuJoCoEnvironments.is_terminated(env.core_env)
end

"""
    DRiL.truncated(env::DRiLMuJoCoEnv) -> Bool

Check if the environment episode has been truncated (e.g., time limit).
"""
function DRiL.truncated(env::DRiLMuJoCoEnv)
    return MuJoCoEnvironments.is_truncated(env.core_env)
end

"""
    DRiL.observation_space(env::DRiLMuJoCoEnv) -> DRiL.Box

Get the observation space specification.
"""
function DRiL.observation_space(env::DRiLMuJoCoEnv)
    return env.observation_space
end

"""
    DRiL.action_space(env::DRiLMuJoCoEnv) -> DRiL.Box

Get the action space specification.
"""
function DRiL.action_space(env::DRiLMuJoCoEnv)
    return env.action_space
end

"""
    DRiL.get_info(env::DRiLMuJoCoEnv) -> Dict

Get additional environment information.
"""
function DRiL.get_info(env::DRiLMuJoCoEnv)
    # Return basic info about the current state
    return Dict{String, Any}(
        "timestep" => env.core_env.timestep,
        "max_episode_steps" => env.core_env.max_episode_steps
    )
end

"""
    Random.seed!(env::DRiLMuJoCoEnv, seed::Int)

Seed the environment's random number generator.
"""
function Random.seed!(env::DRiLMuJoCoEnv, seed::Int)
    Random.seed!(env.core_env, seed)
    return nothing
end

end # module
