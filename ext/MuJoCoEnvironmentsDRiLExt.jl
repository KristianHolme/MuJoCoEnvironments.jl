module MuJoCoEnvironmentsDRiLExt

using MuJoCoEnvironments
using DRiL
using Random

"""
    DRiLMuJoCoEnv{E <: AbstractMuJoCoEnv} <: DRiL.AbstractEnv

Wrapper type that adapts a MuJoCo environment to the DRiL interface.

# Fields
- `core_env::E`: The underlying MuJoCo environment
- `observation_space::DRiL.Box`: The observation space definition
- `action_space::DRiL.Box`: The action space definition
"""
struct DRiLMuJoCoEnv{E<:AbstractMuJoCoEnv} <: DRiL.AbstractEnv
    core_env::E
    observation_space::DRiL.Box
    action_space::DRiL.Box
    
    function DRiLMuJoCoEnv(core_env::E) where {E<:AbstractMuJoCoEnv}
        # Get bounds from the core environment
        obs_low, obs_high = MuJoCoEnvironments.get_observation_bounds(core_env)
        act_low, act_high = MuJoCoEnvironments.get_action_bounds(core_env)
        
        # Create DRiL Box spaces
        observation_space = DRiL.Box(Float32.(obs_low), Float32.(obs_high))
        action_space = DRiL.Box(Float32.(act_low), Float32.(act_high))
        
        return new{E}(core_env, observation_space, action_space)
    end
end

# Export the wrapper type
export DRiLMuJoCoEnv

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
function DRiL.act!(env::DRiLMuJoCoEnv, action::AbstractVector)
    # Convert to appropriate type if needed
    action_vec = convert(Vector{Float64}, action)
    
    # Step the environment
    obs, reward, terminated, truncated, info = MuJoCoEnvironments.step_env!(env.core_env, action_vec)
    
    # Store terminal observation in info if episode ended
    if terminated || truncated
        info["terminal_observation"] = obs
    end
    
    return Float32(reward)
end

"""
    DRiL.observe(env::DRiLMuJoCoEnv) -> observation

Get the current observation from the environment.
"""
function DRiL.observe(env::DRiLMuJoCoEnv)
    obs = MuJoCoEnvironments.get_observation(env.core_env)
    return Float32.(obs)
end

"""
    DRiL.terminated(env::DRiLMuJoCoEnv) -> Bool

Check if the environment episode has terminated.
"""
function DRiL.terminated(env::DRiLMuJoCoEnv)
    # We need to check if we're at a terminal state
    # For now, MuJoCo environments don't have inherent terminal states
    # (except height-based termination which we'll handle per environment)
    return false  # Default: no termination
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
    return Dict{String,Any}(
        "timestep" => env.core_env.timestep,
        "max_episode_steps" => env.core_env.max_episode_steps
    )
end

"""
    Random.seed!(env::DRiLMuJoCoEnv, seed::Int)

Seed the environment's random number generator.
"""
function Random.seed!(env::DRiLMuJoCoEnv, seed::Int)
    Random.seed!(env.core_env.rng, seed)
    return nothing
end

# Convenience constructors for specific environments
export DRiLHalfCheetahEnv

"""
    DRiLHalfCheetahEnv(; kwargs...) -> DRiLMuJoCoEnv

Convenience constructor for HalfCheetah environment with DRiL interface.
"""
function DRiLHalfCheetahEnv(; kwargs...)
    core_env = MuJoCoEnvironments.HalfCheetahEnv(; kwargs...)
    return DRiLMuJoCoEnv(core_env)
end

end # module

