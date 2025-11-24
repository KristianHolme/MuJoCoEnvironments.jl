module MuJoCoEnvironmentsCommonRLInterfaceExt

using MuJoCoEnvironments
using CommonRLInterface
const CRL = CommonRLInterface

"""
    CommonRLMuJoCoEnv{E <: AbstractMuJoCoEnv} <: CommonRLInterface.AbstractEnv

Wrapper type that adapts a MuJoCo environment to the CommonRLInterface.

# Fields
- `core_env::E`: The underlying MuJoCo environment
- `last_observation::Vector{Float64}`: Cached last observation
- `last_reward::Float64`: Cached last reward
- `last_terminated::Bool`: Cached termination status
- `last_truncated::Bool`: Cached truncation status
- `last_info::Dict`: Cached info dictionary
"""
mutable struct CommonRLMuJoCoEnv{E<:AbstractMuJoCoEnv} <: CRL.AbstractEnv
    core_env::E
    last_observation::Vector{Float64}
    last_reward::Float64
    last_terminated::Bool
    last_truncated::Bool
    last_info::Dict{String,Any}
    
    function CommonRLMuJoCoEnv(core_env::E) where {E<:AbstractMuJoCoEnv}
        # Initialize with default values
        obs_low, obs_high = MuJoCoEnvironments.get_observation_bounds(core_env)
        obs_size = length(obs_low)
        
        return new{E}(
            core_env,
            zeros(Float64, obs_size),
            0.0,
            false,
            false,
            Dict{String,Any}()
        )
    end
end

# Export the wrapper type
export CommonRLMuJoCoEnv

"""
    CRL.reset!(env::CommonRLMuJoCoEnv)

Reset the environment to its initial state.
"""
function CRL.reset!(env::CommonRLMuJoCoEnv)
    obs = MuJoCoEnvironments.reset_env!(env.core_env)
    env.last_observation = convert(Vector{Float64}, obs)
    env.last_reward = 0.0
    env.last_terminated = false
    env.last_truncated = false
    env.last_info = Dict{String,Any}()
    return nothing
end

"""
    CRL.act!(env::CommonRLMuJoCoEnv, action)

Take an action in the environment.
"""
function CRL.act!(env::CommonRLMuJoCoEnv, action)
    # Convert action to appropriate format
    action_vec = collect(Float64, action)
    
    # Step the environment
    obs, reward, terminated, truncated, info = MuJoCoEnvironments.step_env!(env.core_env, action_vec)
    
    # Cache the results
    env.last_observation = convert(Vector{Float64}, obs)
    env.last_reward = Float64(reward)
    env.last_terminated = terminated
    env.last_truncated = truncated
    env.last_info = info
    
    return nothing
end

"""
    CRL.observe(env::CommonRLMuJoCoEnv)

Get the current observation from the environment.
"""
function CRL.observe(env::CommonRLMuJoCoEnv)
    return env.last_observation
end

"""
    CRL.terminated(env::CommonRLMuJoCoEnv) -> Bool

Check if the environment episode has terminated.
"""
function CRL.terminated(env::CommonRLMuJoCoEnv)
    return env.last_terminated || env.last_truncated
end

"""
    CRL.actions(env::CommonRLMuJoCoEnv)

Get the action space specification.
Returns a vector of intervals, one for each action dimension.
"""
function CRL.actions(env::CommonRLMuJoCoEnv)
    low, high = MuJoCoEnvironments.get_action_bounds(env.core_env)
    # Return as a vector of intervals
    return [low[i]..high[i] for i in 1:length(low)]
end

"""
    CRL.observations(env::CommonRLMuJoCoEnv)

Get the observation space specification.
Returns a vector of intervals, one for each observation dimension.
"""
function CRL.observations(env::CommonRLMuJoCoEnv)
    low, high = MuJoCoEnvironments.get_observation_bounds(env.core_env)
    # Handle infinite bounds
    low = replace(x -> isinf(x) ? -1e10 : x, low)
    high = replace(x -> isinf(x) ? 1e10 : x, high)
    # Return as a vector of intervals
    return [low[i]..high[i] for i in 1:length(low)]
end

"""
    CRL.state(env::CommonRLMuJoCoEnv)

Get the full state of the environment (qpos and qvel).
"""
function CRL.state(env::CommonRLMuJoCoEnv)
    qpos = MuJoCoEnvironments.get_qpos(env.core_env)
    qvel = MuJoCoEnvironments.get_qvel(env.core_env)
    return vcat(qpos, qvel)
end

"""
    CRL.setstate!(env::CommonRLMuJoCoEnv, state)

Set the full state of the environment (qpos and qvel).
"""
function CRL.setstate!(env::CommonRLMuJoCoEnv, state)
    nq = env.core_env.model.nq
    nv = env.core_env.model.nv
    
    # Split state into qpos and qvel
    qpos = state[1:nq]
    qvel = state[(nq+1):(nq+nv)]
    
    # Set the state
    env.core_env.data.qpos .= qpos
    env.core_env.data.qvel .= qvel
    
    # Forward to update derived quantities
    MuJoCo.forward!(env.core_env.model, env.core_env.data)
    
    # Update observation
    obs = MuJoCoEnvironments.get_observation(env.core_env)
    env.last_observation = convert(Vector{Float64}, obs)
    
    return nothing
end

"""
    CRL.clone(env::CommonRLMuJoCoEnv)

Create a deep copy of the environment.
"""
function CRL.clone(env::CommonRLMuJoCoEnv)
    return deepcopy(env)
end

# Convenience constructors for specific environments
export CommonRLHalfCheetahEnv

"""
    CommonRLHalfCheetahEnv(; kwargs...) -> CommonRLMuJoCoEnv

Convenience constructor for HalfCheetah environment with CommonRLInterface.
"""
function CommonRLHalfCheetahEnv(; kwargs...)
    core_env = MuJoCoEnvironments.HalfCheetahEnv(; kwargs...)
    return CommonRLMuJoCoEnv(core_env)
end

end # module

