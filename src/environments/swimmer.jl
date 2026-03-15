# ------------------------------------------------------------
# Swimmer-v4 Environment
# ------------------------------------------------------------

"""
    SwimmerEnv{T, R} <: DrillInterface.AbstractEnv

Swimmer environment - a 3-link swimming robot that must learn to swim forward.

# Type Parameters
- `T`: Floating point type for weights and parameters (default: `Float64`)
- `R`: Concrete type of the random number generator (default: Julia's default RNG)

# Observation Space
- Position coordinates (qpos): 5 dimensions (excluding x,y position)
- Velocity coordinates (qvel): 5 dimensions
Total: 8 dimensions

# Action Space
- 2 continuous actions in [-1, 1], one for each joint

# Reward
- Forward velocity reward
- Control cost: penalty for large actions
"""
mutable struct SwimmerEnv{T <: AbstractFloat, R <: Random.AbstractRNG} <: _AbstractMuJoCoEnv{T}
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    rng::R

    forward_reward_weight::T
    ctrl_cost_weight::T
    reset_noise_scale::T

    prev_x_pos::T

    function SwimmerEnv(;
            max_episode_steps::Int = 1000,
            forward_reward_weight = 1.0,
            ctrl_cost_weight = 1.0e-4,
            reset_noise_scale = 0.1,
            rng = Random.default_rng(),
            model_path::Union{String, Nothing} = nothing
        )
        T = typeof(forward_reward_weight)
        R = typeof(rng)
        if model_path !== nothing
            model, data = load_model_from_path(model_path)
        else
            path = get_model_path("swimmer")
            model, data = load_model_from_path(path)
        end

        return new{T, R}(
            model, data, 0, max_episode_steps, rng,
            forward_reward_weight, ctrl_cost_weight, reset_noise_scale,
            zero(T)
        )
    end
end

function reset_env!(env::SwimmerEnv)
    env.timestep = 0
    reset_mujoco_state!(env; noise_scale = env.reset_noise_scale)
    env.prev_x_pos = env.data.qpos[1]
    return get_observation(env)
end

function step_env!(env::SwimmerEnv, action::AbstractVector)
    low, high = get_action_bounds(env)
    action = clip_action(action, low, high)

    x_pos_before = env.data.qpos[1]

    apply_action!(env, action)
    forward_step!(env)

    x_pos_after = env.data.qpos[1]
    x_velocity = (x_pos_after - x_pos_before) / env.model.opt.timestep

    forward_reward = env.forward_reward_weight * x_velocity
    ctrl_cost = env.ctrl_cost_weight * sum(action .^ 2)
    reward = forward_reward - ctrl_cost

    env.timestep += 1

    terminated = false  # Swimmer doesn't have termination conditions
    truncated = is_truncated(env)

    observation = get_observation(env)

    info = Dict{String, Any}(
        "x_position" => x_pos_after,
        "x_velocity" => x_velocity,
        "reward_forward" => forward_reward,
        "reward_ctrl" => -ctrl_cost
    )

    env.prev_x_pos = x_pos_after

    return observation, reward, terminated, truncated, info
end

function get_observation(env::SwimmerEnv)
    # Exclude x,y position
    qpos = vec(env.data.qpos[3:end])
    qvel = vec(env.data.qvel)
    return vcat(qpos, qvel)
end

function get_action_bounds(env::SwimmerEnv)
    n_actions = env.model.nu
    return fill(-1.0, n_actions), fill(1.0, n_actions)
end

function get_observation_bounds(env::SwimmerEnv)
    obs_size = (env.model.nq - 2) + env.model.nv
    return fill(-Inf, obs_size), fill(Inf, obs_size)
end

function is_terminated(env::SwimmerEnv)
    return false  # Swimmer doesn't have termination conditions
end

# Drill interface implementation
function DrillInterface.reset!(env::SwimmerEnv)
    reset_env!(env)
    return nothing
end

function DrillInterface.act!(env::SwimmerEnv, action::AbstractVector)
    obs, reward, terminated, truncated, info = step_env!(env, action)
    if terminated || truncated
        info["terminal_observation"] = obs
    end
    return convert(eltype(env.forward_reward_weight), reward)
end

function DrillInterface.observe(env::SwimmerEnv)
    return get_observation(env)
end

function DrillInterface.terminated(env::SwimmerEnv)
    return is_terminated(env)
end

function DrillInterface.truncated(env::SwimmerEnv)
    return is_truncated(env)
end

function DrillInterface.observation_space(env::SwimmerEnv)
    obs_low, obs_high = get_observation_bounds(env)
    T = typeof(env.forward_reward_weight)
    return DrillInterface.Box(T.(obs_low), T.(obs_high))
end

function DrillInterface.action_space(env::SwimmerEnv)
    act_low, act_high = get_action_bounds(env)
    T = typeof(env.forward_reward_weight)
    return DrillInterface.Box(T.(act_low), T.(act_high))
end

function DrillInterface.get_info(env::SwimmerEnv)
    return Dict{String, Any}(
        "timestep" => env.timestep,
        "max_episode_steps" => env.max_episode_steps
    )
end
