# ------------------------------------------------------------
# Walker2d-v4 Environment
# ------------------------------------------------------------

"""
    Walker2dEnv{T, R} <: AbstractMuJoCoEnv

Walker2d environment - a 2D two-legged robot that must learn to walk forward.

# Type Parameters
- `T`: Floating point type for weights and parameters (default: `Float64`)
- `R`: Concrete type of the random number generator (default: Julia's default RNG)

# Observation Space
- Position coordinates (qpos): 9 dimensions (excluding x-position)
- Velocity coordinates (qvel): 9 dimensions
Total: 17 dimensions

# Action Space
- 6 continuous actions in [-1, 1], one for each joint

# Reward
- Forward velocity reward
- Healthy reward: bonus for staying upright
- Control cost: penalty for large actions
"""
mutable struct Walker2dEnv{T, R} <: AbstractMuJoCoEnv{T}
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    rng::R

    forward_reward_weight::T
    ctrl_cost_weight::T
    healthy_reward::T
    reset_noise_scale::T

    healthy_z_range::Tuple{T, T}
    healthy_angle_range::Tuple{T, T}

    prev_x_pos::T

    function Walker2dEnv(;
            max_episode_steps::Int = 1000,
            forward_reward_weight = 1.0,
            ctrl_cost_weight = 1.0e-3,
            healthy_reward = 1.0,
            reset_noise_scale = 5.0e-3,
            healthy_z_range = (0.8, 2.0),
            healthy_angle_range = (-1.0, 1.0),
            rng = Random.default_rng(),
            model_path::Union{String, Nothing} = nothing
        )
        T = typeof(forward_reward_weight)
        R = typeof(rng)
        if model_path !== nothing
            model, data = load_model_from_path(model_path)
        else
            path = get_model_path("walker2d")
            model, data = load_model_from_path(path)
        end

        return new{T, R}(
            model, data, 0, max_episode_steps, rng,
            forward_reward_weight, ctrl_cost_weight, healthy_reward, reset_noise_scale,
            healthy_z_range, healthy_angle_range,
            zero(T)
        )
    end
end

function is_healthy(env::Walker2dEnv)
    z = env.data.qpos[2]
    angle = env.data.qpos[3]

    z_healthy = env.healthy_z_range[1] <= z <= env.healthy_z_range[2]
    angle_healthy = env.healthy_angle_range[1] <= angle <= env.healthy_angle_range[2]

    return z_healthy && angle_healthy
end

function is_terminated(env::Walker2dEnv)
    return !is_healthy(env)
end

function reset_env!(env::Walker2dEnv)
    env.timestep = 0
    reset_mujoco_state!(env; noise_scale = env.reset_noise_scale)
    env.prev_x_pos = env.data.qpos[1]
    return get_observation(env)
end

function step_env!(env::Walker2dEnv, action::AbstractVector)
    low, high = get_action_bounds(env)
    action = clip_action(action, low, high)

    x_pos_before = env.data.qpos[1]

    apply_action!(env, action)
    forward_step!(env)

    x_pos_after = env.data.qpos[1]
    x_velocity = (x_pos_after - x_pos_before) / env.model.opt.timestep

    forward_reward = env.forward_reward_weight * x_velocity
    ctrl_cost = env.ctrl_cost_weight * sum(action .^ 2)
    healthy_reward = is_healthy(env) ? env.healthy_reward : 0.0
    reward = forward_reward + healthy_reward - ctrl_cost

    env.timestep += 1

    terminated = !is_healthy(env)
    truncated = is_truncated(env)

    observation = get_observation(env)

    info = Dict{String, Any}(
        "x_position" => x_pos_after,
        "x_velocity" => x_velocity,
        "reward_forward" => forward_reward,
        "reward_ctrl" => -ctrl_cost,
        "reward_healthy" => healthy_reward
    )

    env.prev_x_pos = x_pos_after

    return observation, reward, terminated, truncated, info
end

function get_observation(env::Walker2dEnv)
    qpos = env.data.qpos[2:end]
    qvel = env.data.qvel
    return vcat(qpos, qvel)
end

function get_action_bounds(env::Walker2dEnv)
    n_actions = env.model.nu
    return fill(-1.0, n_actions), fill(1.0, n_actions)
end

function get_observation_bounds(env::Walker2dEnv)
    obs_size = (env.model.nq - 1) + env.model.nv
    return fill(-Inf, obs_size), fill(Inf, obs_size)
end
