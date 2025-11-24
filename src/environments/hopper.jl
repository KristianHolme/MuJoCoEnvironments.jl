# ------------------------------------------------------------
# Hopper-v4 Environment
# ------------------------------------------------------------

"""
    HopperEnv <: AbstractMuJoCoEnv

Hopper environment - a 2D one-legged robot that must learn to hop forward.

# Observation Space
- Position coordinates (qpos): 6 dimensions (excluding x-position)
- Velocity coordinates (qvel): 6 dimensions
Total: 11 dimensions

# Action Space
- 3 continuous actions in [-1, 1], one for each joint

# Reward
- Forward velocity reward
- Healthy reward: bonus for staying upright
- Control cost: penalty for large actions
"""
mutable struct HopperEnv <: AbstractMuJoCoEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    rng::Random.AbstractRNG

    forward_reward_weight::Float64
    ctrl_cost_weight::Float64
    healthy_reward::Float64
    reset_noise_scale::Float64

    # Healthy constraints
    healthy_z_range::Tuple{Float64, Float64}
    healthy_angle_range::Tuple{Float64, Float64}

    prev_x_pos::Float64

    function HopperEnv(;
            max_episode_steps::Int = 1000,
            forward_reward_weight::Float64 = 1.0,
            ctrl_cost_weight::Float64 = 0.001,
            healthy_reward::Float64 = 1.0,
            reset_noise_scale::Float64 = 5.0e-3,
            healthy_z_range::Tuple{Float64, Float64} = (0.7, Float64(Inf)),
            healthy_angle_range::Tuple{Float64, Float64} = (-0.2, 0.2),
            rng::Random.AbstractRNG = Random.Xoshiro(),
            model_path::Union{String, Nothing} = nothing
        )
        if model_path !== nothing
            model, data = load_model_from_path(model_path)
        else
            try
                path = get_model_path("hopper")
                model, data = load_model_from_path(path)
            catch
                @warn "Could not find hopper model, using sample model"
                model, data = MuJoCo.sample_model_and_data()
            end
        end

        return new(
            model, data, 0, max_episode_steps, rng,
            forward_reward_weight, ctrl_cost_weight, healthy_reward, reset_noise_scale,
            healthy_z_range, healthy_angle_range,
            0.0
        )
    end
end

function is_healthy(env::HopperEnv)
    z = env.data.qpos[2]
    angle = env.data.qpos[3]

    z_healthy = env.healthy_z_range[1] <= z <= env.healthy_z_range[2]
    angle_healthy = env.healthy_angle_range[1] <= angle <= env.healthy_angle_range[2]

    return z_healthy && angle_healthy
end

function reset_env!(env::HopperEnv)
    env.timestep = 0
    reset_mujoco_state!(env; noise_scale = env.reset_noise_scale)
    env.prev_x_pos = env.data.qpos[1]
    return get_observation(env)
end

function step_env!(env::HopperEnv, action::AbstractVector)
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

function get_observation(env::HopperEnv)
    qpos = env.data.qpos[2:end]  # Exclude x position
    qvel = env.data.qvel
    return vcat(qpos, qvel)
end

function get_action_bounds(env::HopperEnv)
    n_actions = env.model.nu
    return fill(-1.0, n_actions), fill(1.0, n_actions)
end

function get_observation_bounds(env::HopperEnv)
    obs_size = (env.model.nq - 1) + env.model.nv
    return fill(-Inf, obs_size), fill(Inf, obs_size)
end
