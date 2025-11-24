# ------------------------------------------------------------
# Ant-v4 Environment
# ------------------------------------------------------------

"""
    AntEnv <: AbstractMuJoCoEnv

Ant environment - a 4-legged robot that must learn to walk forward.

# Observation Space
- Position coordinates (qpos): 13 dimensions (excluding x,y position)
- Velocity coordinates (qvel): 14 dimensions
Total: 27 dimensions

# Action Space
- 8 continuous actions in [-1, 1], two for each leg

# Reward
- Forward velocity reward
- Healthy reward: bonus for staying alive
- Control cost: penalty for large actions
- Contact cost: penalty for undesired contacts
"""
mutable struct AntEnv <: AbstractMuJoCoEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    rng::Random.AbstractRNG

    forward_reward_weight::Float64
    ctrl_cost_weight::Float64
    contact_cost_weight::Float64
    healthy_reward::Float64
    reset_noise_scale::Float64

    healthy_z_range::Tuple{Float64, Float64}

    prev_x_pos::Float64

    function AntEnv(;
            max_episode_steps::Int = 1000,
            forward_reward_weight::Float64 = 1.0,
            ctrl_cost_weight::Float64 = 0.5,
            contact_cost_weight::Float64 = 5.0e-4,
            healthy_reward::Float64 = 1.0,
            reset_noise_scale::Float64 = 0.1,
            healthy_z_range::Tuple{Float64, Float64} = (0.2, 1.0),
            rng::Random.AbstractRNG = Random.Xoshiro(),
            model_path::Union{String, Nothing} = nothing
        )
        if model_path !== nothing
            model, data = load_model_from_path(model_path)
        else
            try
                path = get_model_path("ant")
                model, data = load_model_from_path(path)
            catch
                @warn "Could not find ant model, using sample model"
                model, data = MuJoCo.sample_model_and_data()
            end
        end

        return new(
            model, data, 0, max_episode_steps, rng,
            forward_reward_weight, ctrl_cost_weight, contact_cost_weight,
            healthy_reward, reset_noise_scale,
            healthy_z_range,
            0.0
        )
    end
end

function is_healthy(env::AntEnv)
    z = env.data.qpos[3]  # z position is at index 3 for Ant
    return env.healthy_z_range[1] <= z <= env.healthy_z_range[2]
end

function reset_env!(env::AntEnv)
    env.timestep = 0
    reset_mujoco_state!(env; noise_scale = env.reset_noise_scale)
    env.prev_x_pos = env.data.qpos[1]
    return get_observation(env)
end

function step_env!(env::AntEnv, action::AbstractVector)
    low, high = get_action_bounds(env)
    action = clip_action(action, low, high)

    x_pos_before = env.data.qpos[1]

    apply_action!(env, action)
    forward_step!(env)

    x_pos_after = env.data.qpos[1]
    x_velocity = (x_pos_after - x_pos_before) / env.model.opt.timestep

    forward_reward = env.forward_reward_weight * x_velocity
    ctrl_cost = env.ctrl_cost_weight * sum(action .^ 2)

    # Contact cost (external forces on body)
    contact_cost = env.contact_cost_weight * sum(abs.(env.data.cfrc_ext))

    healthy_reward = is_healthy(env) ? env.healthy_reward : 0.0
    reward = forward_reward + healthy_reward - ctrl_cost - contact_cost

    env.timestep += 1

    terminated = !is_healthy(env)
    truncated = is_truncated(env)

    observation = get_observation(env)

    info = Dict{String, Any}(
        "x_position" => x_pos_after,
        "x_velocity" => x_velocity,
        "reward_forward" => forward_reward,
        "reward_ctrl" => -ctrl_cost,
        "reward_contact" => -contact_cost,
        "reward_healthy" => healthy_reward
    )

    env.prev_x_pos = x_pos_after

    return observation, reward, terminated, truncated, info
end

function get_observation(env::AntEnv)
    # Exclude x,y position (indices 1,2)
    qpos = env.data.qpos[3:end]
    qvel = env.data.qvel
    return vcat(qpos, qvel)
end

function get_action_bounds(env::AntEnv)
    n_actions = env.model.nu
    return fill(-1.0, n_actions), fill(1.0, n_actions)
end

function get_observation_bounds(env::AntEnv)
    obs_size = (env.model.nq - 2) + env.model.nv
    return fill(-Inf, obs_size), fill(Inf, obs_size)
end
