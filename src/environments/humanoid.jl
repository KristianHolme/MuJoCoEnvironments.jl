# ------------------------------------------------------------
# Humanoid-v4 Environment
# ------------------------------------------------------------

"""
    HumanoidEnv{T, R} <: DRiL.AbstractEnv

Humanoid environment - a 3D bipedal robot that must learn to walk forward.

# Type Parameters
- `T`: Floating point type for weights and parameters (default: `Float64`)
- `R`: Concrete type of the random number generator (default: Julia's default RNG)

# Observation Space
- Position coordinates (qpos): 24 dimensions (excluding x,y position)
- Velocity coordinates (qvel): 23 dimensions
- Center of mass velocity: 3 dimensions (com_inertia and com_velocity)
- Actuator forces: Variable dimensions
Total: ~376 dimensions (varies with model specifics)

# Action Space
- 17 continuous actions in [-0.4, 0.4], one for each actuator

# Reward
- Forward velocity reward
- Healthy reward: bonus for staying upright
- Control cost: penalty for large actions
"""
mutable struct HumanoidEnv{T, R} <: _AbstractMuJoCoEnv{T}
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

    prev_x_pos::T

    function HumanoidEnv(;
            max_episode_steps::Int = 1000,
            forward_reward_weight = 1.25,
            ctrl_cost_weight = 0.1,
            healthy_reward = 5.0,
            reset_noise_scale = 1.0e-2,
            healthy_z_range = (1.0, 2.0),
            rng = Random.default_rng(),
            model_path::Union{String, Nothing} = nothing
        )
        T = typeof(forward_reward_weight)
        R = typeof(rng)
        if model_path !== nothing
            model, data = load_model_from_path(model_path)
        else
            path = get_model_path("humanoid")
            model, data = load_model_from_path(path)
        end

        return new{T, R}(
            model, data, 0, max_episode_steps, rng,
            forward_reward_weight, ctrl_cost_weight, healthy_reward, reset_noise_scale,
            healthy_z_range,
            zero(T)
        )
    end
end

function is_healthy(env::HumanoidEnv)
    z = env.data.qpos[3]  # z position
    return env.healthy_z_range[1] <= z <= env.healthy_z_range[2]
end

function is_terminated(env::HumanoidEnv)
    return !is_healthy(env)
end

function reset_env!(env::HumanoidEnv)
    env.timestep = 0
    reset_mujoco_state!(env; noise_scale = env.reset_noise_scale)
    env.prev_x_pos = env.data.qpos[1]
    return get_observation(env)
end

function step_env!(env::HumanoidEnv, action::AbstractVector)
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

function get_observation(env::HumanoidEnv)
    # Exclude x,y position
    qpos = vec(env.data.qpos[3:end])
    qvel = vec(env.data.qvel)

    # Basic observation: qpos (without x,y) and qvel
    return vcat(qpos, qvel)
end

function get_action_bounds(env::HumanoidEnv)
    n_actions = env.model.nu
    # Humanoid typically has action range [-0.4, 0.4]
    return fill(-0.4, n_actions), fill(0.4, n_actions)
end

function get_observation_bounds(env::HumanoidEnv)
    obs_size = (env.model.nq - 2) + env.model.nv
    return fill(-Inf, obs_size), fill(Inf, obs_size)
end

# DRiL interface implementation
function DRiL.reset!(env::HumanoidEnv)
    reset_env!(env)
    return nothing
end

function DRiL.act!(env::HumanoidEnv, action::AbstractVector)
    obs, reward, terminated, truncated, info = step_env!(env, action)
    if terminated || truncated
        info["terminal_observation"] = obs
    end
    return convert(eltype(env.forward_reward_weight), reward)
end

function DRiL.observe(env::HumanoidEnv)
    return get_observation(env)
end

function DRiL.terminated(env::HumanoidEnv)
    return is_terminated(env)
end

function DRiL.truncated(env::HumanoidEnv)
    return is_truncated(env)
end

function DRiL.observation_space(env::HumanoidEnv)
    obs_low, obs_high = get_observation_bounds(env)
    T = typeof(env.forward_reward_weight)
    return DRiL.Box(T.(obs_low), T.(obs_high))
end

function DRiL.action_space(env::HumanoidEnv)
    act_low, act_high = get_action_bounds(env)
    T = typeof(env.forward_reward_weight)
    return DRiL.Box(T.(act_low), T.(act_high))
end

function DRiL.get_info(env::HumanoidEnv)
    return Dict{String, Any}(
        "timestep" => env.timestep,
        "max_episode_steps" => env.max_episode_steps
    )
end
