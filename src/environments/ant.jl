# ------------------------------------------------------------
# Ant-v5 Environment
# ------------------------------------------------------------

"""
    AntEnv{T, R} <: DrillInterface.AbstractEnv

Ant environment - a 4-legged robot that must learn to walk forward.

Matches Gymnasium Ant-v5 specification.

# Type Parameters
- `T`: Floating point type for weights and parameters (default: `Float64`)
- `R`: Concrete type of the random number generator (default: Julia's default RNG)

# Observation Space
- Position coordinates (qpos): 13 dimensions (excluding x,y position)
- Velocity coordinates (qvel): 14 dimensions
- Contact forces (cfrc_ext): 78 dimensions (excluding worldbody)
Total: 105 dimensions by default

# Action Space
- 8 continuous actions in [-1, 1], two for each leg

# Reward
- Forward velocity reward
- Healthy reward: bonus for staying alive (only when healthy)
- Control cost: penalty for large actions
- Contact cost: penalty for undesired contacts
"""
mutable struct AntEnv{T, R} <: _AbstractMuJoCoEnv{T}
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    frame_skip::Int
    rng::R

    forward_reward_weight::T
    ctrl_cost_weight::T
    contact_cost_weight::T
    healthy_reward::T
    reset_noise_scale::T

    healthy_z_range::Tuple{T, T}
    contact_force_range::Tuple{T, T}
    main_body::Int
    include_cfrc_ext_in_observation::Bool

    prev_xy_pos::Vector{T}

    function AntEnv(;
            max_episode_steps::Int = 1000,
            frame_skip::Int = 5,
            forward_reward_weight = 1.0,
            ctrl_cost_weight = 0.5,
            contact_cost_weight = 5.0e-4,
            healthy_reward = 1.0,
            reset_noise_scale = 0.1,
            healthy_z_range = (0.2, 1.0),
            contact_force_range = (-1.0, 1.0),
            main_body::Union{Int, String} = 1,
            include_cfrc_ext_in_observation::Bool = true,
            rng = Random.default_rng(),
            model_path::Union{String, Nothing} = nothing
        )
        T = typeof(forward_reward_weight)
        R = typeof(rng)
        if model_path !== nothing
            model, data = load_model_from_path(model_path)
        else
            path = get_model_path("ant")
            model, data = load_model_from_path(path)
        end

        # Convert main_body name to ID if needed
        main_body_id = if main_body isa String
            MuJoCo.body(model, main_body).id
        else
            main_body
        end

        return new{T, R}(
            model, data, 0, max_episode_steps, frame_skip, rng,
            forward_reward_weight, ctrl_cost_weight, contact_cost_weight,
            healthy_reward, reset_noise_scale,
            healthy_z_range, contact_force_range,
            main_body_id, include_cfrc_ext_in_observation,
            zeros(T, 2)
        )
    end
end

function is_healthy(env::AntEnv)
    state = vcat(env.data.qpos, env.data.qvel)
    min_z, max_z = env.healthy_z_range
    z = env.data.qpos[3]  # z position is at index 3 for Ant
    return all(isfinite, state) && min_z <= z <= max_z
end

function is_terminated(env::AntEnv)
    return !is_healthy(env)
end

function reset_env!(env::AntEnv)
    env.timestep = 0
    reset_mujoco_state!(env; noise_scale = env.reset_noise_scale)
    xy_pos = [env.data.qpos[1], env.data.qpos[2]]
    env.prev_xy_pos = xy_pos
    return get_observation(env)
end

function step_env!(env::AntEnv, action::AbstractVector)
    low, high = get_action_bounds(env)
    action = clip_action(action, low, high)

    # Get body position before step
    body_before = MuJoCo.body(env.data, env.main_body)
    xy_pos_before = [body_before.xpos[1], body_before.xpos[2]]

    # Apply action and step simulation
    apply_action!(env, action)
    forward_step!(env, env.frame_skip)

    # Get body position after step
    body_after = MuJoCo.body(env.data, env.main_body)
    xy_pos_after = [body_after.xpos[1], body_after.xpos[2]]

    # Calculate dt
    dt = env.model.opt.timestep * env.frame_skip

    # Calculate velocity (xy displacement / dt)
    xy_velocity = (xy_pos_after - xy_pos_before) / dt
    x_velocity = xy_velocity[1]

    # Compute rewards
    forward_reward = env.forward_reward_weight * x_velocity
    healthy_reward = is_healthy(env) ? env.healthy_reward : zero(typeof(env.healthy_reward))
    ctrl_cost = env.ctrl_cost_weight * sum(action .^ 2)

    # Contact cost (clipped contact forces, squared)
    contact_forces = clamp.(env.data.cfrc_ext, env.contact_force_range[1], env.contact_force_range[2])
    contact_cost = env.contact_cost_weight * sum(contact_forces .^ 2)

    reward = forward_reward + healthy_reward - ctrl_cost - contact_cost

    env.timestep += 1

    terminated = !is_healthy(env)
    truncated = is_truncated(env)

    observation = get_observation(env)

    info = Dict{String, Any}(
        "x_position" => env.data.qpos[1],
        "y_position" => env.data.qpos[2],
        "distance_from_origin" => sqrt(env.data.qpos[1]^2 + env.data.qpos[2]^2),
        "x_velocity" => x_velocity,
        "y_velocity" => xy_velocity[2],
        "reward_forward" => forward_reward,
        "reward_ctrl" => -ctrl_cost,
        "reward_contact" => -contact_cost,
        "reward_survive" => healthy_reward
    )

    env.prev_xy_pos = xy_pos_after

    return observation, reward, terminated, truncated, info
end

function get_observation(env::AntEnv)
    # Exclude x,y position (indices 1,2)
    qpos = vec(env.data.qpos[3:end])
    qvel = vec(env.data.qvel)

    obs = vcat(qpos, qvel)

    # Add contact forces if requested (excluding worldbody which is index 0)
    if env.include_cfrc_ext_in_observation
        cfrc_ext = env.data.cfrc_ext[2:end, :]  # Exclude worldbody
        contact_forces = clamp.(cfrc_ext, env.contact_force_range[1], env.contact_force_range[2])
        obs = vcat(obs, vec(contact_forces))
    end

    return obs
end

function get_action_bounds(env::AntEnv)
    n_actions = env.model.nu
    return fill(-1.0, n_actions), fill(1.0, n_actions)
end

function get_observation_bounds(env::AntEnv)
    obs_size = (env.model.nq - 2) + env.model.nv
    if env.include_cfrc_ext_in_observation
        # cfrc_ext has shape (nbody, 6), excluding worldbody
        obs_size += (env.model.nbody - 1) * 6
    end
    return fill(-Inf, obs_size), fill(Inf, obs_size)
end

# Drill interface implementation
function DrillInterface.reset!(env::AntEnv; seed = nothing)
    seed isa Integer && Random.seed!(env.rng, seed)
    reset_env!(env)
    return nothing
end

function DrillInterface.act!(env::AntEnv, action::AbstractVector)
    _, reward, _, _, _ = step_env!(env, action)
    return convert(eltype(env.forward_reward_weight), reward)
end

function DrillInterface.observe(env::AntEnv)
    return get_observation(env)
end

function DrillInterface.terminated(env::AntEnv)
    return is_terminated(env)
end

function DrillInterface.truncated(env::AntEnv)
    return is_truncated(env)
end

function DrillInterface.observation_space(env::AntEnv)
    obs_low, obs_high = get_observation_bounds(env)
    T = typeof(env.forward_reward_weight)
    return DrillInterface.Box(T.(obs_low), T.(obs_high))
end

function DrillInterface.action_space(env::AntEnv)
    act_low, act_high = get_action_bounds(env)
    T = typeof(env.forward_reward_weight)
    return DrillInterface.Box(T.(act_low), T.(act_high))
end

function DrillInterface.get_info(env::AntEnv)
    return Dict{String, Any}(
        "timestep" => env.timestep,
        "max_episode_steps" => env.max_episode_steps
    )
end
