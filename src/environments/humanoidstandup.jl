# ------------------------------------------------------------
# HumanoidStandup-v5 Environment
# ------------------------------------------------------------

"""
    HumanoidStandupEnv{T, R} <: DrillInterface.AbstractEnv

Humanoid Standup environment - 3D bipedal robot learning to stand up.

Matches Gymnasium HumanoidStandup-v5 specification.

# Type Parameters
- `T`: Floating point type for weights and parameters (default: `Float64`)
- `R`: Concrete type of the random number generator (default: Julia's default RNG)

# Observation Space
- qpos (excluding x,y): 20 dimensions
- qvel: 23 dimensions
- cinert (excluding worldbody): 130 dimensions
- cvel (excluding worldbody): 78 dimensions
- qfrc_actuator (excluding root): 17 dimensions
- cfrc_ext (excluding worldbody): 78 dimensions
Total: 346 dimensions by default

# Action Space
- 17 continuous actions in [-0.4, 0.4], torques at joints

# Reward
- Uph cost: reward for moving up (standing up)
- Control cost: penalty for large actions
- Impact cost: penalty for large contact forces
"""
mutable struct HumanoidStandupEnv{T, R} <: _AbstractMuJoCoEnv{T}
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    frame_skip::Int
    rng::R

    uph_cost_weight::T
    ctrl_cost_weight::T
    impact_cost_weight::T
    impact_cost_range::Tuple{T, T}
    reset_noise_scale::T

    include_cinert_in_observation::Bool
    include_cvel_in_observation::Bool
    include_qfrc_actuator_in_observation::Bool
    include_cfrc_ext_in_observation::Bool

    init_qpos::Vector{T}

    function HumanoidStandupEnv(;
            max_episode_steps::Int = 1000,
            frame_skip::Int = 5,
            uph_cost_weight = 1.0,
            ctrl_cost_weight = 0.1,
            impact_cost_weight = 0.5e-6,
            impact_cost_range = (-Inf, 10.0),
            reset_noise_scale = 1.0e-2,
            include_cinert_in_observation::Bool = true,
            include_cvel_in_observation::Bool = true,
            include_qfrc_actuator_in_observation::Bool = true,
            include_cfrc_ext_in_observation::Bool = true,
            rng = Random.default_rng(),
            model_path::Union{String, Nothing} = nothing
        )
        T = typeof(uph_cost_weight)
        R = typeof(rng)
        if model_path !== nothing
            model, data = load_model_from_path(model_path)
        else
            path = get_model_path("humanoidstandup")
            model, data = load_model_from_path(path)
        end

        init_qpos = vec(copy(something(data.qpos)))

        return new{T, R}(
            model, data, 0, max_episode_steps, frame_skip, rng,
            uph_cost_weight, ctrl_cost_weight, impact_cost_weight,
            impact_cost_range, reset_noise_scale,
            include_cinert_in_observation, include_cvel_in_observation,
            include_qfrc_actuator_in_observation, include_cfrc_ext_in_observation,
            init_qpos
        )
    end
end

function is_terminated(env::HumanoidStandupEnv)
    return false  # HumanoidStandup never terminates
end

function reset_env!(env::HumanoidStandupEnv)
    env.timestep = 0
    MuJoCo.reset!(env.model, env.data)  # start from the initial state, not the previous episode

    noise_low = -env.reset_noise_scale
    noise_high = env.reset_noise_scale

    qpos = env.init_qpos .+ (noise_low .+ (noise_high - noise_low) .* rand(env.rng, length(env.init_qpos)))
    qvel = something(env.data.qvel) .+ (noise_low .+ (noise_high - noise_low) .* rand(env.rng, length(something(env.data.qvel))))

    state = vcat(vec(qpos), vec(qvel), zeros(env.model.na))
    MuJoCo.set_physics_state!(env.model, env.data, state)
    MuJoCo.forward!(env.model, env.data)

    return get_observation(env)
end

function step_env!(env::HumanoidStandupEnv, action::AbstractVector)
    low, high = get_action_bounds(env)
    action = clip_action(action, low, high)

    apply_action!(env, action)
    forward_step!(env, env.frame_skip)

    pos_after = env.data.qpos[3]  # z position

    observation = get_observation(env)

    # Calculate reward
    uph_cost = (pos_after - 0.0) / env.model.opt.timestep
    quad_ctrl_cost = env.ctrl_cost_weight * sum(env.data.ctrl .^ 2)
    quad_impact_cost = env.impact_cost_weight * sum(env.data.cfrc_ext .^ 2)
    quad_impact_cost = clamp(quad_impact_cost, env.impact_cost_range[1], env.impact_cost_range[2])

    reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1.0

    terminated = false
    truncated = is_truncated(env)

    info = Dict{String, Any}(
        "x_position" => env.data.qpos[1],
        "y_position" => env.data.qpos[2],
        "z_distance_from_origin" => env.data.qpos[3] - env.init_qpos[3],
        "tendon_length" => env.data.ten_length,
        "tendon_velocity" => env.data.ten_velocity,
        "reward_linup" => uph_cost,
        "reward_quadctrl" => -quad_ctrl_cost,
        "reward_impact" => -quad_impact_cost
    )

    env.timestep += 1

    return observation, reward, terminated, truncated, info
end

function get_observation(env::HumanoidStandupEnv)
    position = vec(env.data.qpos[3:end])  # Exclude x, y
    velocity = vec(env.data.qvel)

    obs_parts = [position, velocity]

    if env.include_cinert_in_observation
        com_inertia = vec(env.data.cinert[2:end, :])  # Exclude worldbody
        push!(obs_parts, com_inertia)
    end

    if env.include_cvel_in_observation
        com_velocity = vec(env.data.cvel[2:end, :])  # Exclude worldbody
        push!(obs_parts, com_velocity)
    end

    if env.include_qfrc_actuator_in_observation
        actuator_forces = vec(env.data.qfrc_actuator[7:end])  # Exclude root (6 DOF)
        push!(obs_parts, actuator_forces)
    end

    if env.include_cfrc_ext_in_observation
        external_contact_forces = vec(env.data.cfrc_ext[2:end, :])  # Exclude worldbody
        push!(obs_parts, external_contact_forces)
    end

    return vec(vcat(obs_parts...))
end

function get_action_bounds(env::HumanoidStandupEnv)
    n_actions = env.model.nu
    return fill(-0.4, n_actions), fill(0.4, n_actions)
end

# Drill interface implementation
function DrillInterface.reset!(env::HumanoidStandupEnv; seed = nothing)
    seed isa Integer && Random.seed!(env.rng, seed)
    reset_env!(env)
    return nothing
end

function DrillInterface.act!(env::HumanoidStandupEnv, action::AbstractVector)
    _, reward, _, _, _ = step_env!(env, action)
    return convert(eltype(env.ctrl_cost_weight), reward)
end

function DrillInterface.observe(env::HumanoidStandupEnv)
    return get_observation(env)
end

function DrillInterface.terminated(env::HumanoidStandupEnv)
    return is_terminated(env)
end

function DrillInterface.truncated(env::HumanoidStandupEnv)
    return is_truncated(env)
end

function DrillInterface.observation_space(env::HumanoidStandupEnv)
    obs_low, obs_high = get_observation_bounds(env)
    T = typeof(env.ctrl_cost_weight)
    return DrillInterface.Box(T.(obs_low), T.(obs_high))
end

function DrillInterface.action_space(env::HumanoidStandupEnv)
    act_low, act_high = get_action_bounds(env)
    T = typeof(env.ctrl_cost_weight)
    return DrillInterface.Box(T.(act_low), T.(act_high))
end

function DrillInterface.get_info(env::HumanoidStandupEnv)
    return Dict{String, Any}(
        "timestep" => env.timestep,
        "max_episode_steps" => env.max_episode_steps
    )
end

function get_observation_bounds(env::HumanoidStandupEnv)
    obs_size = (env.model.nq - 2) + env.model.nv
    if env.include_cinert_in_observation
        obs_size += (env.model.nbody - 1) * 10
    end
    if env.include_cvel_in_observation
        obs_size += (env.model.nbody - 1) * 6
    end
    if env.include_qfrc_actuator_in_observation
        obs_size += env.model.nv - 6
    end
    if env.include_cfrc_ext_in_observation
        obs_size += (env.model.nbody - 1) * 6
    end
    return fill(-Inf, obs_size), fill(Inf, obs_size)
end
