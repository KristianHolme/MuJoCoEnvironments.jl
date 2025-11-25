# ------------------------------------------------------------
# Reacher-v5 Environment
# ------------------------------------------------------------

"""
    ReacherEnv{T, R} <: DRiL.AbstractEnv

Reacher environment - a two-jointed robot arm.

Matches Gymnasium Reacher-v5 specification.

# Type Parameters
- `T`: Floating point type for weights and parameters (default: `Float64`)
- `R`: Concrete type of the random number generator (default: Julia's default RNG)

# Observation Space
- cos(qpos): 2 dimensions (cosine of arm angles)
- sin(qpos): 2 dimensions (sine of arm angles)
- qpos[2:]: 2 dimensions (target coordinates)
- qvel[:2]: 2 dimensions (arm angular velocities)
- fingertip - target: 2 dimensions (x, y difference)
Total: 10 dimensions

# Action Space
- 2 continuous actions in [-1, 1], torques at hinge joints

# Reward
- Distance reward: negative distance from fingertip to target
- Control reward: negative squared action norm
"""
mutable struct ReacherEnv{T, R} <: _AbstractMuJoCoEnv{T}
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    frame_skip::Int
    rng::R

    reward_dist_weight::T
    reward_control_weight::T

    goal::Vector{T}

    function ReacherEnv(;
            max_episode_steps::Int = 50,
            frame_skip::Int = 2,
            reward_dist_weight = 1.0,
            reward_control_weight = 0.1,
            rng = Random.default_rng(),
            model_path::Union{String, Nothing} = nothing
        )
        T = typeof(reward_dist_weight)
        R = typeof(rng)
        if model_path !== nothing
            model, data = load_model_from_path(model_path)
        else
            path = get_model_path("reacher")
            model, data = load_model_from_path(path)
        end

        return new{T, R}(
            model, data, 0, max_episode_steps, frame_skip, rng,
            reward_dist_weight, reward_control_weight,
            zeros(T, 2)
        )
    end
end

function is_terminated(env::ReacherEnv)
    return false  # Reacher never terminates
end

function reset_env!(env::ReacherEnv)
    env.timestep = 0

    # Reset arm position with noise
    noise_low = -0.1
    noise_high = 0.1
    qpos = env.data.qpos .+ (noise_low .+ (noise_high - noise_low) .* rand(env.rng, length(env.data.qpos)))

    # Sample goal position (uniform in circle of radius 0.2)
    while true
        env.goal = (-0.2 .+ 0.4 .* rand(env.rng, 2))
        if norm(env.goal) < 0.2
            break
        end
    end

    # Set target position
    qpos[(end - 1):end] = env.goal

    # Reset velocities with noise
    qvel = env.data.qvel .+ (-0.005 .+ 0.01 .* rand(env.rng, length(env.data.qvel)))
    qvel[(end - 1):end] .= 0  # Target has no velocity

    MuJoCo.set_state!(env.model, env.data, qpos, qvel)

    return get_observation(env)
end

function step_env!(env::ReacherEnv, action::AbstractVector)
    low, high = get_action_bounds(env)
    action = clip_action(action, low, high)

    apply_action!(env, action)
    forward_step!(env, env.frame_skip)

    observation = get_observation(env)

    # Calculate reward
    fingertip_pos = MuJoCo.body(env.model, env.data, MuJoCo.bodyid(env.model, "fingertip")).xpos
    target_pos = MuJoCo.body(env.model, env.data, MuJoCo.bodyid(env.model, "target")).xpos
    vec = fingertip_pos - target_pos

    reward_dist = -norm(vec) * env.reward_dist_weight
    reward_ctrl = -sum(action .^ 2) * env.reward_control_weight
    reward = reward_dist + reward_ctrl

    terminated = false
    truncated = is_truncated(env)

    info = Dict{String, Any}(
        "reward_dist" => reward_dist,
        "reward_ctrl" => reward_ctrl
    )

    env.timestep += 1

    return observation, reward, terminated, truncated, info
end

function get_observation(env::ReacherEnv)
    theta = env.data.qpos[1:2]

    fingertip_pos = MuJoCo.body(env.model, env.data, MuJoCo.bodyid(env.model, "fingertip")).xpos
    target_pos = MuJoCo.body(env.model, env.data, MuJoCo.bodyid(env.model, "target")).xpos
    diff = fingertip_pos[1:2] - target_pos[1:2]

    return vec(
        vcat(
            cos.(vec(theta)),
            sin.(vec(theta)),
            vec(env.data.qpos[3:4]),  # Target position
            vec(env.data.qvel[1:2]),  # Arm velocities
            vec(diff)  # fingertip - target (x, y)
        )
    )
end

function get_action_bounds(env::ReacherEnv)
    n_actions = env.model.nu
    return fill(-1.0, n_actions), fill(1.0, n_actions)
end

function get_observation_bounds(env::ReacherEnv)
    obs_size = 10
    return fill(-Inf, obs_size), fill(Inf, obs_size)
end

# DRiL interface implementation
function DRiL.reset!(env::ReacherEnv)
    reset_env!(env)
    return nothing
end

function DRiL.act!(env::ReacherEnv, action::AbstractVector)
    obs, reward, terminated, truncated, info = step_env!(env, action)
    if terminated || truncated
        info["terminal_observation"] = obs
    end
    return convert(eltype(env.reward_dist_weight), reward)
end

function DRiL.observe(env::ReacherEnv)
    return get_observation(env)
end

function DRiL.terminated(env::ReacherEnv)
    return is_terminated(env)
end

function DRiL.truncated(env::ReacherEnv)
    return is_truncated(env)
end

function DRiL.observation_space(env::ReacherEnv)
    obs_low, obs_high = get_observation_bounds(env)
    T = typeof(env.reward_dist_weight)
    return DRiL.Box(T.(obs_low), T.(obs_high))
end

function DRiL.action_space(env::ReacherEnv)
    act_low, act_high = get_action_bounds(env)
    T = typeof(env.reward_dist_weight)
    return DRiL.Box(T.(act_low), T.(act_high))
end

function DRiL.get_info(env::ReacherEnv)
    return Dict{String, Any}(
        "timestep" => env.timestep,
        "max_episode_steps" => env.max_episode_steps
    )
end
