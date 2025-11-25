# ------------------------------------------------------------
# Pusher-v5 Environment
# ------------------------------------------------------------

"""
    PusherEnv{T, R} <: DRiL.AbstractEnv

Pusher environment - a multi-jointed robot arm.

Matches Gymnasium Pusher-v5 specification.

# Type Parameters
- `T`: Floating point type for weights and parameters (default: `Float64`)
- `R`: Concrete type of the random number generator (default: Julia's default RNG)

# Observation Space
- qpos[:7]: 7 dimensions (arm joint positions)
- qvel[:7]: 7 dimensions (arm joint velocities)
- fingertip position: 3 dimensions
- object position: 3 dimensions
- goal position: 3 dimensions
Total: 23 dimensions

# Action Space
- 7 continuous actions in [-2, 2], torques at joints

# Reward
- Distance reward: negative distance from object to goal
- Near reward: negative distance from fingertip to object
- Control reward: negative squared action norm
"""
mutable struct PusherEnv{T, R} <: _AbstractMuJoCoEnv{T}
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    frame_skip::Int
    rng::R

    reward_near_weight::T
    reward_dist_weight::T
    reward_control_weight::T

    cylinder_pos::Vector{T}
    goal_pos::Vector{T}

    function PusherEnv(;
            max_episode_steps::Int = 100,
            frame_skip::Int = 5,
            reward_near_weight = 0.5,
            reward_dist_weight = 1.0,
            reward_control_weight = 0.1,
            rng = Random.default_rng(),
            model_path::Union{String, Nothing} = nothing
        )
        T = typeof(reward_near_weight)
        R = typeof(rng)
        if model_path !== nothing
            model, data = load_model_from_path(model_path)
        else
            path = get_model_path("pusher_v5")
            model, data = load_model_from_path(path)
        end

        return new{T, R}(
            model, data, 0, max_episode_steps, frame_skip, rng,
            reward_near_weight, reward_dist_weight, reward_control_weight,
            zeros(T, 2), zeros(T, 2)
        )
    end
end

function is_terminated(env::PusherEnv)
    return false  # Pusher never terminates
end

function reset_env!(env::PusherEnv)
    env.timestep = 0

    qpos = copy(env.data.qpos)

    # Goal position is fixed (2D for qpos)
    env.goal_pos = [0.0, 0.0]

    # Sample cylinder position until distance to goal > 0.17
    while true
        env.cylinder_pos = [
            -0.3 + 0.3 * rand(env.rng),  # x: [-0.3, 0]
            -0.2 + 0.4 * rand(env.rng),   # y: [-0.2, 0.2]
        ]
        if norm(env.cylinder_pos - env.goal_pos) > 0.17
            break
        end
    end

    # Set object and goal positions
    qpos[(end - 3):(end - 2)] = env.cylinder_pos
    qpos[(end - 1):end] = env.goal_pos

    # Reset velocities with noise
    qvel = env.data.qvel .+ (-0.005 + 0.01 * rand(env.rng, length(env.data.qvel)))
    qvel[(end - 4):end] .= 0  # Object and goal have no velocity

    MuJoCo.set_state!(env.model, env.data, qpos, qvel)

    return get_observation(env)
end

function step_env!(env::PusherEnv, action::AbstractVector)
    low, high = get_action_bounds(env)
    action = clip_action(action, low, high)

    apply_action!(env, action)
    forward_step!(env, env.frame_skip)

    observation = get_observation(env)

    # Calculate reward
    fingertip_pos = MuJoCo.body(env.model, env.data, MuJoCo.bodyid(env.model, "tips_arm")).xpos
    object_pos = MuJoCo.body(env.model, env.data, MuJoCo.bodyid(env.model, "object")).xpos
    goal_pos = MuJoCo.body(env.model, env.data, MuJoCo.bodyid(env.model, "goal")).xpos

    vec_1 = object_pos - fingertip_pos
    vec_2 = object_pos - goal_pos

    reward_near = -norm(vec_1) * env.reward_near_weight
    reward_dist = -norm(vec_2) * env.reward_dist_weight
    reward_ctrl = -sum(action .^ 2) * env.reward_control_weight
    reward = reward_dist + reward_ctrl + reward_near

    terminated = false
    truncated = is_truncated(env)

    info = Dict{String, Any}(
        "reward_dist" => reward_dist,
        "reward_ctrl" => reward_ctrl,
        "reward_near" => reward_near
    )

    env.timestep += 1

    return observation, reward, terminated, truncated, info
end

function get_observation(env::PusherEnv)
    fingertip_pos = MuJoCo.body(env.model, env.data, MuJoCo.bodyid(env.model, "tips_arm")).xpos
    object_pos = MuJoCo.body(env.model, env.data, MuJoCo.bodyid(env.model, "object")).xpos
    goal_pos = MuJoCo.body(env.model, env.data, MuJoCo.bodyid(env.model, "goal")).xpos

    return vec(
        vcat(
            vec(env.data.qpos[1:7]),
            vec(env.data.qvel[1:7]),
            vec(fingertip_pos),
            vec(object_pos),
            vec(goal_pos)
        )
    )
end

function get_action_bounds(env::PusherEnv)
    n_actions = env.model.nu
    return fill(-2.0, n_actions), fill(2.0, n_actions)
end

function get_observation_bounds(env::PusherEnv)
    obs_size = 23
    return fill(-Inf, obs_size), fill(Inf, obs_size)
end

# DRiL interface implementation
function DRiL.reset!(env::PusherEnv)
    reset_env!(env)
    return nothing
end

function DRiL.act!(env::PusherEnv, action::AbstractVector)
    obs, reward, terminated, truncated, info = step_env!(env, action)
    if terminated || truncated
        info["terminal_observation"] = obs
    end
    return convert(eltype(env.reward_near_weight), reward)
end

function DRiL.observe(env::PusherEnv)
    return get_observation(env)
end

function DRiL.terminated(env::PusherEnv)
    return is_terminated(env)
end

function DRiL.truncated(env::PusherEnv)
    return is_truncated(env)
end

function DRiL.observation_space(env::PusherEnv)
    obs_low, obs_high = get_observation_bounds(env)
    T = typeof(env.reward_near_weight)
    return DRiL.Box(T.(obs_low), T.(obs_high))
end

function DRiL.action_space(env::PusherEnv)
    act_low, act_high = get_action_bounds(env)
    T = typeof(env.reward_near_weight)
    return DRiL.Box(T.(act_low), T.(act_high))
end

function DRiL.get_info(env::PusherEnv)
    return Dict{String, Any}(
        "timestep" => env.timestep,
        "max_episode_steps" => env.max_episode_steps
    )
end
