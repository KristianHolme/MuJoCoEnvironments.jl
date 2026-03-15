# ------------------------------------------------------------
# InvertedDoublePendulum-v5 Environment
# ------------------------------------------------------------

"""
    InvertedDoublePendulumEnv{T, R} <: DrillInterface.AbstractEnv

Inverted Double Pendulum environment - 2 pole variation of CartPole.

Matches Gymnasium InvertedDoublePendulum-v5 specification.

# Type Parameters
- `T`: Floating point type for weights and parameters (default: `Float64`)
- `R`: Concrete type of the random number generator (default: Julia's default RNG)

# Observation Space
- Cart position (qpos[1]): 1 dimension
- sin(pole angles): 2 dimensions
- cos(pole angles): 2 dimensions
- Velocities (qvel): 3 dimensions
- Constraint force (qfrc_constraint[1]): 1 dimension
Total: 9 dimensions

# Action Space
- 1 continuous action in [-1, 1], force applied to cart

# Reward
- Alive bonus: +10 for each timestep the pendulum is healthy
- Distance penalty: penalty based on tip position
- Velocity penalty: penalty for moving too fast
"""
mutable struct InvertedDoublePendulumEnv{T, R} <: _AbstractMuJoCoEnv{T}
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    frame_skip::Int
    rng::R

    healthy_reward::T
    reset_noise_scale::T

    function InvertedDoublePendulumEnv(;
            max_episode_steps::Int = 1000,
            frame_skip::Int = 5,
            healthy_reward = 10.0,
            reset_noise_scale = 0.1,
            rng = Random.default_rng(),
            model_path::Union{String, Nothing} = nothing
        )
        T = typeof(healthy_reward)
        R = typeof(rng)
        if model_path !== nothing
            model, data = load_model_from_path(model_path)
        else
            path = get_model_path("inverted_double_pendulum")
            model, data = load_model_from_path(path)
        end

        return new{T, R}(
            model, data, 0, max_episode_steps, frame_skip, rng,
            healthy_reward, reset_noise_scale
        )
    end
end

function is_terminated(env::InvertedDoublePendulumEnv)
    # Terminate if tip y-coordinate <= 1
    # The tip is at site_xpos[0] (first site)
    y_tip = env.data.site_xpos[1, 3]  # y coordinate (index 3)
    return y_tip <= 1.0
end

function reset_env!(env::InvertedDoublePendulumEnv)
    env.timestep = 0
    reset_mujoco_state!(env; noise_scale = env.reset_noise_scale)
    return get_observation(env)
end

function step_env!(env::InvertedDoublePendulumEnv, action::AbstractVector)
    low, high = get_action_bounds(env)
    action = clip_action(action, low, high)

    apply_action!(env, action)
    forward_step!(env, env.frame_skip)

    # Get tip position
    x_tip = env.data.site_xpos[1, 1]  # x coordinate
    y_tip = env.data.site_xpos[1, 3]  # y coordinate

    observation = get_observation(env)
    terminated = is_terminated(env)

    # Calculate reward
    v1, v2 = env.data.qvel[2], env.data.qvel[3]  # Angular velocities of poles
    dist_penalty = 0.01 * x_tip^2 + (y_tip - 2.0)^2
    vel_penalty = 1.0e-3 * v1^2 + 5.0e-3 * v2^2
    alive_bonus = terminated ? zero(typeof(env.healthy_reward)) : env.healthy_reward

    reward = alive_bonus - dist_penalty - vel_penalty
    truncated = is_truncated(env)

    info = Dict{String, Any}(
        "reward_survive" => alive_bonus,
        "distance_penalty" => -dist_penalty,
        "velocity_penalty" => -vel_penalty
    )

    env.timestep += 1

    return observation, reward, terminated, truncated, info
end

function get_observation(env::InvertedDoublePendulumEnv)
    # Cart position
    cart_pos = vec(env.data.qpos[1:1])

    # sin and cos of pole angles
    pole_angles = vec(env.data.qpos[2:3])
    sin_angles = sin.(pole_angles)
    cos_angles = cos.(pole_angles)

    # Velocities (clipped to [-10, 10])
    qvel = vec(clamp.(env.data.qvel, -10.0, 10.0))

    # Constraint force (x component only, clipped)
    qfrc_constraint = vec(clamp.(env.data.qfrc_constraint[1:1], -10.0, 10.0))

    return vec(vcat(cart_pos, sin_angles, cos_angles, qvel, qfrc_constraint))
end

function get_action_bounds(env::InvertedDoublePendulumEnv)
    return [-1.0], [1.0]
end

function get_observation_bounds(env::InvertedDoublePendulumEnv)
    obs_size = 9
    return fill(-Inf, obs_size), fill(Inf, obs_size)
end

# Drill interface implementation
function DrillInterface.reset!(env::InvertedDoublePendulumEnv)
    reset_env!(env)
    return nothing
end

function DrillInterface.act!(env::InvertedDoublePendulumEnv, action::AbstractVector)
    obs, reward, terminated, truncated, info = step_env!(env, action)
    if terminated || truncated
        info["terminal_observation"] = obs
    end
    return convert(eltype(env.healthy_reward), reward)
end

function DrillInterface.observe(env::InvertedDoublePendulumEnv)
    return get_observation(env)
end

function DrillInterface.terminated(env::InvertedDoublePendulumEnv)
    return is_terminated(env)
end

function DrillInterface.truncated(env::InvertedDoublePendulumEnv)
    return is_truncated(env)
end

function DrillInterface.observation_space(env::InvertedDoublePendulumEnv)
    obs_low, obs_high = get_observation_bounds(env)
    T = typeof(env.healthy_reward)
    return DrillInterface.Box(T.(obs_low), T.(obs_high))
end

function DrillInterface.action_space(env::InvertedDoublePendulumEnv)
    act_low, act_high = get_action_bounds(env)
    T = typeof(env.healthy_reward)
    return DrillInterface.Box(T.(act_low), T.(act_high))
end

function DrillInterface.get_info(env::InvertedDoublePendulumEnv)
    return Dict{String, Any}(
        "timestep" => env.timestep,
        "max_episode_steps" => env.max_episode_steps
    )
end
