# ------------------------------------------------------------
# InvertedPendulum-v5 Environment
# ------------------------------------------------------------

"""
    InvertedPendulumEnv{T, R} <: DrillInterface.AbstractEnv

Inverted Pendulum environment - MuJoCo version of CartPole.

Matches Gymnasium InvertedPendulum-v5 specification.

# Type Parameters
- `T`: Floating point type for weights and parameters (default: `Float64`)
- `R`: Concrete type of the random number generator (default: Julia's default RNG)

# Observation Space
- Position coordinates (qpos): 2 dimensions (cart position, pole angle)
- Velocity coordinates (qvel): 2 dimensions
Total: 4 dimensions

# Action Space
- 1 continuous action in [-3, 3], force applied to cart

# Reward
- +1 for each timestep the pole is upright (angle < 0.2 radians)
"""
mutable struct InvertedPendulumEnv{T, R} <: _AbstractMuJoCoEnv{T}
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    frame_skip::Int
    rng::R

    reset_noise_scale::T

    function InvertedPendulumEnv(;
            max_episode_steps::Int = 1000,
            frame_skip::Int = 2,
            reset_noise_scale = 0.01,
            rng = Random.default_rng(),
            model_path::Union{String, Nothing} = nothing
        )
        T = typeof(reset_noise_scale)
        R = typeof(rng)
        if model_path !== nothing
            model, data = load_model_from_path(model_path)
        else
            path = get_model_path("inverted_pendulum")
            model, data = load_model_from_path(path)
        end

        return new{T, R}(
            model, data, 0, max_episode_steps, frame_skip, rng,
            reset_noise_scale
        )
    end
end

function is_terminated(env::InvertedPendulumEnv)
    observation = get_observation(env)
    # Terminate if observation is not finite or pole angle > 0.2
    return !all(isfinite, observation) || abs(observation[2]) > 0.2
end

function reset_env!(env::InvertedPendulumEnv)
    env.timestep = 0
    reset_mujoco_state!(env; noise_scale = env.reset_noise_scale)
    return get_observation(env)
end

function step_env!(env::InvertedPendulumEnv, action::AbstractVector)
    low, high = get_action_bounds(env)
    action = clip_action(action, low, high)

    apply_action!(env, action)
    forward_step!(env, env.frame_skip)

    observation = get_observation(env)
    terminated = is_terminated(env)
    reward = terminated ? zero(typeof(env.reset_noise_scale)) : one(typeof(env.reset_noise_scale))
    truncated = is_truncated(env)

    info = Dict{String, Any}(
        "reward_survive" => reward
    )

    env.timestep += 1

    return observation, reward, terminated, truncated, info
end

function get_observation(env::InvertedPendulumEnv)
    return vcat(vec(env.data.qpos), vec(env.data.qvel))
end

function get_action_bounds(env::InvertedPendulumEnv)
    return [-3.0], [3.0]
end

function get_observation_bounds(env::InvertedPendulumEnv)
    obs_size = env.model.nq + env.model.nv
    return fill(-Inf, obs_size), fill(Inf, obs_size)
end

# Drill interface implementation
function DrillInterface.reset!(env::InvertedPendulumEnv; seed = nothing)
    seed isa Integer && Random.seed!(env.rng, seed)
    reset_env!(env)
    return nothing
end

function DrillInterface.act!(env::InvertedPendulumEnv, action::AbstractVector)
    _, reward, _, _, _ = step_env!(env, action)
    return convert(eltype(env.reset_noise_scale), reward)
end

function DrillInterface.observe(env::InvertedPendulumEnv)
    return get_observation(env)
end

function DrillInterface.terminated(env::InvertedPendulumEnv)
    return is_terminated(env)
end

function DrillInterface.truncated(env::InvertedPendulumEnv)
    return is_truncated(env)
end

function DrillInterface.observation_space(env::InvertedPendulumEnv)
    obs_low, obs_high = get_observation_bounds(env)
    T = typeof(env.reset_noise_scale)
    return DrillInterface.Box(T.(obs_low), T.(obs_high))
end

function DrillInterface.action_space(env::InvertedPendulumEnv)
    act_low, act_high = get_action_bounds(env)
    T = typeof(env.reset_noise_scale)
    return DrillInterface.Box(T.(act_low), T.(act_high))
end

function DrillInterface.get_info(env::InvertedPendulumEnv)
    return Dict{String, Any}(
        "timestep" => env.timestep,
        "max_episode_steps" => env.max_episode_steps
    )
end
