# ------------------------------------------------------------
# HalfCheetah-v4 Environment
# ------------------------------------------------------------

"""
    HalfCheetahEnv <: AbstractMuJoCoEnv

Half-Cheetah environment - a 2D robot that must learn to run forward.

The HalfCheetah is a 2-legged robot that must learn to run forward as fast as
possible. The robot has 6 actuated joints (3 per leg) and 9 degrees of freedom
(including the x and z position and rotation of the torso).

# Observation Space
- Position coordinates (qpos): 9 dimensions (excluding x-position)
- Velocity coordinates (qvel): 9 dimensions
Total: 17 dimensions

# Action Space
- 6 continuous actions in [-1, 1], one for each joint

# Reward
- Forward velocity reward: rewards moving forward
- Control cost: small penalty for using large control inputs
"""
mutable struct HalfCheetahEnv <: AbstractMuJoCoEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    timestep::Int
    max_episode_steps::Int
    rng::Random.AbstractRNG

    # Environment-specific parameters
    forward_reward_weight::Float64
    ctrl_cost_weight::Float64
    reset_noise_scale::Float64

    # For computing forward velocity
    prev_x_pos::Float64

    function HalfCheetahEnv(;
            max_episode_steps::Int = 1000,
            forward_reward_weight::Float64 = 1.0,
            ctrl_cost_weight::Float64 = 0.1,
            reset_noise_scale::Float64 = 0.1,
            rng::Random.AbstractRNG = Random.Xoshiro(),
            model_path::Union{String, Nothing} = nothing
        )
        # Load the model - either from provided path or use built-in
        if model_path !== nothing
            model, data = load_model_from_path(model_path)
        else
            # Try to find the built-in half_cheetah model
            try
                path = get_model_path("half_cheetah")
                model, data = load_model_from_path(path)
            catch
                # If that fails, use sample model as fallback
                @warn "Could not find half_cheetah model, using sample model"
                model, data = MuJoCo.sample_model_and_data()
            end
        end

        env = new(
            model, data, 0, max_episode_steps, rng,
            forward_reward_weight, ctrl_cost_weight, reset_noise_scale,
            0.0
        )

        return env
    end
end

"""
    reset_env!(env::HalfCheetahEnv) -> observation

Reset the HalfCheetah environment to its initial state.
"""
function reset_env!(env::HalfCheetahEnv)
    # Reset timestep
    env.timestep = 0

    # Reset MuJoCo state with noise
    reset_mujoco_state!(env; noise_scale = env.reset_noise_scale)

    # Store initial x position for velocity computation
    env.prev_x_pos = env.data.qpos[1]

    # Return initial observation
    return get_observation(env)
end

"""
    step_env!(env::HalfCheetahEnv, action) -> (observation, reward, terminated, truncated, info)

Take a step in the HalfCheetah environment.
"""
function step_env!(env::HalfCheetahEnv, action::AbstractVector)
    # Get action bounds and clip
    low, high = get_action_bounds(env)
    action = clip_action(action, low, high)

    # Store current x position
    x_pos_before = env.data.qpos[1]

    # Apply action and step simulation
    apply_action!(env, action)
    forward_step!(env)

    # Get new x position and compute forward velocity
    x_pos_after = env.data.qpos[1]
    x_velocity = (x_pos_after - x_pos_before) / env.model.opt.timestep

    # Compute reward
    forward_reward = env.forward_reward_weight * x_velocity
    ctrl_cost = env.ctrl_cost_weight * sum(action .^ 2)
    reward = forward_reward - ctrl_cost

    # Update timestep
    env.timestep += 1

    # Check termination and truncation
    terminated = false  # HalfCheetah doesn't have termination conditions
    truncated = is_truncated(env)

    # Get observation
    observation = get_observation(env)

    # Build info dict
    info = Dict{String, Any}(
        "x_position" => x_pos_after,
        "x_velocity" => x_velocity,
        "reward_forward" => forward_reward,
        "reward_ctrl" => -ctrl_cost
    )

    # Update prev_x_pos for next step
    env.prev_x_pos = x_pos_after

    return observation, reward, terminated, truncated, info
end

"""
    get_observation(env::HalfCheetahEnv) -> Vector

Get the current observation from the HalfCheetah environment.

The observation consists of:
- qpos[2:end]: position coordinates (excluding x position)
- qvel: all velocity coordinates
"""
function get_observation(env::HalfCheetahEnv)
    # qpos without x position (qpos[1])
    qpos = env.data.qpos[2:end]
    qvel = env.data.qvel

    return vcat(qpos, qvel)
end

"""
    get_action_bounds(env::HalfCheetahEnv) -> (low, high)

Get the action bounds for HalfCheetah.
"""
function get_action_bounds(env::HalfCheetahEnv)
    n_actions = env.model.nu
    low = fill(-1.0, n_actions)
    high = fill(1.0, n_actions)
    return low, high
end

"""
    get_observation_bounds(env::HalfCheetahEnv) -> (low, high)

Get the observation bounds for HalfCheetah.
"""
function get_observation_bounds(env::HalfCheetahEnv)
    # Observation size: nq - 1 (no x pos) + nv
    obs_size = (env.model.nq - 1) + env.model.nv

    # Use large bounds since we don't have strict limits
    low = fill(-Inf, obs_size)
    high = fill(Inf, obs_size)

    return low, high
end
