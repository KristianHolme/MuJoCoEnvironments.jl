# ------------------------------------------------------------
# Core Internal Interface
# ------------------------------------------------------------
"""
    AbstractMuJoCoEnv

Abstract base type for all MuJoCo environments.

This minimal interface is intentionally separate from specific RL framework
interfaces (like DRiL, CommonRLInterface, POMDPs, etc.) to maintain flexibility
and avoid coupling.
"""
abstract type AbstractMuJoCoEnv end

"""
    reset_env!(env::AbstractMuJoCoEnv) -> observation

Reset the environment to its initial state and return the initial observation.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment to reset

# Returns
- `observation`: Initial observation after reset
"""
function reset_env! end

"""
    step_env!(env::AbstractMuJoCoEnv, action) -> (observation, reward, terminated, truncated, info)

Take a step in the environment by applying the given action.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment to step
- `action`: Action to apply (typically a vector of floats)

# Returns
- `observation`: New observation after taking the action
- `reward`: Reward obtained from the action
- `terminated`: Whether the episode has terminated (reached terminal state)
- `truncated`: Whether the episode was truncated (e.g., time limit reached)
- `info`: Dictionary with additional information
"""
function step_env! end

"""
    get_observation(env::AbstractMuJoCoEnv) -> observation

Get the current observation from the environment.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `observation`: Current observation
"""
function get_observation end

"""
    get_action_bounds(env::AbstractMuJoCoEnv) -> (low, high)

Get the lower and upper bounds for the action space.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `low`: Vector of lower bounds for each action dimension
- `high`: Vector of upper bounds for each action dimension
"""
function get_action_bounds end

"""
    get_observation_bounds(env::AbstractMuJoCoEnv) -> (low, high)

Get the lower and upper bounds for the observation space.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `low`: Vector of lower bounds for each observation dimension
- `high`: Vector of upper bounds for each observation dimension
"""
function get_observation_bounds end

