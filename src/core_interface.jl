# ------------------------------------------------------------
# Core Internal Interface
# ------------------------------------------------------------
"""
    _AbstractMuJoCoEnv{T} <: DrillInterface.AbstractEnv

Internal abstract base type for MuJoCo environments.

# Type Parameters
- `T`: Floating point type used for weights, parameters, and observations (e.g., `Float32`, `Float64`)

This is an internal interface used for shared functionality. All MuJoCo environments
inherit from this type and thus implement DrillInterface.AbstractEnv.
"""
abstract type _AbstractMuJoCoEnv{T} <: DrillInterface.AbstractEnv end

"""
    reset_env!(env::_AbstractMuJoCoEnv{T}) -> observation

Reset the environment to its initial state and return the initial observation.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment to reset

# Returns
- `observation`: Initial observation after reset
"""
function reset_env! end

"""
    step_env!(env::_AbstractMuJoCoEnv{T}, action) -> (observation, reward, terminated, truncated, info)

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
    get_observation(env::_AbstractMuJoCoEnv{T}) -> observation

Get the current observation from the environment.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `observation`: Current observation
"""
function get_observation end

"""
    get_action_bounds(env::_AbstractMuJoCoEnv{T}) -> (low, high)

Get the lower and upper bounds for the action space.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `low`: Vector of lower bounds for each action dimension
- `high`: Vector of upper bounds for each action dimension
"""
function get_action_bounds end

"""
    get_observation_bounds(env::_AbstractMuJoCoEnv{T}) -> (low, high)

Get the lower and upper bounds for the observation space.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `low`: Vector of lower bounds for each observation dimension
- `high`: Vector of upper bounds for each observation dimension
"""
function get_observation_bounds end

"""
    is_terminated(env::_AbstractMuJoCoEnv{T}) -> Bool

Check if the environment episode has terminated (reached a terminal state).

This function queries the current termination state of the environment.
Termination conditions are environment-specific (e.g., falling down, going out of bounds).

# Arguments
- `env::AbstractMuJoCoEnv`: The environment

# Returns
- `Bool`: `true` if the episode has terminated, `false` otherwise

# Note
This function should be called after `step_env!` to check the termination state
from the most recent step.
"""
function is_terminated end
