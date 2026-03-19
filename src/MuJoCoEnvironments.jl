module MuJoCoEnvironments

using MuJoCo
using Random
using DrillInterface

# Core interface
include("core_interface.jl")
# _AbstractMuJoCoEnv is internal, not exported

# Base functionality
include("utils.jl")
export get_model_path, load_model_from_path
export apply_action!, forward_step!, is_truncated, reset_mujoco_state!
export clip_action, get_ctrl, get_qpos, get_qvel

# Core interface functions
export get_action_bounds, get_observation, get_observation_bounds, is_terminated, reset_env!,
    step_env!
# Random.seed! is exported via Random module, but available for all environments

# Rendering
include("rendering.jl")
export close_render, render

# Environments
include("environments/halfcheetah.jl")
include("environments/hopper.jl")
include("environments/walker2d.jl")
include("environments/ant.jl")
include("environments/humanoid.jl")
include("environments/swimmer.jl")
include("environments/inverted_pendulum.jl")
include("environments/inverted_double_pendulum.jl")
include("environments/reacher.jl")
include("environments/pusher.jl")
include("environments/humanoidstandup.jl")

export AntEnv, HalfCheetahEnv, HopperEnv, HumanoidEnv, SwimmerEnv, Walker2dEnv
export HumanoidStandupEnv, InvertedDoublePendulumEnv, InvertedPendulumEnv, PusherEnv,
    ReacherEnv

end
