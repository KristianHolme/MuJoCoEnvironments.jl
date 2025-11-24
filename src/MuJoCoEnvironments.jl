module MuJoCoEnvironments

using MuJoCo
using Random

# Core interface
include("core_interface.jl")
export AbstractMuJoCoEnv

# Base functionality
include("mujoco_env_base.jl")
export MuJoCoEnvBase, load_model_from_path, get_model_path
export apply_action!, forward_step!, reset_mujoco_state!, is_truncated
export get_qpos, get_qvel, get_ctrl, clip_action

# Core interface functions
export reset_env!, step_env!, get_observation, get_action_bounds, get_observation_bounds

# Rendering
include("rendering.jl")
export render, close_render

# Environments
include("environments/halfcheetah.jl")
include("environments/hopper.jl")
include("environments/walker2d.jl")
include("environments/ant.jl")
include("environments/humanoid.jl")
include("environments/swimmer.jl")

export HalfCheetahEnv, HopperEnv, Walker2dEnv, AntEnv, HumanoidEnv, SwimmerEnv

end
