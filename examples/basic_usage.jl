# Basic Usage Example for MuJoCoEnvironments
# This example demonstrates how to use the core MuJoCo environments

using MuJoCoEnvironments
using Random

# Set random seed for reproducibility
Random.seed!(42)

println("="^60)
println("Basic MuJoCo Environments Usage Example")
println("="^60)

# Example 1: HalfCheetah
println("\n1. HalfCheetah Environment")
println("-"^60)

env = HalfCheetahEnv(max_episode_steps = 100)
println("Created HalfCheetah environment")
println("Action space: ", get_action_bounds(env))
println("Observation space size: ", length(get_observation(env)))

# Reset and run a few steps
obs = reset_env!(env)
println("\nInitial observation (first 5 dims): ", obs[1:min(5, length(obs))])

total_reward = 0.0
for step in 1:10
    # Random action
    action = randn(env.model.nu)
    obs, reward, terminated, truncated, info = step_env!(env, action)
    total_reward += reward

    if step <= 3
        println("Step $step: reward = $(round(reward, digits = 3)), terminated = $terminated")
    end
end

println("Total reward after 10 steps: $(round(total_reward, digits = 3))")

# Example 2: Hopper
println("\n2. Hopper Environment")
println("-"^60)

hopper = HopperEnv(max_episode_steps = 100)
println("Created Hopper environment")

obs = reset_env!(hopper)
println("Observation size: ", length(obs))

# Run until termination or truncation
total_reward = 0.0
step = 0
terminated = false
truncated = false

while !terminated && !truncated && step < 50
    action = randn(hopper.model.nu) * 0.1  # Small random actions
    obs, reward, terminated, truncated, info = step_env!(hopper, action)
    total_reward += reward
    step += 1
end

println("Episode ended after $step steps")
println("Terminated: $terminated, Truncated: $truncated")
println("Total reward: $(round(total_reward, digits = 3))")

# Example 3: Walker2d
println("\n3. Walker2d Environment")
println("-"^60)

walker = Walker2dEnv(max_episode_steps = 100)
println("Created Walker2d environment")

obs = reset_env!(walker)
total_reward = 0.0

for step in 1:20
    action = randn(walker.model.nu) * 0.2
    obs, reward, terminated, truncated, info = step_env!(walker, action)
    total_reward += reward

    if terminated
        println("Walker fell down at step $step")
        break
    end
end

println("Total reward: $(round(total_reward, digits = 3))")

# Example 4: Rendering (if visualizer is initialized)
println("\n4. Rendering Example")
println("-"^60)
println("To render an environment, first initialize the visualizer:")
println("  MuJoCo.install_visualiser()")
println("Then call:")
println("  render(env)")
println("This will display the current state in the MuJoCo visualizer")

println("\n" * "="^60)
println("Example completed successfully!")
println("="^60)
