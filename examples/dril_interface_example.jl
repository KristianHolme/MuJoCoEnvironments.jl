# Drill Interface Example
# This example shows how to use MuJoCo environments with the Drill.jl interface

using MuJoCoEnvironments
using Drill  # This will load the Drill extension automatically
using Random

println("="^60)
println("Drill Interface Example")
println("="^60)

# Create a HalfCheetah environment with Drill interface
println("\nCreating HalfCheetah with Drill interface...")
core_env = HalfCheetahEnv(max_episode_steps = 200)
env = convert(Drill.AbstractEnv, core_env)

println("Environment type: ", typeof(env))
println("Observation space: ", DrillInterface.observation_space(env))
println("Action space: ", DrillInterface.action_space(env))

# Reset the environment
println("\nResetting environment...")
DrillInterface.reset!(env)
obs = DrillInterface.observe(env)
println("Initial observation size: ", size(obs))
println("First 5 observation values: ", obs[1:min(5, length(obs))])

# Run an episode
println("\nRunning episode...")
total_reward = 0.0
episode_length = 0

for step in 1:100
    # Sample random action from action space
    action = rand(env.core_env.rng, DrillInterface.action_space(env))

    # Take a step
    reward = DrillInterface.act!(env, action)
    obs = DrillInterface.observe(env)

    total_reward += reward
    episode_length += 1

    # Check if episode ended
    if DrillInterface.terminated(env) || DrillInterface.truncated(env)
        println("Episode ended at step $episode_length")
        println("Terminated: $(DrillInterface.terminated(env)), Truncated: $(DrillInterface.truncated(env))")
        break
    end

    # Print progress every 20 steps
    if step % 20 == 0
        println("Step $step: reward = $(round(reward, digits = 3)), total = $(round(total_reward, digits = 3))")
    end
end

println("\nEpisode Summary:")
println("  Length: $episode_length steps")
println("  Total reward: $(round(total_reward, digits = 3))")
println("  Average reward: $(round(total_reward / episode_length, digits = 3))")

# Example with seeding
println("\n" * "="^60)
println("Testing reproducibility with seeding...")
println("="^60)

Random.seed!(env, 123)
DrillInterface.reset!(env)
obs1 = DrillInterface.observe(env)

Random.seed!(env, 123)
DrillInterface.reset!(env)
obs2 = DrillInterface.observe(env)

println("Observations match after same seed: ", obs1 ≈ obs2)

# You can also create environments directly from the core types
println("\n" * "="^60)
println("Creating environments with wrapper pattern...")
println("="^60)

# Create core environment
core_hopper = HopperEnv(max_episode_steps = 150)

# Wrap it for Drill
dril_hopper = convert(Drill.AbstractEnv, core_hopper)

println("Created wrapped Hopper environment")
println("Observation space: ", DrillInterface.observation_space(dril_hopper))
println("Action space: ", DrillInterface.action_space(dril_hopper))

# Run a short episode
DrillInterface.reset!(dril_hopper)
for step in 1:10
    action = randn(length(DrillInterface.action_space(dril_hopper).low))
    reward = DrillInterface.act!(dril_hopper, action)

    if DrillInterface.terminated(dril_hopper)
        println("Hopper fell down at step $step")
        break
    end
end

println("\n" * "="^60)
println("Drill interface example completed!")
println("="^60)
