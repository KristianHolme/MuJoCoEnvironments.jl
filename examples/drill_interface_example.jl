# Drill Interface Example
# This example shows how to use MuJoCo environments with the Drill.jl interface

using MuJoCoEnvironments
using Drill
using Random

println("="^60)
println("Drill Interface Example")
println("="^60)

# Create a HalfCheetah environment
println("\nCreating HalfCheetah...")
env = HalfCheetahEnv(max_episode_steps = 200)

println("Environment type: ", typeof(env))
println("Observation space: ", observation_space(env))
println("Action space: ", action_space(env))

# Reset the environment
println("\nResetting environment...")
reset!(env)
obs = observe(env)
println("Initial observation size: ", size(obs))
println("First 5 observation values: ", obs[1:min(5, length(obs))])

# Run an episode
println("\nRunning episode...")
total_reward = 0.0
episode_length = 0

for step in 1:100
    # Sample random action from action space
    action = rand(env.rng, action_space(env))

    # Take a step
    reward = act!(env, action)
    obs = observe(env)

    total_reward += reward
    episode_length += 1

    # Check if episode ended
    if terminated(env) || truncated(env)
        println("Episode ended at step $episode_length")
        println("Terminated: $(terminated(env)), Truncated: $(truncated(env))")
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

reset!(env; seed = 123)
obs1 = copy(observe(env))

reset!(env; seed = 123)
obs2 = observe(env)

println("Observations match after same seed: ", obs1 ≈ obs2)

# Create another environment the same way
println("\n" * "="^60)
println("Creating Hopper environment...")
println("="^60)

dril_hopper = HopperEnv(max_episode_steps = 150)

println("Created Hopper environment")
println("Observation space: ", observation_space(dril_hopper))
println("Action space: ", action_space(dril_hopper))

# Run a short episode
reset!(dril_hopper)
for step in 1:10
    action = randn(length(action_space(dril_hopper).low))
    reward = act!(dril_hopper, action)

    if terminated(dril_hopper)
        println("Hopper fell down at step $step")
        break
    end
end

println("\n" * "="^60)
println("Drill interface example completed!")
println("="^60)
