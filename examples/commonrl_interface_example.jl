# CommonRLInterface Example
# This example shows how to use MuJoCo environments with CommonRLInterface.jl

using MuJoCoEnvironments
using CommonRLInterface
using Random

const CRL = CommonRLInterface

println("="^60)
println("CommonRLInterface Example")
println("="^60)

# Create a HalfCheetah environment with CommonRLInterface
println("\nCreating HalfCheetah with CommonRLInterface...")
core_env = HalfCheetahEnv(max_episode_steps = 200)
env = convert(CommonRLMuJoCoEnv, core_env)

println("Environment type: ", typeof(env))
println("Action space: ", CRL.actions(env))
println("Observation space (first 3 intervals): ", CRL.observations(env)[1:3])

# Reset the environment
println("\nResetting environment...")
CRL.reset!(env)
obs = CRL.observe(env)
println("Initial observation size: ", length(obs))
println("First 5 observation values: ", obs[1:min(5, length(obs))])

# Run an episode
println("\nRunning episode...")
total_reward = 0.0
episode_length = 0

for step in 1:100
    # Sample random action
    action_space = CRL.actions(env)
    action = [
        rand() * (interval.right - interval.left) + interval.left
            for interval in action_space
    ]

    # Take a step
    CRL.act!(env, action)
    obs = CRL.observe(env)

    # Get reward (from the cached last reward)
    reward = env.last_reward
    total_reward += reward
    episode_length += 1

    # Check if episode ended
    if CRL.terminated(env)
        println("Episode ended at step $episode_length")
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

# Example with state manipulation
println("\n" * "="^60)
println("State manipulation example...")
println("="^60)

CRL.reset!(env)
initial_state = CRL.state(env)
println("Initial state size: ", length(initial_state))

# Take a few steps
for _ in 1:5
    action = zeros(length(CRL.actions(env)))
    CRL.act!(env, action)
end

modified_state = CRL.state(env)
println("State after 5 steps (different from initial): ", !(initial_state ≈ modified_state))

# Restore initial state
CRL.setstate!(env, initial_state)
restored_state = CRL.state(env)
println("State restored successfully: ", initial_state ≈ restored_state)

# Example with Walker2d
println("\n" * "="^60)
println("Walker2d example...")
println("="^60)

walker_core = Walker2dEnv(max_episode_steps = 100)
walker = CommonRLMuJoCoEnv(walker_core)

println("Created Walker2d environment")
CRL.reset!(walker)

total_reward = 0.0
for step in 1:30
    action_space = CRL.actions(walker)
    action = [randn() * 0.2 for _ in action_space]

    CRL.act!(walker, action)
    reward = walker.last_reward
    total_reward += reward

    if CRL.terminated(walker)
        println("Walker fell down at step $step")
        break
    end
end

println("Total reward: $(round(total_reward, digits = 3))")

# Cloning example
println("\n" * "="^60)
println("Environment cloning example...")
println("="^60)

CRL.reset!(env)
env_clone = CRL.clone(env)

println("Original environment observation: ", CRL.observe(env)[1:3])
println("Cloned environment observation: ", CRL.observe(env_clone)[1:3])
println("Observations match: ", CRL.observe(env) ≈ CRL.observe(env_clone))

# Take actions in original env only
for _ in 1:5
    action = zeros(length(CRL.actions(env)))
    CRL.act!(env, action)
end

println("\nAfter 5 steps in original:")
println("Original observation: ", CRL.observe(env)[1:3])
println("Clone observation (unchanged): ", CRL.observe(env_clone)[1:3])
println("Observations differ: ", !(CRL.observe(env) ≈ CRL.observe(env_clone)))

println("\n" * "="^60)
println("CommonRLInterface example completed!")
println("="^60)
