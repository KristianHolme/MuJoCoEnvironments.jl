# All Environments Showcase
# This example demonstrates all available MuJoCo environments

using MuJoCoEnvironments
using Random

Random.seed!(42)

println("="^60)
println("MuJoCo Environments Showcase")
println("="^60)

# List of all environments
environments = [
    ("HalfCheetah", HalfCheetahEnv),
    ("Hopper", HopperEnv),
    ("Walker2d", Walker2dEnv),
    ("Ant", AntEnv),
    ("Humanoid", HumanoidEnv),
    ("Swimmer", SwimmerEnv),
]

results = Dict{String, Any}()

for (name, EnvType) in environments
    println("\n" * "="^60)
    println("Testing: $name")
    println("="^60)

    try
        # Create environment
        env = EnvType(max_episode_steps = 100)
        println("✓ Environment created successfully")

        # Get space information
        obs_low, obs_high = get_observation_bounds(env)
        act_low, act_high = get_action_bounds(env)

        println("  Observation space: $(length(obs_low)) dimensions")
        println("  Action space: $(length(act_low)) dimensions")
        println("  Action range: [$(act_low[1]), $(act_high[1])]")

        # Reset
        obs = reset_env!(env)
        println("✓ Reset successful, observation size: $(length(obs))")

        # Run episode
        total_reward = 0.0
        max_steps = 50

        for step in 1:max_steps
            # Random action (scaled down for stability)
            action = randn(length(act_low)) * 0.3

            obs, reward, terminated, truncated, info = step_env!(env, action)
            total_reward += reward

            if terminated || truncated
                println("  Episode ended at step $step")
                println("    Terminated: $terminated")
                println("    Truncated: $truncated")
                break
            end
        end

        println("✓ Episode completed")
        println("  Total reward: $(round(total_reward, digits = 3))")
        println("  Average reward: $(round(total_reward / min(50, env.timestep), digits = 3))")

        # Store results
        results[name] = Dict(
            "success" => true,
            "obs_dim" => length(obs_low),
            "action_dim" => length(act_low),
            "total_reward" => total_reward,
            "steps" => env.timestep
        )

    catch e
        println("✗ Error: $e")
        results[name] = Dict("success" => false, "error" => string(e))
    end
end

# Summary
println("\n" * "="^60)
println("Summary")
println("="^60)

successful = sum(r["success"] for r in values(results))
println("Successful environments: $successful / $(length(environments))")

println("\nEnvironment Details:")
println("-"^60)
println(rpad("Environment", 15), " | ", rpad("Obs Dim", 10), " | ", rpad("Action Dim", 12), " | ", "Status")
println("-"^60)

for (name, _) in environments
    if haskey(results, name) && results[name]["success"]
        obs_dim = results[name]["obs_dim"]
        act_dim = results[name]["action_dim"]
        status = "✓"
    else
        obs_dim = "-"
        act_dim = "-"
        status = "✗"
    end
    println(rpad(name, 15), " | ", rpad(string(obs_dim), 10), " | ", rpad(string(act_dim), 12), " | ", status)
end

println("\n" * "="^60)
println("Showcase completed!")
println("="^60)
println("\nNote: Some environments may show warnings if model files are not found.")
println("This is expected and the environments will fall back to sample models.")
