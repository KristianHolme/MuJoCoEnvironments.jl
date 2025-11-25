@testitem "reset_env!" begin

    env = HalfCheetahEnv(max_episode_steps = 100)
    obs = reset_env!(env)
    @test obs isa Vector
    @test length(obs) > 0
    @test env.timestep == 0
end

@testitem "step_env!" begin

    env = HalfCheetahEnv(max_episode_steps = 100)
    reset_env!(env)
    low, high = get_action_bounds(env)
    action = zeros(length(low))
    obs, reward, terminated, truncated, info = step_env!(env, action)
    @test obs isa Vector
    @test reward isa Real
    @test terminated isa Bool
    @test truncated isa Bool
    @test info isa Dict
    @test env.timestep == 1
end

@testitem "is_terminated" begin

    # Environments without termination
    env_hc = HalfCheetahEnv(max_episode_steps = 100)
    reset_env!(env_hc)
    @test !is_terminated(env_hc)

    env_swim = SwimmerEnv(max_episode_steps = 100)
    reset_env!(env_swim)
    @test !is_terminated(env_swim)

    # Environments with termination (should check after unhealthy state)
    env_ant = AntEnv(max_episode_steps = 100)
    reset_env!(env_ant)
    # Initially should be healthy
    @test !is_terminated(env_ant)
end

@testitem "get_observation" begin

    env = HopperEnv(max_episode_steps = 100)
    reset_env!(env)
    obs = get_observation(env)
    @test obs isa Vector
    @test length(obs) > 0
end

@testitem "get_action_bounds" begin

    env = Walker2dEnv(max_episode_steps = 100)
    low, high = get_action_bounds(env)
    @test low isa Vector
    @test high isa Vector
    @test length(low) == length(high)
    @test all(low .<= high)
end

@testitem "get_observation_bounds" begin

    env = AntEnv(max_episode_steps = 100)
    low, high = get_observation_bounds(env)
    @test low isa Vector
    @test high isa Vector
    @test length(low) == length(high)
end

@testitem "is_truncated" begin

    env = HalfCheetahEnv(max_episode_steps = 5)
    reset_env!(env)
    @test !is_truncated(env)
    low, high = get_action_bounds(env)
    for _ in 1:5
        action = zeros(length(low))
        step_env!(env, action)
    end
    @test is_truncated(env)
end

@testitem "Random.seed!" begin
    using Random

    env = HopperEnv(max_episode_steps = 100)
    Random.seed!(env, 123)
    obs1 = reset_env!(env)

    Random.seed!(env, 123)
    obs2 = reset_env!(env)

    @test obs1 ≈ obs2
end
