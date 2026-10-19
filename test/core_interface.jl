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

@testitem "reset! with seed" begin
    using DrillInterface

    env = HopperEnv(max_episode_steps = 100)
    DrillInterface.reset!(env; seed = 123)
    obs1 = copy(DrillInterface.observe(env))

    DrillInterface.reset!(env; seed = 123)
    obs2 = DrillInterface.observe(env)

    @test obs1 ≈ obs2

    DrillInterface.reset!(env; seed = 456)
    @test !(DrillInterface.observe(env) ≈ obs1)
end

@testitem "All environments - interface smoke" begin
    max_steps = 5
    envs = [
        HalfCheetahEnv(max_episode_steps = max_steps),
        HopperEnv(max_episode_steps = max_steps),
        Walker2dEnv(max_episode_steps = max_steps),
        AntEnv(max_episode_steps = max_steps),
        HumanoidEnv(max_episode_steps = max_steps),
        SwimmerEnv(max_episode_steps = max_steps),
        ReacherEnv(max_episode_steps = max_steps),
        PusherEnv(max_episode_steps = max_steps),
        InvertedPendulumEnv(max_episode_steps = max_steps),
        InvertedDoublePendulumEnv(max_episode_steps = max_steps),
        HumanoidStandupEnv(max_episode_steps = max_steps),
    ]

    for env in envs
        obs = reset_env!(env)
        @test obs isa Vector
        @test length(obs) > 0
        @test env.timestep == 0

        low, high = get_action_bounds(env)
        @test low isa Vector
        @test high isa Vector
        @test length(low) == length(high)
        @test all(low .<= high)

        obs, reward, terminated, truncated, info = step_env!(env, zeros(length(low)))
        @test obs isa Vector
        @test reward isa Real
        @test terminated isa Bool
        @test truncated isa Bool
        @test info isa Dict
        @test env.timestep == 1

        obs = get_observation(env)
        @test obs isa Vector
        @test length(obs) > 0

        obs_low, obs_high = get_observation_bounds(env)
        @test length(obs_low) == length(obs_high) == length(get_observation(env))

        @test is_terminated(env) isa Bool
        @test is_truncated(env) isa Bool
    end
end
