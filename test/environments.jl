@testitem "HalfCheetahEnv" begin
    using MuJoCoEnvironments

    env = HalfCheetahEnv(max_episode_steps = 100)
    @test env isa HalfCheetahEnv
    obs = reset_env!(env)
    @test length(obs) == 17  # nq-1 + nv = 9-1 + 9 = 17
    low, high = get_action_bounds(env)
    @test length(low) == 6
end

@testitem "HopperEnv" begin
    using MuJoCoEnvironments

    env = HopperEnv(max_episode_steps = 100)
    @test env isa HopperEnv
    obs = reset_env!(env)
    @test length(obs) == 11  # nq-1 + nv = 6-1 + 6 = 11
    low, high = get_action_bounds(env)
    @test length(low) == 3
end

@testitem "Walker2dEnv" begin
    using MuJoCoEnvironments

    env = Walker2dEnv(max_episode_steps = 100)
    @test env isa Walker2dEnv
    obs = reset_env!(env)
    @test length(obs) == 17  # nq-1 + nv = 9-1 + 9 = 17
    low, high = get_action_bounds(env)
    @test length(low) == 6
end

@testitem "AntEnv" begin
    using MuJoCoEnvironments

    env = AntEnv(max_episode_steps = 100)
    @test env isa AntEnv
    obs = reset_env!(env)
    @test length(obs) == 27  # nq-2 + nv = 15-2 + 14 = 27
    low, high = get_action_bounds(env)
    @test length(low) == 8
end

@testitem "HumanoidEnv" begin
    using MuJoCoEnvironments

    env = HumanoidEnv(max_episode_steps = 100)
    @test env isa HumanoidEnv
    obs = reset_env!(env)
    low, high = get_action_bounds(env)
    @test length(low) == 17
end

@testitem "SwimmerEnv" begin
    using MuJoCoEnvironments

    env = SwimmerEnv(max_episode_steps = 100)
    @test env isa SwimmerEnv
    obs = reset_env!(env)
    @test length(obs) == 8  # nq-2 + nv = 5-2 + 5 = 8
    low, high = get_action_bounds(env)
    @test length(low) == 2
end

@testitem "Type Parameters - Float32" begin
    using MuJoCoEnvironments

    env = HalfCheetahEnv(
        max_episode_steps = 100,
        forward_reward_weight = Float32(1.0),
        ctrl_cost_weight = Float32(0.1)
    )
    @test env.forward_reward_weight isa Float32
    @test env.ctrl_cost_weight isa Float32
end

@testitem "Type Parameters - Custom RNG" begin
    using MuJoCoEnvironments
    using Random

    rng = Random.MersenneTwister(123)
    env = HopperEnv(max_episode_steps = 100, rng = rng)
    @test env.rng isa Random.MersenneTwister
end
