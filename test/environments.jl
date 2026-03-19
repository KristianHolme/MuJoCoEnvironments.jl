@testitem "HalfCheetahEnv" begin

    env = HalfCheetahEnv(max_episode_steps = 100)
    @test env isa HalfCheetahEnv
    obs = reset_env!(env)
    @test length(obs) == 17  # nq-1 + nv = 9-1 + 9 = 17
    low, high = get_action_bounds(env)
    @test length(low) == 6
end

@testitem "HopperEnv" begin
    env = HopperEnv(max_episode_steps = 100)
    @test env isa HopperEnv
    obs = reset_env!(env)
    @test length(obs) == 11  # nq-1 + nv = 6-1 + 6 = 11
    low, high = get_action_bounds(env)
    @test length(low) == 3
end

@testitem "Walker2dEnv" begin

    env = Walker2dEnv(max_episode_steps = 100)
    @test env isa Walker2dEnv
    obs = reset_env!(env)
    @test length(obs) == 17  # nq-1 + nv = 9-1 + 9 = 17
    low, high = get_action_bounds(env)
    @test length(low) == 6
end

@testitem "AntEnv" begin

    env = AntEnv(max_episode_steps = 100)
    @test env isa AntEnv
    obs = reset_env!(env)
    @test length(obs) == 105
    low, high = get_action_bounds(env)
    @test length(low) == 8
end

@testitem "HumanoidEnv" begin

    env = HumanoidEnv(max_episode_steps = 100)
    @test env isa HumanoidEnv
    obs = reset_env!(env)
    obs_low, obs_high = get_observation_bounds(env)
    @test length(obs) == length(obs_low)
    low, high = get_action_bounds(env)
    @test length(low) == 17
end

@testitem "SwimmerEnv" begin

    env = SwimmerEnv(max_episode_steps = 100)
    @test env isa SwimmerEnv
    obs = reset_env!(env)
    @test length(obs) == 8  # nq-2 + nv = 5-2 + 5 = 8
    low, high = get_action_bounds(env)
    @test length(low) == 2
end

@testitem "ReacherEnv" begin
    env = ReacherEnv(max_episode_steps = 100)
    @test env isa ReacherEnv
    obs = reset_env!(env)
    @test length(obs) == 10
    low, high = get_action_bounds(env)
    @test length(low) == 2
end

@testitem "PusherEnv" begin
    env = PusherEnv(max_episode_steps = 100)
    @test env isa PusherEnv
    obs = reset_env!(env)
    @test length(obs) == 23
    low, high = get_action_bounds(env)
    @test length(low) == 7
end

@testitem "InvertedPendulumEnv" begin
    env = InvertedPendulumEnv(max_episode_steps = 100)
    @test env isa InvertedPendulumEnv
    obs = reset_env!(env)
    @test length(obs) == 4
    low, high = get_action_bounds(env)
    @test length(low) == 1
    @test low[1] == -3.0 && high[1] == 3.0
end

@testitem "InvertedDoublePendulumEnv" begin
    env = InvertedDoublePendulumEnv(max_episode_steps = 100)
    @test env isa InvertedDoublePendulumEnv
    obs = reset_env!(env)
    @test length(obs) == 9
    low, high = get_action_bounds(env)
    @test length(low) == 1
end

@testitem "HumanoidStandupEnv" begin
    env = HumanoidStandupEnv(max_episode_steps = 100)
    @test env isa HumanoidStandupEnv
    obs = reset_env!(env)
    obs_low, obs_high = get_observation_bounds(env)
    @test length(obs) == length(obs_low)
    low, high = get_action_bounds(env)
    @test length(low) == 17
end

@testitem "Type Parameters - Float32" begin

    env = HalfCheetahEnv(
        max_episode_steps = 100,
        forward_reward_weight = Float32(1.0),
        ctrl_cost_weight = Float32(0.1)
    )
    @test env.forward_reward_weight isa Float32
    @test env.ctrl_cost_weight isa Float32
end

@testitem "Type Parameters - Custom RNG" begin
    using Random

    rng = Random.MersenneTwister(123)
    env = HopperEnv(max_episode_steps = 100, rng = rng)
    @test env.rng isa Random.MersenneTwister
end
