@testitem "DRiL wrapper creation" begin
    using MuJoCoEnvironments
    using DRiL

    core_env = HalfCheetahEnv(max_episode_steps = 100)
    env = DRiLMuJoCoEnv(core_env)
    @test env isa DRiLMuJoCoEnv
end

@testitem "DRiL Base.convert" begin
    using MuJoCoEnvironments
    using DRiL

    core_env = HopperEnv(max_episode_steps = 100)
    env = convert(DRiLMuJoCoEnv, core_env)
    @test env isa DRiLMuJoCoEnv
end

@testitem "DRiL interface methods" begin
    using MuJoCoEnvironments
    using DRiL

    core_env = Walker2dEnv(max_episode_steps = 100)
    env = convert(DRiLMuJoCoEnv, core_env)

    DRiL.reset!(env)
    obs = DRiL.observe(env)
    @test obs isa Vector
    # Should use the environment's type parameter (Float64 by default)
    @test eltype(obs) == typeof(core_env.forward_reward_weight)

    action_space = DRiL.action_space(env)
    @test action_space isa DRiL.Box

    observation_space = DRiL.observation_space(env)
    @test observation_space isa DRiL.Box

    T = typeof(core_env.forward_reward_weight)
    action = zeros(T, length(action_space.low))
    reward = DRiL.act!(env, action)
    @test reward isa T

    @test DRiL.terminated(env) isa Bool
    @test DRiL.truncated(env) isa Bool
end

@testitem "DRiL termination check" begin
    using MuJoCoEnvironments
    using DRiL

    core_env = AntEnv(max_episode_steps = 100)
    env = convert(DRiLMuJoCoEnv, core_env)
    DRiL.reset!(env)
    @test DRiL.terminated(env) == is_terminated(core_env)
end

@testitem "DRiL Random.seed!" begin
    using MuJoCoEnvironments
    using DRiL
    using Random

    core_env = HopperEnv(max_episode_steps = 100)
    env = convert(DRiLMuJoCoEnv, core_env)

    Random.seed!(env, 123)
    DRiL.reset!(env)
    obs1 = DRiL.observe(env)

    Random.seed!(env, 123)
    DRiL.reset!(env)
    obs2 = DRiL.observe(env)

    @test obs1 ≈ obs2
end

@testitem "DRiL type parameters" begin
    using MuJoCoEnvironments
    using DRiL

    # Test with Float32 environment
    core_env_f32 = HalfCheetahEnv(
        max_episode_steps = 100,
        forward_reward_weight = Float32(1.0),
        ctrl_cost_weight = Float32(0.1)
    )
    env_f32 = convert(DRiLMuJoCoEnv, core_env_f32)

    DRiL.reset!(env_f32)
    obs_f32 = DRiL.observe(env_f32)
    @test eltype(obs_f32) == Float32

    action_space_f32 = DRiL.action_space(env_f32)
    @test eltype(action_space_f32.low) == Float32

    action_f32 = zeros(Float32, length(action_space_f32.low))
    reward_f32 = DRiL.act!(env_f32, action_f32)
    @test reward_f32 isa Float32
end
