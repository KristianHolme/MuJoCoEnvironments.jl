@testitem "CommonRLInterface wrapper creation" begin
    using MuJoCoEnvironments
    using CommonRLInterface
    const CRL = CommonRLInterface

    core_env = HalfCheetahEnv(max_episode_steps = 100)
    env = CommonRLMuJoCoEnv(core_env)
    @test env isa CommonRLMuJoCoEnv
end

@testitem "CommonRLInterface Base.convert" begin
    using MuJoCoEnvironments
    using CommonRLInterface
    const CRL = CommonRLInterface

    core_env = HopperEnv(max_episode_steps = 100)
    env = convert(CommonRLMuJoCoEnv, core_env)
    @test env isa CommonRLMuJoCoEnv
end

@testitem "CommonRLInterface interface methods" begin
    using MuJoCoEnvironments
    using CommonRLInterface
    const CRL = CommonRLInterface

    core_env = Walker2dEnv(max_episode_steps = 100)
    env = convert(CommonRLMuJoCoEnv, core_env)

    CRL.reset!(env)
    obs = CRL.observe(env)
    @test obs isa Vector

    actions = CRL.actions(env)
    @test actions isa Vector

    observations = CRL.observations(env)
    @test observations isa Vector

    action = [
        rand() * (interval.right - interval.left) + interval.left
            for interval in actions
    ]
    CRL.act!(env, action)
    @test CRL.observe(env) isa Vector

    @test CRL.terminated(env) isa Bool
end

@testitem "CommonRLInterface state manipulation" begin
    using MuJoCoEnvironments
    using CommonRLInterface
    const CRL = CommonRLInterface

    core_env = AntEnv(max_episode_steps = 100)
    env = convert(CommonRLMuJoCoEnv, core_env)
    CRL.reset!(env)

    state1 = CRL.state(env)
    @test state1 isa Vector

    # Take a step
    action = zeros(length(CRL.actions(env)))
    CRL.act!(env, action)
    state2 = CRL.state(env)
    @test state2 isa Vector

    # Restore state
    CRL.setstate!(env, state1)
    state3 = CRL.state(env)
    @test state1 ≈ state3
end

@testitem "CommonRLInterface clone" begin
    using MuJoCoEnvironments
    using CommonRLInterface
    const CRL = CommonRLInterface

    core_env = SwimmerEnv(max_episode_steps = 100)
    env = convert(CommonRLMuJoCoEnv, core_env)
    CRL.reset!(env)

    env_clone = CRL.clone(env)
    @test env_clone isa CommonRLMuJoCoEnv
    @test CRL.observe(env) ≈ CRL.observe(env_clone)
end
