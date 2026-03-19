@testitem "DrillInterface smoke - all environments" begin
    using DrillInterface

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
        DrillInterface.reset!(env)
        obs = DrillInterface.observe(env)
        @test obs isa Vector
        @test length(obs) > 0

        act_space = DrillInterface.action_space(env)
        act_low = act_space.low
        act_high = act_space.high
        action = zeros(length(act_low))
        ret = DrillInterface.act!(env, action)
        @test ret isa Real

        @test DrillInterface.terminated(env) isa Bool
        @test DrillInterface.truncated(env) isa Bool

        obs_space = DrillInterface.observation_space(env)
        @test length(obs_space.low) == length(obs_space.high) == length(DrillInterface.observe(env))
        @test length(act_low) == length(act_high)

        info = DrillInterface.get_info(env)
        @test info isa Dict
    end
end
