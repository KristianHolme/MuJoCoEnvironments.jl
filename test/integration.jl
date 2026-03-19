@testitem "Full episode" begin
    env = HalfCheetahEnv(max_episode_steps = 10)
    reset_env!(env)
    let tot_reward = 0.0
        low, high = get_action_bounds(env)

        for _ in 1:10
            action = zeros(length(low))
            _, reward, terminated, truncated, info = step_env!(env, action)
            tot_reward += reward
            if terminated || truncated
                break
            end
        end

        @test tot_reward isa Real
        @test is_truncated(env) || env.timestep >= 10
    end
end

@testitem "Full episode - all environments" begin
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
        reset_env!(env)
        tot_reward = 0.0
        low, high = get_action_bounds(env)
        for _ in 1:max_steps
            action = zeros(length(low))
            _, reward, terminated, truncated, info = step_env!(env, action)
            tot_reward += reward
            if terminated || truncated
                break
            end
        end
        @test tot_reward isa Real
        @test is_truncated(env) || env.timestep >= max_steps
    end
end

@testitem "Reproducibility" begin
    using Random

    rng1 = Random.Xoshiro(123)
    rng2 = Random.Xoshiro(123)

    env1 = HopperEnv(max_episode_steps = 100, rng = rng1)
    env2 = HopperEnv(max_episode_steps = 100, rng = rng2)

    obs1 = reset_env!(env1)
    obs2 = reset_env!(env2)

    @test obs1 ≈ obs2

    for (EnvT, name) in [(ReacherEnv, "Reacher"), (InvertedPendulumEnv, "InvertedPendulum")]
        r1 = Random.Xoshiro(456)
        r2 = Random.Xoshiro(456)
        e1 = EnvT(max_episode_steps = 100, rng = r1)
        e2 = EnvT(max_episode_steps = 100, rng = r2)
        o1 = reset_env!(e1)
        o2 = reset_env!(e2)
        @test o1 ≈ o2
    end
end

@testitem "Multiple environments" begin
    envs = [
        HalfCheetahEnv(max_episode_steps = 10),
        HopperEnv(max_episode_steps = 10),
        Walker2dEnv(max_episode_steps = 10),
        AntEnv(max_episode_steps = 10),
        HumanoidEnv(max_episode_steps = 10),
        SwimmerEnv(max_episode_steps = 10),
        ReacherEnv(max_episode_steps = 10),
        PusherEnv(max_episode_steps = 10),
        InvertedPendulumEnv(max_episode_steps = 10),
        InvertedDoublePendulumEnv(max_episode_steps = 10),
        HumanoidStandupEnv(max_episode_steps = 10),
    ]

    for env in envs
        obs = reset_env!(env)
        @test obs isa Vector
        low, high = get_action_bounds(env)
        action = zeros(length(low))
        step_env!(env, action)
    end
end
