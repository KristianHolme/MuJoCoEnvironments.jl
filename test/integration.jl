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

@testitem "Reproducibility" begin
    using Random

    rng1 = Random.Xoshiro(123)
    rng2 = Random.Xoshiro(123)

    env1 = HopperEnv(max_episode_steps = 100, rng = rng1)
    env2 = HopperEnv(max_episode_steps = 100, rng = rng2)

    obs1 = reset_env!(env1)
    obs2 = reset_env!(env2)

    @test obs1 ≈ obs2
end

@testitem "Multiple environments" begin
    envs = [
        HalfCheetahEnv(max_episode_steps = 10),
        HopperEnv(max_episode_steps = 10),
        Walker2dEnv(max_episode_steps = 10),
    ]

    for env in envs
        obs = reset_env!(env)
        @test obs isa Vector
        low, high = get_action_bounds(env)
        action = zeros(length(low))
        step_env!(env, action)
    end
end
