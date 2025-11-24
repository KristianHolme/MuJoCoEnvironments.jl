# ------------------------------------------------------------
# Rendering Support using MuJoCo.jl Visualizer
# ------------------------------------------------------------

"""
    render(env::AbstractMuJoCoEnv; mode::Symbol=:human)

Render the current state of the environment.

# Arguments
- `env::AbstractMuJoCoEnv`: The environment to render
- `mode::Symbol`: Rendering mode (currently only `:human` is supported)

# Details
Uses MuJoCo.jl's native visualizer to display the environment state.
Make sure you have called `MuJoCo.init_visualiser()` before using this function.
"""
function render(env::AbstractMuJoCoEnv{T}; mode::Symbol = :human) where {T}
    if mode == :human
        MuJoCo.visualise!(env.model, env.data)
    else
        error("Unsupported render mode: $mode. Currently only :human is supported.")
    end
    return nothing
end

"""
    close_render(env::AbstractMuJoCoEnv)

Close the rendering window (if applicable).

# Arguments
- `env::AbstractMuJoCoEnv`: The environment
"""
function close_render(env::AbstractMuJoCoEnv{T}) where {T}
    # MuJoCo.jl handles closing automatically, but we provide this for API completeness
    return nothing
end
