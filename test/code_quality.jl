@testitem "Code quality (Aqua.jl)" begin
    using Aqua

    Aqua.test_all(MuJoCoEnvironments)
end

@testitem "Code linting (JET.jl)" begin
    using JET

    JET.test_package(MuJoCoEnvironments; target_modules = (MuJoCoEnvironments,))
end
