@testitem "Code quality (Aqua.jl)" begin
    using MuJoCoEnvironments
    using Aqua

    Aqua.test_all(MuJoCoEnvironments)
end

@testitem "Code linting (JET.jl)" begin
    using MuJoCoEnvironments
    using JET

    JET.test_package(MuJoCoEnvironments; target_defined_modules = true)
end
