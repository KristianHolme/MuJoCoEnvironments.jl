using MuJoCoEnvironments
using Test
using Aqua
using JET

@testset "MuJoCoEnvironments.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MuJoCoEnvironments)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(MuJoCoEnvironments; target_defined_modules = true)
    end
    # Write your tests here.
end
