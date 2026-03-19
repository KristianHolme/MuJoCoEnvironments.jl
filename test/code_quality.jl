@testitem "Code quality (Aqua.jl)" begin
    using Aqua

    Aqua.test_all(MuJoCoEnvironments)
end

@testitem "Code linting (JET.jl)" begin
    using JET

    # Custom report filter: keep analysis strict but suppress only the specific
    # MethodErrorReports we accept (MuJoCo wrapper Union{Nothing,...} and visualise! untyped).
    struct MuJoCoEnvironmentsJETFilter end

    function _type_contains_nothing(@nospecialize(typ))
        typ === Nothing && return true
        if typ isa Union
            return _type_contains_nothing(typ.a) || _type_contains_nothing(typ.b)
        end
        if typ isa Type && typ <: Tuple && isdefined(typ, :parameters)
            for i in 2:length(typ.parameters)
                _type_contains_nothing(typ.parameters[i]) && return true
            end
        end
        return false
    end

    function _method_error_report_should_suppress(report)
        t = report.t
        if t isa Vector
            for ti in t
                _type_contains_nothing(ti) && return true
            end
            return false
        end
        if _type_contains_nothing(t)
            return true
        end
        # visualise!(env.model, env.data) with untyped wrapper types (::Any, ::Any)
        vst = report.vst
        isempty(vst) && return false
        file_str = string(last(vst).file)
        if occursin("rendering", file_str)
            return true
        end
        return false
    end

    function JET.configured_reports(::MuJoCoEnvironmentsJETFilter, reports::Vector{JET.InferenceErrorReport})
        # Keep only reports from our package (same as target_modules)
        mod_matcher = JET.LastFrameModule(MuJoCoEnvironments)
        reports = filter(reports) do r
            return JET.match_report(mod_matcher, r)
        end
        # Suppress only the narrow set of MethodErrorReports we accept
        reports = filter(reports) do r
            if r isa JET.MethodErrorReport
                return !_method_error_report_should_suppress(r)
            end
            return true
        end
        return reports
    end

    JET.test_package(
        MuJoCoEnvironments;
        target_modules = (MuJoCoEnvironments,),
        report_config = MuJoCoEnvironmentsJETFilter(),
    )
end
