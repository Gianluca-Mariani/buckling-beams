module FindEquilibria

using HomotopyContinuation, LinearAlgebra, CriticalTransitions, DynamicalSystems, ThreadsX

@kwdef mutable struct RealSolution
    realStable::Matrix{Float64}
    realUnstable::Matrix{Float64}
    timesStable::Vector{Float64}
    timesUnstable::Vector{Float64}
    maskStable::Vector{Bool}
    transitionActions::Vector{Vector{Tuple{Int, Float64}}}
end


"""
is_minimum(Vector{Float64}, Matrix{Expression}, Vector{HomotopyContinuation.ModelKit.Variable}) -> Bool

# Arguments
- `x_sol::Vector{Float64}`: The numerical solution where the Hessian is to be evaluated.
- `H_evaluated::Matrix{Expression}`: The symbolic Hessian with only the q variables left unsubstituted.
- `q::Vector{HomotopyContinuation.ModelKit.Variable}`: coordinate symbolic array, must be of same length as `x_sol`.

Returns true if the Hessian matrix is positive definite at the given solution, false otherwise.
"""
function is_minimum(x_sol::Vector{Float64}, H_evaluated::Matrix{Expression}, q::Vector{HomotopyContinuation.ModelKit.Variable}; eps=1e-1)
    H_num = evaluate(H_evaluated, q=>x_sol)
    d = diag(H_num)
    e = diag(H_num, 1)
    H_mat = SymTridiagonal(d, e) 
    lams = eigvals(H_mat)
    return lams[1] > eps
end

"""
symbolic_potential(n::Int) -> (Vector{Polynomial}, Matrix{Polynomial}, Vector{Polynomial}, Polynomial, Polynomial, Polynomial, Polynomial)

Constructs the symbolic gradient and Hessian of a potential function for a chain of length `n`.

# Returns
- `grad`: Vector of symbolic gradients w.r.t. the coordinates `q`
- `H`: Hessian matrix of second derivatives
- `q`: Vector of symbolic variables `q₁, q₂, ..., qₙ`
- `r₀`, `r₁`, `a`, `off`: Additional symbolic scalar parameters
"""
function symbolic_potential(n::Int)
    # Create symbolic variables
    @var r₀ r₁ a off
    @var q[1:n]

    # Construct the symbolic potential
    phi = i -> (i % 4 == 0) || (i % 4 == 3) ? 1.0 : -1.0
    V = sum(0.5 * ((-1)^(i-1) + r₀ * phi(i-1) * a) * q[i]^2 + 0.25 * q[i]^4 + q[i] * off for i in 1:n) # Non-linear beam potential (plus offset for homotopy stability)
    V += sum(0.5 * r₁ * (q[i+1] - q[i])^2 for i in 1:n-1) # Coupling potential
    V += 0.5 * r₁ * (q[1] - q[n])^2 # Periodic conditions: coupling between first and last beam

    # Calculate gradient and Hessian symbolically
    grad = differentiate(V, q)
    H = differentiate(grad, q)

    # Return all quantities symbolically
    return grad, H, q, r₀, r₁, a, off
end

"""
    find_equilibria_series(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0_val::Float64, r1_val::Float64)

Tracks equilibria of a symbolic potential system as a function of time-varying parameters.

# Arguments
- `n::Int`: The number of variables (number of beams in the chain).
- `times::AbstractVector{Float64}`: A vector of time points at which to evaluate the system.
- `ω_val::Float64`: Frequency used in the time-dependent modulation of parameter `a`.
- `r0_val::Float64`: Numerical value for the dynamic stiffness parameter `r₀`.
- `r1_val::Float64`: Numerical value for the coupling parameter `r₁`.

# Returns
- `unwrapped_data::Vector{Vector{Vector{ComplexF64}}}`: A list of solutions (as vectors of complex numbers) for each time point, corresponding to equilibrium points tracked from the homotopy continuation.
- `H_0::Matrix{Expression}`: The symbolic Hessian matrix of the potential, with all parameters except `a` substituted.
- `a`: a symbolic variable
- `q`: a symbolic variable

# Notes
- Uses homotopy continuation to track solutions from random complex initial parameters to a family of real parameters depending on `sin(ω * t)`.
- Solutions are not filtered to be real; all reachable tracked solutions (including complex) are returned.
- This function uses a symbolic gradient defined via `symbolic_potential`.
- only_finite = false, multiple_results = true are needed to not discard some valid solutions

# Dependencies
Requires a symbolic differentiation library such as `Symbolics.jl`, and a solver package capable of homotopy continuation (e.g., `HomotopyContinuation.jl`).

"""
function find_equilibria_series(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0_val::Float64, r1_val::Float64; fast::Bool=true, N::Int = 20)
    N_times = length(times)
    
    # TODO: Try fixing random seed

    # Initialize variables
    grad, H, q, r0_sym, r1_sym, a_sym, off_sym = symbolic_potential(n)

    # Make substitutions that need to be done only once
    grad_0 = subs(grad, r0_sym => r0_val, r1_sym => r1_val)
    H_0 = subs(H, r0_sym => r0_val, r1_sym => r1_val, off_sym => 0.0)
    S = System(grad_0; variables = q, parameters = [a_sym, off_sym])

    # Initial solve with complex random parameter
    start_parameters = [randn(ComplexF64), randn(ComplexF64)]
    result = solve(S; target_parameters = start_parameters, start_system=:total_degree)
    N_sol = length(result)
    N_fast_sol = N_sol

    # Fast algorithm only keeps solutions in result that are at least once real in a sparse time scan
    if fast
        sampling_points = range(times[1], times[end], length=N) # Define a 
        data_sparse = Vector{Vector{Float64}}(undef, N)
        for (i, t) in enumerate(sampling_points)
            data_sparse[i] = [sin(ω_val * t), 0.0]
        end

        # track p₀ towards the entries of data_sparse
        data_points_fast = solve(
        S,
        solutions(result);
        start_parameters = start_parameters,
        target_parameters = data_sparse,
        start_system=:total_degree,
        transform_result = (r,p) -> results(r; only_finite = false, multiple_results = true)
        )

        flag_real = falses(N_sol)
        for j in 1:N_sol
            for i in 1:N
                if is_solution_real((data_points_fast[i][j]).solution)
                    flag_real[j] = true
                    break
                end
            end
        end
        N_fast_sol = sum(flag_real)
        result = result[flag_real]
    end

    # generate all parameter values for dense time sweep
    data_full = Vector{Vector{Float64}}(undef, N_times)
    for (i, t) in enumerate(times)
        data_full[i] = [sin(ω_val * t), 0.0]
    end

    # track p₀ towards the entries of data_full
    data_points_full = solve(
    S,
    solutions(result);
    start_parameters = start_parameters,
    target_parameters = data_full,
    start_system=:total_degree,
    transform_result = (r,p) -> results(r; only_finite = false, multiple_results = true)
    )

    all_solutions = Array{ComplexF64}(undef, N_fast_sol, N_times, n)
    for j in 1:N_fast_sol
            for i in 1:N_times
                all_solutions[j, i, :] = (data_points_full[i][j]).solution
            end
        end
    
    #unwrapped_data = [[result.solution for result in data_point] for data_point in data_points]
    return all_solutions, (H = H_0, a_sym = a_sym, q = q, grad_1 = subs(grad_0, off_sym => 0))
end


"""
    is_solution_real(solution::Vector{ComplexF64}; tol::Float64 = 1e-8) -> Boolean

returns true if the input solution is real up to a tolerance value

# Arguments
- `solution::Vector{ComplexF64}`: A solution array of complex numbers for all coordinates values
- `tol::Float64`: tolerance for considering the array real

# Description
Returns true if the L2 norm of the imaginary part of the input solution is smaller than the input tolerance
"""
function is_solution_real(solution::Vector{ComplexF64}; tol::Float64 = 1e-6)
    return norm(imag(solution))  < tol

end

"""
    mark_real(data::Vector{Vector{Vector{ComplexF64}}}) -> Vector{Vector{Boolean}}

Maps each vector solution in the input vector to a boolean value, which is true if the solution is real

# Arguments
- `data::Vector{Vector{Vector{ComplexF64}}}`: Array of all solutions at all time steps (possibly re-ordered by `align_solutions`)
- `tol::Float64`: tolerance for considering the array real

# Description
Returns true for each solution if the solution is real, false otherwise
"""
function mark_real(data::Array{ComplexF64, 3}; tol::Float64 = 1e-6)
    #real_mask = [[is_solution_real(solution; tol) for solution in time_step] for time_step in data]
    N_sol, N_times, _ = size(data)
    real_mask = Array{Bool}(undef, N_sol, N_times)
    for i in 1:N_sol
        for j in 1:N_times
            real_mask[i, j] = is_solution_real(data[i, j, :]; tol)
        end
    end
    return real_mask
end

"""
    mark_real_stable(data::Vector{Vector{Vector{ComplexF64}}}, ω_val::Float64, times::AbstractVector{Float64}, sym::NamedTuple, real_mask::Vector{Vector{Bool}})
        -> Vector{Vector{Boolean}}

Maps each vector solution in the input vector to a boolean value, which is true if the solution is real

# Arguments
- `data::Vector{Vector{Vector{ComplexF64}}}`: Array of all solutions at all time steps (possibly re-ordered by `align_solutions`)
- `ω_val::Float64`: numerical value for the frequency
- `times::AbstractVector{Float64}`: time array with all time point values
- `sym::NamedTuple`: symbolic named tuple containing the Hessian H, the parameter a, and the variables vector q
- `real_mask::Vector{Vector{Bool}}`: Nested array of booleans, mapping each solution in data to true if real (possibly returned by `mark_real`)

# Description
Returns true for each solution if the solution is real, false otherwise
"""
function mark_real_stable(data::Array{ComplexF64, 3}, ω_val::Float64, times::AbstractVector{Float64}, sym::NamedTuple, real_mask::Array{Bool, 2}; eps::Float64 = 1e-1)
    N_sol, N_times = size(real_mask)
    real_stable_mask = Array{Bool, 2}(undef, N_sol, N_times)
    for j in 1:N_times
        H_time = subs(sym.H, sym.a_sym => sin(ω_val * times[j]))
        for i in 1:N_sol
            if real_mask[i, j]
                real_stable_mask[i, j] = is_minimum(real(data[i, j, :]), H_time, sym.q; eps)
            else
                real_stable_mask[i, j] = false
            end
        end
    end
    return real_stable_mask
end


"""
get_solutions_flags(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0_val::Float64, r1_val::Float64)
    -> Vector{Vector{Vector{ComplexF64}}}, Vector{Vector{Boolean}}, Vector{Vector{Boolean}}
"""
function get_solutions_flags(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0_val::Float64, r1_val::Float64, fast::Bool = true, N::Int = 10)
    
    result, sym = find_equilibria_series(n, times, ω_val, r0_val, r1_val; fast, N)
    real_result = mark_real(result)
    stablereal_result = mark_real_stable(result, ω_val, times, sym, real_result)

    #actions = get_action(result, stablereal_result, times, sym, ω_val)

    return result, real_result, stablereal_result#, actions
end



function get_action(result::Vector{Vector{Vector{ComplexF64}}}, stablereal_result::Vector{Vector{Bool}}, times::AbstractVector{Float64}, sym::NamedTuple, ω_val::Float64)

    function grad_action(u, p, t)
        return evaluate(sym.grad_1, sym.q => u, sym.a_sym => p[1])
    end

    n = length(sym.q)
    x0 = randn(n)

    actions = zeros(length(times))

    a_values = [sin(ω_val*t) for t in times]

    for (i, a) in enumerate(a_values)
        x_i = real(result[i][1])
        x_f = real(result[i][2])
        sde = CoupledSDEs(grad_action, x0, [a]; noise_strength = 0.05)
        pathStruct = geometric_min_action_method(sde, x_i, x_f; N=10)
        actions[i] = pathStruct.action
    end

    return actions
    
end

function create_structs(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0_val::Float64, r1_val::Float64; fast::Bool = true, N::Int = 10)
    result, sym = find_equilibria_series(n, times, ω_val, r0_val, r1_val; fast, N)
    real_result = mark_real(result)
    stablereal_result = mark_real_stable(result, ω_val, times, sym, real_result)
    unstablereal_result = real_result .&& .!stablereal_result

    N_sol, _, _ = size(result)
    sol_struct_array = Vector{RealSolution}(undef, N_sol)

    for i in 1:N_sol
        in1 = real(result[i, stablereal_result[i, :], :])
        in2 = real(result[i, unstablereal_result[i, :], :])
        in3 = times[stablereal_result[i, :]]
        in4 = times[unstablereal_result[i, :]]
        sol_struct_array[i] = RealSolution(realStable = in1, realUnstable = in2, timesStable = in3, timesUnstable = in4, maskStable = stablereal_result[i, :], transitionActions = Vector{Vector{Tuple{Int, Float64}}}())
    end

    for i in 1:N_sol
        
    end

    return sol_struct_array
end

"""
    sweep_one_parameter(n::Int, times::AbstractVector{Float64}, ω_val::Float64, sweeping::AbstractVector{Float64}, fixed::Float64, sweep_label::String)
        -> Vector{Vector{Vector{Boolean}}}, Vector{Vector{Vector{Boolean}}}

Returns two vectors where each entry is the real_mask and real_stable_mask returned by `mark_real` and `mark_real_stable` for each value in sweeping,
as determined by sweep_label

# Arguments
- `n::Int`: Number of variables in the system (e.g., length of the chain).
- `times::AbstractVector{Float64}`: A vector of time points over which to evaluate the solution count.
- `ω_val::Float64`: Angular frequency used in the time-dependent modulation (e.g., in `a = sin(ωt)`).
- `sweeping::AbstractVector{Float64}`: Parameter vector to sweep
- `fixed::Float64`: Fixed parameter not to sweep
- `sweep_label::String`: Label to discriminate which parameter to sweep, only "r0" and "r1" accepted

"""
function sweep_one_parameter(n::Int, times::AbstractVector{Float64}, ω_val::Float64, sweeping::AbstractVector{Float64}, fixed::Float64, sweep_label::String)
    real_results = Vector{Vector{Vector{Bool}}}(undef, length(sweeping))
    real_stable_results = Vector{Vector{Vector{Bool}}}(undef, length(sweeping))

    for i in eachindex(sweeping)
        if cmp(sweep_label, "r0") == 0
            res, sym = find_equilibria_series(n, times, ω_val, sweeping[i], fixed)
        elseif cmp(sweep_label, "r1") == 0
            res, sym = find_equilibria_series(n, times, ω_val, fixed, sweeping[i])
        else
            throw(ArgumentError("Invalid value for sweep_label: $(sweep_label). Expected \"r0\" or \"r1\"."))
        end
        aligned = align_solutions(res)
        real_results[i] = mark_real(aligned)
        real_stable_results[i] = mark_real_stable(aligned, ω_val, times, sym, real_results[i])
    end

    return real_results, real_stable_results

end


"""
sweep_two_parameters(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0s::AbstractVector{Float64}, r1s::AbstractVector{Float64})
    -> Vector{Vector{Vector{Vector{Boolean}}}}, Vector{Vector{Vector{Vector{Boolean}}}}

Returns two Vector{Vector{..}} where each entry is the real_mask and real_stable_mask returned by `mark_real` and `mark_real_stable`,
for each pair of values (r0s[i], r1s[j]) value in sweeping

# Arguments
- `n::Int`: Number of variables in the system (e.g., length of the chain).
- `times::AbstractVector{Float64}`: A vector of time points over which to evaluate the solution count.
- `ω_val::Float64`: Angular frequency used in the time-dependent modulation (e.g., in `a = sin(ωt)`).
- `r0s::AbstractVector{Float64}`: r0s vector to sweep
- `r1s::AbstractVector{Float64}`: r1s vector to sweep

"""
function sweep_two_parameters(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0s::AbstractVector{Float64}, r1s::AbstractVector{Float64})
    real_results = Vector{Vector{Vector{Vector{Bool}}}}(undef, length(r1s))
    real_stable_results = Vector{Vector{Vector{Vector{Bool}}}}(undef, length(r1s))

    for i in eachindex(r1s)
        real_results[i], real_stable_results[i] = sweep_one_parameter(n, times, ω_val, r0s, r1s[i], "r0")
    end

    return real_results, real_stable_results
end

#=

omegas = [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]],
        [[2.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [2.0, 2.0]]]

r0s = [[[0.5, 0.5], [0.8, 0.8]], [[0.5, 0.5], [0.8, 0.8]],
        [[0.5, 0.5], [0.8, 0.8]], [[0.5, 0.5], [0.8, 0.8]]]

r1s = [[[0.2, 0.5], [0.2, 0.5]], [[0.2, 0.5], [0.2, 0.5]],
        [[0.2, 0.5], [0.2, 0.5]], [[0.2, 0.5], [0.2, 0.5]]]

result = parallel_find_equilibria(2, 0:0.1:10, omegas, r0s, r1s)



x1_over_time = [result[t][1][2] for t in eachindex(result)]
x2_over_time = [result[t][2][2] for t in eachindex(result)]
x3_over_time = [result[t][3][2] for t in eachindex(result)]
x4_over_time = [result[t][4][2] for t in eachindex(result)]


using Plots
plot(times, x1_over_time, xlabel="t", ylabel="x2", label="Solution 1")
plot!(times, x2_over_time, label="Solution 2")
plot!(times, x3_over_time, label="Solution 3")
plot!(times, x4_over_time, label="Solution 4")

=#

## TODO: parameter sweeps


function parallel_find_equilibria(n::Int, times, ω_matrix, r0_matrix, r1_matrix)
    # Flatten parameter arrays (same order)
    ω_vec = vec(ω_matrix)
    r0_vec = vec(r0_matrix)
    r1_vec = vec(r1_matrix)

    # Form parameter list as tuples
    param_list = [(ω, r0, r1) for (ω, r0, r1) in zip(ω_vec, r0_vec, r1_vec)]

    # Map over parameter list
    results_vec = ThreadsX.map(((ω, r0, r1),) -> find_equilibria_series(n, times, ω, r0, r1), param_list)

    # Reshape back to original shape
    results = reshape(results_vec, size(omega))
    return results
end


end