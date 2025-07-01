module FindEquilibria

using HomotopyContinuation, LinearAlgebra, CriticalTransitions, DynamicalSystems, Symbolics, Munkres, ThreadsX

"""
is_minimum(Vector{Float64}, Matrix{Expression}, Vector{HomotopyContinuation.ModelKit.Variable}) -> Bool

# Arguments
- `x_sol::Vector{Float64}`: The numerical solution where the Hessian is to be evaluated.
- `H_evaluated::Matrix{Expression}`: The symbolic Hessian with only the q variables left unsubstituted.
- `q::Vector{HomotopyContinuation.ModelKit.Variable}`: coordinate symbolic array, must be of same length as `x_sol`.

Returns true if the Hessian matrix is positive definite at the given solution, false otherwise.
"""
function is_minimum(x_sol::Vector{Float64}, H_evaluated::Matrix{Expression}, q::Vector{HomotopyContinuation.ModelKit.Variable})
    eps = 1e-1

    H_num = evaluate(H_evaluated, q=>x_sol)
    d = diag(H_num)
    e = diag(H_num, 1)
    H_mat = SymTridiagonal(d, e) 
    lams = eigvals(H_mat)
    return lams[1] > eps
    #return isposdef(H_mat) # return true if the Hessian is positive definite

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
function find_equilibria_series(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0_val::Float64, r1_val::Float64)
    # Initialize variables
    grad, H, q, r0_sym, r1_sym, a_sym, off_sym = symbolic_potential(n)
    H_0 = subs(H, r0_sym => r0_val, r1_sym => r1_val, off_sym => 0.0)

    # Make substitutions that need to be done only once
    grad_0 = subs(grad, r0_sym => r0_val, r1_sym => r1_val)
    S = System(grad_0; variables = q, parameters = [a_sym, off_sym])

    # Initial solve with complex parameter
    start_a = randn(ComplexF64)
    start_off = randn(ComplexF64)
    start_parameters = [start_a, start_off]
    result = solve(S; target_parameters = start_parameters, start_system=:total_degree)

    # generate all parameter values
    data = [[sin(ω_val * times[i]), 0.0] for i in eachindex(times)]

    # track p₀ towards the entries of data
    data_points = solve(
    S,
    solutions(result);
    start_parameters = start_parameters,
    target_parameters = data,
    start_system=:total_degree,
    transform_result = (r,p) -> results(r; only_finite = false, multiple_results = true)
    )
    unwrapped_data = [[result.solution for result in data_point] for data_point in data_points]
    return unwrapped_data, (H = H_0, a_sym = a_sym, q = q, grad_1 = subs(grad_0, off_sym => 0))
end


function find_real_equilibria_fast(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0_val::Float64, r1_val::Float64; N::Int = 10)
    # Much to do here still
    # Initialize variables
    grad, H, q, r0_sym, r1_sym, a_sym, off_sym = symbolic_potential(n)
    H_0 = subs(H, r0_sym => r0_val, r1_sym => r1_val, off_sym => 0.0)

    # Make substitutions that need to be done only once
    grad_0 = subs(grad, r0_sym => r0_val, r1_sym => r1_val)
    S = System(grad_0; variables = q, parameters = [a_sym, off_sym])

    # Initial solve with complex parameter
    start_a = randn(ComplexF64)
    start_off = randn(ComplexF64)
    start_parameters = [start_a, start_off]
    result = solve(S; target_parameters = start_parameters, start_system=:total_degree)

    # generate all parameter values
    sampling_points = range(times[1], times[end], length=N)
    data_sparse = [[sin(ω_val * sampling_points[i]), 0.0] for i in eachindex(sampling_points)]

    # track p₀ towards the entries of data
    data_points = solve(
    S,
    solutions(result);
    start_parameters = start_parameters,
    target_parameters = data_sparse,
    start_system=:total_degree,
    transform_result = (r,p) -> results(r; only_finite = false, multiple_results = true)
    )
    unwrapped_data = [[result.solution for result in data_point] for data_point in data_points]

    n_times = size(unwrapped_data)[1]
    n_solutions = size(unwrapped_data[1])[1]
    flag_real = [any(is_solution_real(vec(unwrapped_data[t][s])) for t in 1:n_times)
          for s in 1:n_solutions]

    data_full = [[sin(ω_val * times[i]), 0.0] for i in eachindex(times)]
    
    final_data_points = solve(
    S,
    solutions(result[flag_real]);
    start_parameters = start_parameters,
    target_parameters = data_full,
    start_system=:total_degree,
    transform_result = (r,p) -> results(r; only_finite = false, multiple_results = true))

    unwrapped_data_final = [[result.solution for result in data_point] for data_point in final_data_points]

    return unwrapped_data_final, (H = H_0, a_sym = a_sym, q = q, grad_1 = subs(grad_0, off_sym => 0))
    
end

"""
generate_combinations(n::Int, z::Vector{ComplexF64}) -> Vector{Vector{ComplexF64}}

Generates all possible combinations (with replacement) of `n` elements taken from vector `z`.

Each combination is represented as a vector of length `n`, where each element is selected from `z`.
The total number of combinations is `length(z)^n`.

# Arguments
- `n::Int`: The number of elements in each combination.
- `z::Vector{ComplexF64}`: The vector of complex values to draw from.

# Returns
- `Vector{Vector{ComplexF64}}`: A vector containing all combinations of length `n`, where each element is a copy of a combination vector.

# Example
```julia
z = [1 + 0im, -1 + 0im, 0 + 1im]
generate_combinations(2, z)
# Returns: [[1+0im, 1+0im], [1+0im, -1+0im], ..., [0+1im, 0+1im]]
"""
function generate_combinations(n::Int, z::Vector{ComplexF64})
    num_combinations = length(z)^n
    x = Vector{Vector{ComplexF64}}(undef, num_combinations)

    indices = Iterators.product(fill(1:length(z), n)...)

    for (i, index_tuple) in enumerate(indices)
        x[i] = [z[j] for j in index_tuple]
    end

  return x
end

"""
Compute the number of solutions for a given potential system, repeated over several runs.

Arguments:
- `n::Int`: Number of degrees of freedom (chain length).
- `time::Float64`: Time at which the equilibrium is evaluated.
- `ω_val::Float64`: Frequency parameter.
- `r0_val::Float64`, `r1_val::Float64`: Non-dimensional constants.
- `reps::Int`: Number of repetitions.

Returns:
- `num_sol::Vector{Int64}`: Number of solutions found in each repetition.
"""
function get_number_of_solutions(n::Int, time::Float64, ω_val::Float64, r0_val::Float64, r1_val::Float64, reps::Int)
    num_sol = Vector{Int}(undef, reps)
    redirect_stdout(devnull) do
        for i in 1:reps
            num_sol[i] = length(find_equilibria_series(n, [time], ω_val, r0_val, r1_val)[1][1])
        end
    end
    return num_sol
end


"""
use_homotopy_tracker(n::Int, time::Float64, ω_val::Float64, r0_val::Float64, r1_val::Float64) 
    -> Vector{Vector{ComplexF64}}

Tracks solutions to a parameterized polynomial system using a manually constructed homotopy and tracker.

This function implements a **manual homotopy continuation** approach using the `HomotopyContinuation.jl` package. It constructs a symbolic potential energy gradient, defines a simple start system with known roots (the complex cube roots of unity), and then tracks these known solutions toward the solutions of the target system. This is a more low-level alternative to calling `solve(...)` directly with parameter tracking.

# Arguments
- `n::Int`: Number of variables in the system (i.e.: number of beams in the chain).
- `time::Float64`: Time value used to compute the time-dependent parameter `a = sin(ω * t)`.
- `ω_val::Float64`: Frequency used in the time-dependent parameter.
- `r0_val::Float64`: Dynamic stiffness parameter for the potential function.
- `r1_val::Float64`: Coupling parameter for nearest-neighbor interactions.

# Returns
- `Vector{Vector{ComplexF64}}`: A vector of successful tracked solutions (complex-valued vectors of length `n`), obtained from homotopy continuation starting at the roots of the simplified start system.

# Notes
- The start system is defined as `y_i^3 - 1 = 0` for each variable, whose known solutions are the complex cube roots of unity (total degree homotopy)
- This method is lower-level and more explicit than using `solve(System(...); target_parameters = ...)`, which handles tracking, solution validation, and path failures more automatically.
- Tracking is done with `track.(...)`, and only successful paths are returned.

# Example
```julia
solutions = use_homotopy_tracker(3, 1.0, 2π, 1.0, 1.0)
"""
function use_homotopy_tracker(n::Int, time::Float64, ω_val::Float64, r0_val::Float64, r1_val::Float64)
    # Get symbolic potential and substitute fixed parameters values
    grad, _, q, r0_sym, r1_sym, a_sym, off_sym = symbolic_potential(n)
    grad_0 = subs(grad, r0_sym => r0_val, r1_sym => r1_val, a_sym => sin(ω_val * time), off_sym => 0.0)
    
    # Construct start system
    @var y[1:n]
    G = System(im * (y.^3 .- 1))

    # Construct start solutions
    z = [1, exp(2im * π / 3), exp(-2im * π / 3)] # Solution vector for single equation
    start_solutions = generate_combinations(n, z) # Construct vector with all system solutions

    # construct tracker
    F = System(grad_0; variables = q)
    H = StraightLineHomotopy(G, F)
    tracker = Tracker(H)

    # track each start solution separetely
    results = track.(tracker, start_solutions)
    tracked_solutions = [result.solution for result in results if is_success(result)]

    return tracked_solutions
end


"""
get_number_solutions_per_time(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0_val::Float64, r1_val::Float64)
    -> Vector{Int64}

Returns the number of equilibrium solutions found at each point in time during the evolution of a parameterized polynomial system.

This function computes the number of solutions obtained by homotopy continuation at each time in the provided time vector. It uses `find_equilibria_series` to compute the tracked solutions for each time-dependent parameter configuration and extracts the count of solutions at each time step.

# Arguments
- `n::Int`: Number of variables in the system (e.g., length of the chain).
- `times::AbstractVector{Float64}`: A vector of time points over which to evaluate the solution count.
- `ω_val::Float64`: Angular frequency used in the time-dependent modulation (e.g., in `a = sin(ωt)`).
- `r0_val::Float64`: Dynamic stiffness parameter for the potential function.
- `r1_val::Float64`: Coupling parameter between neighboring elements.

# Returns
- `Vector{Int64}`: A vector of the same length as `times`, where each entry contains the number of equilibrium solutions found at the corresponding time.

# Example
```julia
times = 0:0.1:10
counts = get_number_solutions_per_time(4, times, 1.0, 0.5, 0.8)
"""
function get_number_solutions_per_time(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0_val::Float64, r1_val::Float64)
    result = find_equilibria_series(n, times, ω_val, r0_val, r1_val)
    
    result_length = Vector{Int64}(undef, length(times))
    for i in eachindex(times)
        result_length[i] = length(result[1][i])
    end
    return result_length
end


"""
    align_solutions(data::Vector{Vector{Vector{ComplexF64}}}) -> Vector{Vector{Vector{ComplexF64}}}

Aligns time-evolving solutions to ensure consistent ordering across time steps by minimizing pairwise squared Euclidean distances using the Hungarian (Munkres) algorithm.

# Arguments
- `data::Vector{Vector{Vector{ComplexF64}}}`: A nested array where:
  - The outer vector has length `T`, representing `T` time steps,
  - Each middle vector has length `S`, representing `S` solutions at that time,
  - Each inner vector contains the complex coordinates of a given solution at that time step.

# Description
This function reorders the solutions at each time step to match the closest solutions from the previous time step, minimizing the total squared Euclidean distance between matched solution vectors.

For each pair of consecutive time steps `t` and `t+1`, it:
1. Computes a cost matrix of pairwise distances between solutions,
2. Solves the optimal assignment problem using the `munkres()` algorithm,
3. Reorders the solutions at time `t+1` based on the assignment.

# Returns
- A reordered deep copy of `data`, with solutions at each time step aligned consistently based on proximity.

# Assumptions
- The number of solutions (`S`) is constant across time steps.
- Each solution is represented as a vector of complex numbers (e.g., multivariate state).

# Example
```julia
aligned_data = align_solutions(solution_data)
"""
function align_solutions(data::Vector{Vector{Vector{ComplexF64}}})
    T = length(data)
    S = length(data[1])

    aligned = deepcopy(data)

    for t in 1:(T - 1)
        A = aligned[t]     # solutions at time t
        B = data[t + 1]    # original solutions at time t+1

        # Build cost matrix (S x S) of squared Euclidean distances
        cost_matrix = zeros(S, S)
        for i in 1:S, j in 1:S
            cost_matrix[i, j] = sum(abs2, A[i] .- B[j])
        end

        # Solve the assignment problem
        col_idx = munkres(cost_matrix)

        # Reorder B according to assignment
        aligned[t + 1] = [B[j] for j in col_idx]
    end

    return aligned
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
function mark_real(data::Vector{Vector{Vector{ComplexF64}}}; tol::Float64 = 1e-6)
    real_mask = [[is_solution_real(solution; tol) for solution in time_step] for time_step in data]
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
function mark_real_stable(data::Vector{Vector{Vector{ComplexF64}}}, ω_val::Float64, times::AbstractVector{Float64}, sym::NamedTuple, real_mask::Vector{Vector{Bool}})
    real_stable_mask = deepcopy(real_mask)
    for (i, time) in enumerate(times)
        H_time = subs(sym.H, sym.a_sym => sin(ω_val * time))
        for (j, sol) in enumerate(data[i])
            if real_mask[i][j]
                real_stable_mask[i][j] = is_minimum(real(sol), H_time, sym.q)
            else
                real_stable_mask[i][j] = false
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
    local result, sym
    
    if fast
        result, sym = find_real_equilibria_fast(n, times, ω_val, r0_val, r1_val; N)
    else
        result, sym = find_equilibria_series(n, times, ω_val, r0_val, r1_val)
    end
    real_result = mark_real(result)
    stablereal_result = mark_real_stable(result, ω_val, times, sym, real_result)

    actions = get_action(result, stablereal_result, times, sym, ω_val)

    return result, real_result, stablereal_result, actions
end



function get_action(result::Vector{Vector{Vector{ComplexF64}}}, stablereal_result::Vector{Vector{Bool}}, times::AbstractVector{Float64}, sym, ω_val::Float64)

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