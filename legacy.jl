using Munkres

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