using DynamicPolynomials, HomotopyContinuation, LinearAlgebra, ThreadsX


"""
is_minimum(Vector{Float64}, Matrix{<:Polynomial}, Vector{<:DynamicPolynomials.Variable}) -> Boolean
Returns true if the Hessian matrix is positive definite at the given solution, false otherwise (also for errors).
"""
function is_minimum(x_sol::Vector{Float64}, H_evaluated::Matrix{<:Polynomial}, q::AbstractVector{<:DynamicPolynomials.Variable})
    
    H_num = coefficient.(subs(H_evaluated, q=>x_sol), q[1]^0) # Unwrap constant polynomials
    d = diag(H_num)
    e = diag(H_num, 1)
    H_mat = SymTridiagonal(d, e)  
    return isposdef(H_mat) # return true if the Hessian is positive definite

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


# Main solver using parameter homotopy
function find_equilibria_series(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0_val::Float64, r1_val::Float64)
    """Missing Documentation"""
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
    return unwrapped_data, H_0
end


function generate_combinations(n::Int, z::Vector{ComplexF64})
    """
    Missing Documentation
    """
  num_combinations = 3^n
  x = Vector{Vector{ComplexF64}}(undef, num_combinations)

  indices = Iterators.product(fill(1:length(z), n)...)

  for (i, index_tuple) in enumerate(indices)
    x[i] = [z[j] for j in index_tuple]
  end

  return x
end

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

"""
Compute the number of solutions for a given potential system, repeated over several runs.

Arguments:
- `n::Int`: Number of degrees of freedom (chain length).
- `time::Float64`: Time at which the equilibrium is evaluated.
- `ω_val::Float64`: Frequency parameter.
- `r0_val::Float64`, `r1_val::Float64`: Coupling constants.
- `reps::Int`: Number of repetitions.

Returns:
- `num_sol::Vector{Int64}`: Number of solutions found in each repetition.

Note:
- Possible improvement: parallelize the for-loop for performance.
"""
function get_number_of_solutions(n::Int, time::Float64, ω_val::Float64, r0_val::Float64, r1_val::Float64, reps::Int)
    num_sol = zeros(Int64, reps)
    for i in 1:reps
        num_sol[i] = length(find_equilibria_series(n, [time], ω_val, r0_val, r1_val)[1][1])
    end
    return num_sol
end

"""Documentation needed"""
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

#=

function group_close_solutions(solutions; tol=1e-8)
    groups = []

    for sol in solutions
        # Check if this solution is close to an existing group
        matched = false
        for group in groups
            if any(norm(sol .- member) < tol for member in group)
                push!(group, sol)
                matched = true
                break
            end
        end
        if !matched
            push!(groups, [sol])
        end
    end

    return groups
end

groups = group_close_solutions(tracked_solutions)

println("Grouped solutions:")
for (i, group) in enumerate(groups)
    println("Group $i (size $(length(group))):")
    for sol in group
        println("  ", sol)
    end
end



omegas = [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]],
        [[2.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [2.0, 2.0]]]

r0s = [[[0.5, 0.5], [0.8, 0.8]], [[0.5, 0.5], [0.8, 0.8]],
        [[0.5, 0.5], [0.8, 0.8]], [[0.5, 0.5], [0.8, 0.8]]]

r1s = [[[0.2, 0.5], [0.2, 0.5]], [[0.2, 0.5], [0.2, 0.5]],
        [[0.2, 0.5], [0.2, 0.5]], [[0.2, 0.5], [0.2, 0.5]]]

result = parallel_find_equilibria(2, 0:0.1:10, omegas, r0s, r1s)

ω = 2.0
r0_val = 0.2
r1_val = 0.5
n = 8
T = 2π / ω
times = 0.0:0.1:0.5
result = find_equilibria_series(n, times, ω, r0_val, r1_val)
result_length = Vector{Int64}(undef, length(times))

for i in eachindex(times)
    result_length[i] = length(result[i])
end


plot(times, result_length, xlabel="t", ylabel="# solutions", label="Stable solutions", legend=:bottom)
vline!([T / 4, T / 2, 3 * T / 4, T], label=nothing) 
display(current()) 


for i in 1:length(result)
    @show i
    println(length(result[i]))
end

x1_over_time = [result[t][1][2] for t in eachindex(result)]
x2_over_time = [result[t][2][2] for t in eachindex(result)]
x3_over_time = [result[t][3][2] for t in eachindex(result)]
x4_over_time = [result[t][4][2] for t in eachindex(result)]


using Plots
plot(times, x1_over_time, xlabel="t", ylabel="x2", label="Solution 1")
plot!(times, x2_over_time, label="Solution 2")
plot!(times, x3_over_time, label="Solution 3")
plot!(times, x4_over_time, label="Solution 4")

#All time points calculated indepedently

ω = 2.0
r0_val = 0.4
r1_val = 0.5
n = 8
T = 2π / ω
times = 0.0:0.05:T
stable_solutions = Vector{Vector{Vector{Float64}}}(undef, length(times))
grad, H, q, r0_sym, r1_sym, a_sym = symbolic_potential(n)

# Make substitutions that need to be done only once
grad_0 = subs(grad, r0_sym => r0_val, r1_sym => r1_val)
H_0 = subs(H, r0_sym => r0_val, r1_sym => r1_val)
S = System(grad_0, variables = q, parameters = [a_sym])
real_len = Vector{Float64}(undef, length(times))
stable_len = Vector{Float64}(undef, length(times))
real_sols = Vector{Vector{Vector{Float64}}}(undef, length(times))
stable_sols = Vector{Vector{Vector{Float64}}}(undef, length(times))

for i in eachindex(times)
    real_sols[i] = find_real_starting_points(times[i], ω, S)
    real_len[i] = length(real_sols[i])
    H_solve = subs(H_0, a_sym => sin(ω * times[i]))
    info = map(x_sol -> (x_sol, is_minimum(x_sol, H_solve, q)), real_sols[i])
    stable_real_sols = [x for (x, is_stable) in info if is_stable]
    stable_sols[i] = stable_real_sols
    stable_len[i] = length(stable_real_sols)
end 

#rounded_sols = map(x -> map(y -> round.(y; digits=2), x), stable_sols)
#println(rounded_sols)

using Plots

plot(times, real_len, xlabel="t", ylabel="# solutions", label="Real solutions")
plot!(times, stable_len, label="Stable solutions")
vline!([T / 4, T / 2, 3 * T / 4, T], label=nothing) 
display(current())

using Plots
using LaTeXStrings
ω = 2.0
r0_val = 0.0
r1_val = LinRange(0.5, 0.6, 200)
n = 8
T = 2π / ω
times = 0.0:0.05:0.0
result_length = Vector{Int64}(undef, length(r1_val))
for i in eachindex(r1_val)
    result_length[i] = length(find_equilibria_series(n, times, ω, r0_val, r1_val[i])[1])
end

plot(r1_val, result_length, xlabel=L"r_1", ylabel="# solutions", label="Stable solutions", ylims=(0,18))

=#


