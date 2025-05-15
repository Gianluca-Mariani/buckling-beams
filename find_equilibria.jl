using DynamicPolynomials, HomotopyContinuation, LinearAlgebra, ThreadsX


# Check if a solution is a minimum based on the Hessian matrix
function is_minimum(x_sol::Vector{Float64}, H_evaluated::Matrix{<:Polynomial}, q::AbstractVector{<:DynamicPolynomials.Variable})
    """
    Vector{Float64}, Matrix{<:Polynomial}, Vector{<:DynamicPolynomials.Variable} -> Boolean
    Returns true if the Hessian matrix is positive definite at the given solution, false otherwise (also for errors).
    """
    H_num = coefficient.(subs(H_evaluated, q=>x_sol), q[1]^0) # Unwrap constant polynomials

    d = diag(H_num)
    e = diag(H_num, 1)
    H_mat = SymTridiagonal(d, e)  
    return isposdef(H_mat) # return true if the Hessian is positive definite

end


# Define symbolic potential with DynamicPolynomials variables
function symbolic_potential(n::Int)
    """
    Integer -> AbstractVector, AbstractMatrix, AbstractVector, DynamicPolynomials.Variable, DynamicPolynomials.Variable, DynamicPolynomials.Variable, DynamicPolynomials.Variable
    Returns the gradient, Hessian, and symbolic variables for the potential function given the input length of chain n.
    """
    @var r₀ r₁ a
    @var q[1:n]

    phi = i -> (i % 4 == 0) || (i % 4 == 3) ? 1.0 : -1.0
    V = sum(0.5 * ((-1)^(i-1) + r₀ * phi(i-1) * a) * q[i]^2 + 0.25 * q[i]^4 for i in 1:n)
    V += sum(0.5 * r₁ * (q[i+1] - q[i])^2 for i in 1:n-1)
    V += 0.5 * r₁ * (q[1] - q[n])^2

    grad = differentiate(V, q)
    H = differentiate(grad, q)
    return grad, H, q, r₀, r₁, a
end




# Main solver using parameter homotopy
function find_equilibria_series(n::Int, times::AbstractVector{Float64}, ω_val::Float64, r0_val::Float64, r1_val::Float64)
    # Initialize variables
    grad, H, q, r0_sym, r1_sym, a_sym = symbolic_potential(n)

    # Make substitutions that need to be done only once
    grad_0 = subs(grad, r0_sym => r0_val, r1_sym => r1_val)
    H_0 = subs(H, r0_sym => r0_val, r1_sym => r1_val)
    S = System(grad_0; variables = q, parameters = [a_sym])

    # Initial solve with complex parameter
    start_parameter = randn(ComplexF64, 1)
    result = solve(S; target_parameters = start_parameter, start_system=:total_degree)
    

    # generate some random data to simulate the parameters
    data = [[sin(ω_val * times[i])] for i in eachindex(times)]

    # track p₀ towards the entries of data
    data_points = solve(
    S,
    solutions(result);
    start_parameters = start_parameter,
    target_parameters = data,
    start_system=:total_degree,
    transform_result = (r,p) -> solutions(r)
    )
    
    return solutions(result)
end


function generate_combinations(n::Int, z::Vector{ComplexF64})
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
    @show ω_vec

    # Form parameter list as tuples
    param_list = [(ω, r0, r1) for (ω, r0, r1) in zip(ω_vec, r0_vec, r1_vec)]

    # Map over parameter list
    results_vec = ThreadsX.map(((ω, r0, r1),) -> find_equilibria_series(n, times, ω, r0, r1), param_list)

    # Reshape back to original shape
    results = reshape(results_vec, size(omega))
    return results
end


num_sol = zeros(Int64, 100)
ω = 2.0
r0_val = 0.5
r1_val = 0.0
n = 4
T = 2π / ω
times = 0.0:0.1:0.0
for i in eachindex(num_sol)
    num_sol[i] = length(find_equilibria_series(n, times, ω, r0_val, r1_val))
end
using Plots
histogram(num_sol, bins= 9, xlabel="Number of solutions", ylabel="Frequency")
#=

@var y[1:n]
grad, H, q, r0_sym, r1_sym, a_sym = symbolic_potential(n)
grad_0 = subs(grad, r0_sym => r0_val, r1_sym => r1_val, a_sym => sin(ω * times[18 ]))
F = System(grad_0; variables = q)

# construct start system and homotopy
G = System(im * (y.^3 .- 1))
H = StraightLineHomotopy(G, F)
z = [1, exp(2im * π / 3), exp(-2im * π / 3)]
start_solutions = generate_combinations(n, z)
# construct tracker
tracker = Tracker(H)
# track each start solution separetely
results = track.(tracker, start_solutions)
println("# successfull: ", count(is_success, results))



omegas = [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]],
        [[2.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [2.0, 2.0]]]

r0s = [[[0.5, 0.5], [0.8, 0.8]], [[0.5, 0.5], [0.8, 0.8]],
        [[0.5, 0.5], [0.8, 0.8]], [[0.5, 0.5], [0.8, 0.8]]]

r1s = [[[0.2, 0.5], [0.2, 0.5]], [[0.2, 0.5], [0.2, 0.5]],
        [[0.2, 0.5], [0.2, 0.5]], [[0.2, 0.5], [0.2, 0.5]]]

result = parallel_find_equilibria(2, 0:0.1:10, omegas, r0s, r1s)

using Plots
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


