using DynamicPolynomials, HomotopyContinuation, LinearAlgebra

"""
Data types used:

Vector{Float64}: 1D array of Float64, short for Array{Float64, 1}, included in base Julia
Num: a symbolic number, a polynomial, or an abstract number, included in the DynamicPolynomials package
Matrix{Num}: 2D array of Num, short for Array{Num, 2}
AbstractVector{<:Num}: a vector of Num or one of its subtypes (<: denotes subtypes), short for Array{Num, N} where N is an integer
AbstractVector: a Julia abstract type for vectors, which can be of any subtype
"""


# Check if a solution is a minimum based on the Hessian matrix
function is_minimum(x_sol::Vector{Float64}, H_evaluated::Matrix{Num}, q::AbstractVector{<:Num})
    """
    Vector{Float64}, Matrix{Num}, Vector{<:Num} -> Boolean
    Returns true if the Hessian matrix is positive definite at the given solution, false otherwise (also for errors).
    """
    q_vals = Dict(q[i] => x_sol[i] for i in eachindex(q)) # Create a dictionary for linking q variables to the numerical values of the solution
    H_num = Symbolics.substitute.(H_evaluated, Ref(q_vals))

    try
        d = diag(H_num)
        e = diag(H_num, 1)
        H_mat = SymTridiagonal(d, e)
        return isposdef(H_mat) # return true if the Hessian is positive definite
    catch
        return false
    end
end


# Define symbolic potential with DynamicPolynomials variables
function symbolic_potential(n::Int)
    """
    Integer -> AbstractVector, AbstractMatrix, AbstractVector, DynamicPolynomials.Variable, DynamicPolynomials.Variable, DynamicPolynomials.Variable, DynamicPolynomials.Variable
    Returns the gradient, Hessian, and symbolic variables for the potential function given the input length of chain n.
    """
    @polyvar r₀ r₁ a
    @polyvar q[1:n]

    phi = i -> (i % 4 == 0) || (i % 4 == 3) ? 1.0 : -1.0
    V = sum(0.5 * ((-1)^(i-1) + r₀ * phi(i-1) * a) * q[i]^2 + 0.25 * q[i]^4 for i in 1:n)
    V += sum(0.5 * r₁ * (q[i+1] - q[i])^2 for i in 1:n-1)
    V += 0.5 * r₁ * (q[1] - q[n])^2

    grad = differentiate(V, q)
    H = differentiate(grad, q)
    return grad, H, q, r₀, r₁, a
end

function compute_gradient_polynomials(grad::Vector{Num}, q::Vector{Num}, r0_sym::Num, r1_sym::Num, ω_sym::Num, t_sym::Num)
    # Define symbolic variables (Symbolics.jl style)
    @variables x[1:length(q)] r0 r1 ω t

    # Substitutions
    q_subs = Dict(q[i] => x[i] for i in eachindex(q))
    param_subs = Dict(r0_sym => r0, r1_sym => r1, ω_sym => ω, t_sym => t)

    grad_subs = Symbolics.substitute.(grad, Ref(param_subs))
    grad_polys = Symbolics.substitute.(grad_subs, Ref(q_subs))

    return grad_polys, x, [r0, r1, ω, t]
end

# Function to substitute values correctly for DynamicPolynomials for Hessian
function evaluate_hessian(H::Matrix{Num}, q::Vector{Num}, t_sym::Num, r0_sym::Num, r1_sym::Num, ω_sym::Num, t_val, r0_val, r1_val, ω_val)
    # Create DynamicPolynomials variables
    @polyvar x[1:length(q)] t

    # Construct substitution dictionary for parameters
    param_subs = Dict(t_sym => t_val, r0_sym => r0_val, r1_sym => r1_val, ω_sym => ω_val)
    H1 = Symbolics.substitute.(H, Ref(param_subs))

    return H1, x
end

# Main solver using parameter homotopy
function find_equilibria_series(n, times, ω_val, r0_val, r1_val)
    grad, H, q, t_sym, r0_sym, r1_sym, ω_sym = symbolic_potential(n)
    stable_solutions = Vector{Vector{Vector{Float64}}}(undef, length(times))

    println("Initial step")
    grad_polys, x_vars, param_vars = compute_gradient_polynomials(grad, q, r0_sym, r1_sym, ω_sym, t_sym)

    # Initial solve
    F = symbolic_system(grad_polys, x_vars, parameters=param_vars)
    param_start = [r0_val, r1_val, ω_val, times[1]]
    result = solve(F; target_parameters=param_start)

    tol = 1e-5
    sols = solutions(result)
    real_sols = [sol for sol in sols if all(x -> abs(imag(x)) < tol, sol)]
    x_sols = [Float64[real(x[j]) for j in 1:n] for x in real_sols]

    H_eval = evaluate_hessian(H, q, t_sym, r0_sym, r1_sym, ω_sym, times[1], r0_val, r1_val, ω_val)
    info = ThreadsX.map(x_sol -> (x_sol, is_minimum(x_sol, H_eval, q)), x_sols)
    stable_real_sols = [x for (x, is_stable) in info if is_stable]

    stable_solutions[1] = stable_real_sols

    println("Tracking parameter homotopy")
    for (i, t_val) in enumerate(times[2:end])
        grad_polys, _, param_vars = compute_gradient_polynomials(grad, q, r0_sym, r1_sym, ω_sym, t_sym)
        F = symbolic_system(grad_polys, x_vars, parameters=param_vars)

        param_start = [r0_val, r1_val, ω_val, times[i]]
        param_target = [r0_val, r1_val, ω_val, t_val]

        tracked = track(F, stable_real_sols, Dict(param_vars .=> param_start), Dict(param_vars .=> param_target))
        sols = solutions(tracked)

        real_sols = [sol for sol in sols if all(x -> abs(imag(x)) < tol, sol)]
        x_sols = [Float64[real(x[j]) for j in 1:n] for x in real_sols]

        H_eval = evaluate_hessian(H, q, t_sym, r0_sym, r1_sym, ω_sym, t_val, r0_val, r1_val, ω_val)
        info = ThreadsX.map(x_sol -> (x_sol, is_minimum(x_sol, H_eval, q)), x_sols)
        stable_real_sols = [x for (x, is_stable) in info if is_stable]

        stable_solutions[i+1] = stable_real_sols
    end

    return stable_solutions
end

# === Parameters and Execution ===
n = 2
r0_val = 0.5
r1_val = 0.0
ω_val = 2.0
times = 0:1:10

#stable_solutions = find_equilibria_series(n, times, ω_val, r0_val, r1_val)