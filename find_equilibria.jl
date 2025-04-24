using Symbolics, DynamicPolynomials, HomotopyContinuation, LinearAlgebra

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
function symbolic_potential(n)
    @variables t r₀ r₁ ω
    @variables q[1:n]
    q = collect(q)

    phi = i -> (i % 4 == 0) || (i % 4 == 3) ? 0.0 : π
    V = sum(0.5 * ((-1)^(i-1) + r₀ * sin(ω * t + phi(i-1))) * q[i]^2 + 0.25 * q[i]^4 for i in 1:n)
    V += sum(0.5 * r₁ * (q[i+1] - q[i])^2 for i in 1:n-1)
    V += 0.5 * r₁ * (q[1] - q[n])^2

    grad = Symbolics.gradient(V, q)
    H = Symbolics.hessian(V, q)
    return grad, H, q, t, r₀, r₁, ω
end

# Function to substitute values correctly for DynamicPolynomials
function compute_gradient_polynomials(grad, q, t_sym, r0_sym, r1_sym, ω_sym, r0_val, r1_val, ω_val, t_val)
    # Create DynamicPolynomials variables
    @polyvar x[1:length(q)] t

    # Construct substitution dictionary for parameters
    param_subs = Dict(r0_sym => r0_val, r1_sym => r1_val, ω_sym => ω_val, t_sym => t_val)
    grad_subs = Symbolics.substitute.(grad, Ref(param_subs))

    # Map q variables to DynamicPolynomials
    q_subs = Dict(q[i] => x[i] for i in eachindex(q))
    grad_polys = Symbolics.substitute.(grad_subs, Ref(q_subs))

    return grad_polys, x
end

# Main solver using parameter homotopy
function find_equilibria_series(n, times, ω_val, r0_val, r1_val)
    grad, H, q, t_sym, r0_sym, r1_sym, ω_sym = symbolic_potential(n)
    stable_solutions = Vector{Vector{Vector{Float64}}}(undef, length(times))

    println("Initial step")
    grad_polys, x_vars = compute_gradient_polynomials(grad, q, t_sym, r0_sym, r1_sym, ω_sym, r0_val, r1_val, ω_val, times[1])
    
    # Ensure symbolic variables are properly passed to HomotopyContinuation
    F = System(grad_polys, variables=x_vars, parameters=[t_sym])
    result = solve(F; start_parameters=[times[1]], target_parameters=[times[1]])

    tol = 1e-5
    sols = solutions(result)
    real_sols = [sol for sol in sols if all(x -> abs(imag(x)) < tol, sol)]
    x_sols = [Float64[real(x[j]) for j in 1:n] for x in real_sols]

    H_eval = compute_Hessian(H, t_sym, r0_sym, r1_sym, ω_sym, times[1], r0_val, r1_val, ω_val)
    info = ThreadsX.map(x_sol -> (x_sol, is_minimum(x_sol, H_eval, q)), x_sols)
    stable_real_sols = [x for (x, is_stable) in info if is_stable]
    stable_solutions[1] = stable_real_sols

    println("Tracking parameter homotopy")
    for (i, t_val) in enumerate(times[2:end])
        grad_polys, _ = compute_gradient_polynomials(grad, q, t_sym, r0_sym, r1_sym, ω_sym, r0_val, r1_val, ω_val, t_val)
        F = System(grad_polys, variables=x_vars)
        result = solve(F, start_solutions=stable_real_sols)
        
        sols = solutions(result)
        real_sols = [sol for sol in sols if all(x -> abs(imag(x)) < tol, sol)]
        x_sols = [Float64[real(x[j]) for j in 1:n] for x in real_sols]

        H_eval = compute_Hessian(H, t_sym, r0_sym, r1_sym, ω_sym, t_val, r0_val, r1_val, ω_val)
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