using DynamicPolynomials, HomotopyContinuation, LinearAlgebra, ThreadsX


# Check if a solution is a minimum based on the Hessian matrix
function is_minimum(x_sol::Vector{Float64}, H_evaluated::Matrix{<:Polynomial}, q::AbstractVector{<:DynamicPolynomials.Variable})
    """
    Vector{Float64}, Matrix{<:Polynomial}, Vector{<:DynamicPolynomials.Variable} -> Boolean
    Returns true if the Hessian matrix is positive definite at the given solution, false otherwise (also for errors).
    """
    #q_vals = Dict(q[i] => x_sol[i] for i in eachindex(q)) # Create a dictionary for linking q variables to the numerical values of the solution
    #H_num = map(p -> subs(p, q_vals), H_evaluated) # Apply substitution element-wise

    H_num = coefficient.(subs(H_evaluated, q=>x_sol), q[1]^0) # Unwrap constant polynomials

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


# Main solver using parameter homotopy
function find_equilibria_series(n::Int, times, ω_val::Float64, r0_val::Float64, r1_val::Float64)
    # Initialize variables
    stable_solutions = Vector{Vector{Vector{Float64}}}(undef, length(times))
    grad, H, q, r0_sym, r1_sym, a_sym = symbolic_potential(n)

    # Make substitutions that need to be done only once
    grad_0 = subs(grad, r0_sym => r0_val, r1_sym => r1_val)
    H_0 = subs(H, r0_sym => r0_val, r1_sym => r1_val)
    S = System(grad_0, variables = q, parameters = [a_sym])

    println("Initial step")

    # The first homotopy won't always find all the solutions, so we repeat it until we find all of them
    while true
        # Initial solve
        result = solve(S; target_parameters = [sin(ω_val * times[1])])
        real_sols = real_solutions(result)

        H_solve = subs(H_0, a_sym => sin(ω_val * times[1]))
        info = ThreadsX.map(x_sol -> (x_sol, is_minimum(x_sol, H_solve, q)), real_sols)
        stable_real_sols = [x for (x, is_stable) in info if is_stable]
        stable_solutions[1] = stable_real_sols
        length(stable_real_sols) === Int(2^(n / 2)) && break
    end

    println("Tracking parameter homotopy")
    for (i, t_val) in enumerate(times[2:end])
        result = solve(S, stable_solutions[i]; start_parameters = [sin(ω_val * times[i])], target_parameters = [sin(ω_val * t_val)])
        stable_real_sols = real_solutions(result)
        stable_solutions[i+1] = stable_real_sols
    end

    return stable_solutions
end


function parallel_find_equilibria(n::Int, times, ω_matrx::Float64, r0_matrx::Float64, r1_matrx::Float64)


end