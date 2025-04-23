# dynamic_lagrangian_equilibria.jl

using Symbolics, DynamicPolynomials, HomotopyContinuation, LinearAlgebra, ThreadsX

function is_minimum(x_sol::Vector{Float64}, H_evaluated::Matrix{Num}, q::AbstractVector{<:Num})
    q_vals = Dict(q[i] => x_sol[i] for i in eachindex(q))
    H_num = Symbolics.substitute.(H_evaluated, Ref(q_vals))

    try
        d = diag(H_num)
        e = diag(H_num, 1)
        H_mat = SymTridiagonal(d, e)

        return isposdef(H_mat)
    catch 
        return false
    end
end

function find_equilibria(n, t_val, ω_val, r0_val, r1_val)
    # 1. Define symbolic variables
    @variables t r₀ r₁ ω
    @variables q[1:n]
    q = collect(q)  # This turns q into a plain Vector{Num}

    # 2. Construct the potential
    phi = i -> (i % 4 == 0) || (i % 4 == 3) ? 0.0 : π  # example with if condition
    V = sum(0.5 * ((-1)^(i-1) + r₀ * sin(ω * t + phi(i-1))) * q[i]^2 + 0.25 * q[i]^4 for i in 1:n)
    V += sum(0.5 * r₁ * (q[i+1] - q[i])^2 for i in 1:n-1)
    V += 0.5 * r₁ * (q[1] - q[n])^2

    # 3. Compute gradient and Hessian symbolically
    grad = Symbolics.gradient(V, q)
    H = Symbolics.hessian(V, q)

    # 4. Substitute parameters
    subs = Dict(t => t_val, r₀ => r0_val, r₁ => r1_val, ω => ω_val)
    grad_eval = Symbolics.substitute.(grad, Ref(subs))  # Resulting in an array of symbolic expressions
    H_eval = Symbolics.substitute.(H, Ref(subs))  # Resulting in a matrix of symbolic expressions

    # 5. Reconstruct as polynomials using DynamicPolynomials
    @polyvar x[1:n]
    q_subs = Dict(q[i] => x[i] for i in 1:n)
    grad_polys = [Symbolics.substitute(g, q_subs) for g in grad_eval]  # Direct substitution into polynomials

    # Flatten the list of polynomials (if necessary)
    grad_polys_flat = vcat(grad_polys...)  # This flattens the array

    # 6. Solve using HomotopyContinuation
    result = solve(grad_polys_flat)

    real_sols = real_solutions(result)  # returns only real-valued solutions
    x_sols = [Float64[x[i] for i in 1:n] for x in real_sols]  # convert to plain vectors

    # 7. Filter out unstable solutions
    stability_info = ThreadsX.map(x_sol -> (x_sol, is_minimum(x_sol, H_eval, q)), x_sols)
    stable_solutions = [x for (x, is_stable) in stability_info if is_stable]
    
    return stable_solutions
end

# Example call
Stable_solutions = find_equilibria(2, 0.0, 2.0, 1, 0.0)
