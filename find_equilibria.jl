# dynamic_lagrangian_equilibria.jl

using Symbolics, DynamicPolynomials, HomotopyContinuation, LinearAlgebra, ThreadsX

using Base.Threads: @threads
using SharedArrays  # or just a regular array if memory is not an issue

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

    tol = 1e-5  # tolerance for "almost real"
    sols = solutions(result)
    real_sols = [sol for sol in sols if all(x -> abs(imag(x)) < tol, sol)]  # accept near-real
    x_sols = [Float64[real(x[i]) for i in 1:n] for x in real_sols]


    # 7. Filter out unstable solutions
    stability_info = ThreadsX.map(x_sol -> (x_sol, is_minimum(x_sol, H_eval, q)), x_sols)
    stable_solutions = [x for (x, is_stable) in stability_info if is_stable]
    
    return stable_solutions
end


times = 0:1:10  # Example time points

function find_stable_points(t)
    r0_val = 0.5
    r1_val = 0.0
    ω_val = 2.0
    n = 2
    return find_equilibria(n, t, ω_val, r0_val, r1_val)
end


# Preallocate the array for stable solutions
stable_solutions = Vector{Vector{Vector{Float64}}}(undef, length(times))

for i in eachindex(times)
    t = times[i]
    stable_solutions[i] = find_stable_points(t)
end


for i in eachindex(stable_solutions)
    println("Length of stable solutions at time $i: ", length(stable_solutions[i]))   
end

function build_paths(stable_solutions)
    n_timepoints = length(stable_solutions)
    n_paths = length(stable_solutions[1])  # assume constant number for now

    paths = [Vector{Vector{Float64}}(undef, n_timepoints) for _ in 1:n_paths]

    # Initialize paths with first timepoint
    for j in 1:n_paths
        paths[j][1] = stable_solutions[1][j]
    end

    # For each following timepoint, assign each solution to closest from previous step
    for i in 2:n_timepoints
        prev_sols = paths .|> x -> x[i - 1]
        curr_sols = copy(stable_solutions[i])
        used = falses(length(curr_sols))

        for j in 1:n_paths
            prev = prev_sols[j]
            # Find closest unused solution
            dists = [norm(sol - prev) for sol in curr_sols]
            for k in sortperm(dists)
                if !used[k]
                    paths[j][i] = curr_sols[k]
                    used[k] = true
                    break
                end
            end
        end
    end

    return paths
end

sorted_paths = build_paths(stable_solutions)