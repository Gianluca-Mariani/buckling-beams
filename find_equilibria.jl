# dynamic_lagrangian_equilibria.jl

using Symbolics
using HomotopyContinuation

function build_lagrangian_gradient(n::Int)
    @variables t ω r₀ r₁
    @variables q[1:n]  # dynamic vector of q₁, q₂, ..., qₙ

    # === Build the components ===

    # time_factor = (-1)^i + r0 * sin(ω * t + phi(i))
    phi = i -> (i % 4 == 0) || (i % 4 == 3) ? 0.0 : π  # example with if condition
    time_factor = [(-1)^i + r₀ * sin(ω * t + phi(i)) for i in 1:n]

    potential_quad = 0.5 * sum(time_factor[i] * q[i]^2 for i in 1:n)
    potential_quartic = 0.25 * sum(q[i]^4 for i in 1:n)

    # q_shifted is q rolled by +1 (circular shift)
    q_shifted = [q[mod1(i - 1, n)] for i in 1:n]  # mod1: Julia is 1-based
    potential_coupling = 0.5 * r₁ * sum((q[i] - q_shifted[i])^2 for i in 1:n)

    # Lagrangian (actually negative potential)
    V = potential_quad + potential_quartic + potential_coupling
    gradV = Symbolics.gradient(V, q)

    return -gradV, q, (t, ω, r₀, r₁)
end

function find_equilibria(n::Int, t_val, ω_val, r0_val, r1_val)
    grad, q_syms, (t, ω, r₀, r₁) = build_lagrangian_gradient(n)
    println(typeof(grad))

    # Substitute numeric values
    subs = Dict(t => t_val, ω => ω_val, r₀ => r0_val, r₁ => r1_val)

    # Apply substitution to all elements of grad using a more direct map function
    grad_eval = map(e -> substitute(e, subs), grad)

    # Now build the HC.jl system
    @polyvar q[1:n]
    funcs = [eval(build_function(grad_eval[i], q)[1]) for i in 1:n]

    result = solve(funcs)
    real_solutions = [sol for sol in result if isreal(sol)]
    return [real.(s) for s in real_solutions]
end

# Example call
find_equilibria(5, 1.0, 2.0, 0.5, 1.5)

