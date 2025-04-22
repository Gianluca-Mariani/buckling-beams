# dynamic_lagrangian_equilibria.jl

using Symbolics
using HomotopyContinuation
using DynamicPolynomials

function build_lagrangian_gradient(n::Int)
    @variables t ω r₀ r₁
    @variables q[1:n]  # dynamic vector of q₁, q₂, ..., qₙ

    # === Build the components ===

    # time_factor = (-1)^i + r0 * sin(ω * t + phi(i))
    phi = i -> (i % 4 == 0) || (i % 4 == 3) ? 0.0 : π  # example with if condition
    time_factor = [(-1)^(i-1) + r₀ * sin(ω * t + phi(i-1)) for i in 1:n]

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

using Symbolics, DynamicPolynomials, HomotopyContinuation

function find_equilibria(n, t_val, ω_val, r0_val, r1_val)
    # 1. Define symbolic variables
    @variables t r₀ r₁ ω
    @variables q[1:n]

    # 2. Construct the potential
    phi = i -> (i % 4 == 0) || (i % 4 == 3) ? 0.0 : π  # example with if condition
    V = sum(0.5 * ((-1)^(i-1) + r₀ * sin(ω * t + phi(i-1))) * q[i]^2 + 0.25 * q[i]^4 for i in 1:n)
    V += sum(0.5 * r₁ * (q[i+1] - q[i])^2 for i in 1:n-1)
    V += 0.5 * r₁ * (q[1] - q[n])^2

    # 3. Compute gradient symbolically
    grad = Symbolics.gradient(V, q)

    # 4. Substitute parameters
    subs = Dict(t => t_val, r₀ => r0_val, r₁ => r1_val, ω => ω_val)
    grad_eval = Symbolics.substitute.(grad, Ref(subs))  # Resulting in an array of symbolic expressions

    # 5. Reconstruct as polynomials using DynamicPolynomials
    @polyvar x[1:n]
    q_subs = Dict(q[i] => x[i] for i in 1:n)
    grad_polys = [Symbolics.substitute(g, q_subs) for g in grad_eval]  # Direct substitution into polynomials

    # Flatten the list of polynomials (if necessary)
    grad_polys_flat = vcat(grad_polys...)  # This flattens the array

    # 6. Solve using HomotopyContinuation
    result = solve(grad_polys_flat)
    return result
end

# Example call
find_equilibria(2, 1.0, 2.0, 0.5, 0.0)
