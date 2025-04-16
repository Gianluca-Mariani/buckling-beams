# dynamic_lagrangian_equilibria.jl

using Symbolics
using HomotopyContinuation

function build_lagrangian_gradient(n::Int)
    @variables t ω r₀ r₁
    @variables q[1:n]  # dynamic vector of q₁, q₂, ..., qₙ

    # === Build the components ===

    # time_factor = (-1)^i + r0 * sin(ω * t + phi(i))
    phi = i -> π * i / n  # example phase shift function
    time_factor = [(-1)^i + r₀ * sin(ω * t + phi(i)) for i in 1:n]

    potential_quad = 0.5 * sum(time_factor[i] * q[i]^2 for i in 1:n)
    potential_quartic = 0.25 * sum(q[i]^4 for i in 1:n)

    # q_shifted is q rolled by +1 (circular shift)
    q_shifted = [q[mod1(i - 1, n)] for i in 1:n]  # mod1: Julia is 1-based
    potential_coupling = 0.5 * r₁ * sum((q[i] - q_shifted[i])^2 for i in 1:n)

    # Lagrangian (actually negative potential)
    V = potential_quad + potential_quartic + potential_coupling
    gradV = Symbolics.gradient(V, q)

    return gradV, q, (t, ω, r₀, r₁)
end