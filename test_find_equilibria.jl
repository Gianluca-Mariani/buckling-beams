using Test
using DynamicPolynomials
using LinearAlgebra

include("find_equilibria.jl")


# Sample test for is_minimum function
function test_is_minimum()
    # Define the solution and Hessian
    x_sol = [1.0, 2.0, 3.0]
    
    # Create a Symbolic Hessian
    @polyvar q1 q2 q3
    H_symbolic = polynomial.([q1 1.0 0.0; 1.0 q2 1.0; 0.0 1.0 q3])
    
    # Test case 1: Hessian is positive definite
    result = is_minimum(x_sol, H_symbolic, [q1, q2, q3])
    @test result == true
    
    # Test case 2: Hessian is not positive definite
    H_symbolic = [q1 - 1 1.0 0.0; 1.0 q2 - 1 1.0; 0.0 1.0 q3 - 1]  # Not positive definite
    result = is_minimum(x_sol, H_symbolic, [q1, q2, q3])
    @test result == false
    
    # Test case 3: Catching errors in Hessian substitution
    H_symbolic = [q1 + q2 1.0; 1.0 q3 + 2.0]  # Simple symbolic example
    
    result = is_minimum(x_sol, H_symbolic, [q1, q2, q3])
    @test result == true
    
    # Test case 4: Invalid input handling
    @test_throws MethodError begin
        is_minimum(x_sol, "invalid_hessian", [q1, q2, q3])  # Passing invalid Hessian input
    end
end


# Define tests for the symbolic_potential function
function test_symbolic_potential()
    n = 2
    grad, H, q, r₀, r₁, a = symbolic_potential(n)

    # Check the returned types
    @test grad isa AbstractVector
    @test H isa AbstractMatrix
    @test q isa AbstractVector
    @test r₀ isa DynamicPolynomials.Variable
    @test r₁ isa DynamicPolynomials.Variable
    @test a isa DynamicPolynomials.Variable

    grad_test = [(1 + r₀ * a) * q[1] + q[1]^3 + 2.0r₁ * (q[1] - q[2]), 
                 -(1 + r₀ * a) * q[2] + q[2]^3 + 2.0r₁ * (q[2] - q[1])]

    H_test = [
    (1 + r₀ * a) + 3.0* q[1]^2 + 2.0 * r₁ -2.0 * r₁;
    -2.0 * r₁ -(1 + r₀ * a) + 3.0 * q[2]^2 + 2.0 * r₁
    ]

            
    @test grad == grad_test
    @test H == H_test

end

function test_find_equilibria_series()
    n = 2
    times = 0:0.1:10
    ω_val = 1.0
    r0_val = 0.5
    r1_val = 0.0

    # Call the function
    stable_solutions = find_equilibria_series(n, times, ω_val, r0_val, r1_val)
    x1_over_time = [stable_solutions[t][1][2] for t in eachindex(stable_solutions)]
    x2_over_time = [stable_solutions[t][2][2] for t in eachindex(stable_solutions)]
    x10_over_time = [stable_solutions[t][1][1] for t in eachindex(stable_solutions)]
    x20_over_time = [stable_solutions[t][2][1] for t in eachindex(stable_solutions)]
    x1_analytical = sqrt.(1 .+ r0_val .* sin.(ω_val .* times))
    x2_analytical = -sqrt.(1 .+ r0_val .* sin.(ω_val .* times))
    x0_analytical = zeros(length(times))

    # Check the output type
    @test stable_solutions isa Vector{Vector{Vector{Float64}}}

    # Check the length of the output
    @test length(stable_solutions) == length(times)

    # Check the values of the solutions
    # This first test is guaranteed to work only with n < 4. With n = 4, there will be at least 4 solutions
    # Since the order of the solutions cannot be predicted, we would have to check all possible combinations
    @test (isapprox(x1_over_time, x1_analytical) && isapprox(x2_over_time, x2_analytical)) || (isapprox(x1_over_time, x2_analytical) && isapprox(x2_over_time, x1_analytical))
    @test isapprox(x10_over_time, x0_analytical) && isapprox(x20_over_time, x0_analytical)

end

# Run all tests
test_is_minimum()
test_symbolic_potential()
test_find_equilibria_series()
