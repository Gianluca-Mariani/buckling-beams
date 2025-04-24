using Test
using Symbolics
using LinearAlgebra

include("find_equilibria.jl")


# Sample test for is_minimum function
function test_is_minimum()
    # Define the solution and Hessian
    x_sol = [1.0, 2.0, 3.0]
    
    # Create a Symbolic Hessian
    @variables q1 q2 q3
    H_symbolic = [q1 1.0 0.0; 1.0 q2 1.0; 0.0 1.0 q3]
    
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
    n = 3
    grad, H, q, t, r₀, r₁, ω = symbolic_potential(n)

    # Check the returned types
    @test typeof(grad) == Vector{Num}
    @test typeof(H) == Matrix{Num}
    @test typeof(q) == Vector{Num}
    @test typeof(t) == Num
    @test typeof(r₀) == Num
    @test typeof(r₁) == Num
    @test typeof(ω) == Num

end

# Run all tests
test_is_minimum()
test_symbolic_potential()