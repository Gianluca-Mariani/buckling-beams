"""
This script defines classes to simulate the propagation of a soliton along buckling beam arrays and logical gates
"""

# Import packages
import diffrax as dfx
import jax.numpy as jnp
import jax


class beam_array:
    """
    Propagates a soliton through a beam array, given the physical parameters and boundary condition
    Requires diffrax, jax and jax.numpy
    """

    def __init__(self, A, epsilon, b, gamma, n, xi):
        """Initialize the physical parameters"""
        self.A = A  # Compression/decompression amplitude
        self.epsilon = epsilon  # Distance from critical buckling point
        self.b = b  # Nonlinear cubic parameter
        self.gamma = gamma  # Beam coupling constant
        self.n = n  # Number of 4-beam units
        self.xi = xi  # Parameters magnification factor between successive elements

    def ODE(self, t, x, args):
        """Defines the system of ODEs for the nonlinear solver"""
        i, x0 = args
        dxdt = jnp.zeros(4 * self.n)  # Immutable preallocated array

        def body(j, dxdt):
            # Connecting beam
            dxdt = dxdt.at[4 * j].set(
                jax.lax.cond(
                    j == 0,
                    lambda _: -self.xi ** (4 * j)
                    * (self.A * (1 - (-1) ** i) + self.epsilon)
                    * x[4 * j]
                    - self.xi ** (4 * j) * self.b * x[4 * j] ** 3
                    - self.gamma * (2 * x[4 * j] - x0 - x[4 * j + 1]),
                    lambda _: -self.xi ** (4 * j)
                    * (self.A * (1 - (-1) ** i) + self.epsilon)
                    * x[4 * j]
                    - self.xi ** (4 * j) * self.b * x[4 * j] ** 3
                    - self.gamma * (2 * x[4 * j] - x[4 * j - 1] - x[4 * j + 1]),
                    operand=None,
                )
            )

            # Encoding beam
            dxdt = dxdt.at[4 * j + 1].set(
                self.xi ** (4 * j + 1)
                * (self.A * (1 + (-1) ** (i + 1)) + self.epsilon)
                * x[4 * j + 1]
                - self.xi ** (4 * j + 1) * self.b * x[4 * j + 1] ** 3
                - self.gamma * (2 * x[4 * j + 1] - x[4 * j] - x[4 * j + 2])
            )

            # Connecting beam
            dxdt = dxdt.at[4 * j + 2].set(
                -self.xi ** (4 * j + 2)
                * (self.A * (1 - (-1) ** (i + 1)) + self.epsilon)
                * x[4 * j + 2]
                - self.xi ** (4 * j + 2) * self.b * x[4 * j + 2] ** 3
                - self.gamma * (2 * x[4 * j + 2] - x[4 * j + 1] - x[4 * j + 3])
            )

            # Encoding beam
            dxdt = dxdt.at[4 * j + 3].set(
                jax.lax.cond(
                    j == self.n - 1,
                    lambda _: self.xi ** (4 * j + 3)
                    * (self.A * (1 + (-1) ** i) + self.epsilon)
                    * x[4 * j + 3]
                    - self.xi ** (4 * j + 3) * self.b * x[4 * j + 3] ** 3
                    - self.gamma * (x[4 * j + 3] - x[4 * j + 2]),
                    lambda _: self.xi ** (4 * j + 3)
                    * (self.A * (1 + (-1) ** i) + self.epsilon)
                    * x[4 * j + 3]
                    - self.xi ** (4 * j + 3) * self.b * x[4 * j + 3] ** 3
                    - self.gamma * (2 * x[4 * j + 3] - x[4 * j + 2] - x[4 * j + 4]),
                    operand=None,
                )
            )

            return dxdt

        dxdt = jax.lax.fori_loop(0, self.n, body, dxdt)
        return dxdt

    def solve(self, i_stop, x0_array, i0=0, init="Down", print_b=False):
        """Solves the system of nonlinear ODE at different pseudo times i"""

        def set_initial_up(j, y0):
            """Sets all encoding beams initially buckling up"""
            y0 = y0.at[4 * j].set(0)
            y0 = y0.at[4 * j + 1].set(jnp.sqrt(self.epsilon / self.b))
            y0 = y0.at[4 * j + 2].set(0)
            y0 = y0.at[4 * j + 3].set(jnp.sqrt(self.epsilon / self.b))
            return y0

        def set_initial_down(j, y0):
            """Sets all encoding beams initially buckling down"""
            y0 = y0.at[4 * j].set(0)
            y0 = y0.at[4 * j + 1].set(-jnp.sqrt(self.epsilon / self.b))
            y0 = y0.at[4 * j + 2].set(0)
            y0 = y0.at[4 * j + 3].set(-jnp.sqrt(self.epsilon / self.b))
            return y0

        # Sets the initial array condition
        y0 = jnp.zeros(4 * self.n)
        if init == "Up":
            y0 = jax.lax.fori_loop(0, self.n, set_initial_up, y0)
        elif init == "Down":
            y0 = jax.lax.fori_loop(0, self.n, set_initial_down, y0)
        else:
            raise ValueError("Initial condition not supported!")

        # Define the solver
        solver = dfx.Tsit5()

        for i in range(i_stop):
            args = (i + i0, x0_array[i])
            term = dfx.ODETerm(self.ODE)
            solver_state = dfx.diffeqsolve(
                term,
                solver,
                t0=0.0,
                t1=100.0,
                dt0=1e-2,
                y0=y0,
                stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-8),
                args=args,
            )
            # Extract result
            x_final = solver_state.ys[-1]
            if print_b:
                rounded_values = jnp.round(x_final, 2)
                formatted_values = [f"{float(val):+.2f}" for val in rounded_values]
                print("Stationary points found at:", formatted_values)
            y0 = x_final


class gate:
    """
    Propagates two signals through a logical gate
    Requires diffrax, jax and jax.numpy
    """

    def __init__(self, A, epsilon, b, gamma, F):
        self.A = A  # Compression/decompression amplitude
        self.epsilon = epsilon  # Distance from critical buckling point
        self.b = b  # Nonlinear cubic parameter
        self.gamma = gamma  # Beam coupling constant
        self.F = F  # Offset force on the gate beam

    def ODE(self, t, x, args):
        """Defines the system of ODEs for the nonlinear solver"""
        i, x0 = args
        x0A = x0[0]
        x0B = x0[1]

        # Connecting beam A
        dxdt0 = (
            -(self.A * (1 - (-1) ** i) + self.epsilon) * x[0]
            - self.b * x[0] ** 3
            - self.gamma * (2 * x[0] - x0A - x[2])
        )

        # Connecting beam B
        dxdt1 = (
            -(self.A * (1 - (-1) ** i) + self.epsilon) * x[1]
            - self.b * x[1] ** 3
            - self.gamma * (2 * x[1] - x0B - x[2])
        )

        # Gate (encoding) beam
        dxdt2 = (
            (self.A * (1 + (-1) ** (i + 1)) + self.epsilon) * x[2]
            - self.b * x[2] ** 3
            - self.gamma * (3 * x[2] - x[0] - x[1] - x[3])
            - self.F
        )

        # Connecting beam
        dxdt3 = (
            -(self.A * (1 - (-1) ** (i + 1)) + self.epsilon) * x[3]
            - self.b * x[3] ** 3
            - self.gamma * (2 * x[3] - x[2] - x[4])
        )

        # Encoding beam
        dxdt4 = (
            (self.A * (1 + (-1) ** i) + self.epsilon) * x[4]
            - self.b * x[4] ** 3
            - self.gamma * (x[4] - x[3])
        )

        return jnp.array([dxdt0, dxdt1, dxdt2, dxdt3, dxdt4])

    def solve(self, i_stop, x0_array, i0=0, init="Down", print_b=False):
        """Solves the system of nonlinear ODE at different pseudo times i"""

        if init == "Up":
            y00 = 0
            y01 = 0
            y02 = jnp.sqrt(self.epsilon / self.b)
            y03 = 0
            y04 = jnp.sqrt(self.epsilon / self.b)
            y0 = jnp.array([y00, y01, y02, y03, y04])
        elif init == "Down":
            y00 = 0
            y01 = 0
            y02 = -jnp.sqrt(self.epsilon / self.b)
            y03 = 0
            y04 = -jnp.sqrt(self.epsilon / self.b)
            y0 = jnp.array([y00, y01, y02, y03, y04])
        else:
            raise ValueError("Initial condition not supported!")

        # Define the solver
        solver = dfx.Tsit5()

        for i in range(i_stop):
            args = (i + i0, x0_array[:, i])
            term = dfx.ODETerm(self.ODE)
            solver_state = dfx.diffeqsolve(
                term,
                solver,
                t0=0.0,
                t1=100.0,
                dt0=1e-2,
                y0=y0,
                stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-8),
                args=args,
            )
            # Extract result
            x_final = solver_state.ys[-1]
            if print_b:
                print("Stationary point found at:", jnp.round(x_final, 1))
            y0 = x_final


# Constants definition
A = 1
epsilon = 0.2
b = 0.5
gamma = 1
n = 2
i_max = 10
xi = 1.45
F = 0.15

# Solve system (beam_array)
single_cell = beam_array(A=A, epsilon=epsilon, b=b, gamma=gamma, n=n, xi=xi)
i_vals = jnp.arange(i_max)
x0 = jnp.where(
    (i_vals % 4 == 0) | (i_vals % 4 == 3),
    -jnp.sqrt((2 * A + epsilon) / b),
    jnp.sqrt((2 * A + epsilon) / b),
)
single_cell.solve(i_max, x0, print_b=True)

# Solve system (gate)
# single_gate = gate(A=A, epsilon=epsilon, b=b, gamma=gamma, F=F)
# phase_shift = 0
# i_vals = jnp.arange(i_max)
# x0A = jnp.where((i_vals % 4 == 0) | (i_vals % 4 == 3), -jnp.sqrt((2*A+epsilon)/b), jnp.sqrt((2*A+epsilon)/b))
# x0B = jnp.where((i_vals % 4 == 0 + phase_shift) | (i_vals % 4 == 3 - phase_shift), -jnp.sqrt((2*A+epsilon)/b), jnp.sqrt((2*A+epsilon)/b))
# x0 = jnp.stack([x0A, x0B], axis=0)
# single_gate.solve(i_max, x0, print_b=True)
