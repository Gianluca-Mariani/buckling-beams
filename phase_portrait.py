import numpy as np
import matplotlib.pyplot as plt

r0 = 0.5
r1 = 0.5
a = -1

# Define the system
def f(x, y):
    dx = -(1 + r0 * a) * x - x**3 - 2 * r1 * (x - y)
    dy = (1 + r0 * a) * y - y**3 - 2 * r1 * (y - x)
    return dx, dy



# Create a grid of points
x_vals = np.linspace(-2, 2, 50)
y_vals = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x_vals, y_vals)

# Evaluate the vector field
U, V = f(X, Y)
N = np.sqrt(U**2 + V**2)
U /= N
V /= N


x_array = np.linspace(-2, 2, 100)
# Plot the vector field
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, U, V, N, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase Portrait')
plt.grid(True)
plt.axis('equal')
plt.plot(x_array, ((1+r0*a)*x_array + x_array**3)/(2*r1) + x_array, 'r--', label='x=y')
plt.plot((-(1+r0*a)*x_array + x_array**3)/(2*r1) + x_array, x_array, 'r--', label='x=y')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()