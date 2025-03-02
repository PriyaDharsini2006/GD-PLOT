import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(0)

# True parameters of the linear model: y = m1_true * x + m2_true + noise
m1_true, m2_true = 3.5, 1.5

# Generate synthetic data
x = np.linspace(-10, 10, 100)
noise = np.random.normal(0, 1, x.shape)
y = m1_true * x + m2_true + noise

def MSE(y, y_pred):
    """
    Mean Squared Error (MSE) formula:
    MSE = (1/n) * Σ(y - y_pred)^2
    """
    return np.mean((y - y_pred) ** 2)

def linear_search(x, y, m2_fixed=m2_true, m1_range=(-16, 16), num_points=100):
    """
    Perform a linear search to find the best m1 that minimizes MSE.
    """
    m1_vals = np.linspace(m1_range[0], m1_range[1], num_points)
    mse_vals = [MSE(y, m1 * x + m2_fixed) for m1 in m1_vals]

    min_mse = min(mse_vals)
    best_m1 = m1_vals[np.argmin(mse_vals)]

    return m1_vals, mse_vals, best_m1, min_mse

# Perform linear search
m1_vals, mse_vals, best_m1_ls, min_mse_ls = linear_search(x, y)

def plot_linear_search(m1_vals, mse_vals, best_m1, min_mse):
    """
    Plot the results of the linear search.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(m1_vals, mse_vals, color='green', label='MSE vs m1 (Linear Search)')
    plt.scatter(best_m1, min_mse, color='red', label=f'Best m1: {best_m1:.4f}')
    plt.xlabel('m1')
    plt.ylabel('MSE')
    plt.title('Linear Search: MSE vs m1')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_linear_search(m1_vals, mse_vals, best_m1_ls, min_mse_ls)

def gradient_descent(x, y, learning_rate=0.001, epochs=100, initial_m1=-12, initial_m2=12):
    """
    Perform gradient descent to optimize m1 and m2.
    
    Gradient update rules:
    - grad_m1 = (-2/n) * Σ(x * (y - y_pred))
    - grad_m2 = (-2/n) * Σ(y - y_pred))
    """
    m1, m2 = initial_m1, initial_m2
    n = len(x)

    m1_history = []
    mse_history = []

    for _ in range(epochs):
        y_pred = m1 * x + m2

        # Compute gradients
        grad_m1 = (-2 / n) * np.sum(x * (y - y_pred))
        grad_m2 = (-2 / n) * np.sum(y - y_pred)

        # Update parameters
        m1 -= learning_rate * grad_m1
        m2 -= learning_rate * grad_m2

        # Store values for visualization
        m1_history.append(m1)
        mse_history.append(MSE(y, y_pred))

    return m1, m2, m1_history, mse_history

# Perform gradient descent
m1_gd, m2_gd, m1_steps, mse_steps = gradient_descent(x, y)

def plot_gradient_descent_with_steps(m1_vals, mse_vals, m1_steps, mse_steps):
    """
    Plot the results of gradient descent with gradient steps on the GD graph line.
    """
    # Evaluate MSE for the same m1 values as in linear search, using final m2 from GD
    mse_gd_vals = [MSE(y, m1 * x + m2_gd) for m1 in m1_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(m1_vals, mse_gd_vals, color='green', label='MSE vs m1 (Gradient Descent)')
    plt.scatter(m1_steps, [MSE(y, m1 * x + m2_gd) for m1 in m1_steps], color='blue', label='Gradient Steps', alpha=0.8, marker='o')
    plt.scatter(m1_steps[-1], MSE(y, m1_steps[-1] * x + m2_gd), color='red', label=f'Final m1: {m1_steps[-1]:.4f}', zorder=3)
    plt.xlabel('m1')
    plt.ylabel('MSE')
    plt.title('Gradient Descent: MSE vs m1 with Gradient Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_gradient_descent_with_steps(m1_vals, mse_vals, m1_steps, mse_steps)

# Print results
print(f"True parameters: m1 = {m1_true}, m2 = {m2_true}")
print(f"Linear Search: Best m1 = {best_m1_ls:.4f}, Minimum MSE = {min_mse_ls:.4f}")
print(f"Gradient Descent: Final m1 = {m1_gd:.4f}, Minimum MSE = {min(mse_steps):.4f}")
