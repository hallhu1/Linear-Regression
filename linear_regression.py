from numpy import *
from matplotlib import pyplot as plt

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def step_gradient(b_current, m_current, points, learning_rate):
    # starting points for the gradient
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # calculate partial derivatives for gradient
        # gradient is the steepest increase in the error function
        b_gradient += -(2/n) * x * (y - ((m_current * x) + b_current))
        m_gradient += -(2/n) * (y - ((m_current * x) + b_current))
    # update the parameters, go in the opposite direction of the gradient
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def run():
    # Load the data
    points = genfromtxt("data.csv", delimiter=",")

    # Hyperparameters
    learning_rate = 0.0001
    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    # Train the model
    print(f"Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {compute_error_for_line_given_points(initial_b, initial_m, points)}")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(f"After {num_iterations} iterations b = {b}, m = {m}, error = {compute_error_for_line_given_points(b, m, points)}")

    # Plot the data
    plt.scatter(points[:, 0], points[:, 1], color="red")
    plt.plot(points[:, 0], points[:, 0] * m + b, color="blue")
    plt.title("Linear Regression")
    plt.legend()
    plt.savefig("linear_regression.png")
    plt.show()
    plt.close()
    

if __name__ == "__main__":
    exit(run())
