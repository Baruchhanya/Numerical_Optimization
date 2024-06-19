import numpy as np

def line_search_minimization(f, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method='gradient', backtracking=True):
    """
    Perform line search minimization using either Gradient Descent or Newton's Method.
    Parameters:
    f : The objective function to minimize. Should return a tuple (f_val, grad, hessian).
    x0 : Initial starting point for the minimization.
    obj_tol : Tolerance for changes in the objective function value for termination.
    param_tol : Tolerance for changes in parameter values for termination.
    max_iter : Maximum allowed number of iterations.
    method : 'gradient' or 'newton'.
    backtracking : Whether to use backtracking line search for step size determination.

    Returns:
    tuple: Final parameter values, final objective value, success flag, path of parameter values, objective values.
    """

    def backtracking_wolfe(f, x, p, grad, alpha=0.01, beta=0.5):
        """
        Perform backtracking line search to satisfy Wolfe conditions.

        Parameters:
        f : The objective function to minimize.
        x : Current parameter values.
        p : Descent direction.
        grad: Current gradient.
        alpha : Wolfe condition constant for sufficient decrease.
        beta : Backtracking constant for reducing step size.

        Returns:
        float: Step size satisfying Wolfe conditions.
        """
        t = 1.0  # Initial step size
        while f(x + t * p)[0] > f(x)[0] + alpha * t * np.dot(grad, p):
            t *= beta  # Reduce step size
        return t

    x = np.array(x0, dtype=float)  # Convert initial point to numpy array
    path = [x.copy()]  # Store the path of x values
    objective_values = [f(x)[0]]  # Store the path of objective values
    flag = True
    for i in range(max_iter):
        if method == 'gradient':
            f_val, grad = f(x)[:2]
            step_size = backtracking_wolfe(f, x, -grad, grad) if backtracking else 0.01
            x_next = x - step_size * grad
        elif method == 'newton':
            f_val, grad, hessian = f(x, hessian=True)
            # Ensure Hessian is positive definite
            if np.all(np.linalg.eigvals(hessian) > 0):
                p = -np.linalg.inv(hessian).dot(grad)
            else:
                print(f'Hessian is not positive definite at iteration {i+1}. Applying Levenberg-Marquardt modification.')
                hessian += np.eye(len(hessian)) * 1e-3  # Adding a small value to the diagonal elements
                p = -np.linalg.inv(hessian).dot(grad)
            step_size = backtracking_wolfe(f, x, p, grad) if backtracking else 1.0
            x_next = x + step_size * p
        else:
            raise ValueError("Unsupported method")

        path.append(x_next.copy())
        objective_values.append(f(x_next)[0])

        # Debugging information
        print(f"Iteration {i+1}: x = {x_next}, f(x) = {f(x_next)[0]}, step_size = {step_size}")

        # Check for convergence
        if np.linalg.norm(x_next - x) < param_tol or np.abs(f(x_next)[0] - f(x)[0]) < obj_tol:
            print(f'This is the final iteration, the algorirhm does converged: {method}, Iteration {i+1}: x = {x_next}, f(x) = {f(x_next)[0]}, success flag = {flag}')
            return x_next, f(x_next)[0], flag, path, objective_values

        x = x_next
    flag = False
    print(f'This is the final iteration, the algorirhm does not converged: {method}, Iteration {i+1}: x = {x}, f(x) = {f(x)[0]}, success flag = {flag}')
    return x, f(x)[0], flag, path, objective_values
