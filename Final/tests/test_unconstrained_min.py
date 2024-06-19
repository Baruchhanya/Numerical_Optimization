import numpy as np
import unittest
import sys

sys.path.append("../")
sys.path.append("../srcs")

from srcs.unconsrtained_min import line_search_minimization
from srcs.utils import plot_paths, plot_function_values
from examples import (
    quadratic_circle, quadratic_ellipse, quadratic_rotated_ellipse,
    rosenbrock, linear_function, exponential_function
)

class TestUnconstrainedMinimization(unittest.TestCase):
    def _test_functions_on_examples(self, example_name: str, example_func: callable):
        lst_minimization_funcs_paramerters = [
            ('gradient', {'method': 'gradient', 'backtracking':True}),
            ('newton', {'method': 'newton', 'backtracking':True}),
        ]
        print(f'\nTesting: {example_name}')
        output_paths_plot_path = f'../out/example_{example_name}_paths.png'
        output_objective_values_plot_path = f'../out/example_{example_name}_objective_values.png'
        if example_name == 'rosenbrock':
            x0 = [-1,2]
        else:
            x0 = [1,1]

        minimizers_paths = {}
        minimizers_objective_values = {}
        minimizers_converges = {}
        all_converges = True
        for minimizer_params_name, minimizer_kwargs in lst_minimization_funcs_paramerters:
            if ('linear' in example_name) and ('newton' in minimizer_params_name):
                continue
            
            print(f'Minimizer: {minimizer_params_name}')
            result = line_search_minimization(example_func,  x0, **minimizer_kwargs)
            minimizers_paths[minimizer_params_name] = result[3]
            minimizers_objective_values[minimizer_params_name] = result[4]
            minimizers_converges[minimizer_params_name] = result[2]
        
        plot_paths(example_func, minimizers_paths, (-2, 2), (-2, 2), title=f'Contour of objective function: {example_name}',output_path=output_paths_plot_path)
        plot_function_values(minimizers_objective_values, title=f'Optimization iterations vs. Objective function values of: {example_name}', output_path=output_objective_values_plot_path)
        for minimizer_name, is_convered in minimizers_converges.items():
            self.assertTrue(is_convered, f'Minimizer: {minimizer_name}')

if __name__ == '__main__':
    lst_examples = [
        ('quadratic_circle', quadratic_circle),
        ('quadratic_ellipse', quadratic_ellipse),
        ('quadratic_rotated_ellipse', quadratic_rotated_ellipse),
        ('rosenbrock', rosenbrock),
        ('linear_function', linear_function),
        ('exponential_function', exponential_function)
    ]

    # Dynamically create test methods
    def create_test_method(example_name, example_func):
        def test_method(self):
            self._test_functions_on_examples(example_name, example_func)
        return test_method

    # Add test methods to the test class
    for example_name, example_func in lst_examples:
        test_method = create_test_method(example_name, example_func)
        test_method_name = f'test_{example_name}'
        setattr(TestUnconstrainedMinimization, test_method_name, test_method)
    

    unittest.main()
