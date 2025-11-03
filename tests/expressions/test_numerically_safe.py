import unittest

import numpy as np

from biogeme.expressions import Beta, log, logzero
from biogeme.function_output import FunctionOutput
from biogeme.jax_calculator import create_function_simple_expression
from biogeme.likelihood.negative_likelihood import NegativeLikelihood


class test_expressions(unittest.TestCase):
    def test_log(self):
        argument = Beta('argument', 2, None, None, 0)
        the_log = log(argument)
        expression_function = create_function_simple_expression(
            expression=the_log, numerically_safe=True, use_jit=True
        )
        the_function_output: FunctionOutput = expression_function(
            [10.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, np.log(10), 3)
        optimization_function = NegativeLikelihood(
            dimension=1,
            loglikelihood=create_function_simple_expression(
                expression=the_log, numerically_safe=True, use_jit=True
            ),
        )
        check_results = optimization_function.check_derivatives(x=[1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient, check_results.finite_differences.gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, places=2)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

        # Log of a negative number
        the_function_output: FunctionOutput = expression_function(
            [-1.0], gradient=True, hessian=False, bhhh=False
        )
        self.assertTrue(the_function_output.function < -1e10)
        self.assertGreater(the_function_output.gradient[0], 0.0)
        optimization_function = NegativeLikelihood(
            dimension=1,
            loglikelihood=create_function_simple_expression(
                expression=the_log, numerically_safe=True, use_jit=True
            ),
        )
        check_results = optimization_function.check_derivatives(x=[-1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient, check_results.finite_differences.gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, places=2)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

        # Log of zero
        the_function_output: FunctionOutput = expression_function(
            [0.0], gradient=True, hessian=False, bhhh=False
        )
        self.assertTrue(the_function_output.function < -10)
        self.assertGreater(the_function_output.gradient[0], 0.0)
        optimization_function = NegativeLikelihood(
            dimension=1,
            loglikelihood=create_function_simple_expression(
                expression=the_log, numerically_safe=True, use_jit=True
            ),
        )
        check_results = optimization_function.check_derivatives(x=[-1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient,
            check_results.finite_differences.gradient,
        ):
            self.assertAlmostEqual(a_grad, fd_grad, places=2)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

    def test_logzero(self):
        argument = Beta('argument', 2, None, None, 0)
        the_log = logzero(argument)
        expression_function = create_function_simple_expression(
            expression=the_log, numerically_safe=True, use_jit=True
        )
        the_function_output: FunctionOutput = expression_function(
            [10.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, np.log(10), 3)
        optimization_function = NegativeLikelihood(
            dimension=1,
            loglikelihood=create_function_simple_expression(
                expression=the_log, numerically_safe=True, use_jit=True
            ),
        )
        check_results = optimization_function.check_derivatives(x=[1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient, check_results.finite_differences.gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, places=2)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

        # Log of a negative number
        the_function_output: FunctionOutput = expression_function(
            [-1.0], gradient=True, hessian=False, bhhh=False
        )
        self.assertTrue(the_function_output.function < -1e10)
        self.assertGreater(the_function_output.gradient[0], 0.0)
        optimization_function = NegativeLikelihood(
            dimension=1,
            loglikelihood=create_function_simple_expression(
                expression=the_log, numerically_safe=True, use_jit=True
            ),
        )
        check_results = optimization_function.check_derivatives(x=[-1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient, check_results.finite_differences.gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, places=2)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

        # Log of zero
        the_function_output: FunctionOutput = expression_function(
            [0.0], gradient=True, hessian=False, bhhh=False
        )
        self.assertEqual(the_function_output.function, 0)
        self.assertEqual(the_function_output.gradient[0], 0.0)
        optimization_function = NegativeLikelihood(
            dimension=1,
            loglikelihood=create_function_simple_expression(
                expression=the_log, numerically_safe=True, use_jit=True
            ),
        )
        check_results = optimization_function.check_derivatives(x=[-1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient,
            check_results.finite_differences.gradient,
        ):
            self.assertAlmostEqual(a_grad, fd_grad, places=2)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

    def test_divide(self):
        numerator = Beta('numerator', 1, None, None, 0)
        denominator = Beta('denominator', 1, None, None, 0)
        ratio = numerator / denominator
        expression_function = create_function_simple_expression(
            expression=ratio, numerically_safe=True, use_jit=True
        )
        the_function_output: FunctionOutput = expression_function(
            [1, 1], gradient=True, hessian=False, bhhh=False
        )
        expected_f = 1
        expected_g = [1, -1]
        self.assertAlmostEqual(the_function_output.function, expected_f, 3)
        self.assertAlmostEqual(the_function_output.gradient[0], expected_g[0], 3)
        self.assertAlmostEqual(the_function_output.gradient[1], expected_g[1], 3)

        # Division by zero
        the_function_output: FunctionOutput = expression_function(
            [1, 0], gradient=True, hessian=False, bhhh=False
        )
        expected_f = 0
        expected_g = [0, 4503599627370496.0]
        self.assertAlmostEqual(the_function_output.function, expected_f, 3)
        self.assertAlmostEqual(the_function_output.gradient[0], expected_g[0], 3)
        self.assertAlmostEqual(
            the_function_output.gradient[1], expected_g[1], delta=1e2
        )

    def test_power(self):
        base = Beta('base', 1, None, None, 0)
        exponent = Beta('exponent', 1, None, None, 0)
        power = base**exponent
        expression_function = create_function_simple_expression(
            expression=power, numerically_safe=True, use_jit=True
        )
        the_function_output: FunctionOutput = expression_function(
            [1, 1], gradient=True, hessian=False, bhhh=False
        )
        expected_f = 1
        expected_g = [1, 0]
        self.assertAlmostEqual(the_function_output.function, expected_f, 3)
        self.assertAlmostEqual(the_function_output.gradient[0], expected_g[0], 3)
        self.assertAlmostEqual(the_function_output.gradient[1], expected_g[1], 3)

        optimization_function = NegativeLikelihood(
            dimension=1,
            loglikelihood=create_function_simple_expression(
                expression=power, numerically_safe=True, use_jit=True
            ),
        )
        check_results = optimization_function.check_derivatives(x=[1.0, 1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient,
            check_results.finite_differences.gradient,
        ):
            self.assertAlmostEqual(a_grad, fd_grad, places=2)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

        base_value = 2.0
        exponent_value = 3.0
        base = Beta('base', base_value, None, None, 0)
        exponent = Beta('exponent', exponent_value, None, None, 0)
        power = base**exponent
        expression_function = create_function_simple_expression(
            expression=power, numerically_safe=True, use_jit=True
        )
        the_function_output: FunctionOutput = expression_function(
            [base_value, exponent_value], gradient=True, hessian=False, bhhh=False
        )
        expected_f = base_value**exponent_value
        expected_g = [12, 5.545]
        self.assertAlmostEqual(the_function_output.function, expected_f, 3)
        self.assertAlmostEqual(the_function_output.gradient[0], expected_g[0], 3)
        self.assertAlmostEqual(the_function_output.gradient[1], expected_g[1], 3)

        base_value = 0.0
        exponent_value = 0.0
        base = Beta('base', base_value, None, None, 0)
        exponent = Beta('exponent', exponent_value, None, None, 0)
        power = base**exponent
        expression_function = create_function_simple_expression(
            expression=power, numerically_safe=True, use_jit=True
        )
        the_function_output: FunctionOutput = expression_function(
            [base_value, exponent_value], gradient=True, hessian=False, bhhh=False
        )
        expected_f = 0
        expected_g = [0, 0]
        self.assertAlmostEqual(the_function_output.function, expected_f, 3)
        self.assertAlmostEqual(the_function_output.gradient[0], expected_g[0], 3)
        self.assertAlmostEqual(the_function_output.gradient[1], expected_g[1], 3)

        base_value = 1.0e-6
        exponent_value = -2
        base = Beta('base', base_value, None, None, 0)
        exponent = Beta('exponent', exponent_value, None, None, 0)
        power = base**exponent
        expression_function = create_function_simple_expression(
            expression=power, numerically_safe=True, use_jit=True
        )
        the_function_output: FunctionOutput = expression_function(
            [base_value, exponent_value], gradient=True, hessian=False, bhhh=False
        )
        expected_f = 1.0e12
        expected_g = [0, 0]
        self.assertAlmostEqual(the_function_output.function, expected_f, 2)
        self.assertTrue(the_function_output.gradient[0] < 0)
        self.assertTrue(the_function_output.gradient[1] < 0)
