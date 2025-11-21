"""
Test the expressions module

:author: Michel Bierlaire
:data: Wed Apr 29 17:47:53 2020

"""

import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

from biogeme import models
from biogeme.audit_tuple import AuditTuple
from biogeme.database import Database
from biogeme.distributions import normalpdf
from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    BelongsTo,
    Beta,
    BinaryMax,
    BinaryMin,
    ConditionalSum,
    ConditionalTermTuple,
    Derive,
    Draws,
    Elem,
    Expression,
    IntegrateNormal,
    LinearTermTuple,
    LinearUtility,
    LogLogit,
    MonteCarlo,
    MultipleSum,
    NormalCdf,
    Numeric,
    PanelLikelihoodTrajectory,
    RandomVariable,
    Variable,
    audit_expression,
    cos,
    exp,
    expression_to_value,
    get_dict_expressions,
    get_dict_values,
    is_numeric,
    list_of_all_betas_in_expression,
    list_of_fixed_betas_in_expression,
    list_of_free_betas_in_expression,
    list_of_random_variables_in_expression,
    list_of_variables_in_expression,
    log,
    logzero,
    sin,
    validate_and_convert,
)
from biogeme.expressions.collectors import list_of_draws_in_expression
from biogeme.expressions.minus import Minus
from biogeme.expressions_registry import ExpressionRegistry
from biogeme.function_output import (
    FunctionOutput,
    NamedFunctionOutput,
    convert_to_dict,
)
from biogeme.jax_calculator import (
    CallableExpression,
    create_function_simple_expression,
    get_value_and_derivatives,
    get_value_c,
)
from biogeme.likelihood.negative_likelihood import NegativeLikelihood
from biogeme.tools import check_derivatives
from biogeme.tools.derivatives import CheckDerivativesResults
from test_data import getData

EPSILON: float = np.finfo(np.float64).eps


class test_expressions(unittest.TestCase):
    def setUp(self):
        self.myData = getData(2)
        self.Person = Variable('Person')
        self.Variable1 = Variable('Variable1')
        self.Variable2 = Variable('Variable2')
        self.Choice = Variable('Choice')
        self.Av1 = Variable('Av1')
        self.Av2 = Variable('Av2')
        self.Av3 = Variable('Av3')
        self.beta1 = Beta('beta1', 0.2, None, None, 0)
        self.beta2 = Beta('beta2', 0.4, None, None, 0)
        self.beta3 = Beta('beta3', 1, None, None, 1)
        self.beta4 = Beta('beta4', 0, None, None, 1)
        self.omega1 = RandomVariable('omega1')
        self.omega2 = RandomVariable('omega2')
        self.xi1 = Draws('xi1', 'NORMAL')
        self.xi2 = Draws('xi2', 'UNIF')
        self.xi3 = Draws('xi3', 'WRONGTYPE')
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)
        self.newvar = self.myData.define_variable(
            'newvar', self.Variable1 + self.Variable2
        )

    def run_test_create_function(self, use_jit: bool):
        beta1 = Beta('beta1', 0, None, None, 0)
        beta2 = Beta('beta2', 0, None, None, 0)
        quotient = beta1 / beta2
        the_function: CallableExpression = create_function_simple_expression(
            expression=quotient, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = the_function(
            [2, 4], gradient=False, hessian=False, bhhh=False
        )
        self.assertEqual(2.0 / 4.0, the_function_output.function)

    def test_create_function(self):
        self.run_test_create_function(use_jit=True)

    def test_create_function_no_jit(self):
        self.run_test_create_function(use_jit=False)

    def test_errors(self):
        with self.assertRaises(BiogemeError):
            _ = Numeric(1) / 'ert'

    #        with self.assertRaises(exceptions.BiogemeError):
    #            _ = 'ert' / Numeric(1)

    def assertDataframeEqual(self, a, b, msg):
        try:
            pd.testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def test_is_numeric(self):
        result = is_numeric(1)
        self.assertTrue(result)
        result = is_numeric(0.1)
        self.assertTrue(result)
        result = is_numeric(True)
        self.assertTrue(result)
        result = is_numeric(self)
        self.assertFalse(result)

    def test_add(self):
        result = self.Variable1 + self.Variable2
        self.assertEqual(str(result), '(Variable1 + Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 + 1
        self.assertEqual(str(result), '(Variable1 + `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 + self.Variable1
        self.assertEqual(str(result), '(`1.0` + Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(BiogemeError):
            _ = self + self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1 + self

    def test_sub(self):
        result = self.Variable1 - self.Variable2
        self.assertEqual(str(result), '(Variable1 - Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 - 1
        self.assertEqual(str(result), '(Variable1 - `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 - self.Variable1
        self.assertEqual(str(result), '(`1.0` - Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(BiogemeError):
            _ = self - self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1 - self

    def test_mul(self):
        result = self.Variable1 * self.Variable2
        self.assertEqual(str(result), '(Variable1 * Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 * 1
        self.assertEqual(str(result), '(Variable1 * `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 * self.Variable1
        self.assertEqual(str(result), '(`1.0` * Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(BiogemeError):
            _ = self * self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1 * self

    def run_test_exp(self, use_jit: bool):
        argument = Beta('argument', 2, None, None, 0)
        the_exp = exp(argument)
        expression_function: CallableExpression = create_function_simple_expression(
            expression=the_exp, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [10.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, np.exp(10), 3)
        optimization_function: CallableExpression = create_function_simple_expression(
            expression=the_exp, numerically_safe=False, use_jit=use_jit
        )
        check_results: CheckDerivativesResults = check_derivatives(
            the_function=optimization_function, x=[1.0]
        )
        for a_grad, fd_grad in zip(
            check_results.analytical_gradient, check_results.finite_differences_gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, delta=1.0e-4)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical_hessian),
            np.nditer(check_results.finite_differences_hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

        # exp of a large number
        check_results = check_derivatives(
            the_function=optimization_function, x=[1800.0]
        )
        self.assertAlmostEqual(check_results.function, np.inf)

    def test_exp(self):
        self.run_test_exp(use_jit=True)

    def test_exp_no_jit(self):
        self.run_test_exp(use_jit=False)

    def run_test_loglogit(self, use_jit: bool):
        V1 = Beta('V1', 2, None, None, 0)
        V2 = Beta('V2', 2, None, None, 0)
        V = {1: V1, 2: V2}
        av = {1: 1, 2: 1}
        the_logit = LogLogit(V, av, 1)
        expression_function = create_function_simple_expression(
            expression=the_logit, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [10.0, 30], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, -20.000000002061153, 3)
        optimization_function = create_function_simple_expression(
            expression=the_logit, numerically_safe=False, use_jit=use_jit
        )
        check_results = check_derivatives(
            the_function=optimization_function, x=[10, 11]
        )
        for a_grad, fd_grad in zip(
            check_results.analytical_gradient, check_results.finite_differences_gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, delta=1.0e-4)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical_hessian),
            np.nditer(check_results.finite_differences_hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

    def test_loglogit(self):
        self.run_test_loglogit(use_jit=True)

    def test_loglogit_no_jit(self):
        self.run_test_loglogit(use_jit=False)

    def run_test_conjunction(self, use_jit: bool):

        zero = Numeric(0)
        non_zero = Numeric(12)
        other_non_zero = Numeric(-1)
        and_result = non_zero & other_non_zero
        should_be_true = get_value_and_derivatives(
            expression=and_result, numerically_safe=False, use_jit=use_jit
        )
        self.assertEqual(1, should_be_true.function)
        and_result_with_other_syntax = non_zero & other_non_zero
        should_be_true = get_value_and_derivatives(
            expression=and_result, numerically_safe=False, use_jit=use_jit
        )
        self.assertEqual(should_be_true.function, 1)
        other_result = zero & other_non_zero
        should_be_false = get_value_and_derivatives(
            expression=other_result, numerically_safe=False, use_jit=use_jit
        )
        self.assertEqual(should_be_false.function, 0)

    def test_conjunction_no_jit(self):
        self.run_test_conjunction(use_jit=False)

    def test_conjunction(self):
        self.run_test_conjunction(use_jit=True)

    def run_test_disjunction(self, use_jit: bool):
        zero = Numeric(0)
        zero_two = Numeric(0)
        non_zero = Numeric(12)
        or_result = non_zero | zero
        should_be_true = get_value_and_derivatives(
            expression=or_result, numerically_safe=False, use_jit=use_jit
        )
        self.assertEqual(should_be_true.function, 1)
        other_result = zero | zero_two
        should_be_false = get_value_and_derivatives(
            expression=other_result, numerically_safe=False, use_jit=use_jit
        )
        self.assertEqual(should_be_false.function, 0)

    def test_disjunction(self):
        self.run_test_disjunction(use_jit=True)

    def test_disjunction_no_jit(self):
        self.run_test_disjunction(use_jit=False)

    def run_test_loglogit_full_choice_set(self, use_jit: bool):
        V1 = Beta('V1', 2, None, None, 0)
        V2 = Beta('V2', 2, None, None, 0)
        V = {1: V1, 2: V2}
        the_logit = LogLogit(util=V, av=None, choice=1)
        expression_function: CallableExpression = create_function_simple_expression(
            expression=the_logit, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [10.0, 30], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, -20.000000002061153, 3)
        optimization_function = create_function_simple_expression(
            expression=the_logit, numerically_safe=False, use_jit=use_jit
        )
        check_results = check_derivatives(
            the_function=optimization_function, x=[10, 11]
        )
        for a_grad, fd_grad in zip(
            check_results.analytical_gradient, check_results.finite_differences_gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, delta=1.0e-4)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical_hessian),
            np.nditer(check_results.finite_differences_hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

    def test_loglogit_full_choice_set(self):
        self.run_test_loglogit_full_choice_set(use_jit=True)

    def test_loglogit_full_choice_set_no_jit(self):
        self.run_test_loglogit_full_choice_set(use_jit=False)

    def run_test_normal_cdf(self, use_jit: bool):
        argument = Beta('argument', 2, None, None, 0)
        the_cdf = NormalCdf(argument)
        expression_function: CallableExpression = create_function_simple_expression(
            expression=the_cdf, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [10.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, norm.cdf(10), 3)
        the_function_output: FunctionOutput = expression_function(
            [-0.01], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, norm.cdf(-0.01), 3)

        optimization_function = create_function_simple_expression(
            expression=the_cdf, numerically_safe=False, use_jit=use_jit
        )
        check_results = check_derivatives(the_function=optimization_function, x=[1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical_gradient, check_results.finite_differences_gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, delta=1.0e-4)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical_hessian),
            np.nditer(check_results.finite_differences_hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

    def test_normal_cdf(self):
        self.run_test_normal_cdf(use_jit=True)

    def test_normal_cdf_no_jit(self):
        self.run_test_normal_cdf(use_jit=False)

    def run_test_power_constant(self, use_jit: bool):
        x = Beta('x', 2, None, None, 0)
        x_square = x * x
        the_power = x_square**2
        expression_function: CallableExpression = create_function_simple_expression(
            expression=the_power, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [2.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, 16, 3)
        check_results = check_derivatives(the_function=expression_function, x=[1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical_gradient, check_results.finite_differences_gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, delta=1.0e-4)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical_hessian),
            np.nditer(check_results.finite_differences_hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

        # Test a negative argument and an integer
        check_results = check_derivatives(the_function=expression_function, x=[-1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical_gradient, check_results.finite_differences_gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, delta=1.0e-4)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical_hessian),
            np.nditer(check_results.finite_differences_hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

    def test_power_constant(self):
        self.run_test_power_constant(use_jit=True)

    def test_power_constant_no_jit(self):
        self.run_test_power_constant(use_jit=False)

    def run_test_power_constant_neg_integer(self, use_jit: bool):
        x = Beta('x', 2, None, None, 0)
        # Test a negative argument and an integer
        the_power = x**-2
        expression_function: CallableExpression = create_function_simple_expression(
            expression=the_power, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [-2.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, 0.25, 3)
        check_results: CheckDerivativesResults = check_derivatives(
            the_function=expression_function, x=[-1.0]
        )
        for a_grad, fd_grad in zip(
            check_results.analytical_gradient, check_results.finite_differences_gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, delta=1.0e-4)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical_hessian),
            np.nditer(check_results.finite_differences_hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

    def test_power_constant_neg_integer(self):
        self.run_test_power_constant_neg_integer(use_jit=True)

    def test_power_constant_neg_integer_no_jit(self):
        self.run_test_power_constant_neg_integer(use_jit=False)

    def run_test_power_constant_epsilon(self, use_jit: bool):
        x = Beta('x', 2, None, None, 0)
        # Test a negative argument and an integer
        the_power = x**2
        small_x = 1.0e-34
        expression_function: CallableExpression = create_function_simple_expression(
            expression=the_power, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [small_x], gradient=True, hessian=True, bhhh=True
        )
        self.assertAlmostEqual(the_function_output.function, 0)
        self.assertAlmostEqual(the_function_output.gradient[0], 0)
        self.assertAlmostEqual(the_function_output.hessian[0][0], 2)

    def test_power_constant_epsilon(self):
        self.run_test_power_constant_epsilon(use_jit=True)

    def test_power_constant_epsilon_no_jit(self):
        self.run_test_power_constant_epsilon(use_jit=False)

    def run_test_power_constant_neg_non_integer(self, use_jit: bool):
        # -2 ** -2.5: this must raise an exception
        x = Beta('x', 2, None, None, 0)
        # Test a negative argument and an integer
        the_power = x**-2.5
        expression_function = create_function_simple_expression(
            expression=the_power, numerically_safe=False, use_jit=use_jit
        )

        negative_number = expression_function(
            [-2.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertTrue(np.isnan(negative_number.function), 'The value is not NaN')

    def test_power_constant_neg_non_integer(self):
        self.run_test_power_constant_neg_non_integer(use_jit=True)

    def test_power_constant_neg_non_integer_no_jit(self):
        self.run_test_power_constant_neg_non_integer(use_jit=False)

    def run_test_power(self, use_jit: bool):
        x = Beta('x', 2, None, None, 0)
        y = Beta('y', 2, None, None, 0)

        other_power = (x * x) ** (y + y)
        expression_function: CallableExpression = create_function_simple_expression(
            expression=other_power, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [2, 1], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, 16, 3)
        check_results = check_derivatives(
            the_function=expression_function, x=[1.0, 1.0]
        )
        for a_grad, fd_grad in zip(
            check_results.analytical_gradient, check_results.finite_differences_gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, delta=1.0e-4)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical_hessian),
            np.nditer(check_results.finite_differences_hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

    def test_power(self):
        self.run_test_power(use_jit=True)

    def test_power_no_jit(self):
        self.run_test_power(use_jit=False)

    def run_test_power_small_base(self, use_jit: bool):
        x = Beta('x', 2, None, None, 0)
        y = Beta('y', 2, None, None, 0)

        other_power = (x * x) ** (y + y)
        expression_function: CallableExpression = create_function_simple_expression(
            expression=other_power, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [2, 1], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, 16, 3)
        check_results = check_derivatives(
            the_function=expression_function, x=[1.0, 1.0]
        )
        for a_grad, fd_grad in zip(
            check_results.analytical_gradient, check_results.finite_differences_gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, delta=1.0e-4)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical_hessian),
            np.nditer(check_results.finite_differences_hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

    def test_power_small_base(self):
        self.run_test_power_small_base(use_jit=True)

    def test_power_small_base_no_jit(self):
        self.run_test_power_small_base(use_jit=False)

    def run_test_power_neg_non_integer(self, use_jit: bool):
        # -2 ** -2.5: this must raise an exception
        x = Beta('x', 2, None, None, 0)
        exponent = Beta('exponent', -2.5, None, None, 0)
        # Test a negative argument and an integer
        the_power = x**exponent
        expression_function = create_function_simple_expression(
            expression=the_power, numerically_safe=False, use_jit=use_jit
        )

        negative_number = expression_function(
            [-2.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertTrue(np.isnan(negative_number.function), 'The value is not NaN')

    def test_power_neg_non_integer(self):
        self.run_test_power_neg_non_integer(use_jit=True)
        self.run_test_power_neg_non_integer(use_jit=False)

    def run_test_power_using_constant(self, use_jit: bool):
        x = Beta('x', 2, None, None, 0)
        y = Numeric(1) * Numeric(1)

        the_power = (x * x) ** (y + y)
        optimization_function = NegativeLikelihood(
            dimension=2,
            loglikelihood=create_function_simple_expression(
                expression=the_power, numerically_safe=False, use_jit=use_jit
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

    def test_power_using_constant(self):
        self.run_test_power_using_constant(use_jit=True)
        self.run_test_power_using_constant(use_jit=False)

    def run_test_complex_expression(self, use_jit: bool):

        df1 = pd.DataFrame(
            {
                'Person': [1, 1, 1, 2, 2],
                'Exclude': [0, 0, 1, 0, 1],
                'Variable1': [1, 2, 3, 4, 5],
                'Variable2': [10, 20, 30, 40, 50],
                'Choice': [1, 2, 3, 1, 2],
                'Av1': [0, 1, 1, 1, 1],
                'Av2': [1, 1, 1, 1, 1],
                'Av3': [0, 1, 1, 1, 1],
            }
        )
        data = Database(f'test', df1)
        variable1 = Variable('Variable1')
        variable2 = Variable('Variable2')
        beta1 = Beta('beta1', -1.0, None, None, 0)
        beta2 = Beta('beta2', 2.0, None, None, 0)
        likelihood = -((beta1 * variable1) ** 2) - (beta2 * variable2) ** 2
        optimization_function = NegativeLikelihood(
            dimension=2,
            loglikelihood=create_function_simple_expression(
                expression=likelihood,
                database=data,
                numerically_safe=False,
                use_jit=use_jit,
            ),
        )
        check_results = optimization_function.check_derivatives(x=[-1.0, -1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient, check_results.finite_differences.gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, places=2)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

    def test_complex_expression(self):
        self.run_test_complex_expression(use_jit=True)

    def test_complex_expression_no_jit(self):
        self.run_test_complex_expression(use_jit=False)

    def run_test_log(self, use_jit: bool):
        argument = Beta('argument', 2, None, None, 0)
        the_log = log(argument)
        expression_function = create_function_simple_expression(
            expression=the_log, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [10.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, np.log(10), 3)
        optimization_function = NegativeLikelihood(
            dimension=1,
            loglikelihood=create_function_simple_expression(
                expression=the_log, numerically_safe=False, use_jit=use_jit
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
        log_negative = expression_function(
            [-10.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertTrue(np.isnan(log_negative.function), 'The value is not NaN')

        # Log of zero
        log_of_zero = expression_function(
            [0.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertTrue(np.isinf(log_of_zero.function), 'The value is not infinity')

        # Log of a number close to zero
        log_small_number = expression_function(
            [EPSILON], gradient=False, hessian=False, bhhh=False
        )
        self.assertEqual(
            np.log(EPSILON),
            log_small_number.function,
        )

        # Log of a number very close to zero
        log_small_number = expression_function(
            [EPSILON / 2], gradient=False, hessian=False, bhhh=False
        )
        self.assertEqual(np.log(EPSILON / 2), log_small_number.function)

    def test_log(self):
        self.run_test_log(use_jit=True)
        self.run_test_log(use_jit=False)

    def run_test_logzero(self, use_jit: bool):
        argument = Beta('argument', 2, None, None, 0)
        the_log = logzero(argument)
        expression_function = create_function_simple_expression(
            expression=the_log, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [10.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, np.log(10), 3)
        optimization_function = NegativeLikelihood(
            dimension=1,
            loglikelihood=expression_function,
        )
        check_results = optimization_function.check_derivatives(x=[1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient, check_results.finite_differences.gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, places=3)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

        # Log of a negative number
        log_negative = expression_function(
            [-10.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertTrue(np.isnan(log_negative.function), 'The value is not NaN')

        # Log of zero
        log_of_zero = expression_function(
            [0.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertEqual(
            0,
            log_of_zero.function,
        )

        # Log of a number close to zero
        log_small_number = expression_function(
            [EPSILON], gradient=False, hessian=False, bhhh=False
        )
        self.assertEqual(
            np.log(EPSILON),
            log_small_number.function,
        )

        # Log of a number very close to zero
        log_small_number = expression_function(
            [EPSILON / 2], gradient=False, hessian=False, bhhh=False
        )
        self.assertEqual(np.log(EPSILON / 2), log_small_number.function)

    def test_logzero(self):
        self.run_test_logzero(use_jit=True)
        self.run_test_logzero(use_jit=False)

    def run_test_div(self, use_jit: bool):
        result = self.Variable1 / self.Variable2
        self.assertEqual(str(result), '(Variable1 / Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        # Numbering of the variables is by alphabetical order.
        numerator = Beta('a_numerator', 2, None, None, 0)
        denominator = Beta('b_denominator', 2, None, None, 0)
        the_ratio = numerator / denominator
        expression_function = create_function_simple_expression(
            expression=the_ratio, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [2.0, 2.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, 1, 3)
        optimization_function = NegativeLikelihood(
            dimension=2,
            loglikelihood=expression_function,
        )
        optimization_function.x = [2.0, 4.0]
        check_results = optimization_function.check_derivatives(x=[2.0, 4.0])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient, check_results.finite_differences.gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, delta=1.0e-4)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)
        # Special case where numerator is 0
        the_function_output: FunctionOutput = expression_function(
            [0.0, 2.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, 0, 3)
        check_results = optimization_function.check_derivatives(x=[0.0, 2.0])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient, check_results.finite_differences.gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, delta=1.0e-4)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

        # Special case where denominator is 1
        the_function_output: FunctionOutput = expression_function(
            [2.0, 1.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, 2, 3)
        check_results = optimization_function.check_derivatives(x=[2.0, 1.0])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient, check_results.finite_differences.gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, places=2)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

        # Special case where denominator is close to zero
        the_function_output: FunctionOutput = expression_function(
            [2.0, 1.0e-18], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, 2e18, delta=1e12)
        # Here, no way to verify the derivatives with finite difference, due to the numerical difficulties.

        # Special case where denominator is zero
        the_function_output: FunctionOutput = expression_function(
            [2, 0], gradient=False, hessian=False, bhhh=False
        )
        self.assertTrue(
            np.isinf(the_function_output.function), 'The value is not infinity'
        )
        # Here, no way to verify the derivatives with finite difference, due to the numerical difficulties.

        # Special case where denominator is close to zero
        the_function_output: FunctionOutput = expression_function(
            [2.0, -1.0e-18], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, -2e18, delta=1e12)
        # Here, no way to verify the derivatives with finite difference, due to the numerical difficulties.

        result = self.Variable1 / 1
        self.assertEqual(str(result), '(Variable1 / `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 / self.Variable1
        self.assertEqual(str(result), '(`1.0` / Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(BiogemeError):
            _ = self / self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1 / self

    def test_div(self):
        self.run_test_div(use_jit=True)
        self.run_test_div(use_jit=False)

    def test_neg(self):
        result = -self.Variable1
        self.assertEqual(str(result), '(-Variable1)')
        self.assertTrue(result.children[0] is self.Variable1)

    def test_pow(self):
        result = self.Variable1**self.Variable2
        self.assertEqual(str(result), '(Variable1 ** Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1**1
        self.assertEqual(str(result), 'Variable1**1.0')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1**self.Variable1
        self.assertEqual(str(result), '(`1.0` ** Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        result = Numeric(3) ** Numeric(2)
        nine = result.get_value()
        self.assertEqual(9, nine)

        with self.assertRaises(BiogemeError):
            _ = self**self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1**self

    def test_and_1(self):
        result = self.Variable1 & self.Variable2
        self.assertEqual(str(result), '(Variable1 and Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 & 1
        self.assertEqual(str(result), '(Variable1 and `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 & self.Variable1
        self.assertEqual(str(result), '(`1.0` and Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(BiogemeError):
            _ = self & self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1 & self

    def test_or(self):
        result = self.Variable1 | self.Variable2
        self.assertEqual(str(result), '(Variable1 or Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 | 1
        self.assertEqual(str(result), '(Variable1 or `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 | self.Variable1
        self.assertEqual(str(result), '(`1.0` or Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(BiogemeError):
            _ = self | self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1 | self

    def test_eq(self):
        result = self.Variable1 == self.Variable2
        self.assertEqual(str(result), '(Variable1 == Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 == 1
        self.assertEqual(str(result), '(Variable1 == `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 == self.Variable1
        self.assertEqual(str(result), '(Variable1 == `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(BiogemeError):
            _ = self == self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1 == self

    def test_neq(self):
        result = self.Variable1 != self.Variable2
        self.assertEqual(str(result), '(Variable1 != Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 != 1
        self.assertEqual(str(result), '(Variable1 != `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 != self.Variable1
        self.assertEqual(str(result), '(Variable1 != `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(BiogemeError):
            _ = self != self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1 != self

    def test_le(self):
        result = self.Variable1 <= self.Variable2
        self.assertEqual(str(result), '(Variable1 <= Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 <= 1
        self.assertEqual(str(result), '(Variable1 <= `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 <= self.Variable1
        self.assertEqual(str(result), '(Variable1 >= `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(BiogemeError):
            _ = self <= self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1 <= self

    def test_ge(self):
        result = self.Variable1 >= self.Variable2
        self.assertEqual(str(result), '(Variable1 >= Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 >= 1
        self.assertEqual(str(result), '(Variable1 >= `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 >= self.Variable1
        self.assertEqual(str(result), '(Variable1 <= `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(BiogemeError):
            _ = self >= self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1 >= self

    def test_lt(self):
        result = self.Variable1 < self.Variable2
        self.assertEqual(str(result), '(Variable1 < Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 < 1
        self.assertEqual(str(result), '(Variable1 < `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 < self.Variable1
        self.assertEqual(str(result), '(Variable1 > `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(BiogemeError):
            _ = self < self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1 < self

    def test_gt(self):
        result = self.Variable1 > self.Variable2
        self.assertEqual(str(result), '(Variable1 > Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 > 1
        self.assertEqual(str(result), '(Variable1 > `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 > self.Variable1
        self.assertEqual(str(result), '(Variable1 < `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(BiogemeError):
            _ = self > self.Variable1

        with self.assertRaises(BiogemeError):
            _ = self.Variable1 > self

    def run_test_get_value_c(self, use_jit: bool):
        result = get_value_c(
            expression=self.Variable1,
            database=self.myData,
            numerically_safe=False,
            use_jit=use_jit,
        )
        np.testing.assert_equal(result, [10, 20, 30, 40, 50])

    def test_get_value_c(self):
        self.run_test_get_value_c(use_jit=True)
        self.run_test_get_value_c(use_jit=False)

    def test_DefineVariable(self):
        _ = self.myData.define_variable('newvar_b', self.Variable1 + self.Variable2)
        cols = self.myData.dataframe.columns
        added = 'newvar_b' in cols
        self.assertTrue(added)
        self.myData.dataframe['newvar_p'] = (
            self.myData.dataframe['Variable1'] + self.myData.dataframe['Variable2']
        )
        pd.testing.assert_series_equal(
            self.myData.dataframe['newvar_b'],
            self.myData.dataframe['newvar_p'],
            check_dtype=False,
            check_names=False,
        )

    def run_test_expr1(self, use_jit: bool):
        self.beta1.status = 1
        self.beta2.status = 1
        expr1 = 2 * self.beta1 - exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        self.assertAlmostEqual(expr1.get_value(), -1.275800115089098, 5)
        res = get_value_c(expression=expr1, numerically_safe=False, use_jit=use_jit)
        self.assertAlmostEqual(res, -1.275800115089098, 5)

    def test_expr1(self):
        self.run_test_expr1(use_jit=True)
        self.run_test_expr1(use_jit=False)

    def run_test_expr1_newvalues(self, use_jit: bool):
        self.beta1.status = 1
        self.beta2.status = 1
        expr1 = 2 * self.beta1 - exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        new_values = {'beta1': 1, 'beta2': 2, 'beta3': 3, 'beta4': 2}
        expr1.change_init_values(new_values)
        self.assertAlmostEqual(expr1.get_value(), 1.9323323583816936, 5)
        res = get_value_c(expression=expr1, numerically_safe=False, use_jit=use_jit)
        self.assertAlmostEqual(res, 1.9323323583816936, 5)

    def test_expr1_newvalues(self):
        self.run_test_expr1_newvalues(use_jit=True)
        self.run_test_expr1_newvalues(use_jit=False)

    def run_test_expr1_derivatives(self, use_jit: bool):
        expr1 = 2 * self.beta1 - exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        newvalues = {'beta1': 1, 'beta2': 2, 'beta3': 3, 'beta4': 2}
        expr1.change_init_values(newvalues)
        the_function_output: FunctionOutput = get_value_and_derivatives(
            expression=expr1, numerically_safe=False, use_jit=use_jit
        )
        self.assertAlmostEqual(the_function_output.function, 1.9323323583816936, 5)
        g_ok = [2.0, 0.10150146242745953]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, -0.16916910404576588]
        bhhh0_ok = [4, 0.20300292]
        bhhh1_ok = [0.20300292, 0.01030255]
        for check_left, check_right in zip(the_function_output.gradient, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[0], h0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[1], h1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.bhhh[0], bhhh0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.bhhh[1], bhhh1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

    def test_expr1_derivatives(self):
        self.run_test_expr1_derivatives(use_jit=True)
        self.run_test_expr1_derivatives(use_jit=False)

    def run_test_expr1_named_derivatives(self, use_jit: bool):
        expr1 = 2 * self.beta1 - exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        new_values = {'beta1': 1, 'beta2': 2, 'beta3': 3, 'beta4': 2}
        expr1.change_init_values(new_values)
        the_function_output: NamedFunctionOutput = get_value_and_derivatives(
            expression=expr1,
            named_results=True,
            numerically_safe=False,
            use_jit=use_jit,
        )
        self.assertAlmostEqual(the_function_output.function, 1.9323323583816936, 5)
        g_ok = {'beta1': 2.0, 'beta2': 0.10150146242745953}
        self.assertDictEqual(the_function_output.gradient, g_ok)
        h_ok = {
            'beta1': {'beta1': 0.0, 'beta2': 0.0},
            'beta2': {'beta1': 0.0, 'beta2': -0.16916910404576588},
        }
        self.assertDictEqual(the_function_output.hessian, h_ok)
        bhhh_ok = {
            'beta1': {'beta1': 4, 'beta2': 0.20300292485491905},
            'beta2': {'beta1': 0.20300292485491905, 'beta2': 0.010302546874912978},
        }
        self.assertDictEqual(the_function_output.bhhh, bhhh_ok)

    def test_expr1_named_derivatives(self):
        self.run_test_expr1_named_derivatives(use_jit=True)
        self.run_test_expr1_named_derivatives(use_jit=False)

    def run_test_expr1_gradient(self, use_jit: bool):
        expr1 = 2 * self.beta1 - exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        the_function_output: FunctionOutput = get_value_and_derivatives(
            expression=expr1,
            gradient=True,
            hessian=False,
            bhhh=False,
            numerically_safe=False,
            use_jit=use_jit,
        )
        self.assertAlmostEqual(the_function_output.function, -1.275800115089098, 5)
        g_ok = [2.0, 5.865300402811844]
        for check_left, check_right in zip(the_function_output.gradient, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        self.assertIsNone(the_function_output.hessian)
        self.assertIsNone(the_function_output.bhhh)

    def test_expr1_gradient(self):
        self.run_test_expr1_gradient(use_jit=True)
        self.run_test_expr1_gradient(use_jit=False)

    def run_test_expr1_function(self, use_jit: bool):
        expr1 = 2 * self.beta1 - exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        self.assertFalse(expr1.is_complex())

        the_function = create_function_simple_expression(
            expression=expr1, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = the_function(
            [1, 2], gradient=True, hessian=True, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, 1.9323323583816936, 5)
        g_ok = [2.0, 0.10150146242745953]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, -0.16916910404576588]
        for check_left, check_right in zip(the_function_output.gradient, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[0], h0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[1], h1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

        the_function_output: FunctionOutput = the_function(
            [10, -2], gradient=True, hessian=True, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, 23.694528049465326, 5)
        g_ok = [2, -1.84726402]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, 1.84726402]
        for check_left, check_right in zip(the_function_output.gradient, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[0], h0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[1], h1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

    def test_expr1_function(self):
        self.run_test_expr1_function(use_jit=True)
        self.run_test_expr1_function(use_jit=False)

    def run_test_expr1_database_agg(self, use_jit: bool):
        expr1 = 2 * self.beta1 - exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        the_function_output: FunctionOutput = get_value_and_derivatives(
            expression=expr1,
            database=self.myData,
            numerically_safe=False,
            use_jit=use_jit,
        )
        f_ok = -6.37900057544549
        g_ok = [10.0, 29.32650201405922]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, -155.01151064574157]
        b0_ok = [20.0, 58.65300403]
        b1_ok = [58.65300403, 172.00874408]
        self.assertAlmostEqual(the_function_output.function, f_ok, 5)
        for check_left, check_right in zip(the_function_output.gradient, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[0], h0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[1], h1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.bhhh[0], b0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.bhhh[1], b1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

    def test_expr1_database_agg(self):
        self.run_test_expr1_database_agg(use_jit=True)
        self.run_test_expr1_database_agg(use_jit=False)

    def run_test_expr2(self, use_jit: bool):
        expr2 = 2 * self.beta1 * self.Variable1 - exp(-self.beta2 * self.Variable2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        with self.assertRaises(BiogemeError):
            expr2.get_value()
        res = list(
            get_value_c(
                expression=expr2,
                database=self.myData,
                numerically_safe=False,
                use_jit=use_jit,
            )
        )
        self.assertListEqual(res, [4.0, 8.0, 12.0, 16.0, 20.0])

    def test_expr2(self):
        self.run_test_expr2(use_jit=True)
        self.run_test_expr2(use_jit=False)

    def test_list_of_variables(self):
        expr2 = 2 * self.beta1 * self.Variable1 - exp(-self.beta2 * self.Variable2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        the_list = list_of_variables_in_expression(expr2)
        the_expected_list = [self.Variable1, self.Variable2]
        self.assertCountEqual(
            [str(x) for x in the_list],
            [str(x) for x in the_expected_list],
        )

    def test_list_of_betas(self):
        expr2 = 2 * self.beta1 * self.Variable1 - exp(-self.beta2 * self.Variable2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        the_list = list_of_all_betas_in_expression(expr2)
        the_expected_list = [self.beta1, self.beta2, self.beta3]
        self.assertCountEqual(
            {str(x) for x in the_list},
            {str(x) for x in the_expected_list},
        )
        the_list_free = list_of_free_betas_in_expression(expr2)
        the_expected_list_free = [self.beta1, self.beta2]
        self.assertCountEqual(
            {str(x) for x in the_list_free},
            {str(x) for x in the_expected_list_free},
        )
        the_list_fixed = list_of_fixed_betas_in_expression(expr2)
        the_expected_list_fixed = [self.beta3]
        self.assertCountEqual(
            {str(x) for x in the_list_fixed},
            {str(x) for x in the_expected_list_fixed},
        )
        expr_numeric = Numeric(0)
        the_list_empty = list_of_all_betas_in_expression(expr_numeric)
        self.assertFalse(the_list_empty)

    def test_list_of_random_variables(self):
        rv1 = RandomVariable('test')
        rv2 = RandomVariable('test2')
        dr1 = Draws('test', draw_type='NORMAL')
        dr2 = Draws('test2', draw_type='UNIFORM')
        expression = rv1 * rv2 + dr1 * dr2
        the_list = list_of_random_variables_in_expression(expression)
        the_expected_list = [rv1, rv2]
        self.assertCountEqual(
            {str(x) for x in the_list},
            {str(x) for x in the_expected_list},
        )

    def test_list_of_draws(self):
        rv1 = RandomVariable('test')
        rv2 = RandomVariable('test2')
        dr1 = Draws('test', draw_type='NORMAL')
        dr2 = Draws('test2', draw_type='UNIFORM')
        expression = rv1 * rv2 + dr1 * dr2
        the_list = list_of_draws_in_expression(expression)
        the_expected_list = [dr1, dr2]
        self.assertCountEqual(
            {str(x) for x in the_list},
            {str(x) for x in the_expected_list},
        )

    def test_getClassName(self):
        expr2 = 2 * self.beta1 * self.Variable1 - exp(-self.beta2 * self.Variable2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        c = expr2.get_class_name()
        self.assertEqual(c, 'Minus')

    def test_embedExpression(self):
        expr2 = 2 * self.beta1 * self.Variable1 - exp(-self.beta2 * self.Variable2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        self.assertTrue(expr2.embed_expression(Minus))
        self.assertFalse(expr2.embed_expression(Draws))

    def test_panel_variables(self):
        expr_ok = PanelLikelihoodTrajectory(self.Variable1)
        audit_tuple: AuditTuple = audit_expression(expr_ok)
        self.assertFalse(audit_tuple.warnings)
        self.assertFalse(audit_tuple.errors)

    def test_check_draws(self):
        d1 = Draws('d1', 'UNIFORM')
        d2 = Draws('d2', 'UNIFORM')
        expr_ok = MonteCarlo(d1 * d2)
        audit_tuple: AuditTuple = audit_expression(expr_ok)
        self.assertFalse(audit_tuple.warnings)
        self.assertFalse(audit_tuple.errors)
        expr_not_ok = d1 + MonteCarlo(d1 * d2)
        audit_tuple: AuditTuple = audit_expression(expr_not_ok)
        self.assertTrue(audit_tuple.warnings)

    def test_check_rv(self):
        d1 = RandomVariable('d1')
        d2 = RandomVariable('d2')
        expr_ok = IntegrateNormal(d1, d1)
        audit_tuple: AuditTuple = audit_expression(expr_ok)
        self.assertFalse(audit_tuple.warnings)
        self.assertFalse(audit_tuple.errors)
        expr_warning = IntegrateNormal(1, d1)
        audit_tuple: AuditTuple = audit_expression(expr_warning)
        self.assertTrue(audit_tuple.warnings)
        self.assertFalse(audit_tuple.errors)
        expr_not_ok = d1 + IntegrateNormal(1, d2)
        audit_tuple: AuditTuple = audit_expression(expr_not_ok)
        self.assertTrue(audit_tuple.errors)

    def test_ids_multiple_formulas(self):
        expr1 = 2 * self.beta1 - exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        expr2 = 2 * self.beta1 * self.Variable1 - exp(-self.beta2 * self.Variable2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        collection_of_formulas = [expr1, expr2]
        registry = ExpressionRegistry(
            expressions=collection_of_formulas,
            database=self.myData,
        )
        expected_free_betas_indices = {'beta1': 0, 'beta2': 1}
        self.assertDictEqual(registry.free_betas_indices, expected_free_betas_indices)
        expected_fixed_betas_indices = {'beta3': 0, 'beta4': 1}
        self.assertDictEqual(registry.fixed_betas_indices, expected_fixed_betas_indices)
        expected_variables_indices = {
            'Av1': 5,
            'Av2': 6,
            'Av3': 7,
            'Choice': 4,
            'Exclude': 1,
            'Person': 0,
            'Variable1': 2,
            'Variable2': 3,
            'newvar': 8,
        }
        self.assertDictEqual(registry.variables_indices, expected_variables_indices)
        expected_draws_indices = {}
        self.assertDictEqual(registry.draws_indices, expected_draws_indices)
        expected_random_variables_indices = {}
        self.assertDictEqual(
            registry.random_variables_indices, expected_random_variables_indices
        )

    def run_test_expr3(self, use_jit: bool):
        my_draws = Draws('myDraws', 'UNIFORM')
        expr3 = MonteCarlo(my_draws * my_draws)
        res = get_value_c(
            expression=expr3,
            database=self.myData,
            number_of_draws=100000,
            numerically_safe=False,
            use_jit=use_jit,
        )
        for v in res:
            self.assertAlmostEqual(v, 1.0 / 3.0, 2)

    def test_expr3(self):
        self.run_test_expr3(use_jit=True)
        self.run_test_expr3(use_jit=False)

    def run_test_expr4(self, use_jit: bool):
        omega = RandomVariable('omega')
        density = normalpdf(omega)
        a = 0
        b = 1
        x = a + (b - a) / (1 + exp(-omega))
        dx = (b - a) * exp(-omega) * (1 + exp(-omega)) ** (-2)
        integrand = x * x
        expr4 = IntegrateNormal((integrand * dx / (b - a)) / density, 'omega')
        res = get_value_c(
            expression=expr4,
            database=self.myData,
            numerically_safe=False,
            use_jit=use_jit,
        )
        self.assertTrue(expr4.is_complex())
        for v in res:
            self.assertAlmostEqual(v, 1.0 / 3.0, 2)

    def test_expr4(self):
        self.run_test_expr4(use_jit=True)
        self.run_test_expr4(use_jit=False)

    def run_test_expr5(self, use_jit: bool):
        expr1 = 2 * self.beta1 - exp(-self.beta2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        expr2 = 2 * self.beta1 * self.Variable1 - exp(-self.beta2 * self.Variable2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        expr5 = Elem({1: expr1, 2: expr2}, self.Person) / 10
        res = list(
            get_value_c(
                expression=expr5,
                database=self.myData,
                numerically_safe=False,
                use_jit=use_jit,
            )
        )
        res_ok = [
            -0.02703200460356393,
            -0.02703200460356393,
            -0.02703200460356393,
            1.6,
            2.0,
        ]
        for check_left, check_right in zip(res, res_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

    def test_expr5(self):
        self.run_test_expr5(use_jit=True)
        self.run_test_expr5(use_jit=False)

    def run_test_expr6(self, use_jit: bool):
        expr1 = 2 * self.beta1 - exp(-self.beta2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        expr2 = 2 * self.beta1 * self.Variable1 - exp(-self.beta2 * self.Variable2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        omega = RandomVariable('omega')
        density = normalpdf(omega)
        a = 0
        b = 1
        x = a + (b - a) / (1 + exp(-omega))
        dx = (b - a) * exp(-omega) * (1 + exp(-omega)) ** (-2)
        integrand = x * x
        expr4 = IntegrateNormal(integrand * dx / (b - a) / density, 'omega')
        expr6 = MultipleSum([expr1, expr2, expr4])
        res = list(
            get_value_c(
                expression=expr6,
                database=self.myData,
                numerically_safe=False,
                use_jit=use_jit,
            )
        )
        res_ok = [
            4.063012266030643,
            8.063012266030643,
            12.063012266030643,
            16.063012266030643,
            20.063012266030643,
        ]
        for check_left, check_right in zip(res, res_ok):
            self.assertAlmostEqual(check_left, check_right, 3)

    def test_expr6(self):
        self.run_test_expr6(use_jit=True)
        self.run_test_expr6(use_jit=False)

    def run_test_expr7(self, use_jit: bool):
        self.beta1.status = 1
        self.beta2.status = 1
        V = {0: -self.beta1, 1: -self.beta2, 2: -self.beta1}
        av = {0: 1, 1: 1, 2: 1}
        expr7 = LogLogit(V, av, 1)
        r = expr7.get_value()
        self.assertAlmostEqual(r, -1.2362866960692134, 5)
        expr8 = models.loglogit(V, av, 1)
        self.assertFalse(expr8.is_complex())
        res = get_value_c(
            expression=expr8,
            database=self.myData,
            numerically_safe=False,
            use_jit=use_jit,
        )
        for v in res:
            self.assertAlmostEqual(v, -1.2362866960692136, 5)

    def test_expr7(self):
        self.run_test_expr7(use_jit=True)
        self.run_test_expr7(use_jit=False)

    def run_test_expr9(self, use_jit: bool):
        V = {0: -self.beta1, 1: -self.beta2, 2: -self.beta1}
        av = {0: 1, 1: 1, 2: 1}
        expr8 = models.loglogit(V, av, 1)
        expr9 = Derive(expr8, 'beta2')
        res = get_value_c(
            expression=expr9,
            database=self.myData,
            numerically_safe=False,
            use_jit=use_jit,
        )
        for v in res:
            self.assertAlmostEqual(v, -0.7095392129298093, 5)

    def test_expr9(self):
        self.run_test_expr9(use_jit=True)
        self.run_test_expr9(use_jit=False)

    def run_test_expr10(self, use_jit: bool):
        expr10 = NormalCdf(self.Variable1 / 10 - 1)
        res = get_value_c(
            expression=expr10,
            database=self.myData,
            numerically_safe=False,
            use_jit=use_jit,
        )
        for i, j in zip(
            res,
            [
                0.5,
                0.8413447460685283,
                0.9772498680518218,
                0.99865010196837,
                0.9999683287581669,
            ],
        ):
            self.assertAlmostEqual(i, j, 5)

    def test_expr10(self):
        self.run_test_expr10(use_jit=True)
        self.run_test_expr10(use_jit=False)

    def run_test_expr11(self, use_jit: bool):
        expr1 = 2 * self.beta1 - exp(-self.beta2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        expr2 = 2 * self.beta1 * self.Variable1 - exp(-self.beta2 * self.Variable2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        expr5 = Elem({1: expr1, 2: expr2}, self.Person) / 10
        expr10 = NormalCdf(self.Variable1 / 10 - 1)
        expr11 = BinaryMin(expr5, expr10)
        res = get_value_c(
            expression=expr11,
            database=self.myData,
            numerically_safe=False,
            use_jit=use_jit,
        )
        res_ok = [-0.027032, -0.027032, -0.027032, 0.9986501, 0.99996833]
        for i, j in zip(
            res,
            res_ok,
        ):
            self.assertAlmostEqual(i, j, 5)

    def test_expr11(self):
        self.run_test_expr11(use_jit=True)
        self.run_test_expr11(use_jit=False)

    def run_test_expr12(self, use_jit: bool):
        expr1 = 2 * self.beta1 - exp(-self.beta2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        expr2 = 2 * self.beta1 * self.Variable1 - exp(-self.beta2 * self.Variable2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        expr5 = Elem({1: expr1, 2: expr2}, self.Person) / 10
        expr10 = NormalCdf(self.Variable1 / 10 - 1)
        expr12 = BinaryMax(expr5, expr10)
        res = get_value_c(
            expression=expr12,
            database=self.myData,
            numerically_safe=False,
            use_jit=use_jit,
        )
        for i, j in zip(res, [0.5, 0.8413447460685283, 0.9772498680518218, 1.6, 2.0]):
            self.assertAlmostEqual(i, j, 5)

    def test_expr12(self):
        self.run_test_expr12(use_jit=True)
        self.run_test_expr12(use_jit=False)

    def run_test_linear_utility(self, use_jit: bool):
        terms = [
            LinearTermTuple(beta=self.beta1, x=Variable('Variable1')),
            LinearTermTuple(beta=self.beta2, x=Variable('Variable2')),
        ]
        the_utility = LinearUtility(terms)
        expression_function = create_function_simple_expression(
            expression=the_utility,
            database=self.myData,
            numerically_safe=False,
            use_jit=use_jit,
        )
        the_function_output: FunctionOutput = expression_function(
            [0, 0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, 0, 3)
        optimization_function = NegativeLikelihood(
            dimension=2,
            loglikelihood=expression_function,
        )
        check_results = optimization_function.check_derivatives(x=[10, 11])
        for a_grad, fd_grad in zip(
            check_results.analytical.gradient, check_results.finite_differences.gradient
        ):
            self.assertAlmostEqual(a_grad, fd_grad, delta=1.0e-4)
        for a_hess, fd_hess in zip(
            np.nditer(check_results.analytical.hessian),
            np.nditer(check_results.finite_differences.hessian),
        ):
            self.assertAlmostEqual(a_hess, fd_hess, delta=1.0e-4)

    def test_linear_utility(self):
        self.run_test_linear_utility(use_jit=True)
        self.run_test_linear_utility(use_jit=False)

    def run_test_expr13(self, use_jit: bool):
        terms = [
            LinearTermTuple(beta=self.beta1, x=Variable('Variable1')),
            LinearTermTuple(beta=self.beta2, x=Variable('Variable2')),
            LinearTermTuple(beta=self.beta3, x=self.newvar),
        ]
        expr13 = LinearUtility(terms)
        res = get_value_c(
            expression=expr13,
            database=self.myData,
            numerically_safe=False,
            use_jit=use_jit,
        )
        res_ok = [
            152,
            304,
            456,
            608,
            760,
        ]
        for i, j in zip(res, res_ok):
            self.assertAlmostEqual(i, j, 5)
        expr13bis = (
            self.beta1 * Variable('Variable1')
            + self.beta2 * Variable('Variable2')
            + self.beta3 * self.newvar
        )
        res = get_value_c(
            expression=expr13bis,
            database=self.myData,
            numerically_safe=False,
            use_jit=use_jit,
        )
        res_ok = [
            152,
            304,
            456,
            608,
            760,
        ]
        for i, j in zip(res, res_ok):
            self.assertAlmostEqual(i, j, 5)

    def test_expr13(self):
        self.run_test_expr13(use_jit=True)
        self.run_test_expr13(use_jit=False)

    def run_test_expr14(self, use_jit: bool):
        c1 = Draws('draws1', 'NORMAL_HALTON2')
        c2 = Draws('draws2', 'NORMAL_HALTON2')
        U1 = Beta('beta1', 0, None, None, 0) * Variable('Variable1') + 10 * c1
        U2 = Beta('beta2', 0, None, None, 0) * Variable('Variable2') + 10 * c2
        U3 = 0
        U = {1: U1, 2: U2, 3: U3}
        av = {1: self.Av1, 2: self.Av2, 3: self.Av3}
        expr14 = log(
            MonteCarlo(PanelLikelihoodTrajectory(models.logit(U, av, self.Choice)))
        )
        self.myData.panel('Person')

        res = get_value_c(
            expression=expr14,
            database=self.myData,
            number_of_draws=100000,
            numerically_safe=False,
            use_jit=use_jit,
        )
        res_ok = [-3.93304323, -2.10368902]
        for i, j in zip(res, res_ok):
            self.assertAlmostEqual(i, j, 3)
        # f_list, g_list, h_list, b_list
        the_function_output: FunctionOutput = get_value_and_derivatives(
            expression=expr14,
            database=self.myData,
            number_of_draws=10_000_000,
            gradient=True,
            hessian=True,
            bhhh=True,
            numerically_safe=False,
            use_jit=use_jit,
        )
        expected_function = -6.0367212197720335
        expected_gradient = [-15.76597701, 142.34022993]
        expected_hessian = [
            [-1154.32134784, 11400.42932874],
            [11400.42932874, -115432.13478364],
        ]
        expected_bhhh = [
            [168.48075656, -1150.18334338],
            [-1150.18334338, 10148.25747682],
        ]
        self.assertAlmostEqual(
            the_function_output.function, expected_function, places=3
        )
        for actual, expected in zip(the_function_output.gradient, expected_gradient):
            self.assertAlmostEqual(actual, expected, places=3)
        for actual_row, expected_row in zip(
            the_function_output.hessian, expected_hessian
        ):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=3)
        for actual_row, expected_row in zip(the_function_output.bhhh, expected_bhhh):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=3)

    def test_expr14(self):
        self.run_test_expr14(use_jit=True)
        self.run_test_expr14(use_jit=False)

    def run_test_belongs_to(self, use_jit: bool):
        yes = BelongsTo(Numeric(2), {1, 2, 3})
        self.assertEqual(
            get_value_c(expression=yes, numerically_safe=False, use_jit=use_jit), 1.0
        )
        no = BelongsTo(Numeric(4), {1, 2, 3})
        self.assertEqual(
            get_value_c(expression=no, numerically_safe=False, use_jit=use_jit), 0.0
        )

    def test_belongs_to_no_jit(self):
        self.run_test_belongs_to(use_jit=False)

    def test_belongs_to(self):
        self.run_test_belongs_to(use_jit=True)

    def run_test_conditional_sum(self, use_jit: bool):

        the_terms = [
            ConditionalTermTuple(condition=Numeric(0), term=Numeric(10)),
            ConditionalTermTuple(condition=Numeric(1), term=Numeric(20)),
        ]
        the_sum = ConditionalSum(the_terms)
        self.assertEqual(
            get_value_c(expression=the_sum, numerically_safe=False, use_jit=use_jit), 20
        )

    def test_conditional_sum(self):
        self.run_test_conditional_sum(use_jit=True)

    def test_conditional_sum_no_jit(self):
        self.run_test_conditional_sum(use_jit=False)

    def run_test_sin(self, use_jit: bool):
        beta = Beta('Beta', 0.11, None, None, 0)
        the_sin = sin(2 * beta)
        expression_function = create_function_simple_expression(
            expression=the_sin, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [1.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, 0.909297, 3)
        optimization_function = NegativeLikelihood(
            dimension=1,
            loglikelihood=expression_function,
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
            self.assertAlmostEqual(a_hess, fd_hess, places=2)

    def test_sin(self):
        self.run_test_sin(use_jit=True)
        self.run_test_sin(use_jit=False)

    def run_test_cos(self, use_jit: bool):
        beta = Beta('Beta', 0.11, None, None, 0)
        the_cos = cos(2 * beta)
        expression_function = create_function_simple_expression(
            expression=the_cos, numerically_safe=False, use_jit=use_jit
        )
        the_function_output: FunctionOutput = expression_function(
            [1.0], gradient=False, hessian=False, bhhh=False
        )
        self.assertAlmostEqual(the_function_output.function, -0.4161468365471424, 3)
        optimization_function = NegativeLikelihood(
            dimension=1,
            loglikelihood=expression_function,
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
            self.assertAlmostEqual(a_hess, fd_hess, places=2)

    def test_cos(self):
        self.run_test_cos(use_jit=True)
        self.run_test_cos(use_jit=False)

    def test_bioMin(self):
        expr = BinaryMin(Numeric(3), Numeric(2))
        result = expr.get_value()
        expected_result = 2
        self.assertEqual(expected_result, result)

    def test_bioMax(self):
        expr = BinaryMax(Numeric(3), Numeric(2))
        result = expr.get_value()
        expected_result = 3
        self.assertEqual(expected_result, result)

    def test_and_2(self):
        true_expr_1: Expression = Numeric(3) == Numeric(3)
        true_expr_2: Expression = Numeric(2) == Numeric(2)
        false_expr: Expression = Numeric(3) == Numeric(2)
        result = true_expr_1 & true_expr_2
        self.assertEqual(result.get_value(), 1)
        result = true_expr_1 & false_expr
        self.assertEqual(result.get_value(), 0)

    def test_and_3(self):
        true_expr: Expression = Numeric(3) == Numeric(3)
        false_expr_1: Expression = Numeric(3) == Numeric(2)
        false_expr_2: Expression = Numeric(2) == Numeric(3)
        result: Expression = true_expr | false_expr_1
        self.assertEqual(result.get_value(), 1)
        result = false_expr_1 | false_expr_2
        self.assertEqual(result.get_value(), 0)


class TestExpressionConversion(unittest.TestCase):
    def setUp(self):
        self.beta = Beta('beta', 42, None, None, 0)

    def test_validate_and_convert_numeric(self):
        self.assertIsInstance(validate_and_convert(3.14), Numeric)

    def test_validate_and_convert_boolean(self):
        self.assertEqual(validate_and_convert(True).value, Numeric(1).value)
        self.assertEqual(validate_and_convert(False).value, Numeric(0).value)

    def test_validate_and_convert_invalid_type(self):
        with self.assertRaises(TypeError):
            validate_and_convert("invalid")

    def test_expression_to_value_numeric(self):
        self.assertEqual(expression_to_value(3.14), 3.14)

    def test_expression_to_value_boolean(self):
        self.assertEqual(expression_to_value(True), 1.0)

    def test_expression_to_value_invalid_type(self):
        with self.assertRaises(TypeError):
            expression_to_value("invalid")

    def test_expression_to_value_expression(self):
        self.assertEqual(expression_to_value(self.beta), 42.0)

    def test_get_dict_values(self):
        test_dict = {1: 3.14, 2: True, 3: self.beta}
        expected = {1: 3.14, 2: 1.0, 3: 42.0}
        self.assertEqual(get_dict_values(test_dict), expected)

    def test_get_dict_expressions(self):
        test_dict = {1: 3.14, 2: True, 3: self.beta}
        result = get_dict_expressions(test_dict)
        self.assertIsInstance(result[1], Numeric)
        self.assertIsInstance(result[2], Numeric)
        self.assertIsInstance(result[3], Beta)


class TestConvertToDict(unittest.TestCase):
    def test_basic_conversion(self):
        sequence = ['apple', 'banana', 'cherry']
        name_to_index = {'first': 0, 'second': 1, 'third': 2}
        expected_result = {'first': 'apple', 'second': 'banana', 'third': 'cherry'}
        self.assertEqual(convert_to_dict(sequence, name_to_index), expected_result)

    def test_empty_sequence(self):
        sequence = []
        name_to_index = {}
        expected_result = {}
        self.assertEqual(convert_to_dict(sequence, name_to_index), expected_result)

    def test_empty_map(self):
        sequence = ['apple', 'banana', 'cherry']
        name_to_index = {}
        expected_result = {}
        self.assertEqual(convert_to_dict(sequence, name_to_index), expected_result)

    def test_index_out_of_bounds(self):
        sequence = ['apple', 'banana']
        name_to_index = {'first': 0, 'third': 2}
        with self.assertRaises(IndexError):
            convert_to_dict(sequence, name_to_index)

    def test_negative_index(self):
        sequence = ['apple', 'banana', 'cherry']
        name_to_index = {'first': -1}
        with self.assertRaises(IndexError):
            convert_to_dict(sequence, name_to_index)

    def test_large_index_in_small_list(self):
        sequence = ['apple']
        name_to_index = {'second': 1}
        with self.assertRaises(IndexError):
            convert_to_dict(sequence, name_to_index)

    def test_with_non_integer_indices(self):
        sequence = ['apple', 'banana', 'cherry']
        name_to_index = {'first': '0'}  # index as a string by mistake
        with self.assertRaises(TypeError):  # Assumes TypeError for non-integers
            convert_to_dict(sequence, name_to_index)


if __name__ == '__main__':
    unittest.main()
