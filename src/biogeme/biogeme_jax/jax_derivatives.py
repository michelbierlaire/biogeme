"""Calculate the derivative of the likelihood function using Jax"""
from biogeme.expressions import Expression
from biogeme.function_output import BiogemeFunctionOutputSmartOutputProxy


def calculate_derivatives(loglikelihood: Expression, gradient: bool, hessian: bool, bhhh: bool) -> (
        BiogemeFunctionOutputSmartOutputProxy):

