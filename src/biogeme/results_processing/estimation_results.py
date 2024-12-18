"""
Stores and process the estimation results

Michel Bierlaire
Mon Sep 30 15:39:24 2024
"""

from __future__ import annotations

import warnings
from enum import auto, Enum
from typing import Any

import numpy as np
import pandas as pd
from numpy.linalg import eigh, LinAlgError, pinv, inv, norm
from scipy.stats import norm as normal_distribution
from biogeme.exceptions import BiogemeError
from .recycle_pickle import read_pickle_biogeme
from .raw_estimation_results import (
    RawEstimationResults,
    serialize_to_yaml,
    deserialize_from_yaml,
)
from ..tools import likelihood_ratio


class EstimateVarianceCovariance(Enum):
    """Identifies the estimate of the variance-covariance matrix to be used."""

    RAO_CRAMER = auto()
    ROBUST = auto()
    BOOTSTRAP = auto()


def calc_p_value(t: float) -> float:
    """Calculates the p value of a parameter from its t-statistic.

    The formula is

    .. math:: 2(1-\\Phi(|t|))

    where :math:`\\Phi(\\cdot)` is the CDF of a normal distribution.

    :param t: t-statistics
    :type t: float

    :return: p-value
    :rtype: float
    """
    p_value = 2.0 * (1.0 - normal_distribution.cdf(abs(t)))
    return p_value


def calculates_correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    """Calculates the correlation matrix."""
    d = np.diag(covariance)
    if (d > 0).all():
        diagonal = np.diag(np.sqrt(d))
        inverse_diagonal = inv(diagonal)
        correlation = inverse_diagonal @ covariance @ inverse_diagonal
        return correlation
    return np.full_like(covariance, np.finfo(float).max)


class EstimationResults:
    """Extension of the raw estimation results."""

    def __init__(self, raw_estimation_results: RawEstimationResults | None) -> None:
        """Ctor. with data from the object

        :param raw_estimation_results: raw estimation results
        """
        self.raw_estimation_results = raw_estimation_results

    @classmethod
    def from_yaml_file(
        cls,
        filename: str,
    ) -> EstimationResults:
        """Ctor. with data from the file"""
        restored_results: RawEstimationResults = deserialize_from_yaml(
            filename=filename
        )
        return cls(raw_estimation_results=restored_results)

    @classmethod
    def from_pickle_file(
        cls,
        filename: str,
    ) -> EstimationResults:
        """Ctor. with data from the file"""
        restored_results: RawEstimationResults = read_pickle_biogeme(filename=filename)
        return cls(raw_estimation_results=restored_results)

    def __getattr__(self, name: str) -> Any:
        # If the attribute is not found in the current instance, check raw_estimation_results
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available.')
        return getattr(self.raw_estimation_results, name)

    @property
    def number_of_parameters(self) -> int:
        """Number of estimated parameters"""
        if self.raw_estimation_results is None:
            return 0
        return len(self.raw_estimation_results.beta_names)

    @property
    def number_of_free_parameters(self) -> int:
        """This is the number of estimated parameters, minus those that are at
        their bounds
        """
        if self.raw_estimation_results is None:
            return 0
        return sum(
            not self.is_bound_active(parameter_name=beta) for beta in self.beta_names
        )

    def is_any_bound_active(self, threshold: float = 1.0e-6) -> bool:
        """

        :param threshold: distance below which the bound is considered to be
            active.
        :return: True is any bound constraint is active
        """
        if self.raw_estimation_results is None:
            return False
        number_of_active_constraints = sum(
            self.is_bound_active(parameter_name=beta, threshold=threshold)
            for beta in self.beta_names
        )
        return number_of_active_constraints > 0

    def is_bound_active(self, parameter_name: str, threshold: float = 1.0e-6) -> bool:
        """Check if one of the two bound is 'numerically' active. Being
        numerically active means that the distance between the value of
        the parameter and one of its bounds is below the threshold.

        :param parameter_name: name of the parameter
        :param threshold: distance below which the bound is considered to be
            active.

        :return: True is one of the two bounds is numerically active.

        :raise BiogemeError: if ``threshold`` is negative.
        :raise BiogemeError: if no result is available
        :raise ValueError: if the parameter does not exist

        """
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        if threshold < 0:
            raise BiogemeError(f'Threshold ({threshold}) must be non negative')
        index_param = self.beta_names.index(parameter_name)
        if (
            self.lower_bounds[index_param] is not None
            and np.abs(self.beta_values[index_param] - self.lower_bounds[index_param])
            <= threshold
        ):
            return True
        if (
            self.upper_bounds[index_param] is not None
            and np.abs(self.beta_values[index_param] - self.upper_bounds[index_param])
            <= threshold
        ):
            return True
        return False

    def get_beta_values(self, my_betas: list[str] | None = None) -> dict[str, float]:
        """Retrieve the values of the estimated parameters, by names.

        :param my_betas: names of the requested parameters. If None, all
                  available parameters will be reported. Default: None.

        :return: dict containing the values, where the keys are the names.
        """
        if self.raw_estimation_results is None:
            return {}
        beta_values = {
            name: self.beta_values[index]
            for index, name in enumerate(self.beta_names)
            if my_betas is None or name in my_betas
        }
        return beta_values

    def get_betas_for_sensitivity_analysis(
        self,
        my_betas: list[str],
        size: int = 100,
        use_bootstrap: bool = True,
        variance_covariance_type: EstimateVarianceCovariance = EstimateVarianceCovariance.ROBUST,
    ) -> list[dict[str, float]]:
        """Generate draws from the distribution of the estimates, for
        sensitivity analysis.

        :param my_betas: names of the parameters for which draws are requested.
        :param size: number of draws. If use_bootstrap is True, the value is
            ignored and a warning is issued. Default: 100.
        :param use_bootstrap: if True, the bootstrap estimates are
                  directly used. The advantage is that it does not rely on the
                  assumption that the estimates follow a normal
                  distribution.
        :param variance_covariance_type: type of variance covariance matrix to use.

        :raise biogeme.exceptions.BiogemeError: if use_bootstrap is True and
            the bootstrap results are not available

        :return: list of dict. Each dict has a many entries as parameters.
                The list has as many entries as draws.

        """
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        if use_bootstrap and self.raw_estimation_results.bootstrap is None:
            err = (
                'Bootstrap results are not available for simulation. '
                'Use use_bootstrap=False.'
            )
            raise BiogemeError(err)

        index = [self.get_parameter_index(parameter_name=b) for b in my_betas]
        # If we use bootstrap, we simply return the bootstrap draws
        if use_bootstrap:
            results = [
                {my_betas[i]: value for i, value in enumerate([row[j] for j in index])}
                for row in self.raw_estimation_results.bootstrap
            ]
            return results

        the_matrix = self.get_variance_covariance_matrix(
            variance_covariance_type=variance_covariance_type
        )
        simulated_betas = np.random.multivariate_normal(
            self.beta_values, the_matrix, size
        )

        results = [
            {my_betas[i]: value for i, value in enumerate(row)}
            for row in simulated_betas[:, index]
        ]
        return results

    @property
    def akaike_information_criterion(self) -> float:
        """Calculates the AIC

        :return: AIC
        """
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        akaike = (
            2.0 * self.number_of_parameters
            - 2 * self.raw_estimation_results.final_log_likelihood
        )
        return akaike

    @property
    def bayesian_information_criterion(self) -> float:
        """Calculates the BIC

        :return: BIC
        """
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        bayesian = (
            -2.0 * self.raw_estimation_results.final_log_likelihood
            + self.number_of_parameters
            * np.log(self.raw_estimation_results.sample_size)
        )
        return bayesian

    @property
    def algorithm_has_converged(self) -> bool:
        """Reports if the algorithm has indeed converged

        :return: True if the algorithm has converged.
        :rtype: bool
        """
        if self.raw_estimation_results is None:
            return False
        return self.raw_estimation_results.convergence

    @property
    def variance_covariance_missing(self) -> bool:
        """Check if the variance covariance matrix is missing

        :return: True if missing.
        :rtype: bool
        """
        if self.raw_estimation_results is None:
            return True
        return self.raw_estimation_results.hessian is None

    def calculate_test(self, i: int, j: int, matrix: np.ndarray) -> float:
        """Calculates a t-test comparing two coefficients

        Args:
           i: index of first coefficient \f$\\beta_i\f$.
           j: index of second coefficient \f$\\beta_i\f$.
           matrix: estimate of the variance-covariance matrix \f$m\f$.

        :return: t test
            ..math::  \f[\\frac{\\beta_i-\\beta_j}
                  {\\sqrt{m_{ii}+m_{jj} - 2 m_{ij} }}\f]
        :rtype: float

        """
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        vi = self.raw_estimation_results.beta_values[i]
        vj = self.raw_estimation_results.beta_values[j]
        var_i = matrix[i, i]
        var_j = matrix[j, j]
        covar = matrix[i, j]
        r = var_i + var_j - 2.0 * covar
        if r <= 0:
            test = np.finfo(float).max
        else:
            test = (vi - vj) / np.sqrt(r)
        return test

    @property
    def likelihood_ratio_null(self) -> float:
        """Likelihood ratio test against the null model"""
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        if self.raw_estimation_results.null_log_likelihood is None:
            raise BiogemeError('Null log likelihood has not been calculated')
        test = -2.0 * (
            self.raw_estimation_results.null_log_likelihood
            - self.raw_estimation_results.final_log_likelihood
        )
        return test

    @property
    def likelihood_ratio_init(self) -> float:
        """Likelihood ratio test against the initial model"""
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        if self.raw_estimation_results.initial_log_likelihood is None:
            raise BiogemeError('Initial log likelihood not available')
        test = -2.0 * (
            self.raw_estimation_results.initial_log_likelihood
            - self.raw_estimation_results.final_log_likelihood
        )
        return test

    @property
    def rho_square_init(self) -> float:
        """McFadden rho square normalized to the initial model"""
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        if self.raw_estimation_results.initial_log_likelihood is None:
            raise BiogemeError('Initial log likelihood not available')
        try:
            rho_square = np.nan_to_num(
                1.0
                - self.raw_estimation_results.final_log_likelihood
                / self.raw_estimation_results.initial_log_likelihood
            )
            return rho_square
        except ZeroDivisionError:
            raise BiogemeError('Initial log likelihood is zero')

    @property
    def rho_square_null(self) -> float:
        """McFadden rho square normalized to the null model"""
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        if self.raw_estimation_results.null_log_likelihood is None:
            raise BiogemeError('Null log likelihood not available')
        try:
            rho_square = np.nan_to_num(
                1.0
                - self.raw_estimation_results.final_log_likelihood
                / self.raw_estimation_results.null_log_likelihood
            )
            return rho_square
        except ZeroDivisionError:
            raise BiogemeError('Null log likelihood is zero')

    @property
    def rho_bar_square_init(self) -> float:
        """Corrected McFadden rho square normalized to the initial model"""
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        if self.raw_estimation_results.initial_log_likelihood is None:
            raise BiogemeError('Initial log likelihood not available')
        try:
            rho_bar_square = np.nan_to_num(
                1.0
                - (
                    self.raw_estimation_results.final_log_likelihood
                    - self.number_of_parameters
                )
                / self.raw_estimation_results.initial_log_likelihood
            )
            return rho_bar_square
        except ZeroDivisionError:
            raise BiogemeError('Initial log likelihood is zero')

    @property
    def rho_bar_square_null(self) -> float:
        """Corrected McFadden rho square normalized to the null model"""
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        if self.raw_estimation_results.null_log_likelihood is None:
            raise BiogemeError('Null log likelihood not available')
        try:
            rho_square = np.nan_to_num(
                1.0
                - (
                    self.raw_estimation_results.final_log_likelihood
                    - self.number_of_parameters
                )
                / self.raw_estimation_results.null_log_likelihood
            )
            return rho_square
        except ZeroDivisionError:
            raise BiogemeError('Null log likelihood is zero')

    @property
    def gradient_norm(self) -> float:
        """Norm of the final gradient"""
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available.')
        if self.raw_estimation_results.gradient is None:
            raise BiogemeError('No gradient available')
        gradient_norm = float(norm(self.raw_estimation_results.gradient))
        return gradient_norm

    def eigen_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the eigen structure of the hessian matrix"""
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        try:
            eigenvalues, eigenvectors = eigh(
                -np.nan_to_num(np.array(self.raw_estimation_results.hessian))
            )
            return eigenvalues, eigenvectors
        except LinAlgError as e:
            raise BiogemeError(f'Error in calculating the eigen structure: {e}')

    @property
    def smallest_eigenvalue(self) -> float:
        """Calculates the smallest eigen value of the hessian matrix."""
        eigen_structure = self.eigen_structure()
        eigenvalues, _ = eigen_structure
        return float(np.min(eigenvalues))

    @property
    def smallest_eigenvector(self) -> np.ndarray:
        """Calculates the eigenvector corresponding to the smallest eigen value of the hessian matrix."""
        eigen_structure = self.eigen_structure()
        eigenvalues, eigenvectors = eigen_structure
        min_eigen_index = np.argmin(eigenvalues)
        smallest_eigen_vector = eigenvectors[:, min_eigen_index]
        return smallest_eigen_vector

    @property
    def largest_eigenvalue(self) -> float:
        """Calculates the largest eigen value of the hessian matrix."""
        eigen_structure = self.eigen_structure()
        eigenvalues, _ = eigen_structure
        return float(np.max(eigenvalues))

    @property
    def largest_eigenvector(self) -> np.ndarray:
        """Calculates the eigenvector corresponding to the largest eigen value of the hessian matrix."""
        eigen_structure = self.eigen_structure()
        eigenvalues, eigenvectors = eigen_structure
        max_eigen_index = np.argmax(eigenvalues)
        smallest_eigen_vector = eigenvectors[:, max_eigen_index]
        return smallest_eigen_vector

    @property
    def condition_number(self) -> float:
        """Calculates the condition number of the hessian matrix"""
        try:
            condition_number = self.largest_eigenvalue / self.smallest_eigenvalue
            return condition_number
        except (ZeroDivisionError, TypeError):
            return float(np.finfo(np.float64).max)

    @property
    def rao_cramer_variance_covariance_matrix(self) -> np.ndarray:
        """Calculates the variance-covariance matrix as the (pseudo) inverse of the hessian. We use the pseudo
        inverse in case the matrix is singular

        """
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available.')
        var_covar = -pinv(np.nan_to_num(np.array(self.raw_estimation_results.hessian)))
        return var_covar

    def get_variance_covariance_matrix(
        self, variance_covariance_type: EstimateVarianceCovariance
    ) -> np.ndarray:
        """Returns the variance-covariance matrix of a given type"""
        if variance_covariance_type == EstimateVarianceCovariance.RAO_CRAMER:
            return self.rao_cramer_variance_covariance_matrix
        if variance_covariance_type == EstimateVarianceCovariance.ROBUST:
            return self.robust_variance_covariance_matrix
        if variance_covariance_type == EstimateVarianceCovariance.BOOTSTRAP:
            return self.bootstrap_variance_covariance_matrix
        raise ValueError(f'Unknown type: {variance_covariance_type}')

    @property
    def robust_variance_covariance_matrix(self) -> np.ndarray:
        """Calculates the "sandwich" estimate of the variance-covariance matrix"""

        robust_var_covar = (
            self.rao_cramer_variance_covariance_matrix
            @ np.array(self.raw_estimation_results.bhhh)
            @ self.rao_cramer_variance_covariance_matrix
        )
        return robust_var_covar

    @property
    def bootstrap_variance_covariance_matrix(self) -> np.ndarray:
        """Calculates the bootstrap variance-covariance matrix"""

        if self.raw_estimation_results.bootstrap is None:
            error_msg = f'No bootstrap samples is available.'
            raise BiogemeError(error_msg)

        bootstrap_var_covar = np.cov(
            np.array(self.raw_estimation_results.bootstrap), rowvar=False
        )
        return bootstrap_var_covar

    def dump_yaml_file(self, filename: str) -> None:
        """Dump the raw estimation results in a Yaml file

        :param filename: name of the file
        """
        if self.raw_estimation_results is None:
            raise BiogemeError('No result available')
        serialize_to_yaml(data=self.raw_estimation_results, filename=filename)

    @property
    def monte_carlo(self) -> bool:
        """Verifies if the model involves Monte-Carlo simulation"""
        if self.raw_estimation_results is None:
            return False
        return self.raw_estimation_results.monte_carlo

    def short_summary(self) -> str:
        """Provides a short summary of the estimation results"""
        if self.raw_estimation_results is None:
            text = 'No estimation result is available.'
            return text

        text = ''
        text += f'Results for model {self.raw_estimation_results.model_name}\n'
        text += f'Nbr of parameters:\t\t{self.number_of_parameters}\n'
        text += f'Sample size:\t\t\t{self.raw_estimation_results.sample_size}\n'
        if (
            self.raw_estimation_results.sample_size
            != self.raw_estimation_results.number_of_observations
        ):
            text += f'Observations:\t\t\t{self.raw_estimation_results.number_of_observations}\n'
        text += f'Excluded data:\t\t\t{self.raw_estimation_results.number_of_excluded_data}\n'
        if self.raw_estimation_results.null_log_likelihood is not None:
            text += f'Null log likelihood:\t\t{self.raw_estimation_results.null_log_likelihood:.7g}\n'
        text += f'Final log likelihood:\t\t{self.raw_estimation_results.final_log_likelihood:.7g}\n'
        if self.raw_estimation_results.null_log_likelihood is not None:
            text += (
                f'Likelihood ratio test (null):\t\t'
                f'{self.likelihood_ratio_null:.7g}\n'
            )
            text += f'Rho square (null):\t\t\t{self.rho_square_null:.3g}\n'
            text += f'Rho bar square (null):\t\t\t' f'{self.rho_bar_square_null:.3g}\n'
        text += (
            f'Akaike Information Criterion:\t{self.akaike_information_criterion:.7g}\n'
        )
        text += f'Bayesian Information Criterion:\t{self.bayesian_information_criterion:.7g}\n'
        return text

    def __str__(self) -> str:
        return self.short_summary()

    def get_general_statistics(self) -> dict[str, str]:
        """Format the results in a dict

        :return: dict with the results. The keys describe each
                 content.

        """
        d = {'Number of estimated parameters': f'{self.number_of_parameters}'}
        if self.number_of_free_parameters != self.number_of_parameters:
            d['Number of free parameters'] = f'{self.number_of_free_parameters}'
        d['Sample size'] = f'{self.sample_size}'
        if self.sample_size != self.number_of_observations:
            d['Observations'] = f'{self.number_of_observations}'
        d['Excluded observations'] = f'{self.number_of_excluded_data}'
        if self.null_log_likelihood is not None:
            d['Null log likelihood'] = f'{self.null_log_likelihood:.7g}'
        d['Init log likelihood'] = f'{self.initial_log_likelihood:.7g}'
        d['Final log likelihood'] = f'{self.final_log_likelihood:.7g}'
        if self.null_log_likelihood is not None:
            d['Likelihood ratio test for the null model'] = (
                f'{self.likelihood_ratio_null:.7g}'
            )
            d['Rho-square for the null model'] = f'{self.rho_square_null:.3g}'
            d['Rho-square-bar for the null model'] = f'{self.rho_bar_square_null:.3g}'
        d['Likelihood ratio test for the init. model'] = (
            f'{self.likelihood_ratio_init:.7g}'
        )
        try:
            rho_square_init = f'{self.rho_square_init:.3g}'
        except (BiogemeError, ZeroDivisionError):
            rho_square_init = ''
        d['Rho-square for the init. model'] = rho_square_init
        try:
            rho_bar_square_init = f'{self.rho_bar_square_init:.3g}'
        except (BiogemeError, ZeroDivisionError):
            rho_bar_square_init = ''
        d['Rho-square-bar for the init. model'] = rho_bar_square_init
        try:
            akaike = f'{self.akaike_information_criterion:.7g}'
        except BiogemeError:
            akaike = ''
        d['Akaike Information Criterion'] = akaike
        try:
            bayesian = f'{self.bayesian_information_criterion:.7g}'
        except BiogemeError:
            bayesian = ''
        d['Bayesian Information Criterion'] = bayesian

        try:
            gradient_norm = f'{self.gradient_norm:.4E}'
        except BiogemeError:
            gradient_norm = ''
        d['Final gradient norm'] = gradient_norm
        if self.monte_carlo:
            d['Number of draws'] = f'{self.number_of_draws}'
            d['Draws generation time'] = f'{self.draws_processing_time}'
            d['Types of draws'] = ', '.join(
                [f'{i}: {k}' for i, k in self.types_of_draws.items()]
            )
        if self.bootstrap is not None:
            d['Bootstrapping time'] = f'{self.bootstrap_time}'
        d['Nbr of threads'] = f'{self.number_of_threads}'
        return d

    def get_parameter_index(self, parameter_name: str) -> int:
        """Retrieve the index of a parameter

        :param parameter_name: name of the parameter
        :return: index of the parameter
        """
        index_param = self.beta_names.index(parameter_name)
        return index_param

    def get_parameter_value(self, parameter_index: int) -> float:
        """Retrieve the estimated value of a parameter

        :param parameter_index: index of the parameter
        """
        if parameter_index < 0 or parameter_index >= len(self.beta_values):
            error_msg = (
                f'Invalid parameter index {parameter_index}. Valid range: 0- '
                f'{len(self.beta_values)-1}'
            )
            raise ValueError(error_msg)

        return self.beta_values[parameter_index]

    def get_parameter_std_err(
        self, parameter_index: int, estimate_var_covar: EstimateVarianceCovariance
    ) -> float:
        """Calculates the standard error of the parameter estimate"""
        if parameter_index < 0 or parameter_index >= len(self.beta_values):
            error_msg = (
                f'Invalid parameter index {parameter_index}. Valid range: 0- '
                f'{len(self.beta_values)-1}'
            )
            raise ValueError(error_msg)
        var_covar = self.get_variance_covariance_matrix(
            variance_covariance_type=estimate_var_covar
        )
        variance = var_covar[parameter_index, parameter_index]
        if variance < 0:
            return float(np.finfo(float).max)
        return np.sqrt(variance)

    def get_parameter_t_test(
        self,
        parameter_index: int,
        estimate_var_covar: EstimateVarianceCovariance,
        target: float = 0.0,
    ) -> float:
        """Calculates the t-test of the parameter estimate"""
        if parameter_index < 0 or parameter_index >= len(self.beta_values):
            error_msg = (
                f'Invalid parameter index {parameter_index}. Valid range: 0- '
                f'{len(self.beta_values)-1}'
            )
            raise ValueError(error_msg)
        value = self.get_parameter_value(parameter_index=parameter_index)
        std_err = self.get_parameter_std_err(
            parameter_index=parameter_index, estimate_var_covar=estimate_var_covar
        )
        if std_err == 0:
            return float(np.finfo(float).max)

        return float(np.nan_to_num((value - target) / std_err))

    def get_parameter_p_value(
        self,
        parameter_index: int,
        estimate_var_covar: EstimateVarianceCovariance,
        target: float = 0.0,
    ) -> float:
        """Calculates the p-value of the parameter estimate"""
        if parameter_index < 0 or parameter_index >= len(self.beta_values):
            error_msg = (
                f'Invalid parameter index {parameter_index}. Valid range: 0- '
                f'{len(self.beta_values)-1}'
            )
            raise ValueError(error_msg)
        t_test = self.get_parameter_t_test(
            parameter_index=parameter_index,
            estimate_var_covar=estimate_var_covar,
            target=target,
        )
        return calc_p_value(t_test)

    def likelihood_ratio_test(
        self, other_model: EstimationResults, significance_level: float = 0.05
    ) -> likelihood_ratio.LRTuple:
        """This function performs a likelihood ratio test between a restricted
        and an unrestricted model. The "self" model can be either the
        restricted or the unrestricted.

        :param other_model: other model to perform the test.
        :param significance_level: level of significance of the
            test. Default: 0.05
        :return: a tuple containing:

                  - a message with the outcome of the test
                  - the statistic, that is minus two times the difference
                    between the loglikelihood  of the two models
                  - the threshold of the chi square distribution.

        """
        lr = self.final_log_likelihood
        lu = other_model.final_log_likelihood
        kr = self.number_of_parameters
        ku = other_model.number_of_parameters
        return likelihood_ratio.likelihood_ratio_test(
            (lu, ku), (lr, kr), significance_level
        )

    def get_estimated_parameters(self) -> pd.DataFrame:
        """For backward compatibility"""
        from . import get_pandas_estimated_parameters

        msg = (
            'get_estimated_parameters is deprecated. '
            'Use get_pandas_estimated_parameters(estimation_results=my_results) instead'
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return get_pandas_estimated_parameters(estimation_results=self)
