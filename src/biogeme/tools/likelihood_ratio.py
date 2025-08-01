from typing import NamedTuple

from scipy.stats import chi2

from biogeme.exceptions import BiogemeError


class LRTuple(NamedTuple):
    """Tuple for the likelihood ratio test"""

    message: str
    statistic: float
    threshold: float

    def __str__(self):
        return f'{self.message} (statistic: {self.statistic:.2f}, threshold: {self.threshold:.2f})'


def likelihood_ratio_test(
    model1: tuple[float, int],
    model2: tuple[float, int],
    significance_level: float = 0.05,
) -> LRTuple:
    """This function performs a likelihood ratio test between a
    restricted and an unrestricted model.

    :param model1: the final loglikelihood of one model, and the number of
                   estimated parameters.

    :param model2: the final loglikelihood of the other model, and
                   the number of estimated parameters.

    :param significance_level: level of significance of the test. Default: 0.05

    :return: a tuple containing:

                  - a message with the outcome of the test
                  - the statistic, that is minus two times the difference
                    between the loglikelihood  of the two models
                  - the threshold of the chi square distribution.

    :raise BiogemeError: if the unrestricted model has a lower log
        likelihood than the restricted model.

    """

    log_like_m1, df_m1 = model1
    log_like_m2, df_m2 = model2
    if log_like_m1 > log_like_m2:
        if df_m1 < df_m2:
            raise BiogemeError(
                f'The unrestricted model {model2} has a '
                f'lower log likelihood than the restricted one {model1}'
            )
        log_like_ur = log_like_m1
        log_like_r = log_like_m2
        df_ur = df_m1
        df_r = df_m2
    else:
        if df_m1 >= df_m2:
            raise BiogemeError(
                f'The unrestricted model {model1} has a '
                f'lower log likelihood than the restricted one {model2}'
            )
        log_like_ur = log_like_m2
        log_like_r = log_like_m1
        df_ur = df_m2
        df_r = df_m1

    stat = -2 * (log_like_r - log_like_ur)
    chi_df = df_ur - df_r
    threshold = chi2.ppf(1 - significance_level, chi_df)
    if stat <= threshold:
        final_msg = f'H0 cannot be rejected at level {100*significance_level:.1f}%'
    else:
        final_msg = f'H0 can be rejected at level {100*significance_level:.1f}%'
    return LRTuple(message=final_msg, statistic=stat, threshold=threshold)
