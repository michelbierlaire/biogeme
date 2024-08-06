"""Functions managing the dict of formulas used by Biogeme


:author: Michel Bierlaire

:date: Mon May 13 13:42:50 2024

"""

import logging
from typing import Any

from fuzzywuzzy import fuzz

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression

logger = logging.getLogger(__name__)

# Valid names for the log likelihood.  Only one of them can be present in the dict.
log_like_names = ['log_like', 'loglike']

# Valid names for the weights.  Only one of them can be present in the dict.
weight_names = ['log_like', 'loglike']


def check_validity(dict_of_formulas: dict[str, Any]) -> None:
    """Verifies if the formulas are Biogeme expressions. If not, an exception is raised"""
    for k, f in dict_of_formulas.items():
        if not isinstance(f, Expression):
            raise BiogemeError(
                f'Expression for "{k}" is not of type '
                f"biogeme.expressions.Expression. "
                f"It is of type {type(f)}"
            )


def is_similar_to(word_1: str, word_2: str) -> bool:
    """Checks if two words are similar."""
    return fuzz.ratio(word_1, word_2) >= 80


def get_expression(
    dict_of_formulas: dict[str, Expression], valid_keywords: list[str]
) -> Expression | None:
    """Extract the formula for specific keywords

    :param dict_of_formulas: as the name says...
    :param valid_keywords: keywords that are considered valid to represent the expression
    :return: the requested expression
    """
    found_name = None
    for valid_name in valid_keywords:
        the_expression = dict_of_formulas.get(valid_name)
        if the_expression is not None:
            if found_name is not None:
                error_msg = (
                    f"This expression can be defined with the keyword "
                    f"'{found_name}' or '{valid_name}' but not both."
                )
                raise BiogemeError(error_msg)
            found_name = valid_name

    if found_name is None:
        for valid_name in valid_keywords:
            for key in dict_of_formulas:
                if is_similar_to(valid_name, key):
                    warning_msg = f'In the formulas, one key is "{key}". Should it be "{valid_name}" instead?'
                    logger.warning(warning_msg)
        return None
    return dict_of_formulas[found_name]
