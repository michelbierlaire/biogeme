"""Functions managing the dict of formulas used by Biogeme


:author: Michel Bierlaire

:date: Mon May 13 13:42:50 2024

"""

import difflib
import logging
from typing import Any

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression
from fuzzywuzzy import fuzz

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


def insert_valid_keyword(
    dict_of_formulas: dict[str, Expression],
    reference_keyword: str,
    valid_keywords: list[str],
) -> dict[str, Expression]:
    """Insert the reference keyword if a valid keyword is used."""
    key_log_like_expression: str | None = None
    for key in dict_of_formulas:
        if key == reference_keyword:
            continue
        if key in valid_keywords:
            if reference_keyword in dict_of_formulas:
                warning_msg = f'Both {reference_keyword} and {key} are defined. Only {reference_keyword} is considered.'
            else:
                warning_msg = (
                    f'As {key} is defined, it is used to define {reference_keyword}.'
                )
                key_log_like_expression = key
            logger.warning(warning_msg)

        else:
            matches = difflib.get_close_matches(
                reference_keyword, [key], n=1, cutoff=0.8
            )
            if matches:
                logger.warning(
                    f'Formula key "{key}" is similar to "{reference_keyword}". '
                    f'Did you mean to use "{reference_keyword}"?',
                )
    if key_log_like_expression is not None:
        dict_of_formulas[reference_keyword] = dict_of_formulas.pop(
            key_log_like_expression
        )
    return dict_of_formulas


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
