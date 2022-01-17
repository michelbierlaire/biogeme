"""Class that provides some automatic specification for segmented parameters

:author: Michel Bierlaire
:date: Fri Dec 31 10:41:33 2021

"""

from collections import namedtuple, deque
from biogeme.expressions import Beta, bioMultSum

SegmentationTuple = namedtuple('SegmentationTuple', 'variable mapping')


def combine_segmented_expressions(variable, mapping_of_expressions):
    """Create an expressions that combines all the segments
    :param variable: variable that characterizes the segmentation
    :type variable: biogeme.expressions.Variable

    :param mapping_of_expressions: dictionary that maps each value of
        the variable with the expression for the corresponding
        segment.
    :type mapping_of_expressions: dict(int: biogeme.expressions.Expression)

    :return: combined expression
    :rtype: biogeme.expressions.bioMultSum

    """
    terms = [
        expr * (variable == value)
        for value, expr in mapping_of_expressions.items()
    ]
    return bioMultSum(terms)


def code_to_combine_segmented_expressions(
    variable, mapping_of_expressions, prefix
):
    """Create the Python code for an expressions that combines all the segments

    :param variable: variable that characterizes the segmentation
    :type variable: biogeme.expressions.Variable

    :param mapping_of_expressions: dictionary that maps each value of
        the variable with the Python code of the expression for the
        corresponding segment.
    :type mapping_of_expressions: dict(int: biogeme.expressions.Expression)

    :param prefix: name of the current expression, used as prefix
    :type prefix: str

    :return: code for the combined expression
    :rtype: str

    """
    result = ''
    for value, _ in mapping_of_expressions.items():
        result += (
            f'{prefix}_{variable}_{value} = '
            f'{prefix} * ({variable} == {value})\n'
        )
    terms = ', '.join(
        [
            f'{prefix}_{variable}_{value}'
            for value, expr in mapping_of_expressions.items()
        ]
    )
    result += f'{prefix}_{variable} = bioMultSum([{terms}])\n'
    return result


def create_segmented_parameter(parameter, mapping):
    """Create a version of the parameter for each segment

    :param parameter: parameter
    :type parameter: biogeme.expressions.Beta

    :param mapping: dictionary that maps each segment id with the name
        of the segment.
    :type mapping: dict(int: str)

    :return: a dictionary that maps each segment id with the created parameters
    :rtype: dict(int: biogeme.expressions.Beta)

    """
    segmented_parameters = {
        value: Beta(
            f'{parameter.name}_{name}',
            parameter.initValue,
            parameter.lb,
            parameter.ub,
            parameter.status,
        )
        for value, name in mapping.items()
    }
    return segmented_parameters


def segment_parameter(parameter, list_of_segmentations):
    """Segment a parameter expression along several dimensions of segmentation

    :param parameter: parameter to segment
    :type parameter: biogeme.expressions.Beta

    :param list_of_segmentations: each element of the list is a tuple
        with the variable characterizing the segmentation, and a
        dictionary mapping the values with the names of the segments.
    :type list_of_segmentations:
        tuple(SegmentationTuple(biogeme.expressions.Variable, dict(int:str)))

    :return: expression involving all the segments
    :rtype: biogeme.expressions.Expression

    """
    stack_of_segmentations = deque(list_of_segmentations)
    if not stack_of_segmentations:
        return parameter

    next_segment = stack_of_segmentations.pop()
    segmented_parameters = create_segmented_parameter(
        parameter, next_segment.mapping
    )
    map_of_expressions = {
        key: segment_parameter(value, stack_of_segmentations)
        for key, value in segmented_parameters.items()
    }
    return combine_segmented_expressions(
        next_segment.variable, map_of_expressions
    )


def code_to_segment_parameter(parameter, list_of_segmentations, prefix=''):
    """Generate the Python code to segment a parameter along several
        dimensions of segmentation

    :param parameter: parameter to segment
    :type parameter: biogeme.expressions.Beta

    :param list_of_segmentations: each element of the list is a tuple
        with the variable characterizing the segmentation, and a
        dictionary mapping the values with the names of the segments.
    :type list_of_segmentations:
        tuple(SegmentationTuple(biogeme.expressions.Variable, dict(int:str)))

    :param prefix: name of the current expression, used as prefix
    :type prefix: str

    :return: code for the segmentation
    :rtype: str

    """
    stack_of_segmentations = deque(list_of_segmentations)
    next_segment = stack_of_segmentations.pop()

    if prefix == '':
        prefix = parameter.name
    # prefix = 'beta_var_1'
    result = ''
    if stack_of_segmentations:
        for value in next_segment.mapping.values():
            result += code_to_segment_parameter(
                value, stack_of_segmentations, f'{prefix}_{value}'
            )
            result += '\n'
    else:
        for key, value in next_segment.mapping.items():
            param_name = f'{prefix}_{value}'
            result += f"{param_name} = Beta('{param_name}', 0, None, None)\n"
    terms = ', '.join(
        [
            f'{prefix}_{value} * ({next_segment.variable.name} == {key}))'
            for key, value in next_segment.mapping.items()
        ]
    )
    result += f'{prefix} = bioMultSum([{terms}])'
    return result
