"""

biogeme.segmentation
====================

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

:author: Michel Bierlaire
:date: Thu Dec  7 15:18:45 2023
"""
from biogeme.version import getText
import biogeme.segmentation as seg
from biogeme.expressions import Variable, Beta

# %%
# Version of Biogeme.
print(getText())

# %%
socio_eco_1 = Variable('socio_eco_1')

# %%
segmentation_1 = seg.DiscreteSegmentationTuple(
    variable=socio_eco_1,
    mapping={
        1: 'high',
        2: 'medium',
        3: 'low',
    },
)

# %%
socio_eco_2 = Variable('socio_eco_2')

# %%
segmentation_2 = seg.DiscreteSegmentationTuple(
    variable=socio_eco_2,
    mapping={
        1: 'level_1',
        2: 'level_2',
        3: 'level_3',
        4: 'level_4',
    },
)

# %%
beta_x = Beta('beta_x', 0, None, None, 0)

# %%
segmented_parameter = seg.Segmentation(beta_x, (segmentation_1,))

# %%
# The expressions for the segmented parameter is obtained as follows:
segmented_parameter.segmented_beta()

# %%
# The corresponding Python code can be obtained as follows.
print(segmented_parameter.segmented_code())

# %%
# The code of the original beta is also available:
print(segmented_parameter.beta_code())

# %%
# Same with the second segmentation
segmented_parameter = seg.Segmentation(beta_x, (segmentation_2,))

# %%
segmented_parameter.segmented_beta()

# %%
print(segmented_parameter.segmented_code())

# %%
# The two segmentations can be combined together
segmented_parameter = seg.Segmentation(
    beta_x,
    (
        segmentation_1,
        segmentation_2,
    ),
)

# %%
segmented_parameter.segmented_beta()

# %%
print(segmented_parameter.segmented_code())
