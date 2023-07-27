"""File b06everything.py

:author: Michel Bierlaire, EPFL
:date: Fri Jul 14 09:55:33 2023

Investigate various specifications:
- 3 models
    - logit
    - nested logit with two nests: public and private transportation
    - nested logit with two nests existing and future modes
- 3 functional form for the travel time variables
    - linear specification,
    - Box-Cox transform,
    - power series,
- 2 specification for the cost coefficients:
    - generic
    - alternative specific
- 2 specification for the travel time coefficients:
    - generic
    - alternative specific
- 4 segmentations for the constants:
    - not segmented
    - segmented by GA (yearly subscription to public transport)
    - segmented by luggage
    - segmented both by GA and luggage
-  3 segmentations for the time coefficients:
    - not segmented
    - segmented with first class
    - segmented with trip purpose

This leads to a total of 432 specifications.
When the total number of specifications exceeds 100, Biogeme raises an
ValueOutOfRange exception when the estimate_catalog is called.

"""
import biogeme.biogeme as bio
from biogeme.exceptions import BiogemeError
from everything_spec import model_catalog, database

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, model_catalog)
the_biogeme.modelName = 'b06everything'
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False

# Estimate the parameters.
# It won't work as there are two many specifications
try:
    dict_of_results = the_biogeme.estimate_catalog()
except BiogemeError as e:
    print(e)
