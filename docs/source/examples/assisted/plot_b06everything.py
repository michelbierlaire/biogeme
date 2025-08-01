"""

Combine many specifications: exception is raised
================================================

We combine many specifications, defined in :ref:`everything_spec_section`.
This leads to a total of 432 specifications.
When the total number of specifications exceeds 100, Biogeme raises an
ValueOutOfRange exception when the estimate_catalog is called.
See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_.

Michel Bierlaire, EPFL
Sun Apr 27 2025, 15:57:10
"""

from biogeme.biogeme import BIOGEME
from biogeme.exceptions import BiogemeError

# %%
# See :ref:`everything_spec_section`.
from everything_spec import database, model_catalog

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, model_catalog, generate_html=False, generate_yaml=False)
the_biogeme.model_name = 'b06everything'

# %%
# Estimate the parameters.
# It does not work as there are too many specifications
try:
    dict_of_results = the_biogeme.estimate_catalog()
except BiogemeError as e:
    print(e)
