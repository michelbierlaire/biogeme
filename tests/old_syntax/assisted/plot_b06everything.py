"""

Combine many specifications: exception is raised
================================================

We combine many specifications, defined in :ref:`everything_spec_section`.
This leads to a total of 432 specifications.
When the total number of specifications exceeds 100, Biogeme raises an
ValueOutOfRange exception when the estimate_catalog is called.
See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_.

:author: Michel Bierlaire, EPFL
:date: Fri Jul 14 09:55:33 2023

"""
import biogeme.biogeme as bio
from biogeme.exceptions import BiogemeError

# %%
# See :ref:`everything_spec_section`.
from everything_spec import model_catalog, database

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, model_catalog)
the_biogeme.modelName = 'b06everything'
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False

# %%
# Estimate the parameters.
# It does not work as there are two many specifications
try:
    dict_of_results = the_biogeme.estimate_catalog()
except BiogemeError as e:
    print(e)
