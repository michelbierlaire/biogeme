"""
Old estimation result object.

Michel Bierlaire
Mon Oct 7 13:53:50 2024
"""

from biogeme.exceptions import BiogemeError

text = """
The old and unstructured `bioResults` object is now obsolete. Biogeme has introduced a new, more streamlined object 
called `EstimationResults`, which you can access as follows:

    from biogeme.results_processing.estimation_results import EstimationResults

Transition from Pickle to YAML

Biogeme has discontinued using the `pickle` format for saving estimation results due to its lack of transparency 
and inconsistencies across different software versions. Instead, Biogeme now saves results in a YAML format, 
which is more transparent and readable.

Converting Pickle Files to YAML

If you have an existing pickle file, you can easily convert it to the YAML format with the following command:

    from biogeme.results_processing import pickle_to_yaml
    pickle_to_yaml(pickle_filename='my_model.pickle', yaml_filename='my_model.yaml')

Accessing Estimation Results from Pickle or YAML

You can create an `EstimationResults` object directly from either a pickle file or a YAML file as follows:

    from biogeme.results_processing import EstimationResults
    results = EstimationResults.from_pickle_file(filename='my_model.pickle')
    results = EstimationResults.from_yaml_file(filename='my_model.yaml')
"""


class bioResults:
    """Class managing the estimation results"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        raise BiogemeError(text)
