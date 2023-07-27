"""File results_analysis

:author: Michel Bierlaire, EPFL
:date: Thu Jul 13 16:32:45 2023

Reports the results of the catalog estimation
"""


from biogeme.results import compile_estimation_results, pareto_optimal


def report(dict_of_results):
    """Reports the results of the estimared catalogs"""
    print(f'A total of {len(dict_of_results)} models have been estimated')
    print('== Estimation results ==')

    compiled_results, specs = compile_estimation_results(
        dict_of_results, use_short_names=True
    )
    print(compiled_results)
    for short_name, spec in specs.items():
        print(f'{short_name}\t{spec}')

    pareto_results = pareto_optimal(dict_of_results)
    compiled_pareto_results, pareto_specs = compile_estimation_results(
        pareto_results, use_short_names=True
    )
    print(compiled_pareto_results)
    for short_name, spec in pareto_specs.items():
        print(f'{short_name}\t{spec}')
