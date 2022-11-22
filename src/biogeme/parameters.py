from collections import namedtuple
import tomlkit

ParameterTuple = namedtuple(
    'ParameterTuple', 'name default section description'
)


class parameters:
    def __init__(self, parameter_file=None):
        self.document = None

    def default(self):
        self.parameters = [
            ParameterTuple(
                name = 'number_of_threads'
                default = None,
                section = 'MultiThreading',
                description= (
                    'Multi-threading can be used for estimation and '
                    'simulation. This parameter defines the number of '
                    'threads to be used. If the parameter is set to None, '
                    'the number of available threads is calculated using  '
                    'os.cpu_count().
                )
            ),
            ParameterTuple(
                name = 'number_of_draws'
                default = 1000,
                section = 'MonteCarlo',
                description= (
                    'Number of draws for Monte-Carlo integration.'
                )
            )
            ParameterTuple(
                name = 'skip_audit'
                default = False,
                section = 'Specification',
                description= (
                    'If True, does not check the validity of the '
                    'formulas. It may save significant amount of time '
                    'for large models and large data sets.'
                )
            )
            ParameterTuple(
                name = 'suggest_scales'
                default = True,
                section = 'Specification',
                description= (
                    'If True, Biogeme suggests the scaling of the '
                    'variables in the database.'
                )
            )
            ParameterTuple(
                name = 'missing_data'
                default = 99999,
                section = 'Specification',
                description= (
                    'If one variable has this value, it is assumed '
                    'that a data is missing and an exception will be triggered.'
                )
            )
            ParameterTuple(
                name = 'seed'
                default = None,
                section = 'MonteCarlo',
                description= (
                    'Seed used for the pseudo-random number generation.'
                    ' It is useful only when each run should generate '
                    'the exact same result. If None, a new seed is used '
                    'at each run.'
                )
            )

            

        self.save_iterationd = True
        

