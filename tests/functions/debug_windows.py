import pandas as pd

from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import Variable, Beta, Numeric

df1 = pd.DataFrame(
    {
        'Person': [1, 1, 1, 2, 2],
        'Exclude': [0, 0, 1, 0, 1],
        'Variable1': [1, 2, 3, 4, 5],
        'Variable2': [10, 20, 30, 40, 50],
        'Choice': [1, 2, 3, 1, 2],
        'Av1': [0, 1, 1, 1, 1],
        'Av2': [1, 1, 1, 1, 1],
        'Av3': [0, 1, 1, 1, 1],
    }
)
Variable1 = Variable('Variable1')
Variable2 = Variable('Variable2')
beta1 = Beta('beta1', -1.0, -3, 3, 0)
beta2 = Beta('beta2', 2.0, -3, 10, 0)
likelihood = -((beta1 * Variable1) ** 2) - (beta2 * Variable2) ** 2
simul = (beta1 + 2 * beta2) / Variable1 + (beta2 + 2 * beta1) / Variable2
# dict_of_expressions = {
#    'log_like': likelihood,
#    'weight': Numeric(1),
#    'beta1': beta1,
#    'simul': simul,
# }

dict_of_expressions = {
    'log_like': likelihood,
}
data = Database(f'test_1', df1)

my_biogeme = BIOGEME(data, dict_of_expressions)
results = my_biogeme.estimate()
my_biosim = BIOGEME(data, dict_of_expressions)
s = my_biosim.simulate(results.get_beta_values())
print('******** DONE **************************')
# draws_from_betas = results.get_betas_for_sensitivity_analysis(
#    my_biogeme.id_manager.free_betas.names
# )
# s = my_biogeme.simulate(results.get_beta_values())
# left, right = my_biogeme.confidence_intervals(draws_from_betas)
