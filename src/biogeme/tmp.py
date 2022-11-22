beta_var_1_my_10 = Beta('beta_var_1_my_10', 0, None, None, 0)
beta_var_1_my_20 = Beta('beta_var_1_my_20', 0, None, None, 0)
beta_var_2_my_10 = Beta('beta_var_2_my_10', 0, None, None, 0)
beta_var_2_my_20 = Beta('beta_var_2_my_20', 0, None, None, 0)

term_beta_var_1_my_10 = beta_var_1_my_10 * (Variable2 == 10)
term_beta_var_1_my_20 = beta_var_1_my_20 * (Variable2 == 20)
beta_var_1 = bioMultSum([term_beta_var_1_my_10, term_beta_var_1_my_20])

term_beta_var_2_my_10 = beta_var_2_my_10 * (Variable2 == 10)
term_beta_var_2_my_20 = beta_var_2_my_20 * (Variable2 == 20)
beta_var_2 = bioMultSum([term_beta_var_2_my_10, term_beta_var_2_my_20])

term_beta_var_1 = beta_var_1 * (Variable1 == 1)
term_beta_var_2 = beta_var_2 * (Variable1 == 2)
beta = bioMultSum([term_beta_var_1, term_beta_var_2])
