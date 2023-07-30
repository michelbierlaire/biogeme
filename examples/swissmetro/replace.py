import fileinput
import os


def replace_in_files(file_list, search_string, replacement_string):
    for file_path in file_list:
        # Create a backup of the original file
        backup_file_path = file_path + '.bak'
        os.rename(file_path, backup_file_path)

        # Open the original file for reading and a temporary file for writing
        with fileinput.input(backup_file_path, inplace=True) as file, open(
            file_path, 'w'
        ) as temp_file:
            for line in file:
                # Replace the search string with the replacement string
                updated_line = line.replace(search_string, replacement_string)
                # Write the updated line to the temporary file
                temp_file.write(updated_line)

        # Remove the backup file
        os.remove(backup_file_path)


# Example usage
file_list = [
    'b01logit.py',
    'b02weight.py',
    'b03scale.py',
    'b05normal_mixture.py',
    'b05normal_mixture_integral.py',
    'b06unif_mixture.py',
    'b06unif_mixture_MHLS.py',
    'b06unif_mixture_integral.py',
    'b07discrete_mixture.py',
    'b08boxcox.py',
    'b09nested.py',
    'b10nested_bottom.py',
    'b11cnl.py',
    'b11cnl_sparse.py',
    'b12panel.py',
    'b12panel_flat.py',
    'b14nested_endogenous_sampling.py',
    'b15panel_discrete.py',
    'b15panel_discrete_bis.py',
    'b16panel_discrete_socio_eco.py',
    'b17lognormal_mixture.py',
    'b17lognormal_mixture_integral.py',
    'b18ordinal_logit.py',
    'b23logit.py',
    'b23probit.py',
    'b24halton_mixture.py',
    'b25triangular_mixture.py',
    'b26triangular_panel_mixture.py',
]

search_string = 'shortSummary'  # String to be replaced
replacement_string = 'short_summary'  # String to replace with

replace_in_files(file_list, search_string, replacement_string)
