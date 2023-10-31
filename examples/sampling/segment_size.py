"""segment_size.py

:author: Michel Bierlaire
:date: Tue Aug 29 17:52:38 2023

"""


def generate_segment_size(sample_size, k):
    """This function calculates the size of the k segments, so that
    they are as close to each other as possible, and cover the full sample


    """
    # Calculate the base value and the remainder
    base_value = sample_size // k
    remainder = sample_size % k

    # Distribute the base value across the list
    segment_sizes = [base_value] * k

    # Distribute the remainder across the first few elements
    for i in range(remainder):
        segment_sizes[i] += 1

    return segment_sizes
