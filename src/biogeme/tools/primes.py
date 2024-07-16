import logging

import numpy as np

from biogeme.exceptions import BiogemeError

logger = logging.getLogger(__name__)


def get_prime_numbers(n: int) -> list[int]:
    """Get a given number of prime numbers.

    :param n: Number of primes that are requested.
    :return: List with prime numbers.
    :raise BiogemeError: If the requested number is non-positive.
    """
    if not isinstance(n, int) or n <= 0:
        raise BiogemeError(f'Incorrect number: {n}')

    # Initial setup based on known distribution of primes for efficiency
    if n == 1:
        return [2]  # The first prime number is 2
    estimated_upper_bound = n * (
        np.log(n) + np.log(np.log(n))
    )  # Estimation based on the prime number theorem
    primes = calculate_prime_numbers(int(estimated_upper_bound))

    # In case the estimation was too low, which is very rare, increase until enough primes are found
    while len(primes) < n:
        estimated_upper_bound *= 1.5
        primes = calculate_prime_numbers(int(estimated_upper_bound))

    return primes[:n]


def calculate_prime_numbers(upper_bound: int) -> list[int]:
    """Calculate prime numbers up to a specified upper bound using the Sieve of Eratosthenes.

    :param upper_bound: Prime numbers up to this value will be computed.
    :return: A list of prime numbers up to the upper bound.
    :raise BiogemeError: If the sqrt_max is incorrectly defined (e.g., negative number).
    """
    if not isinstance(upper_bound, int) or upper_bound < 0:
        raise BiogemeError(f'Incorrect value: {upper_bound}')
    if upper_bound < 2:
        return []  # There are no prime numbers less than 2

    # Initialize a list indicating whether numbers are prime
    is_prime = [False, False] + [True] * (
        upper_bound - 1
    )  # 0 and 1 are not prime, others assumed prime initially
    for number in range(
        2, int(np.sqrt(upper_bound)) + 1
    ):  # Check up to the square root of the sqrt_max
        if is_prime[number]:  # Found a prime
            for multiple in range(
                number * number, upper_bound + 1, number
            ):  # Mark multiples of the prime as non-prime
                is_prime[multiple] = False

    # Extract prime numbers based on the boolean list
    primes = [num for num, prime in enumerate(is_prime) if prime]

    return primes
