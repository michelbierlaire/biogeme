import sys
from collections import defaultdict
from itertools import product
from typing import Iterable, Iterator


class ModelNames:
    """Class generating model names from unique configuration string"""

    def __init__(self, prefix: str = 'Model'):
        self.prefix = prefix
        self.dict_of_names = {}
        self.current_number = 0

    def __call__(self, the_id: str) -> str:
        """Get a short model name from a unique ID

        :param the_id: id of the model
        :type the_id: str (or anything that can be used as a key for a dict)
        """
        the_name = self.dict_of_names.get(the_id)
        if the_name is None:
            the_name = f'{self.prefix}_{self.current_number:06d}'
            self.current_number += 1
            self.dict_of_names[the_id] = the_name
        return the_name


def generate_unique_ids(list_of_ids: list[str]) -> dict[str, str]:
    """If there are duplicates in the list, a new list is generated
    where there are renamed to obtain a list with unique IDs.

    :param list_of_ids: list of ids
    :type list_of_ids: list[str]

    :return: a dict that maps the unique names with the original name
    """
    counts = defaultdict(int)
    for the_id in list_of_ids:
        counts[the_id] += 1

    results = {}
    for name, count in counts.items():
        if count == 1:
            results[name] = name
        else:
            substitutes = [f'{name}_{i}' for i in range(count)]
            for new_name in substitutes:
                results[new_name] = name
    return results


def unique_product(*iterables: Iterable, max_memory_mb: int = 1024) -> Iterator[tuple]:
    """Generate the Cartesian product of multiple iterables, keeping
    only the unique entries.  Raises a MemoryError if memory usage
    exceeds the specified threshold.

    :param iterables: Variable number of iterables to compute the
        Cartesian product from.
    :type iterables: Iterable

    :param max_memory_mb: Maximum memory usage in megabytes (default: 1024MB).
    :type max_memory_mb: int

    :return: Yields unique entries from the Cartesian product.
    :rtype: Iterator[tuple]

    """

    mb_to_bytes = 1024 * 1024
    max_memory_bytes = max_memory_mb * mb_to_bytes  # Convert MB to bytes
    seen = set()  # Set to store seen entries
    total_memory = 0  # Track memory usage

    for items in product(*iterables):
        if items not in seen:
            seen.add(items)
            item_size = sum(sys.getsizeof(item) for item in items)
            total_memory += item_size
            if total_memory > max_memory_bytes:
                raise MemoryError(
                    f'Memory usage exceeded the specified threshold: '
                    f'{total_memory/mb_to_bytes:.1f} MB > '
                    f'{max_memory_bytes/mb_to_bytes} MB.'
                )
            yield items
