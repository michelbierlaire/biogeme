"""Decorator for the builder of PyMc model

Michel Bierlaire
Mon Nov 03 2025, 15:44:53
"""

import pandas as pd
import pytensor.tensor as pt
from pytensor.raise_op import Assert


def check_shape(func):
    def wrapper(dataframe: pd.DataFrame, *args, **kwargs):
        result = func(dataframe, *args, **kwargs)
        # Ensure a PyTensor variable
        result = pt.as_tensor_variable(result)

        # Static rank check: must be 1-D (per-observation vector)
        if result.ndim != 1:
            raise ValueError(
                f"Numeric builder must return a 1-D tensor (N,), got ndim={result.ndim} with static shape {result.type.shape}."
            )

        # Runtime length check (symbolic): length == len(dataframe)

        n_obs = pt.as_tensor_variable(len(dataframe), dtype="int64")
        cond = pt.eq(result.shape[0], n_obs)
        result = Assert(
            f"Numeric builder length mismatch: expected {len(dataframe)}, got dynamic length"
        )(result, cond)
        return result

    return wrapper
