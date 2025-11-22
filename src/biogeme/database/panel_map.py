from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContiguousPanelMap:
    """Map for panel data with contiguous blocks per individual."""

    unique_ids: np.ndarray  # shape (K,), individual labels, in order of appearance
    starts: np.ndarray  # shape (K,), start row index of each individual block
    counts: np.ndarray  # shape (K,), number of rows per individual
    indptr: np.ndarray  # shape (K+1,), cumulative pointers: [starts] + [N]

    def rows_slice(self, i: int) -> slice:
        a = int(self.starts[i])
        b = int(self.starts[i] + self.counts[i])
        return slice(a, b)


def build_contiguous_panel_map(
    df: pd.DataFrame, panel_column: str
) -> ContiguousPanelMap:
    """
    Build a panel map assuming each individual's rows are contiguous.
    Raises an error if any individual's rows are non-contiguous.
    """
    if panel_column not in df.columns:
        raise KeyError(f"'{panel_column}' not in dataframe columns.")

    # Work on a 0-based, monotonic row index view
    idx = np.arange(len(df), dtype=np.int64)
    tmp = pd.DataFrame({panel_column: df[panel_column].values, "_pos": idx})

    # First/last positions and counts per individual
    stats = (
        tmp.groupby(panel_column, sort=False)["_pos"]
        .agg(["min", "max", "count"])
        .reset_index()
    )

    # Contiguity check: for each id, max - min + 1 must equal count
    contiguous = (stats["max"] - stats["min"] + 1) == stats["count"]
    if not bool(np.all(contiguous.values)):
        bad = stats.loc[~contiguous, panel_column].tolist()
        raise ValueError(
            "Panel rows are not contiguous for the following IDs: "
            + ", ".join(repr(b) for b in bad)
        )

    unique_ids = stats[panel_column].to_numpy()
    starts = stats["min"].to_numpy(dtype=np.int64)
    counts = stats["count"].to_numpy(dtype=np.int64)

    # Build CSR-like pointer array: indptr[i] = start of block i, with trailing N
    N = len(df)
    # If starts are strictly increasing and counts contiguous, the last pointer is N
    # (contiguity guarantees no gaps between starts except different block lengths)
    indptr = np.empty(len(starts) + 1, dtype=np.int64)
    indptr[:-1] = starts
    indptr[-1] = N

    return ContiguousPanelMap(
        unique_ids=unique_ids, starts=starts, counts=counts, indptr=indptr
    )
