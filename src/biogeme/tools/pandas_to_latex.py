import math
from typing import Iterable

import numpy as np
import pandas as pd


def _add_trailing_zero(formatted_number: str) -> str:
    """Ensure there is at least one decimal digit."""
    if not formatted_number:
        return "0.0"
    if "." in formatted_number:
        if formatted_number[-1] == ".":
            return f"{formatted_number}0"
        return formatted_number
    return f"{formatted_number}.0"


def _format_real_number(value: float) -> str:
    """Format a real number like Biogeme, for LaTeX tables.

    - Use .3g formatting.
    - Avoid losing the decimal point.
    - Preserve scientific notation if used.
    """
    formatted_value = f"{value:.3g}"

    # If scientific notation is used, protect the mantissa
    if "e" in formatted_value:
        left, right = formatted_value.split("e")
        return f"{_add_trailing_zero(left)}e{right}"
    if "E" in formatted_value:
        left, right = formatted_value.split("E")
        return f"{_add_trailing_zero(left)}E{right}"

    return _add_trailing_zero(formatted_value)


def dataframe_to_latex_decimal(
    df: pd.DataFrame,
    float_columns: Iterable[str] | None = None,
    include_index: bool = True,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Generate a LaTeX tabular with r@{.}l alignment for float columns.

    Parameters
    ----------
    df
        Input DataFrame.
    float_columns
        Names of columns to treat as numeric with decimal alignment.
        If None, all columns with float dtype are used.
    include_index
        If True, include the index as the first column (left aligned).
    caption
        Optional LaTeX caption (without \\caption{} wrapper).
    label
        Optional LaTeX label, used as \\label{...} if provided.

    Returns
    -------
    latex
        A LaTeX string with \\begin{tabular} ... \\end{tabular}.
    """
    if float_columns is None:
        float_columns = [
            c for c in df.columns if np.issubdtype(df[c].dtype, np.floating)
        ]
    float_columns = list(float_columns)

    # Column alignment specification
    col_specs: list[str] = []
    if include_index:
        col_specs.append("l")
    for col in df.columns:
        if col in float_columns:
            col_specs.append("r@{.}l")
        else:
            col_specs.append("l")
    col_spec_str = "".join(col_specs)

    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{col_spec_str}}}")

    # Optional caption/label for a standalone table environment
    if caption is not None or label is not None:
        lines.append("\\hline")
        if caption is not None:
            lines.append(f"\\multicolumn{{{len(col_specs)}}}{{c}}{{{caption}}}\\\\")
        if label is not None:
            lines.append(
                f"\\multicolumn{{{len(col_specs)}}}{{c}}{{\\label{{{label}}}}}\\\\"
            )
        lines.append("\\hline")

    # Header row
    header_cells: list[str] = []
    if include_index:
        header_cells.append("")  # index column has no header

    for col in df.columns:
        safe_name = str(col).replace("_", r"\_")
        if col in float_columns:
            # Span the two decimal-aligned columns
            header_cells.append(f"\\multicolumn{{2}}{{c}}{{{safe_name}}}")
        else:
            header_cells.append(safe_name)

    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\hline")

    # Data rows
    for idx, row in df.iterrows():
        row_cells: list[str] = []

        if include_index:
            idx_str = str(idx).replace("_", r"\_")
            row_cells.append(idx_str)

        for col in df.columns:
            val = row[col]
            if col in float_columns:
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    # Empty numeric cell: keep both parts empty
                    row_cells.append("")  # integer part
                    row_cells.append("")  # fractional part
                else:
                    formatted = _format_real_number(float(val))
                    # r@{.}l: replace the dot by '&' so we supply two cells
                    left_right = formatted.split(".", maxsplit=1)
                    if len(left_right) == 2:
                        left, right = left_right
                    else:
                        # Should not happen thanks to _format_real_number, but be safe
                        left, right = formatted, ""
                    row_cells.append(left)
                    row_cells.append(right)
            else:
                cell = str(val)
                cell = cell.replace("_", r"\_")
                row_cells.append(cell)

        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\end{tabular}")

    return "\n".join(lines)
