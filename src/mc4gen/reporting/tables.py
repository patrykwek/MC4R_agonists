"""Stargazer-style LaTeX tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


_BOOKTABS_PREAMBLE = r"""\begin{table}[ht]
  \centering
  \small
  \caption{{{caption}}}
  \label{{tab:{label}}}
  \begin{{tabular}}{{{colspec}}}
  \toprule
"""
_BOOKTABS_CLOSE = r"""  \bottomrule
  \end{tabular}
\end{table}
"""


@dataclass(frozen=True, slots=True)
class LatexTable:
    caption: str
    label: str
    body: str

    def render(self) -> str:
        return self.body


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "--"
        return f"{value:.3g}"
    return str(value).replace("_", r"\_")


def dataframe_to_booktabs(
    df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    columns: Sequence[str] | None = None,
    header: Sequence[str] | None = None,
) -> LatexTable:
    cols = list(columns) if columns is not None else list(df.columns)
    head = list(header) if header is not None else cols
    colspec = "l" + "r" * (len(cols) - 1)
    header_row = "  " + " & ".join(head) + r" \\" + "\n"
    header_row += r"  \midrule" + "\n"
    rows = [
        "  " + " & ".join(_format_cell(df.iloc[i][c]) for c in cols) + r" \\" + "\n"
        for i in range(len(df))
    ]
    body = (
        _BOOKTABS_PREAMBLE.format(caption=caption, label=label, colspec=colspec)
        + header_row
        + "".join(rows)
        + _BOOKTABS_CLOSE
    )
    return LatexTable(caption=caption, label=label, body=body)


def concat_tables(tables: Iterable[LatexTable]) -> str:
    return "\n".join(t.body for t in tables)
