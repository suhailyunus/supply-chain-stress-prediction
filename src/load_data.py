from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class M5Data:
    """Container for the three source tables used by the project."""

    sales: pd.DataFrame
    calendar: pd.DataFrame
    prices: pd.DataFrame


DEFAULT_FILENAMES = {
    "sales": "sales_train_validation.csv",
    "calendar": "calendar.csv",
    "prices": "sell_prices.csv",
}


def load_m5_data(
    data_dir: str | Path,
    *,
    sales_filename: str = DEFAULT_FILENAMES["sales"],
    calendar_filename: str = DEFAULT_FILENAMES["calendar"],
    prices_filename: str = DEFAULT_FILENAMES["prices"],
) -> M5Data:
    """Load M5 sales, calendar, and price tables from one directory."""

    directory = Path(data_dir)

    paths = {
        "sales": directory / sales_filename,
        "calendar": directory / calendar_filename,
        "prices": directory / prices_filename,
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following required data files were not found:\n- "
            + "\n- ".join(missing)
        )

    return M5Data(
        sales=pd.read_csv(paths["sales"]),
        calendar=pd.read_csv(paths["calendar"]),
        prices=pd.read_csv(paths["prices"]),
    )
