from __future__ import annotations

import pandas as pd

ID_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]


def select_items(
    sales: pd.DataFrame,
    *,
    max_items: int | None = 100,
) -> pd.DataFrame:
    """Optionally restrict development work to the first N unique items."""

    if max_items is None:
        return sales.copy()

    items = sales["item_id"].drop_duplicates().head(max_items)
    return sales[sales["item_id"].isin(items)].copy()


def reshape_sales_long(sales: pd.DataFrame) -> pd.DataFrame:
    """Convert daily wide sales columns (d_1, d_2, ...) to long format."""

    missing = [column for column in ID_COLUMNS if column not in sales.columns]
    if missing:
        raise ValueError(f"Sales data is missing ID columns: {missing}")

    day_columns = [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]
    if not day_columns:
        raise ValueError("No daily columns matching 'd_*' were found.")

    long_data = sales.melt(
        id_vars=ID_COLUMNS,
        value_vars=day_columns,
        var_name="day",
        value_name="sales",
    )

    long_data["day_num"] = (
        long_data["day"]
        .str.replace("d_", "", regex=False)
        .astype("int32")
    )

    return long_data


def add_stress_target(
    data: pd.DataFrame,
    *,
    quantile: float = 0.90,
    grouping: tuple[str, ...] = ("item_id",),
) -> pd.DataFrame:
    """
    Create an item-relative high-demand stress proxy.

    The M5 data does not contain direct inventory or stockout labels.
    This target therefore identifies unusually high sales relative to
    the selected grouping's own historical distribution.
    """

    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1.")

    df = data.copy()
    thresholds = (
        df.groupby(list(grouping))["sales"]
        .transform(lambda values: values.quantile(quantile))
    )

    df["stress_threshold"] = thresholds
    df["stress_event"] = (df["sales"] > df["stress_threshold"]).astype("int8")

    return df


def merge_calendar(
    sales_long: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Attach calendar, event, week, and SNAP context."""

    calendar_columns = [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]

    missing = [
        column for column in calendar_columns if column not in calendar.columns
    ]
    if missing:
        raise ValueError(f"Calendar data is missing columns: {missing}")

    return sales_long.merge(
        calendar[calendar_columns],
        left_on="day",
        right_on="d",
        how="left",
        validate="many_to_one",
    )


def merge_prices(
    data: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Attach weekly item-store selling prices."""

    price_columns = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    missing = [column for column in price_columns if column not in prices.columns]
    if missing:
        raise ValueError(f"Price data is missing columns: {missing}")

    return data.merge(
        prices[price_columns],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )


def build_analytical_table(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    max_items: int | None = 100,
    stress_quantile: float = 0.90,
) -> pd.DataFrame:
    """Create the merged long-format analytical table."""

    selected_sales = select_items(sales, max_items=max_items)
    long_sales = reshape_sales_long(selected_sales)
    targeted = add_stress_target(
        long_sales,
        quantile=stress_quantile,
        grouping=("item_id",),
    )
    with_calendar = merge_calendar(targeted, calendar)
    with_prices = merge_prices(with_calendar, prices)

    return with_prices.sort_values(
        ["item_id", "store_id", "day_num"]
    ).reset_index(drop=True)
