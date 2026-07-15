from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

BASE_MODEL_FEATURES = [
    "sales_lag_1",
    "sales_lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "is_event_day",
    "is_weekend",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "sell_price",
    "price_change_1",
]


def create_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create demand, calendar, price, and location features.

    Lag and rolling calculations are isolated by item and store to
    prevent demand histories from different stores being mixed.
    """

    required = {
        "item_id",
        "store_id",
        "state_id",
        "day_num",
        "sales",
        "weekday",
        "event_name_1",
        "sell_price",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    df = data.copy().sort_values(
        ["item_id", "store_id", "day_num"]
    )
    group_keys = ["item_id", "store_id"]

    grouped_sales = df.groupby(group_keys)["sales"]
    df["sales_lag_1"] = grouped_sales.shift(1)
    df["sales_lag_7"] = grouped_sales.shift(7)

    df["rolling_mean_7"] = grouped_sales.transform(
        lambda series: series.shift(1).rolling(7).mean()
    )
    df["rolling_std_7"] = grouped_sales.transform(
        lambda series: series.shift(1).rolling(7).std()
    )

    df["is_weekend"] = (
        df["weekday"].isin(["Saturday", "Sunday"]).astype("int8")
    )
    df["is_event_day"] = df["event_name_1"].notna().astype("int8")

    grouped_price = df.groupby(group_keys)["sell_price"]
    df["price_lag_1"] = grouped_price.shift(1)
    df["price_change_1"] = df["sell_price"] - df["price_lag_1"]
    df["price_pct_change_1"] = grouped_price.pct_change(fill_method=None)
    df["is_discount"] = (df["price_pct_change_1"] < 0).astype("int8")

    location_features = pd.get_dummies(
        df[["state_id", "store_id"]],
        prefix=["state_id", "store_id"],
        drop_first=True,
        dtype="int8",
    )

    return pd.concat([df, location_features], axis=1)


def infer_model_features(featured_data: pd.DataFrame) -> list[str]:
    """Return the standard numeric features plus encoded locations."""

    location_columns = sorted(
        column
        for column in featured_data.columns
        if column.startswith("state_id_") or column.startswith("store_id_")
    )
    return BASE_MODEL_FEATURES + location_columns


def prepare_model_input(
    data: pd.DataFrame,
    expected_features: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Engineer features, align the schema, and remove incomplete rows.

    Missing dummy columns are added as zeros so small inference batches
    remain compatible with the training schema.
    """

    featured = create_model_features(data)

    feature_names = (
        list(expected_features)
        if expected_features is not None
        else infer_model_features(featured)
    )

    for column in feature_names:
        if column not in featured.columns:
            featured[column] = 0

    ready = featured.dropna(subset=feature_names).copy()
    X_ready = ready[feature_names].copy()

    return ready, X_ready, feature_names
