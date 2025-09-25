import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from google.cloud import bigquery
from darts.utils.timeseries_generation import datetime_attribute_timeseries

def preprocess_data(df: pd.DataFrame):
    """
    Convert raw dataframe into Darts TimeSeries objects.
    Creates unique series ID using ond + cabin_type + platform + mobile_devices.
    Splits into target, past covariates, future covariates, and static covariates.
    """
    series_list = []
    past_covariates_list = []
    future_covariates_list = []
    static_covariates_list = []

    # Unique identifier
    df["unique_series_id"] = (
        df["ond"].astype(str) + "_" +
        df["cabin_type"].astype(str) + "_" +
        df["platform"].astype(str) + "_" +
        df["mobile_devices"].astype(str)
    )

    for series_id, group in df.groupby("unique_series_id"):
        group = group.sort_values("hit_datetime_gmt")
        group = group.set_index("hit_datetime_gmt").asfreq("D").fillna(method="ffill").reset_index()

        # Target
        ts = TimeSeries.from_dataframe(
            group,
            time_col="hit_datetime_gmt",
            value_cols="fsv_candidate_revenue_per_search",
            freq="D"
        )
        series_list.append(ts)

        # Past covariates
        past_cov = TimeSeries.from_dataframe(
            group,
            time_col="hit_datetime_gmt",
            value_cols=[
                "total_flight_searches",
                "total_bookings",
                "conversion_rate_bookings_per_search",
                "avg_revenue_per_booking",
                "total_sales_inc_yq"
            ],
            freq="D"
        )
        past_covariates_list.append(past_cov)

        # Future covariates (calendar features as example)
        future_cov = datetime_attribute_timeseries(
            pd.date_range(start=group["hit_datetime_gmt"].min(), 
                          end=group["hit_datetime_gmt"].max(), freq="D"),
            attribute="day_of_week",
            one_hot=True
        )
        future_covariates_list.append(future_cov)

        # Static covariates
        static_cov = pd.DataFrame({
            "ond": [group["ond"].iloc[0]],
            "cabin_type": [group["cabin_type"].iloc[0]],
            "platform": [group["platform"].iloc[0]],
            "mobile_devices": [group["mobile_devices"].iloc[0]]
        })
        static_covariates_list.append(static_cov)

    return series_list, past_covariates_list, future_covariates_list, static_covariates_list
