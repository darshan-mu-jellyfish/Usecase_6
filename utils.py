import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from google.cloud import bigquery

# ------------------------
# Load data from BigQuery
# ------------------------
def load_data_from_bq(project_id: str, dataset: str, table: str, where: str = None) -> pd.DataFrame:
    """
    Load data from BigQuery into pandas DataFrame.
    """
    client = bigquery.Client(project=project_id)

    query = f"""
        SELECT
            ond,  
            hit_datetime_gmt,
            cabin_type,
            platform,
            mobile_devices,
            nth_search_in_visit,
            total_flight_searches,
            total_bookings,
            total_sales_inc_yq,
            unique_visitors_in_segment,
            conversion_rate_bookings_per_search,
            avg_revenue_per_booking,
            fsv_candidate_revenue_per_search
        FROM `{project_id}.{dataset}.{table}`
    """
    if where:
        query += f" WHERE {where}"

    df = client.query(query).to_dataframe()
    df["hit_datetime_gmt"] = pd.to_datetime(df["hit_datetime_gmt"])
    return df

# ------------------------
# Preprocess and create Darts TimeSeries
# ------------------------
def preprocess_data(df: pd.DataFrame):
    """
    Convert raw dataframe into Darts TimeSeries objects.
    Creates unique series ID using ond + cabin_type + platform + mobile_devices.
    """
    series_list = []
    covariates_list = []

    # Create unique series identifier
    df["unique_series_id"] = (
        df["ond"].astype(str) + "_" +
        df["cabin_type"].astype(str) + "_" +
        df["platform"].astype(str) + "_" +
        df["mobile_devices"].astype(str)
    )

    # Group by unique_series_id
    for series_id, group in df.groupby("unique_series_id"):
        group = group.sort_values("hit_datetime_gmt")
        group = group.set_index("hit_datetime_gmt").asfreq("D").fillna(method="ffill").reset_index()

        # Target TimeSeries (fsv_candidate_revenue_per_search)
        ts = TimeSeries.from_dataframe(
            group,
            time_col="hit_datetime_gmt",
            value_cols="fsv_candidate_revenue_per_search",
            freq="D"
        )
        series_list.append(ts)

        # Dynamic covariates TimeSeries
        cov = TimeSeries.from_dataframe(
            group,
            time_col="hit_datetime_gmt",
            value_cols=[
                "total_flight_searches",
                "total_bookings",
                "conversion_rate_bookings_per_search",
                "avg_revenue_per_booking",
                "total_sales_inc_yq"  # keep as input feature
            ],
            freq="D"
        )
        covariates_list.append(cov)

    return series_list, covariates_list

# ------------------------
# Scaling
# ------------------------
def scale_series(series_list, covariates_list):
    """
    Scale target and covariates using Darts Scaler
    """
    scaler_y = Scaler()
    scaler_x = Scaler()

    series_scaled = [scaler_y.fit_transform(s) for s in series_list]
    covs_scaled = [scaler_x.fit_transform(c) for c in covariates_list]

    return series_scaled, covs_scaled, scaler_y, scaler_x
