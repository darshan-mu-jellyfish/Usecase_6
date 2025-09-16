# import pandas as pd
# from darts import TimeSeries
# from darts.dataprocessing.transformers import Scaler

# from google.cloud import bigquery
# import pandas as pd

# def load_data_from_bq(project_id: str, dataset: str, table: str, where: str = None):
#     """
#     Load data from BigQuery into pandas DataFrame.
#     """
#     client = bigquery.Client(project=project_id)

#     query = f"""
#         SELECT series_id_encoded, timestamp, sales, on_promotion, price, category_encoded
#         FROM `{project_id}.{dataset}.{table}`
#     """
#     if where:
#         query += f" WHERE {where}"

#     df = client.query(query).to_dataframe()
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     return df

# def load_and_preprocess(df: pd.DataFrame):
#     """
#     Convert raw dataframe into Darts TimeSeries objects.
#     Schema: series_id_encoded, timestamp, sales, on_promotion, price, category_encoded
#     Combines series_id_encoded and category_encoded to create a unique series ID.
#     """
#     series_list = []
#     covariates_list = []

#     # Create unique series identifier
#     df["unique_series_id"] = df["series_id_encoded"].astype(str) + "_" + df["category_encoded"].astype(str)

#     # Group by the new unique_series_id
#     for series_id, group in df.groupby("unique_series_id"):
#         group = group.sort_values("timestamp")

#         # Set timestamp as index and ensure daily frequency
#         group = group.set_index("timestamp").asfreq("D").fillna(method="ffill").reset_index()

#         # Target TimeSeries (sales)
#         ts = TimeSeries.from_dataframe(
#             group,
#             time_col="timestamp",
#             value_cols="sales",
#             freq="D"
#         )
#         series_list.append(ts)

#         # Dynamic covariates TimeSeries (on_promotion, price)
#         cov = TimeSeries.from_dataframe(
#             group,
#             time_col="timestamp",
#             value_cols=["on_promotion", "price"],
#             freq="D"
#         )
#         covariates_list.append(cov)

#     return series_list, covariates_list


# def scale_series(series_list, covariates_list):
#     """Apply scaling"""
#     scaler_y = Scaler()
#     scaler_x = Scaler()

#     series_scaled = [scaler_y.fit_transform(s) for s in series_list]
#     covs_scaled = [scaler_x.fit_transform(c) for c in covariates_list]

#     return series_scaled, covs_scaled, scaler_y, scaler_x



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
        SELECT series_id_encoded, timestamp, sales, on_promotion, price, category_encoded
        FROM `{project_id}.{dataset}.{table}`
    """
    if where:
        query += f" WHERE {where}"

    df = client.query(query).to_dataframe()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# ------------------------
# Preprocess and create Darts TimeSeries
# ------------------------
def preprocess_data(df: pd.DataFrame):
    """
    Convert raw dataframe into Darts TimeSeries objects.
    Combines series_id_encoded and category_encoded to create a unique series ID.
    """
    series_list = []
    covariates_list = []

    # Create unique series identifier
    df["unique_series_id"] = df["series_id_encoded"].astype(str) + "_" + df["category_encoded"].astype(str)

    # Group by unique_series_id
    for series_id, group in df.groupby("unique_series_id"):
        group = group.sort_values("timestamp")
        # Ensure daily frequency and fill missing
        group = group.set_index("timestamp").asfreq("D").fillna(method="ffill").reset_index()

        # Target TimeSeries (sales)
        ts = TimeSeries.from_dataframe(
            group,
            time_col="timestamp",
            value_cols="sales",
            freq="D"
        )
        series_list.append(ts)

        # Dynamic covariates TimeSeries
        cov = TimeSeries.from_dataframe(
            group,
            time_col="timestamp",
            value_cols=["on_promotion", "price"],
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
