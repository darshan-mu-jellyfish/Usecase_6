import os
import pickle
from google.cloud import storage
from darts.models import TFTModel


def predict(bucket_name, model_dir, project_id, dataset, table, where=None):
    """
    Load latest model from GCS and make batch predictions.
    """
    from utils import load_data_from_bq, preprocess_data, scale_series

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Download model + scalers
    local_model_path = "/tmp/tft_model.pth.tar"
    local_scaler_path = "/tmp/scalers.pkl"
    os.makedirs("/tmp", exist_ok=True)

    bucket.blob(f"{model_dir}/tft_model.pth.tar").download_to_filename(local_model_path)
    bucket.blob(f"{model_dir}/scalers.pkl").download_to_filename(local_scaler_path)

    # Load scalers
    with open(local_scaler_path, "rb") as f:
        scaler_y, scaler_x = pickle.load(f)

    # Load model
    model = TFTModel.load(local_model_path)

    # Load + preprocess data
    df = load_data_from_bq(project_id, dataset, table, where)
    series_list, covariates_list = preprocess_data(df)
    series_scaled, covs_scaled, _, _ = scale_series(series_list, covariates_list)

    # Forecast
    forecasts = [model.predict(n=7, past_covariates=cov) for cov in covs_scaled]
    forecasts = [scaler_y.inverse_transform(f) for f in forecasts]

    print(f"✅ Generated forecasts for {len(forecasts)} series")
    return forecasts
