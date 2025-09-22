import pickle
from darts.models import TFTModel
from utils import load_data_from_bq, preprocess_data, scale_series
from google.cloud import storage
import os

def predict(bucket_name, model_dir, project_id, dataset, table, where=None):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Download model and scalers
    os.makedirs("/tmp", exist_ok=True)
    model_path = "/tmp/tft_model.pth.tar"
    scalers_path = "/tmp/scalers.pkl"
    bucket.blob(f"{model_dir}/tft_model.pth.tar").download_to_filename(model_path)
    bucket.blob(f"{model_dir}/scalers.pkl").download_to_filename(scalers_path)

    with open(scalers_path, "rb") as f:
        scaler_y, scaler_x = pickle.load(f)

    model = TFTModel.load(model_path)

    df = load_data_from_bq(project_id, dataset, table, where)
    series_list, covariates_list = preprocess_data(df)
    series_scaled, covs_scaled, _, _ = scale_series(series_list, covariates_list)

    forecasts = [model.predict(n=7, past_covariates=cov) for cov in covs_scaled]
    forecasts = [scaler_y.inverse_transform(f) for f in forecasts]
    return forecasts
