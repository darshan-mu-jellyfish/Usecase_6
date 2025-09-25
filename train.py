import pickle
from datetime import datetime
from darts.models import TFTModel
from utils import load_data_from_bq, preprocess_data, scale_series
from google.cloud import storage

def train_tft_model(project_id, dataset, table, bucket_name, where=None):
    # Load
    df = load_data_from_bq(project_id, dataset, table, where)

    # Preprocess: now returns 4 sets
    (
        series_list,
        past_covariates_list,
        future_covariates_list,
        static_covariates_list
    ) = preprocess_data(df)

    # Scale series + past covariates only
    series_scaled, past_covs_scaled, scaler_y, scaler_x = scale_series(
        series_list, past_covariates_list
    )

    # Init TFT
    model = TFTModel(
        input_chunk_length=30,
        output_chunk_length=7,
        hidden_size=64,
        lstm_layers=1,
        batch_size=32,
        n_epochs=5,
        add_relative_index=True
    )

    # Train with all covariates
    model.fit(
        series_scaled,
        past_covariates=past_covs_scaled,
        future_covariates=future_covariates_list,
        static_covariates=static_covariates_list,
        verbose=True
    )

    # Save locally
    local_model_path = "/tmp/tft_model.pth.tar"
    local_scaler_path = "/tmp/scalers.pkl"
    model.save(local_model_path)
    with open(local_scaler_path, "wb") as f:
        pickle.dump((scaler_y, scaler_x), f)

    # Upload to GCS
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    version_folder = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_dir = f"darts_models/tft_model/Archive/{version_folder}"

    bucket.blob(f"{archive_dir}/tft_model.pth.tar").upload_from_filename(local_model_path)
    bucket.blob(f"{archive_dir}/scalers.pkl").upload_from_filename(local_scaler_path)

    new_model_dir = "darts_models/tft_model/New_Models"
    old_model_dir = "darts_models/tft_model/Old_Models"

    # Move any existing "new_models" → "old_models"
    for blob in bucket.list_blobs(prefix=new_model_dir):
        relative_name = blob.name[len(new_model_dir):].lstrip("/")
        dest_blob_name = f"{old_model_dir}/{relative_name}"
        bucket.copy_blob(blob, bucket, dest_blob_name)
        blob.delete()  

    # Upload fresh model
    bucket.blob(f"{new_model_dir}/tft_model.pth.tar").upload_from_filename(local_model_path)
    bucket.blob(f"{new_model_dir}/scalers.pkl").upload_from_filename(local_scaler_path)

    return model
