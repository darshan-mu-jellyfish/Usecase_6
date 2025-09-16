from kfp import dsl
from kfp.dsl import component
from datetime import datetime
from google.cloud import aiplatform

PROJECT_ID = "jfdev-jellyfish-poc"
REGION = "us-central1"
BUCKET = "test_tft-training-bucket-central1"
PIPELINE_ROOT = f"gs://{BUCKET}/tft_pipeline_root/"
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/forecasting/forecasting:latest"

# ------------------------------
# Training component
# ------------------------------
@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-storage", "darts", "pandas"],
)
def train_component(project_id: str, dataset: str, table: str, bucket_name: str) -> str:
    """Train TFT model and return the GCS folder path for the trained model."""
    from train import train_tft_model
    from datetime import datetime

    # versioned folder
    version_folder = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_folder = f"darts_models/tft_model/{version_folder}"

    train_tft_model(
        project_id=project_id,
        dataset=dataset,
        table=table,
        bucket_name=bucket_name,
        model_dir=model_folder
    )
    print(f"✅ Model saved to {model_folder}")
    return model_folder

# ------------------------------
# Batch prediction component
# ------------------------------
@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-storage", "darts", "pandas"],
)
def batch_predict_component(
    project_id: str,
    dataset: str,
    table: str,
    bucket_name: str,
    model_folder: str,
    output_path: str = "predictions.csv"
):
    """Run batch prediction using a specified model folder."""
    from batch_predict import predict
    import pandas as pd

    forecasts = predict(
        bucket_name=bucket_name,
        model_folder=model_folder,
        project_id=project_id,
        dataset=dataset,
        table=table
    )

    all_forecasts = []
    for idx, ts in enumerate(forecasts):
        df_pred = ts.pd_dataframe()
        df_pred["series_idx"] = idx
        all_forecasts.append(df_pred)

    pd.concat(all_forecasts).to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

# ------------------------------
# Kubeflow pipeline
# ------------------------------
@dsl.pipeline(
    name="tft-train-and-predict-pipeline",
    pipeline_root=PIPELINE_ROOT
)
def pipeline(
    project_id: str = PROJECT_ID,
    dataset: str = "test_model_export",
    table: str = "test_table_daily",
    bucket_name: str = BUCKET,
    output_path: str = "predictions.csv"
):
    # Step 1: Train model
    train_task = train_component(
        project_id=project_id,
        dataset=dataset,
        table=table,
        bucket_name=bucket_name
    )

    # Step 2: Batch prediction
    predict_task = batch_predict_component(
        project_id=project_id,
        dataset=dataset,
        table=table,
        bucket_name=bucket_name,
        model_folder=train_task.output,
        output_path=output_path
    )
    predict_task.after(train_task)
