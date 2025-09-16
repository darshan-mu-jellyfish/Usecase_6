import kfp
from google.cloud import aiplatform
from tft_pipeline import pipeline, PIPELINE_ROOT, PROJECT_ID, REGION

PIPELINE_JSON = "tft_train_predict_pipeline.yaml"

# Compile pipeline
kfp.compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=PIPELINE_JSON
)

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=PIPELINE_ROOT)

# Submit pipeline
job = aiplatform.PipelineJob(
    display_name="tft-train-predict-pipeline-job",
    template_path=PIPELINE_JSON,
    pipeline_root=PIPELINE_ROOT,
)
job.run(sync=True)
