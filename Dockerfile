FROM me-central1-docker.pkg.dev/gcp-npd-prj-data-shd01/gcp-npd-data-repo-shd02/python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "-m", "pipeline_forecast"]
