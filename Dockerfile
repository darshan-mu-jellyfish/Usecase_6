# Base image
FROM python:3.11-slim

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential wget unzip ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy repo files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "u8darts[torch]"

# Ensure Python can see package
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Entrypoint for CLI
ENTRYPOINT ["python", "-m", "Usecase_6.pipeline_forecast"]
