FROM gcr.io/distroless/python3:3.11

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential wget unzip ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "u8darts[torch]"

ENV PYTHONPATH="/app:${PYTHONPATH}"
ENTRYPOINT ["python", "-m", "uc.pipeline_forecast"]
