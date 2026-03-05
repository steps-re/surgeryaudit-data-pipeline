FROM python:3.12-slim

WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scrape_reddit.py generate_synthetic.py build_dataset.py ./

# Default: run full pipeline
COPY run_pipeline.sh .
RUN chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh"]
