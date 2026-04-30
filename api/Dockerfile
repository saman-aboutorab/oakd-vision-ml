FROM python:3.12-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Source + checkpoint
COPY oakd_vision/ oakd_vision/
COPY training/configs/fusion_config.yaml training/configs/fusion_config.yaml
COPY runs/fusion/gated_f3/best.pt runs/fusion/gated_f3/best.pt

ENV CHECKPOINT=runs/fusion/gated_f3/best.pt
ENV CONFIG=training/configs/fusion_config.yaml

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
