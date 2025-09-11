# Use Python 3.10 base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy all application files
COPY get_pimax_h.py .
COPY config.py .
COPY forecasting_pipeline.py .
COPY get_lyap_exp.py .
COPY main.py .
COPY utils.py .
COPY plot_corr.py .
COPY plot_corr_uncorr.py .

# Create all required directories
RUN mkdir -p /app/res /app/results /cache/huggingface

# Create a script to run all scripts in two stages
RUN echo '#!/bin/bash\n\
# Stage 1: run three main scripts in parallel\n\
python3 get_pimax_h.py &\n\
python3 main.py &\n\
python3 get_lyap_exp.py &\n\
wait\n\
echo "Stage 1 completed: all three processes finished."\n\
\n\
# Stage 2: run visualization scripts sequentially\n\
python3 plot_corr.py\n\
python3 plot_uncorr.py\n\
echo "Stage 2 completed: visualizations finished."' > /app/run_all.sh && chmod +x /app/run_all.sh

# Set the entrypoint to run all scripts
ENTRYPOINT ["/app/run_all.sh"]
