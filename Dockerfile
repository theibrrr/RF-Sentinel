FROM python:3.11-slim

LABEL maintainer="RF-Sentinel Contributors"
LABEL description="RF-Sentinel: Robust RF Modulation Recognition from Raw I/Q Data"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml requirements.txt ./
COPY src/ src/
COPY configs/ configs/
COPY app/ app/
COPY tests/ tests/

# Install package in editable mode
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e ".[all]"

# Create data directories
RUN mkdir -p data/raw data/processed data/splits \
    reports/figures artifacts/checkpoints artifacts/predictions

# Default command
CMD ["bash"]
