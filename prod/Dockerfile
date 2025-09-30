# PRODUCTION DOCKERFILE - Winner Configuration ONLY
FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
# COPY data/ ./data/

# Create necessary directories
RUN mkdir -p logs models_cache

# Set environment variables for HuggingFace model caching
ENV HF_HOME=/app/models_cache
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the FastAPI application
CMD ["python", "-m", "src.main"]