FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set CUDA environment variables for flash attention compilation
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3-pip \
        default-jre-headless \
        build-essential \
        git \
        wget \
        curl \
        ca-certificates \
        && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install packages in optimized order to leverage Docker layer caching
# Install core ML frameworks first (biggest packages) - PyTorch ONLY, NO TENSORFLOW
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CRITICAL: Install transformers and sentence-transformers - PYTORCH ONLY
# Explicitly prevent TensorFlow installation during pip install
RUN pip install --no-cache-dir transformers sentence-transformers

# Set environment variables to prevent TensorFlow usage
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV USE_TF=0
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Remove TensorFlow if it was installed as a dependency
RUN pip uninstall -y tensorflow tensorflow-cpu tf-keras keras || true

# Install scientific computing packages
RUN pip install --no-cache-dir numpy scipy scikit-learn pandas

# Install langchain ecosystem
RUN pip install --no-cache-dir langchain langchain-community langchain_huggingface

# Install vector search backends (FAISS now in requirements.txt)
RUN pip install --no-cache-dir hnswlib

# Install remaining packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set HuggingFace token and login
# ENV HF_TOKEN=your_huggingface_token_here
# RUN huggingface-cli login --token $HF_TOKEN || echo "⚠️ HuggingFace login failed"

# Install flash attention for Jina v3/v4 models - Latest September 2025 version
RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation || echo "⚠️ Flash attention install failed - Jina models may use fallback"

COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
RUN mkdir -p logs models_cache notebooks

ENV HF_HOME=/app/models_cache
ENV PYTHONPATH=/app
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

EXPOSE 9000

# Make test script executable
RUN chmod +x src/test_system.py

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=9000", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]