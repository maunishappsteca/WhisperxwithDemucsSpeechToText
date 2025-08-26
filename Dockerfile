# Base image with CUDA 11.8 and Python 3.10
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Set environment variables
ENV WHISPER_MODEL_CACHE=/app/models
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Copy files
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 
RUN pip install --no-cache-dir -r requirements.txt


# Pre-download nltk punkt tokenizer to image
RUN mkdir -p /usr/share/nltk_data && \
    python -c "import nltk; nltk.download('punkt', download_dir='/usr/share/nltk_data')" && \
    python -c "import nltk; nltk.data.path.append('/usr/share/nltk_data')"
    

# Install huggingface hub for preloading Whisper model
RUN pip install huggingface-hub

# Download Whisper model to cache
RUN python -c "\
import os; \
from huggingface_hub import snapshot_download; \
model_size = os.getenv('WHISPER_MODEL', 'large-v3'); \
cache_dir = os.getenv('WHISPER_MODEL_CACHE', '/app/models'); \
print(f'Downloading {model_size} to {cache_dir}'); \
snapshot_download(repo_id=f'openai/whisper-{model_size}', \
                  local_dir=os.path.join(cache_dir, model_size), \
                  local_dir_use_symlinks=False, \
                  resume_download=True)"

# Run the app
CMD ["python", "app.py"]