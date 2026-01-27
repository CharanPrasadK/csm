# Use NVIDIA CUDA 12.4 base image (Development version for compilation support)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV NO_TORCH_COMPILE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Update apt and install Python 3.10, pip, and system dependencies
# - ffmpeg: Required for audio processing
# - git: Required for silentcipher
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python so 'python' command works
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Upgrade pip to ensure modern wheel support
RUN python -m pip install --upgrade pip

# Copy requirements first
COPY requirements.txt .

# Install dependencies
# Using the --no-cache-dir option to keep the image smaller
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 5006

# Run the server
CMD ["python", "server.py"]