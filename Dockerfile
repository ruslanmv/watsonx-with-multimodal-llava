# ARG for the base image
ARG IMAGE_NAME=nvidia/cuda

# Base Image
FROM ${IMAGE_NAME}:12.5.1-devel-ubuntu22.04 as base

# Different stages for different architectures
FROM base as base-amd64

ENV NV_CUDNN_VERSION 9.2.1.18-1
ENV NV_CUDNN_PACKAGE_NAME libcudnn9-cuda-12
ENV NV_CUDNN_PACKAGE libcudnn9-cuda-12=${NV_CUDNN_VERSION}
ENV NV_CUDNN_PACKAGE_DEV libcudnn9-dev-cuda-12=${NV_CUDNN_VERSION}

FROM base as base-arm64

ENV NV_CUDNN_VERSION 9.2.1.18-1
ENV NV_CUDNN_PACKAGE_NAME libcudnn9-cuda-12
ENV NV_CUDNN_PACKAGE libcudnn9-cuda-12=${NV_CUDNN_VERSION}
ENV NV_CUDNN_PACKAGE_DEV libcudnn9-dev-cuda-12=${NV_CUDNN_VERSION}

# Using the appropriate base based on architecture
FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

# Install cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

# Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV POETRY_VERSION=1.5.1
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

# System Updates and Essential Packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    build-essential \
    python3-pip \
    python3.10 \
    python3.10-venv \
    python3-dev \
    nodejs \
    npm \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment 
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Poetry (Using pip in the virtual environment)
RUN pip install poetry==${POETRY_VERSION}

# Set working director
WORKDIR /app

# Copy project files
COPY pyproject.toml ./

# Install Project Dependencies (Avoid creating a separate virtualenv)
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi 

# Install JupyterLab and Elyra using Poetry
RUN poetry add jupyterlab

# Activate the Poetry environment (You can put this in a separate script)
#RUN poetry shell

# Copy Project Files
COPY . .

# Install additional Python packages using requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Setup JupyterLab kernel
RUN python -m ipykernel install --user --name=watsonx

# Expose JupyterLab Port
EXPOSE 8888

# Start JupyterLab with Elyra on container launch
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
