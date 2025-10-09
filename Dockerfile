FROM python:3.11-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tcsh \
    wget \
    curl \
    git \
    build-essential \
    libgl1-mesa-dri \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxrender-dev \
    libgomp1 \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libfftw3-dev \
    pkg-config \
    ca-certificates \
    libx11-6 \
    libxau6 \
    libxdmcp6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

# Install IMOD
RUN wget https://bio3d.colorado.edu/imod/AMD64-RHEL5/imod_5.1.6_RHEL8-64_CUDA12.0.sh
RUN chmod +x imod_5.1.6_RHEL8-64_CUDA12.0.sh
RUN sh imod_5.1.6_RHEL8-64_CUDA12.0.sh -yes

# Add IMOD environment variables
ENV IMOD_DIR=/usr/local/IMOD
ENV PATH="${IMOD_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${IMOD_DIR}/lib"
ENV PYTHONPATH="${IMOD_DIR}/pylib"

# Verify IMOD installation
RUN ls -la /usr/local/IMOD/bin/ || echo "IMOD not found in expected location"

# Add environment variables to prevent pip hanging
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Fix pip installation
RUN python -m ensurepip --upgrade
RUN python -m pip install --upgrade pip setuptools wheel

# Install base scientific packages first
RUN python -m pip install numpy scipy

# Install other packages
RUN python -m pip install jupyter opencv-python-headless matplotlib mrcfile

# # Install imodmodel last to ensure all dependencies are available
# RUN python -m pip install --no-cache-dir imodmodel

# Verify imodmodel installation and available methods
# RUN python -c "import imodmodel; print('Available methods:'); print([method for method in dir(imodmodel) if not method.startswith('_')]); print('Testing basic functionality...'); print('imodmodel imported successfully')"

WORKDIR /workspace
EXPOSE 8888

# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

