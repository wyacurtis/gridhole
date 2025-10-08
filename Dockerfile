FROM ubuntu:20.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    tcsh \
    wget \
    curl \ 
    build-essential \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /tmp

RUN wget https://bio3d.colorado.edu/imod/AMD64-RHEL5/imod_5.1.6_RHEL8-64_CUDA12.0.sh

RUN chmod +x imod_5.1.6_RHEL8-64_CUDA12.0.sh
RUN sh imod_5.1.6_RHEL8-64_CUDA12.0.sh -yes

# Add IMOD environment variables for default installation
ENV IMOD_DIR=/usr/local/IMOD
ENV PATH="${IMOD_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${IMOD_DIR}/lib:${LD_LIBRARY_PATH}"

# Verify installation
RUN ls -la /usr/local/IMOD/bin/ || echo "IMOD not found in expected location"


RUN pip3 install jupyter opencv-python-headless matplotlib numpy mrcfile

WORKDIR /workspace
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

