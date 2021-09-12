FROM nvidia/cuda:10.0-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    wget \
    gcc \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh \
    && conda install -y python=3.7.0 \
    && conda config --add channels conda-forge \
    && conda clean -ya

RUN conda update --all

# CUDA 10.0-specific steps
RUN conda install -y \
    cudatoolkit=10.0.130 \
    "tensorflow-gpu=1.15.0"\
    "scipy=1.1.0=py37_blas_openblashb06ca3d_202" \
    "scikit-learn=0.20.4=py37_blas_openblashebff5e3_0" \
    && conda clean -ya

RUN pip install webrtcvad

RUN conda install librosa

# Set the default command to python3
# CMD ["python3"]
