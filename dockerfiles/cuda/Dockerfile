FROM nvidia/cuda:10.0-base-ubuntu20.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    wget \
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
    && conda install -y python==3.7.10 \
    && conda config --add channels conda-forge \
    && conda clean -ya

# CUDA 11.0-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.2.89 \
    "pytorch=1.7.1=py3.7_cuda10.2.89_cudnn7.6.5_0" \
    "torchvision=0.9.2=py37_cu102" \
    "torchaudio=0.7.2=py37" \
    "tensorflow-gpu=1.15.0"\
    "scipy" \
    "scikit-learn" \
    "tqdm" \
    "pyyaml" \
    "pysoundfile" \
    && conda clean -ya

# Set the default command to python3
# CMD ["python3"]
