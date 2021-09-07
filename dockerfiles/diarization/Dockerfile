FROM nvidia/cuda:11.0-base-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    ca-certificates \
    sudo \
    wget \
    git \
    bzip2 \
    libx11-6 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

ENV PYANNOTE_DATABASE_CONFIG=/app/voxsrc21-dia/pyannote/database.yml
ENV EXP_DIR=/app/voxsrc21-dia/pyannote

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh \
    && conda install -y python==3.8.0 \
    && conda config --add channels conda-forge \
    && conda clean -ya

# CUDA 11.0-specific steps
RUN conda install -y -c pytorch -c nvidia \
    cudatoolkit=11.1 \
    pytorch \
    torchvision \
    torchaudio \
    tqdm \
    pysoundfile

RUN pip3 install pyannote.audio==1.1.1 

CMD ["python3" , "voxsrc21-dia/pyannote-dia-voxsrc21.py"]
