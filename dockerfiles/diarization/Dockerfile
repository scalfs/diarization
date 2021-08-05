FROM nvidia/cuda:11.0-base-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN pip3 install pyannote.audio==1.1.1
RUN pip3 install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get install -y \
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