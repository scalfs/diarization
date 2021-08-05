FROM nvidia/cuda:11.0-base-ubuntu20.04

RUN pip3 install pyannote.audio==1.1.1
RUN pip3 install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

ENV PYANNOTE_DATABASE_CONFIG=/home/jovyan/work/uis-rnn-voxsrc20/pyannote/database.yml
ENV EXP_DIR=/home/jovyan/work/uis-rnn-voxsrc20/pyannote

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
