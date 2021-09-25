FROM jupyter/tensorflow-notebook:814ef10d64fb

RUN pip install pyannote.audio==1.1.1 easydict
RUN pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

ENV PYANNOTE_DATABASE_CONFIG=/home/jovyan/work/voxsrc21-dia/pyannote/database.yml
ENV EXP_DIR=/home/jovyan/work/voxsrc21-dia/pyannote

USER root

RUN apt-get update && apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

RUN pip install webrtcvad
# RUN apt update && apt install -y --no-install-recommends \
#     git build-essential \
#     python3-dev python3-pip python3-setuptools
# RUN pip -q install pip --upgrade

RUN pip install uisrnn spectralcluster

# FROM jupyter-base
# RUN pip3 install transformers barbar