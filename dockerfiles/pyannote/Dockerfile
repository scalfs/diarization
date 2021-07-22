FROM jupyter/scipy-notebook:latest

RUN pip3 install pyannote.audio==1.1.1
RUN pip3 install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

ENV PYANNOTE_DATABASE_CONFIG=/home/jovyan/work/uis-rnn-voxsrc20/pyannote/database.yml
ENV EXP_DIR=/home/jovyan/work/uis-rnn-voxsrc20/pyannote
# RUN apt update && apt install -y --no-install-recommends \
#     git build-essential \
#     python3-dev python3-pip python3-setuptools
# RUN pip3 -q install pip --upgrade

# FROM jupyter-base
# RUN pip3 install transformers barbar