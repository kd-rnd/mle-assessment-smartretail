FROM python:3.7.10-slim-buster

RUN export DEBIAN_FRONTEND=noninteractive \
  && echo "LC_ALL=en_US.UTF-8" >> /etc/environment \
  && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
  && echo "LANG=en_US.UTF-8" > /etc/locale.conf \
  && apt update && apt install -y locales \
  && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx gcc\
  && apt-get clean \
  && locale-gen en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*
  

ENV LANG=en_US.UTF-8 \
  LANGUAGE=en_US:en \
  LC_ALL=en_US.UTF-8

RUN pip install   torch==1.10.0 \
	torchvision==0.11.1 \
	torchaudio==0.10.0 \
	--extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace


COPY ./requirements.txt /workspace/requirements.txt



RUN pip install --no-cache-dir -r requirements.txt

RUN mim install mmengine "mmcv>=2.0.0"  "mmdet>=3.0.0" "mmpose>=1.0.0"



# CMD [ "python", "./your-daemon-or-script.py" ]
