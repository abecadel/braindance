FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        imagemagick \
        ffmpeg \
        git \
        wget \
        unzip \
        python3 \
        python3-pip \
        python3-dev \
&& apt-get autoremove -y \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean \
&& rm -rf /tmp/tmp* \
&& rm -iRf /root/.cache

RUN pip3 install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126


