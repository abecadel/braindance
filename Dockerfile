FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04 as builder

ARG DEBIAN_FRONTEND=noninteractive

ARG FORCE_CUDA=1
ENV FORCE_CUDA=${FORCE_CUDA}

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


WORKDIR /app

ARG CUDA_ARCHITECTURES="75;80;86"
ARG TCNN_CUDA_ARCHITECTURES="75;80;86"
ENV TORCH_CUDA_ARCH_LIST="7.5+PTX 8.0 8.6"
ENV QT_XCB_GL_INTEGRATION=xcb_egl
ENV MAX_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        imagemagick \
        ffmpeg \
        git \
        wget \
        unzip \
        libgl1-mesa-dev \
        libglib2.0-0 \
        libopencv-dev \
        colmap \
        python3 \
        python3-pip \
        python3-packaging \
&& apt-get autoremove -y \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean \
&& rm -rf /tmp/tmp* \
&& rm -iRf /root/.cache

RUN pip3 install --break-system-packages torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

COPY dependencies dependencies

COPY requirements.txt .
RUN pip3 install --break-system-packages -r requirements.txt

RUN cd dependencies/gradio-rerun-viewer && pip install .
RUN cd dependencies/mega-sam/base && python setup.py install
#
#COPY preload.py .
#COPY download_models.sh .
#COPY src src