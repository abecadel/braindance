FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ARG CLOMAP_GIT_COMMIT=main
ARG GLOMAP_GIT_COMMIT=main
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
        python3 \
        python3-pip \
        python3-dev \
&& apt-get autoremove -y \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean \
&& rm -rf /tmp/tmp* \
&& rm -iRf /root/.cache

RUN pip3 install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN pip3 install --break-system-packages git+https://github.com/huggingface/transformers accelerate qwen-vl-utils[decord]==0.0.8 fastapi gradio
