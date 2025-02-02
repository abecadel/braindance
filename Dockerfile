FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

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
&& apt-get autoremove -y \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean \
&& rm -rf /tmp/tmp* \
&& rm -iRf /root/.cache


RUN pip3 install --break-system-packages git+https://github.com/huggingface/transformers accelerate qwen-vl-utils[decord]==0.0.8 fastapi gradio
RUN pip3 install --break-system-packages -U \
    gradio \
    packaging \
    ninja \
    setuptools \
    open3d \
    plyfile


RUN git clone --recursive https://github.com/NVlabs/InstantSplat.git && \
    cd InstantSplat && \
    mkdir -p mast3r/checkpoints/ && \
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P mast3r/checkpoints/

WORKDIR /app/InstantSplat
RUN pip3 install --break-system-packages -r requirements.txt
RUN pip3 install --break-system-packages submodules/simple-knn
RUN pip3 install --break-system-packages submodules/diff-gaussian-rasterization
RUN pip3 install --break-system-packages submodules/fused-ssim

RUN cd croco/models/curope/ &&\
    python3 setup.py build_ext --inplace

RUN pip3 install --break-system-packages flash-attn --no-build-isolation
