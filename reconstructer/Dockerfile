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

RUN git clone --recursive https://github.com/NVlabs/InstantSplat.git && \
    cd InstantSplat && \
    mkdir -p mast3r/checkpoints/ && \
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P mast3r/checkpoints/

RUN cd InstantSplat && \
    pip3 install --break-system-packages -r requirements.txt && \
    pip3 install --break-system-packages submodules/simple-knn && \
    pip3 install --break-system-packages submodules/diff-gaussian-rasterization && \
    pip3 install --break-system-packages submodules/fused-ssim

RUN cd InstantSplat/croco/models/curope/ && \
    python3 setup.py build_ext --inplace

RUN pip3 install --break-system-packages gradio fastapi

EXPOSE 80
#ENTRYPOINT python3 src/app.py