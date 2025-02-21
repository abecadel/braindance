#!/bin/bash

cd dependencies/InstantSplat && \
mkdir -p mast3r/checkpoints/ && \
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P mast3r/checkpoints/

python3 preload.py