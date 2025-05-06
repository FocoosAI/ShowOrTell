FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel AS builder
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/

ENV FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.7;8.9" \
    PIP_CACHE_DIR=/opt/conda/pkgs \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=TRUE

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        make cmake git ffmpeg libsm6 libxext6 libmagic1 build-essential python3-pip ninja-build && \
    apt-get clean

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install 'git+https://github.com/facebookresearch/detectron2.git' && \
    uv pip install --system git+https://github.com/cocodataset/panopticapi.git && \
    uv pip install --system git+https://github.com/mcordts/cityscapesScripts.git

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    uv pip install --system wandb pillow opencv-python timm einops regex imageio setuptools ftfy scipy prettytable seaborn jupyterlab \
                ipywidgets scikit-learn gdown nvitop tqdm tensorboardX scikit-image ninja torchmetrics matplotlib omegaconf \
                numpy

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    uv pip install --system kornia progressbar ipycanvas POT lvis transformers

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    uv pip install --system -U xformers==0.0.29 --index-url https://download.pytorch.org/whl/cu118

# ------------------------------------------------------------------------------------------------
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# Copy dependencies from previous stage
COPY --from=builder /opt/conda/lib/python3.11/site-packages/ /opt/conda/lib/python3.11/site-packages/
COPY --from=builder /opt/conda/share/jupyter/lab /opt/conda/share/jupyter/lab
ENV FORCE_CUDA="1" \
    PYTHONUNBUFFERED=TRUE \
    DEBIAN_FRONTEND=noninteractive

# Required for opencv
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        git ffmpeg libsm6 libxext6 python3-pip nano psmisc wget unzip curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace