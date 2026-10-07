# Direction 2 runtime image.  The repository is NOT copied into the image: it is mounted at /work (see deploy/README_zh.md), so results are
# written straight back to the host and the git state stays on the host.
#
#   docker build -t d2-runtime .                                                    # default: CUDA 12.4 runtime + torch 2.5.1 (cu124)
#   docker build -t d2-runtime --build-arg CUDA_VERSION=12.1.1 --build-arg TORCH_CUDA=cu121 .   # older driver (>= 525.60)
#
# The torch wheels carry their own CUDA libraries, the base image only has to match the host driver generation.
# (The sm_75 kernels cannot be inspected at build time, there is no GPU during `docker build`: torch.cuda.get_arch_list() is empty there;
#  scripts/d2_target_check.sh step 1 checks the arch list on the real GPU.)
# Requirement: torch wheel with sm_75 (Turing, Quadro RTX 8000): every cu121 / cu124 / cu126 wheel has it.
ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ARG TORCH_CUDA=cu124
ARG TORCH_VERSION=2.5.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/mpl \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip python3-venv git ca-certificates tmux procps less \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3 /usr/local/bin/python

# torch first, from the official wheel index of the chosen CUDA version, then the rest of the requirements (torch is then already satisfied).
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
 && python3 -m pip install -r /tmp/requirements.txt \
 && python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.version.cuda, 'this torch wheel has no CUDA support'"

# The host directory is owned by another uid than the container user: let git work in any directory (also for `docker run --user UID:GID`).
RUN git config --system --add safe.directory '*'

WORKDIR /work
CMD ["bash"]
