# Define Base Image
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel
# Install dependencies

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y git cmake build-essential ninja-build libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev curl ffmpeg pkg-config python3 python3-dev rsync software-properties-common unzip libopencv-dev imagemagick python3-tk

#RUN cd /opt
WORKDIR /opt
RUN git clone https://github.com/colmap/colmap.git
#RUN cd colmap
WORKDIR /opt/colmap
RUN git checkout tags/3.11.0
RUN mkdir build
WORKDIR /opt/colmap/build
RUN cmake .. -GNinja
RUN ninja
RUN ninja install

RUN apt-get update && apt-get install -y python3-venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN python -m pip install --upgrade pip wheel
#RUN pip install numpy==1.26.4 opencv-python pyyaml plyfile tqdm termcolor kornia imgaug lpips tensorboardX ipdb scikit-image imageio imageio[ffmpeg] imageio[pyav] argparse pytorch_msssim open3d einops decord pycocotools
RUN pip install numpy==1.26.4 opencv-python pyyaml plyfile tqdm termcolor kornia imgaug lpips tensorboardX ipdb scikit-image imageio imageio[ffmpeg] imageio[pyav] argparse pytorch_msssim open3d 
COPY ./sam3 /opt/sam3
WORKDIR /opt/sam3
RUN pip install -e .
RUN pip install -e ".[notebooks]"
RUN pip install -e ".[train,dev]"
RUN pip install "setuptools==69.5.1"

RUN pip install huggingface_hub

# SAM 3 weights: two ways in, no token stored anywhere in the image either way.
#
#   1. Local weights (preferred, no token at all).
#      Put a HuggingFace hub cache at ./hf_cache so that
#      ./hf_cache/hub/models--facebook--sam3 exists, then:
#          docker build -t scsam3 .
#      Populate it once per machine with:
#          cd SCSam3 && HF_HOME=./hf_cache hf download facebook/sam3 --token <token>
#
#   2. Fresh clone with no weights: pass a token as a BuildKit secret.
#          export HF_TOKEN=...
#          DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t scsam3 .
#      The secret is mounted for that one command and never lands in a layer.
#
# ./hf_cache is committed as an empty directory (.gitkeep); its contents are
# gitignored because the weights are 6.5 GB.
COPY ./hf_cache /root/.cache/huggingface

RUN --mount=type=secret,id=hf_token,required=false \
    if [ -f /root/.cache/huggingface/hub/models--facebook--sam3/refs/main ]; then \
        echo "SAM 3 weights: using the copy from ./hf_cache"; \
    elif [ -s /run/secrets/hf_token ]; then \
        echo "SAM 3 weights: not in ./hf_cache, downloading with the mounted token"; \
        HF_TOKEN="$(cat /run/secrets/hf_token)" hf download facebook/sam3 --token "${HF_TOKEN}"; \
    else \
        echo "ERROR: SAM 3 weights are neither in ./hf_cache nor downloadable." >&2; \
        echo "       Populate ./hf_cache, or build with --secret id=hf_token,env=HF_TOKEN." >&2; \
        exit 1; \
    fi

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
RUN echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc
