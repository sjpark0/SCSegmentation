# Define Base Image
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel
# Install dependencies

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y git cmake build-essential ninja-build libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev curl ffmpeg pkg-config python3 python3-dev rsync software-properties-common unzip libopencv-dev imagemagick

#RUN cd /opt
WORKDIR /opt
RUN git clone https://github.com/colmap/colmap.git
#RUN cd colmap
WORKDIR /opt/colmap
RUN git checkout tags/3.11.0
RUN mkdir build
#RUN cd build
WORKDIR /opt/colmap/build
RUN cmake .. -GNinja
RUN ninja
RUN ninja install
#RUN cd /
RUN python -m pip install --upgrade pip
RUN pip install numpy==1.26.4
RUN pip install opencv-python pyyaml plyfile tqdm termcolor kornia imgaug lpips tensorboardX ipdb scikit-image imageio imageio[ffmpeg] imageio[pyav] mmcv==1.6.0 argparse pytorch_msssim open3d

RUN pip install matplotlib tikzplotlib jpeg4py lmdb pandas scipy loguru

COPY sam2 /opt/sam2
WORKDIR /opt/sam2
RUN pip install -e .
RUN pip install -e ".[notebooks]"

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

RUN echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc
