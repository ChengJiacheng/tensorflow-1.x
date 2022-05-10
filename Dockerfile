# FROM tensorflow/tensorflow:1.14.0-gpu-py3
# FROM tensorflow/tensorflow:1.4.1-gpu-py3
# FROM tensorflow/tensorflow:1.14.0-gpu
FROM tensorflow/tensorflow:1.14.0-gpu-py3
ENV DEBIAN_FRONTEND noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get install -y wget
RUN apt-key del 7fa2af80 && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update && apt-get install -y sudo psmisc locales cmake vim zip htop git  libsm6 libxext6 libxrender-dev openexr libopenexr-dev
RUN locale-gen en_US.UTF-8

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
    
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

ADD https://raw.githubusercontent.com/ChengJiacheng/DeepOpticsHDR/master/environment.yml /tmp/environment.yml
RUN conda env create --file /tmp/environment.yml && conda init bash

## ADD CONDA ENV PATH TO LINUX PATH 

RUN echo "source activate DeepOpticsHDR" >> ~/.bashrc
ENV PATH /opt/conda/envs/DeepOpticsHDR/bin:$PATH

ADD https://github.com/coder/code-server/releases/download/v4.3.0/code-server_4.3.0_amd64.deb ./ 
RUN sudo dpkg -i code-server_4.3.0_amd64.deb && code-server --install-extension ms-python.python

# Install OpenCV
RUN apt-get update && apt-get install -y build-essential libgtk-3-dev libvtk7-dev   \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev   \
    gfortran libopenexr-dev libatlas-base-dev libtbb2 libtbb-dev
ADD https://github.com/opencv/opencv/archive/refs/tags/3.4.15.zip /tmp/opencv.zip
RUN cd /tmp && unzip opencv.zip
RUN cd /tmp/opencv-3.4.15/ && mkdir -p build && cd build && cmake ..
RUN cd /tmp/opencv-3.4.15/build && make -j4 && sudo make install
