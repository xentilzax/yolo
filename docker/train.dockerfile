FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04
LABEL maintainer="d.kravchenko@ivideon.ru"

#
# setup environment
#
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

#FIX GPG NVIDIA key error 
RUN rm -rf /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

#solve trouble with tzdata
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#install OpenCV, Qt
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
	libopencv-dev \
	ssh \
	wget \
	nano \
	libssl-dev \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
        libavresample-dev \
        libtbb-dev \
        libtbb2 \
        libx264-dev \
        pkg-config

	
RUN apt-get install -y --no-install-recommends git

#install CMake 3.20
ARG CMAKE_VERSION=3.20.0

RUN cd /opt && \
    git clone --depth 1 --branch master https://github.com/Kitware/CMake.git
 
RUN cd /opt/CMake && \
    git fetch --all --tags && \
    git checkout tags/v${CMAKE_VERSION} -b v${CMAKE_VERSION}-branch && \
    ./bootstrap
RUN cd /opt/CMake && \
	make -j$(nproc)
RUN cd /opt/CMake && \
	make install
RUN rm -rf /opt/CMake

#install darknet(AlexeyAB)
RUN cd / && git clone https://github.com/AlexeyAB/darknet.git
RUN cd /darknet/ && \
	sed -i -e "s/GPU=0/GPU=1/g" Makefile && \
	sed -i -e "s/CUDNN=0/CUDNN=1/g" Makefile && \
	sed -i -e "s/OPENCV=0/OPENCV=1/g" Makefile && \
	sed -i -e "s/AVX=0/AVX=1/g" Makefile && \
    make -j4

RUN apt-get install -y --no-install-recommends python3

RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean

