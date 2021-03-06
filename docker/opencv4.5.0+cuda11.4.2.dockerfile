FROM xentilzax/base:cuda11.4.2

ARG SRC=/opt
ARG OPENCV_VERSION="4.5.0"
ARG CUDA="11.4.2"
ARG CUDA_ARCH_BIN="6.1,7.0,7.2,7.5,8.6"

WORKDIR /opt

RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git
RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git

RUN cd "${SRC}/opencv" && \
rm -rf build && \ 
mkdir build && \
cd build && \
cmake \
    -D BUILD_EXAMPLES=OFF \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D CUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
    -D CUDA_ARCH_PTX="" \
    -D CUDA_FAST_MATH=ON \
    -D CUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
    -D ENABLE_FAST_MATH=ON \
    -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH="${SRC}/opencv_contrib/modules" \
    -D WITH_EIGEN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D BUILD_opencv_cudacodec=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
   	-D OPENCV_GENERATE_PKGCONFIG=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_TBB=ON \
    -D WITH_CUDNN=ON \
    ..

RUN cd "${SRC}/opencv/build" && \
    make -j$(nproc) && \
    make package
RUN rm -rf "${SRC}/packages" && \
    mkdir -p "${SRC}/packages" && \
    mv "${SRC}/opencv/build/OpenCV-${OPENCV_VERSION}-x86_64.tar.gz" "${SRC}/packages" && \
    cd "${SRC}/packages" && \
    tar -xzvf OpenCV-${OPENCV_VERSION}-x86_64.tar.gz

#install .deb packager builder FPM
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y \
    ruby \
    ruby-dev \
    rubygems   
RUN gem install rake
RUN gem install fpm

RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN fpm -f \
    -t deb \
    -s dir \
    -C "${SRC}/packages/OpenCV-${OPENCV_VERSION}-x86_64" \
    --name opencv-cuda${CUDA} \
    --version "${OPENCV_VERSION}" \
    --description "OpenCV compiled with CUDA version ${CUDA} for CUDA architecture ${CUDA_ARCH_BIN}" \
    --prefix /usr/local/ \
    -p "${SRC}/packages"
