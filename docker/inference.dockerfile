FROM xentilzax/base:cuda11.4.2

#
# Install OpenCV
#
ARG OPENCV_VERSION="4.5.0"
ARG CUDA="11.4.2"

ADD opencv-cuda${CUDA}_${OPENCV_VERSION}_amd64.deb /opt/opencv.deb

RUN DEBIAN_FRONTEND=noninteractive dpkg -i /opt/opencv.deb
RUN rm /opt/opencv.deb
RUN ldconfig

RUN pip3 install pandas

RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean
