docker build \
    -f ubuntu20.04-amd64-cuda11.4.2.dockerfile \
    -t xentilzax-cuda11.4.2-base \
    .
    
    
docker build \
    -f opencv4.5.0+cuda11.4.2.dockerfile \
    -t xentilzax-opencv-cuda11.4.2:build \
    .

    
#Make .deb package OpenCV
docker run -ti \
-v "${PWD}":/workspace \
--gpus all \
--rm \
xentilzax-opencv-cuda11.4.2:build \
bash /workspace/copy.sh

docker build \
    -f inference.dockerfile \
    -t xentilzax-opencv-cuda11.4.2 \
    .

docker build \
    -f train.dockerfile \
    -t xentilzax-darknet \
    .

