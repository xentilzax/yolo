docker build \
    -f ubuntu20.04-amd64-cuda11.4.2.dockerfile \
    -t xentilzax/base:cuda11.4.2 \
    .
    
    
docker build \
    -f opencv4.5.0+cuda11.4.2.dockerfile \
    -t xentilzax/opencv-make-deb:opencv4.5.0-cuda11.4.2 \
    .

    
#Make .deb package OpenCV
docker run -ti \
-v "${PWD}":/workspace \
--gpus all \
--rm \
xentilzax/opencv-make-deb:opencv4.5.0-cuda11.4.2 \
bash /workspace/copy.sh

docker build \
    -f inference.dockerfile \
    -t xentilzax/opencv-cuda:opencv4.5.0-cuda11.4.2 \
    .

docker build \
    -f train.dockerfile \
    -t xentilzax/darknet \
    .

