docker build \
    -f ubuntu20.04-amd64-cuda11.4.2 \
    -t xentilzax-cuda11.4.2-base \
    .
    
    
docker build \
    -f opencv4.5.0+cuda11.4.2 \
    -t xentilzax-opencv-cuda11.4.2 \
    .

    
#Make .deb package OpenCV
docker run -ti \
-v "${PWD}":/workspace \
--gpus all \
--rm \
xentilzax-opencv-cuda11.4.2 \
bash /workspace/copy.sh
