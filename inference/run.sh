docker run -ti \
-v $PWD:/workspace \
--gpus all \
--rm \
-e INPUT_VIDEO=$1 \
-e OUTPUT_VIDEO=$2 \
xentilzax/opencv-cuda:opencv4.5.0-cuda11.4.2 \
bash /workspace/script.sh
