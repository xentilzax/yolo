docker run -ti \
-v $PWD:/workspace \
--gpus all \
--rm \
-e INPUT_VIDEO=$1 \
-e OUTPUT_VIDEO=$2 \
xentilzax-opencv-cuda11.4.2:latest \
bash /workspace/script.sh
