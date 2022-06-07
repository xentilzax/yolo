docker run -ti \
-v $PWD:/workspace \
--gpus all \
--rm \
xentilzax-opencv-cuda11.4.2:latest \
bash python3 /workspace/inference.py -i=video3.mp4 -o=res-video3.mp4
