docker run -ti \
-v $PWD/$1:/workspace \
-v $PWD/tools:/tools \
-e CLASSES_NUMBER=$2 \
--gpus all \
--rm \
-p 8090:8090 \
xentilzax-darknet:latest \
bash /tools/train-detector.sh
