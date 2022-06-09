WS=${1:-""}
NUMBER=${2:-""}
if [ "$WS" == "" ]; then
	echo "Need two argument: <workspace name> <class number>"
        exit 1
fi
if [ "$NUMBER" == "" ]; then
        echo "Need two argument: <workspace name> <class number>"
        exit 1
fi




docker run -ti \
-v $PWD/AlexeyAB/darknet:/darknet \
-v $PWD/$WS:/workspace \
-v $PWD/tools:/tools \
-e CLASSES_NUMBER=$NUMBER \
--gpus all \
--rm \
-p 8090:8090 \
xentilzax/darknet:latest \
bash /tools/train-classifier.sh
