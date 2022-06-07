#make dir backup
mkdir -p /workspace/backup

#make file obj.data
echo "classes = ${CLASSES_NUMBER}" > /workspace/obj.data
echo "train  = /workspace/train.txt" >> /workspace/obj.data
echo "valid  = /workspace/test.txt" >> /workspace/obj.data
echo "names = /workspace/obj.names" >> /workspace/obj.data
echo "backup = /workspace/backup/" >> /workspace/obj.data

#make file obj.names
rm -f /workspace/obj.names
for i in $(seq 1 1 $CLASSES_NUMBER)
do
   echo "obj_$i" >> /workspace/obj.names
done

#make YOLO-Tiny config
cp -n /tools/custom.cfg /workspace/

sed -i -e "s/classes=classes_number/classes=${CLASSES_NUMBER}/g" /workspace/custom.cfg

FILTERS_NUMBER=$((($CLASSES_NUMBER+5)*3))
sed -i -e "s/filters=func_classes_number/filters=${FILTERS_NUMBER}/g" /workspace/custom.cfg

ITERATION_NUMBER=$(($CLASSES_NUMBER * 2000))
sed -i -e "s/max_batches=ITERATION_NUMBER/max_batches=${ITERATION_NUMBER}/g" /workspace/custom.cfg

STEP1=$(($CLASSES_NUMBER * 1000))
STEP2=$(($CLASSES_NUMBER * 1500))
sed -i -e "s/steps=STEP1,STEP2/steps=${STEP1},${STEP2}/g" /workspace/custom.cfg


#make train.txt and test.txt
/tools/data_for_train.sh

cd /tools && wget -nc  https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29

#run training
/darknet/darknet detector train \
/workspace/obj.data \
/workspace/custom.cfg \
/tools/yolov4-tiny.conv.29 \
-dont_show \
-mjpeg_port 8090 \
-map
