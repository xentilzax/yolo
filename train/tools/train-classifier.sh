#make dir backup
mkdir -p /workspace/backup

#make file setting.txt
echo "classes = ${CLASSES_NUMBER}" > /workspace/settings.txt
echo "train  = /workspace/train.txt" >> /workspace/settings.txt
echo "valid  = /workspace/test.txt" >> /workspace/settings.txt
echo "labels = /workspace/names.txt" >> /workspace/settings.txt
echo "backup = /workspace/backup" >> /workspace/settings.txt
echo "top=2" >> /workspace/settings.txt

#make file names.txt
#rm -f /workspace/obj.names
#for i in $(seq 1 1 $CLASSES_NUMBER)
#do
#   echo "obj_$i" >> /workspace/names.txt
#done

#Check exist config.txt
FILE=/workspace/custom.cfg
if [ -f "$FILE" ]; then
    echo "Fount config: $FILE"
else 
    echo "Not found config file: $FILE"
fi
#cp -n /tools/yolo-tiny-custom.cfg /workspace/

#make train.txt and test.txt
/tools/data_for_train.sh

#run training
/darknet/darknet classifier train \
/workspace/settings.txt \
/workspace/custom.cfg \
-dont_show \
-mjpeg_port 8090 
