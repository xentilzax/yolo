cd /workspace
find $PWD -type f -name "*.jpg" > data.txt
python3 /tools/train_test_val.py
cd /
