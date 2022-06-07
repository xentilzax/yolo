from random import shuffle

text_file = open("data.txt", "r")
line = text_file.readlines()
#print(lines[0])

shuffle(line)

#print(lines[0])

test_portion = 0.1
val_portion = 0
n = len(line)
with open("train.txt", 'w') as f:
    for i in range(int(n*(test_portion + val_portion)), n):
        f.write(line[i])
        
with open("test.txt", 'w') as f:
    for i in range(0, int(n * test_portion)):
        f.write(line[i])
        
with open("val.txt", 'w') as f:
    for i in range(int(n * test_portion), int(n*(test_portion + val_portion))):
        f.write(line[i])
