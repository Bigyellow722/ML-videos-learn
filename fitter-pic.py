import os


dir = '/workspace/rasp-space/videos/machine-learn/25-degree/output/'

files = os.listdir(dir)
files.sort()

print(files)

train = open('train.txt', 'a')
test = open('test.txt', 'a')
train_index = 0
test_index = 0
for file in files:
    if file.endswith("png"):
        #print(file[:4])
        file_index = int(file[:4])
        print(file_index)

        if file_index % 2 == 0:
            entry = dir + file + ' ' + str(train_index) + '\n'
            train.write(entry)
            train_index = train_index + 1
        else:
            entry = dir + file + ' ' + str(test_index) + '\n'
            test.write(entry)
            test_index = test_index + 1
