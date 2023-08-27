import os
import csv

dir = '/workspace/rasp-space/videos/machine-learn/25-degree/output/'

files = os.listdir(dir)
files.sort()

print(files)

train = open('train.csv', 'w', newline='')
test = open('test.csv', 'w', newline='')
train_index = 0
test_index = 0
for file in files:
    if file.endswith("png"):
        #print(file[:4])
        file_index = int(file[:4])
        print(file_index)

        if file_index % 2 == 0:
            entry = file
            train_csv = csv.writer(train)
            train_csv.writerow([entry, str(train_index)])
            train_index = train_index + 1
        else:
            entry = file
            test_csv = csv.writer(test)
            test_csv.writerow([entry, str(test_index)])
            test_index = test_index + 1
