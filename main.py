import torch
import torch.nn as nn
import os
import cv2 as cv

'''
data_dir = '/home/wqy/rasp-space/videos/50-degree'

img_path = data_dir + "/0001.png"

img = cv.imread(img_path)
cv.imshow("test", img)
cv.waitKey(0)
'''
m = nn.Conv2d(16, 33, 3, stride=2)
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input_test = torch.randn(20, 16, 50, 100)
output = m(input_test)
print("============== INPUT ==============")
print(input_test)
print("============== OUTPUT ==============")
print(output)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
