from PIL import Image
import os

odir = '/workspace/rasp-space/videos/machine-learn/25-degree/output/'

files = os.listdir(odir)
files.sort()

for image_path in files:
    if image_path.endswith("png"):
        image = Image.open(odir + image_path)
        new_image = image.resize((100, 250))
        new_image.save(odir + image_path)
