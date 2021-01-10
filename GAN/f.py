import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image
import skimage
import numpy as np
from skimage import filters
import scipy
from skimage import morphology
from scipy import ndimage
import matplotlib.pyplot as plt
from collections import Counter


def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def pre_processing(images):
    val = filters.threshold_otsu(images)
    rest = np.where(images>val,0,1)
    close = scipy.ndimage.morphology.binary_closing(rest).astype(np.int)
    dilation = ndimage.morphology.binary_dilation(close).astype(np.int)
    labeled_array, num_features  = ndimage.measurements.label(dilation)
    re = morphology.remove_small_objects(labeled_array, min_size=600, connectivity=1, in_place=False).astype(np.int)
    img = morphology.binary_erosion(re).astype(np.int)
    plt.subplot(122)
    plt.imshow(img)   

    return img

m = ['a','a','b','c']
b = Counter(m).most_common(1)[0][1]
c = Counter(m).most_common(1)[0][0]
print(c)
folder="/Users/rolin/Desktop/Projcv/forschung/Schwabacher_Glyphs_first_batch/large_m"
m = load_images(folder)
split_1 = int(0.8 * len(m))
train_filenames = m[:split_1]
for i in range(len(m)):
# m = np.array(Image.open('/Users/rolin/Desktop/Projcv/forschung/Schwabacher_Glyphs_first_batch/large_s/large_s-m-86.jpg'))
    plt.subplot(121)
    plt.imshow(m[i])
    pre_processing(m[i])
    plt.show()
