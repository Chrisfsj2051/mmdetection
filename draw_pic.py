import cv2
import numpy as np
import matplotlib.pyplot as plt

from mmdet.datasets.pipelines import Rotate, Shear

img = cv2.imread('008.jpg')
results = {'img':img, 'img_shape':img.shape}
# results = Rotate(level=8, prob=1.0)(results)
results = Shear(prob=1.0, level=8)(results)
img = results['img']
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()