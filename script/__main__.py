import sys
import os

import imageio
import numpy
from nibabel.viewers import OrthoSlicer3D
from nibabel import nifti1
import nibabel as nib
from matplotlib import pylab as plt
import matplotlib


filename = sys.argv[1]
img = nib.load(filename)
img0 = img.get_fdata()
# img0 -= img0.min()
# img0 *= 255.0 / img0.max()
img0 = img0.astype(numpy.uint8)
data_shape = img.dataobj.shape
print(f'图像资源形状：{data_shape}')
if len(data_shape) != 3:
    exit(0)
width, height, queue = data_shape
# for i in range(queue):
#     this_img = img0[:, :, i]
#     imageio.imwrite(os.path.join(sys.argv[2], f'{i}.png'), this_img)

OrthoSlicer3D(img.dataobj).show()
x = int((queue / 10) ** 0.5) + 1
print(f'x: {x}')
num = 1
for i in range(0, queue, 5):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(x, x, num)
    num += 1
plt.show()
