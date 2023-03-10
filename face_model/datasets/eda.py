import os
import imageio
import json
import numpy as np

# data practice
_datadir = './data/nerf_synthetic/lego'

with open(os.path.join(_datadir, 'transforms_test.json'), 'r') as fp:
    meta = json.load(fp)


print(str(meta)[:300])

_frames = meta['frames']
_frame = _frames[0]
print(_frame)

_fname = os.path.join(_datadir, _frame['file_path']+'.png')
print(_fname)

img = imageio.imread(_fname)
print(img.shape) # 4 channel rgba

import matplotlib.pyplot as plt
fig, axis = plt.subplots(1, 10, figsize=(30,300))
for i in range(10):
    _frame = _frames[i]
    _fname = os.path.join(_datadir, _frame['file_path']+'.png')
    img = imageio.imread(_fname)
    axis[i].imshow(img)

import random
random.shuffle(_frames)
fig, axis = plt.subplots(1, 10, figsize=(30,300))
imgs = []
for i in range(10):
    _frame = _frames[i]
    _fname = os.path.join(_datadir, _frame['file_path']+'.png')
    img = imageio.imread(_fname)
    imgs.append(img)
    axis[i].imshow(img)

imgs = (np.array(imgs) / 255.).astype(np.float32)
print(imgs.shape)

imgs_orig = imgs[...,:3]
imgs_mask = imgs[...,-1]
imgs = imgs[...,:3]*imgs[...,-1:] + (1-imgs[...,-1:])

fig, axis = plt.subplots(1,3)
axis[0].imshow(imgs_orig[0])
axis[1].imshow(imgs_mask[0], cmap='gray')
axis[2].imshow(imgs[0])