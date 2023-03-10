# NeRF

import os
import json
import imageio

# data practice
_datadir = './data/nerf_synthetic/lego'

with open(os.path.join(_datadir, 'transforms_test.json'), 'r') as fp:   # 'r' : read only mode
    meta = json.load(fp)

print(str(meta)[:300])
 # 파일 구성을 보면, 각 이미지마다 file_name, rotation, transform_matrix로 구성되어 있다.
 # transform_matrix : rotation과 translation을 시켜주는 matrix. 4x4로 구성.
 # cam2world(=camera pose) matrix : 카메라 기준의 좌표에 transform matrix를 곱해주면 world 좌표로 변경된다.

# 첫 번째 frame 확인
_frames = meta['frames']
_frame = _frames[0]
print()
print(_frame)

# full name으로 변경
_fname = os.path.join(_datadir, _frame['file_path']+'.png')
print()
print(_fname)

# image load 및 shape 확인
img = imageio.imread(_fname)
print()
print(img.shape) # 4 channel rgba(a : mask channel)

# image 시각화
import matplotlib.pyplot as plt
fig, axis = plt.subplots(1, 10, figsize=(30,300))
for i in range(10):
    _frame = _frames[i]
    _fname = os.path.join(_datadir, _frame['file_path']+'.png')
    img = imageio.imread(_fname)
    axis[i].imshow(img)
    
# random으로 image 시각화
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

imgs = (np.array(imgs) / 255.).astype(np.float32) # 0~1사이 값으로 변경
print(imgs.shape)

imgs_orig = imgs[...,:3] # 처음 3개 channel(original image)
imgs_mask = imgs[...,-1] # 추가된 1개 channel(mask image)
imgs = imgs[...,:3]*imgs[...,-1:] + (1-imgs[...,-1:])
 # (1-imgs[...,-1:]) : object 부분은 0. background는 1. -> 배경이 흰 색이 된다.

fig, axis = plt.subplots(1,3)
axis[0].imshow(imgs_orig[0]) # original image
axis[1].imshow(imgs_mask[0], cmap='gray') # mask image
axis[2].imshow(imgs[0]) # original image + mask image