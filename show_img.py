import h5py
import numpy as np
import cv2
import random
import os

h5_file_path = '/mnt/sda/zsh/dataset/AberrationCorrection-MIPI-dataset/train/train_dataset_pertubed.h5'

save_dir = '/home/ubuntu13/zsh/code/Aberration_Correction_MIPI-main/train_img'


h5f = h5py.File(h5_file_path, 'r')

keys = list(h5f.keys())

# 随机选择三个键
random_keys = random.sample(keys, 3)


for i, key in enumerate(random_keys):
    data = np.array(h5f[key]).reshape(h5f[key].shape)

    lq = data[:, :, 0:3]

    H, W, _ = lq.shape

    print(f"Image {i+1}: Height (H) = {H}, Width (W) = {W}")

    lq = (lq * 255).astype(np.uint8)

    # 保存图像
    save_path = os.path.join(save_dir, f'image_{i+1}.png')
    cv2.imwrite(save_path, cv2.cvtColor(lq, cv2.COLOR_RGB2BGR))

# 关闭h5文件
h5f.close()