#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: benchmark-opencv-resize.py


import cv2
import time
import numpy as np

"""
Some prebuilt opencv is much slower than others.
You should check with this script and make sure it prints < 1s.
On E5-2680v3, archlinux, this script prints:
    0.61s for system opencv 3.4.0-2.
    >5 s for anaconda opencv 3.3.1 py36h6cbbc71_1.
On E5-2650v4, this script prints:
    0.6s for opencv built locally with -DWITH_OPENMP=OFF
    0.6s for opencv from `pip install opencv-python`.
    1.3s for opencv built locally with -DWITH_OPENMP=ON
    2s for opencv from `conda install`.
"""


img = (np.random.rand(256, 256, 3) * 255).astype('uint8')

start = time.time()
for k in range(1000):
    out = cv2.resize(img, (384, 384))
print(time.time() - start)
