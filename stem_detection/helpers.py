import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
import cv2

def hello_helpers():
  print('hello helpers!')

def plot_bboxes(image, bboxes):

    img_np = np.array(image)
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)

    for bbox in bboxes:
        x, y, w, h = bbox
        x1, y1 = x, y
        x2, y2 = x + w, y + h

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')

        ax.add_patch(rect)

    plt.show()