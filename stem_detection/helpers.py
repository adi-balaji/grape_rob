import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
import cv2

def hello_helpers():
  print('hello helpers!')

def plot_bboxes(image_tensor, bboxes_tensor):
    
    image = image_tensor.permute(1,2,0)
    plt.imshow(image)
    
    for bbox in bboxes_tensor:
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        rect = plt.Rectangle((xmin, ymin), width, height, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    
    plt.axis('off')
    plt.show()