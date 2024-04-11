import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import json
import cv2

def hello_grape_net():
  print('hello grape_net!')

##########################################################################################################################################################################################

class GrapeDataset(Dataset):
  def __init__(self, img_folder, ann_folder):
    self.img_folder = img_folder
    self.ann_folder = ann_folder
    self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    self.labels = []
    self.images = []

    for file in os.listdir(img_folder):
      img = cv2.imread(os.path.join(img_folder,file))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      self.images.append(img)
    print(f'Loaded {len(self.images)} images from folder {self.img_folder}')

    for ann in os.listdir('grape_stem_data/grape_dataset/ann'):
      with open(os.path.join('grape_stem_data/grape_dataset/ann', ann), 'r') as f:
        objects = json.load(f)['objects']
        img_label = []
        for obj in objects:
          listofbboxs = obj['points']['exterior']
          image_bbox_coords = [point for bbox in listofbboxs for point in bbox]
          img_label.append(image_bbox_coords)

        self.labels.append(img_label)
    
    #small set for testing
    # self.images = self.images[:20]
    # self.labels = self.labels[:20]

    print(f'Loaded {len(self.labels)} labels from folder {self.ann_folder}')
        
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    annotation = self.labels[idx]
    raw_image = self.images[idx]

    bbox = torch.tensor(annotation)
    targets = {'boxes':bbox}

    np_image = np.array(raw_image)
    image = torch.tensor(np_image).permute(2,0,1)

    labels = torch.tensor([0] * len(annotation))
    targets['labels'] = labels
    

    # if self.transform:
    #     image = self.transform(image)
        
    return image, targets


##########################################################################################################################################################################################
  
class FeatureExtractor(nn.Module):
  def __init__(self):

        super(FeatureExtractor, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.vgg19 = nn.Sequential(*list(vgg.features.children())[:-1])

  def forward(self, image):

        features = self.vgg19(image)

        return features

##########################################################################################################################################################################################

class StemDetector(nn.Module):
    def __init__(self):
        super(StemDetector, self).__init__()

        #feature pyramid hetwork
        self.fpn = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bbox_det = nn.Linear(256, 4)  # [N, (x, y, w, h)]

    def forward(self, features):

        fpn_features = self.fpn(features)
        pooled = self.global_avgpool(fpn_features)
        pooled = pooled.view(pooled.size(0), -1)
        bbox_preds = self.bbox_det(pooled)


        return bbox_preds

##########################################################################################################################################################################################

class GrapeNet(nn.Module):
    def __init__(self):
        super(GrapeNet, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.stem_detector = StemDetector()

    def forward(self, image):
        features = self.feature_extractor.forward(image)
        bboxes = self.stem_detector.forward(features)
        return bboxes

##########################################################################################################################################################################################