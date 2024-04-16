import matplotlib.pyplot as plt
import torch

from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torch
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import faster_rcnn
from torchvision.models.detection import mask_rcnn

import matplotlib.pyplot as plt

def get_instance_segmentation_model(num_classes):
    
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    
    hidden_layer = 256
    model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

num_classes = 1
grape_model = get_instance_segmentation_model(num_classes)
grape_model.load_state_dict(torch.load('wgisd_v1.5.pth'))
grape_model.eval()