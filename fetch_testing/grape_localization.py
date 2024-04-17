import torch
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import faster_rcnn
from torchvision.models.detection import mask_rcnn
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline
import numpy as np

grapes_model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = grapes_model.roi_heads.box_predictor.cls_score.in_features
num_classes = 2
grapes_model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
in_features_mask = grapes_model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
grapes_model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
grapes_model.load_state_dict(torch.load('/Users/adibalaji/Desktop/grape_juice/stem_detection/stem_mask1.4.pth'))
grapes_model.eval()

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

def plot_point_cloud(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mean_point = [np.mean(x), np.mean(y), np.mean(z)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=z, cmap='Purples', alpha=0.5, s=30) # grape point cloud
    ax.scatter(mean_point[0], mean_point[1], mean_point[2], c='red', marker='o') # bunch center


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D reconstruction')
    ax.view_init(elev=45, azim=90, roll=0)
    ax.autoscale_view()

    plt.show()

def point_cloud_reconstruction(depth_tensor, K_inv, sparse_point_cloud=False, sparse_points=500):
    points = []
    for y, row in enumerate(depth_tensor):
        for x, val in enumerate(row):
            if val != 0:
                
                point_pixel = np.array([x, y, 1])
                point_camera = np.matmul(K_inv, point_pixel)
                depth_value = depth_tensor[y, x]
                if depth_value < 0.2 or depth_value > 0.95:
                    continue

                
                point_camera[2] = depth_value
                points.append(point_camera)

    points = np.array(points)
    
    if not sparse_point_cloud:
        plot_point_cloud(points)
        return points
    else:
        sample_indices = np.random.choice(len(points), size=sparse_points, replace=False)
        points = points[sample_indices]
        plot_point_cloud(points)
        return points

def predict_stem_mask(stem_model, image_tensor):
    with torch.no_grad():
        pred = stem_model(image_tensor)
        pred = pred.pop()
        masks = pred['masks']
    return masks

def get_stem_pose(stem_mask, image_pil):
    depth_out = pipe(image_pil)
    depth_map = to_tensor(depth_out['depth'])  * stem_mask
    depth_map = depth_map.squeeze(0)

    plt.imshow(depth_map.numpy())
    plt.show()

    #Camera matrix K: [527.1341414037195, 0.0, 323.8974379222906, 0.0, 525.9099904918304, 227.2282369544078, 0.0, 0.0, 1.0]
    fx = 527.1341414037195
    fy = 525.9099904918304
    cx = 227.2282369544078
    cy = stem_mask.shape[1] / 2

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    K_inv = np.linalg.inv(K)

    point_cloud = point_cloud_reconstruction(depth_map, K_inv, sparse_point_cloud=False)

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    mean_point = [np.mean(x), np.mean(y), np.mean(z)]

    return depth_map, mean_point

test_img_file = '/Users/adibalaji/Desktop/grape_juice/fetch_cam/rgb_image.jpg'
test_image = Image.open(test_img_file)
test_tensor = to_tensor(test_image).unsqueeze(0)

stem_masks = predict_stem_mask(grapes_model, test_tensor)

plt.imshow(test_image)
plt.imshow(stem_masks[0].permute(1,2,0).numpy(), alpha=0.5)
plt.show()

depth_map, stem_pose = get_stem_pose(stem_masks[0], test_image)

fetch_pose = np.array([stem_pose[2], stem_pose[0], stem_pose[1]])

print(f'Fetch pose: {fetch_pose}')






