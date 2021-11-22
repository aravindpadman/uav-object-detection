import cv2
import os
import torch
import numpy as np
import torch.nn as nn
import glob
from tqdm import tqdm
from train import evaluate, get_transform
from checkpoint import load_test_checkpoint
from train import get_faster_rcnn_model
import tensorboard_util as tb
from config import cfg

label_map = {
    0: 'ignored regions', 
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor',
    11: 'others'
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#load_test_checkpoint(cfg, model)

model = get_faster_rcnn_model(pretrained=False)
model.to(device)
model.eval()

load_test_checkpoint(cfg, model)

from PIL import Image
from torchvision.transforms import functional as func

images_path = sorted(glob.glob('/home/welcome/Downloads/VisDrone2019-MOT-val/sequences/uav0000117_02622_v/*.jpg'))

# Get class list
classes = label_map

transform = get_transform(False)

image_array = []
for file_path in tqdm(images_path):
    # Load image
    image = Image.open(file_path).convert('RGB')
    img_tensor = func.to_tensor(image).to(device)
    c, h, w = img_tensor.shape


    image, _ = transform(image, target=None)
    #image = image.squeeze(1)

    # Perform inference
    preds = model([image.to(device)])
    labels = preds[0]["labels"].cpu().detach().numpy()
    scores = preds[0]["scores"].cpu().detach().numpy()
    boxes = preds[0]["boxes"].cpu().detach().numpy()
    detections = []
    thickness = 1
    for label, score, box in zip(labels, scores, boxes):
        # Convert to [top-left-x, top-left-y, width, height]
        # in relative coordinates in [0, 1] x [0, 1]
        if score > 0.6:
            x1, y1, x2, y2 = (int(i) for i in box)
            if isinstance(image, torch.Tensor):
                image = image.permute(1,2,0).cpu().numpy()
                image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            else:
                image = cv2.rectangle(np.float32(image), (x1, y1), (x2, y2), (0, 255, 0), thickness)
            label_txt = f"{classes[label]}({score:1.2f})"
            origin = (x1, y1-10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = 0.6
            image = cv2.putText(image, label_txt, origin, font, fontscale, (0, 0, 255), 2, cv2.LINE_AA)
    image_array.append(image)

vid_name = "video_detection"
out = cv2.VideoWriter(f'{vid_name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w, h))

for i in range(len(image_array)):
    out.write(image_array[i].astype(np.uint8))
out.release()
