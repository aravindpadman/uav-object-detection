import os
import torch
import torch.nn as nn

from train import evaluate, get_transform
from checkpoint import load_test_checkpoint
from train import get_faster_rcnn_model
import tensorboard_util as tb
from config import cfg
from dataloader import DetectionDataset


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

dataset_test = DetectionDataset(cfg.DATA_DIR, 'test', transforms=get_transform(False))

cfg.TEST.BATCH_SIZE = 16
data_loader = torch.utils.data.DataLoader(                      
         dataset_test, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
         collate_fn=dataset_test.collate_fn)

load_test_checkpoint(cfg, model)


evaluate(model, data_loader, device, writer=None)

from PIL import Image
from torchvision.transforms import functional as func

import fiftyone as fo



dashboard = False
if dashboard:
    # Get class list
    classes = label_map

    dataset = fo.load_dataset('vizdrone-val-set')

    transform = get_transform(False)

    # Add predictions to samples
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            # Load image
            image = Image.open(sample.filepath).convert('RGB')
            img_tensor = func.to_tensor(image).to(device)
            c, h, w = img_tensor.shape


            image, _ = transform(image, target=None)
            #image = image.squeeze(1)

            # Perform inference
            preds = model([image.to(device)])
            labels = preds[0]["labels"].cpu().detach().numpy()
            scores = preds[0]["scores"].cpu().detach().numpy()
            boxes = preds[0]["boxes"].cpu().detach().numpy()

            # Convert detections to FiftyOne format
            detections = []
            for label, score, box in zip(labels, scores, boxes):
                # Convert to [top-left-x, top-left-y, width, height]
                # in relative coordinates in [0, 1] x [0, 1]
                x1, y1, x2, y2 = box
                rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

                detections.append(
                    fo.Detection(
                        label=classes[label],
                        bounding_box=rel_box,
                        confidence=score
                    )
                )

            # Save predictions to dataset
            sample["predictions"] = fo.Detections(detections=detections)
            sample.save()

    session = fo.launch_app(dataset)

    session.wait()
