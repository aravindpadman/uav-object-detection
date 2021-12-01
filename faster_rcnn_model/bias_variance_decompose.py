"""
plot training, validation and test error/mAP vs epochs
"""
import os
import sys
import time
import math
import glob
import pickle
import pandas as pd
from tqdm import tqdm
import utils
import torch
import torch.nn as nn

from config import cfg
import transforms as T
from tensorboard_util import TensorboardWriter
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from dataloader import DetectionDataset
from faster_rcnn import fasterrcnn_resnet50_fpn
from checkpoint import save_checkpoint, load_test_checkpoint

from train import evaluate

def get_faster_rcnn_model(pretrained=True):
    model = fasterrcnn_resnet50_fpn(pretrained)
    return model



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        pass
        #transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main(cfg):
    # set device
    device =torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create train, val and test datasets
    dataset = DetectionDataset(cfg.DATA_DIR, 'train', transforms=get_transform(True))
    dataset_val = DetectionDataset(cfg.DATA_DIR, 'val', transforms=get_transform(False))
    dataset_test = DetectionDataset(cfg.DATA_DIR, 'test', transforms=get_transform(False))

    # create tensorboard writer
    writer = TensorboardWriter(cfg)

    cfg.TRAIN.BATCH_SIZE = 16
    cfg.TEST.BATCH_SIZE = 16
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=1, num_workers=4,
        collate_fn=dataset.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
        collate_fn=dataset_val.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
        collate_fn=dataset_test.collate_fn)

    dffs = []
    checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, 'checkpoints/*')
    for epoch, checkpoint_path in enumerate(sorted(glob.glob(checkpoint_dir))):
        print(checkpoint_path)
        cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_path
        model = get_faster_rcnn_model(pretrained=False)
        model.to(device)
        model.eval()
        load_test_checkpoint(cfg, model)
        coco_eval_train = evaluate(model, data_loader, device)
        coco_eval_val = evaluate(model, data_loader_val, device)
        coco_eval_test = evaluate(model, data_loader_test, device)
        train_mAP = coco_eval_train.coco_eval['bbox'].stats[0]
        val_mAP = coco_eval_val.coco_eval['bbox'].stats[0]
        test_mAP = coco_eval_test.coco_eval['bbox'].stats[0]
        # log to tensorboard
        if writer:
            writer.add_scalars("mAP@[0.5:0.95]", {'Train': train_mAP, 'Val': val_mAP, 'Test': test_mAP}, epoch)
        dffs.append({'epoch': epoch, 'train_mAPs': coco_eval_train.coco_eval['bbox'], 'val_mAPs': coco_eval_val.coco_eval['bbox'], 'test_mAPs': coco_eval_test.coco_eval['bbox']})
        with open(os.path.join(cfg.OUTPUT_DIR, 'bias_variance_decompose.pkl'), 'wb') as f:
            pickle.dump(dffs, f)
    dt = pd.DataFrame(dffs)
    dt.to_csv(os.path.join(cfg.OUTPUT_DIR, 'map_vs_epochs.csv'), index=False)


            
if __name__ == '__main__':
    main(cfg)
