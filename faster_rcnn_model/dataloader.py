import cv2
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np

import transforms as T

def get_faster_rcnn_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        pass
        #transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)




class DetectionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_dir, split, keep_difficult=False, transforms=None):
        """
        :param data_dir: dir where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split

        assert self.split in {'train', 'val', 'test'}

        self.data_dir = data_dir
        self.keep_difficult = keep_difficult
        self.transforms = transforms

        # Read data files
        self.image_files = sorted(os.listdir(os.path.join(data_dir, f'{self.split}/images')))
        self.image_files = [os.path.join(data_dir, f'{self.split}/images/{i}') for i in self.image_files]
        self.annot_files = sorted(os.listdir(os.path.join(data_dir, f'{self.split}/annotations')))
        self.annot_files = [os.path.join(data_dir, f'{self.split}/annotations/{i}') for i in self.annot_files]
        assert len(self.image_files) == len(self.annot_files)

    def __getitem__(self, idx):
        # Read image
        image = Image.open(self.image_files[idx]).convert("RGB")
        # NOTE: opencv load image
        #image = cv2.imread(self.image_files[idx])
        #image = image.transpose((2,0,1))
        #image = torch.FloatTensor(image)

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.preprocess_annotation(self.annot_files[idx])
        boxes = torch.FloatTensor(objects[0])  # (n_objects, 4)
        obj_score = torch.FloatTensor(objects[1])  # (n_objects)
        labels = torch.LongTensor([int(i) for i in objects[2]])  # (n_objects)
        truncation = torch.FloatTensor(objects[3])  # (n_objects)
        #truncation = torch.as_tensor(objects[3], dtype=torch.uint8)  # (n_objects)
        occlusion = torch.as_tensor(objects[4], dtype=torch.uint8)  # (n_objects)
        #occlusion = torch.FloatTensor(objects[4])  # (n_objects)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        
        targets = {}
        targets['boxes'] = boxes
        targets['obj_score'] = obj_score
        targets['labels'] = labels
        targets['truncation'] = truncation
        targets['iscrowd'] = occlusion
        targets['image_id'] = torch.tensor(idx)
        targets['area'] = area

        if self.transforms is not None:
            image, targets = self.transforms(image, targets)

        return image, targets

    def __len__(self):
        return len(self.image_files)

    def preprocess_annotation(self, annot_file_path):
        objects = []
        objectness_score = []
        labels = []
        truncation = []
        occlusion = []

        with open(annot_file_path, 'r') as f:
            for line in f.readlines():
                annots = line.strip().split(',')
                annots = [float(i) for i in annots if i != '']
                assert len(annots) == 8
                if annots[0] >= annots[0] + annots[2] or annots[1] >= annots[1] + annots[3]:
                    print(annot_file_path, line)
                    continue
                objects.append([int(annots[0]), int(annots[1]), int(annots[0]+annots[2]), int(annots[1]+annots[3])]) # (N, 4)
                objectness_score.append(annots[4]) # (N)
                labels.append(annots[5]) # (N)
                truncation.append(annots[6]) # (N)
                occlusion.append(annots[7]) # (N)

        return objects, objectness_score, labels, truncation, occlusion



    def remove_invalid_annotations(self):
        """remove annotations with invalid formats"""
        pass


    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = []
        targets = []
        for i in batch:
            if i[1]['boxes'].size(0) != 0:
                images.append(i[0])
                targets.append(i[1])
                
        return images, targets

if __name__ == '__main__':
    image_dataset = DetectionDataset('/home/welcome/uav-object-detection/data', 'train', transforms=get_transform(True))
    dataloader = torch.utils.data.DataLoader(
                                image_dataset,
                                batch_size=32, 
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True,
                                collate_fn=image_dataset.collate_fn,
                                )
    #for i in dataloader:
    #    pass
    print(image_dataset[0])
