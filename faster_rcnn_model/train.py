import time
import math
import torch
import torchvision
import transforms as T
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from dataloader import DetectionDataset

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from faster_rcnn import fasterrcnn_resnet50_fpn
from checkpoint import save_checkpoint
import tensorboard_util as tb
from config import cfg

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


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    num_batches = len(data_loader)

    lr_scheduler = None
    
    i = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        if i == 1: break
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # tensorboard writer
        if writer:
            for name, meter in metric_logger.meters.items():
                writer.add_scalar(f'Train/{name}', meter.avg, epoch*num_batches + i)

        i += 1

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, writer=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    i = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        #if i == 2: break
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        i += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def main(cfg):

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 12
    # use our dataset and defined transformations
    dataset = DetectionDataset(cfg.DATA_DIR, 'train', transforms=get_transform(True))
    dataset_test = DetectionDataset(cfg.DATA_DIR, 'val', transforms=get_transform(False))


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4,
        collate_fn=dataset.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
        collate_fn=dataset_test.collate_fn)

    # TODO add tensorboard writer
    if cfg.TENSORBOARD.VISUALIZE:
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # get the model using our helper function
    model = get_faster_rcnn_model()

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=3e-4)
                           
    # let's train it for 10 epochs
    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, writer=writer)
        # update the learning rate
        #lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, data_loader_test, device=device, writer=writer)
        # TODO add model checkpoint logic here
        if cfg.SAVE_CHECKPOINT:
            save_checkpoint(model, optimizer, epoch, cfg, scaler=None)

    if writer:
        writer.close()

    print("That's it!")

if __name__ == '__main__':
    main(cfg)
