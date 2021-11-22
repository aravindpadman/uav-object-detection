"""Functions that handle saving and loading of checkpoints."""

import copy
import numpy as np
import os
import pickle
from collections import OrderedDict
import torch


def get_checkpoint_dir(path_to_job):
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "checkpoints")


def get_path_to_checkpoint(checkpoint_dir, epoch):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(checkpoint_dir, name)


#def get_last_checkpoint(path_to_job):
#    """
#    Get the last checkpoint from the checkpointing folder.
#    Args:
#        path_to_job (string): the path to the folder of the current job.
#    """
#    names = os.listdir(path_to_job)
#    names = [f for f in names if "checkpoint" in f]
#    assert len(names), "No checkpoints found in '{}'.".format(d)
#    # Sort the checkpoints by epoch.
#    name = sorted(names)[-1]
#    return os.path.join(d, name)

def get_last_checkpoint(path_to_job):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job)
    names = os.listdir(d)
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)


def has_checkpoint(path_to_job):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    files = os.listdir(get_checkpoint_dir(path_to_job))
    return any("checkpoint" in f for f in files)


def save_checkpoint(model, optimizer, epoch, cfg, scaler=None):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
        scaler (GradScaler): the mixed precision scale.
    """
    sd = model.state_dict()
    #normalized_sd = sub_to_normal_bn(sd)

    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    # Write the checkpoint.
    checkpoint_dir = get_checkpoint_dir(cfg.OUTPUT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    path_to_checkpoint = get_path_to_checkpoint(checkpoint_dir, epoch)
    with open(path_to_checkpoint, "wb") as f:
        torch.save(checkpoint, f)
    return path_to_checkpoint


def load_checkpoint(
    path_to_checkpoint,
    model,
    optimizer=None,
    scaler=None,
    epoch_reset=False,
):
    """
    Load the checkpoint from the given file. 
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        scaler (GradScaler): GradScaler to load the mixed precision scale.
        inflation (bool): if True, inflate the weights from the checkpoint.
        epoch_reset (bool): if True, reset #train iterations from the checkpoint.
        clear_name_pattern (string): if given, this (sub)string will be cleared
            from a layer name if it can be matched.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    # Load the checkpoint on CPU to avoid GPU mem spike.
    with open(path_to_checkpoint, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    pre_train_dict = checkpoint["model_state"]
    model_dict = model.state_dict()
    # Match pre-trained weights that have same shape as current model.
    pre_train_dict_match = {
        k: v
        for k, v in pre_train_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    # Weights that do not have match from the pre-trained model.
    not_load_layers = [
        k
        for k in model_dict.keys()
        if k not in pre_train_dict_match.keys()
    ]
    # Log weights that are not loaded with the pre-trained weights.
    if not_load_layers:
        for k in not_load_layers:
            logger.info("Network weights {} not loaded.".format(k))
    # Load pre-trained weights.
    model.load_state_dict(pre_train_dict_match, strict=False)
    epoch = -1

    # Load the optimizer state (commonly not done when fine-tuning)
    if "epoch" in checkpoint.keys() and not epoch_reset:
        epoch = checkpoint["epoch"]
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler_state"])
    else:
        epoch = -1
    return epoch


def load_test_checkpoint(cfg, model):
    """
    Loading checkpoint logic for testing.
    """
    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in MODEL_VIS.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TEST.CHECKPOINT_FILE_PATH and test it.
        load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            optimizer=None,
            scaler=None,
            epoch_reset=False,
        )
    elif has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
        load_checkpoint(last_checkpoint, model)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
        )
    else:
        Exception(
            "Unknown way of loading checkpoint. Using with random initialization, only for debugging."
        )


def load_train_checkpoint(cfg, model, optimizer, scaler=None):
    """
    Loading checkpoint logic for training.
    """
    if cfg.TRAIN.AUTO_RESUME and has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)
        logger.info("Load from last checkpoint, {}.".format(last_checkpoint))
        checkpoint_epoch = load_checkpoint(
            last_checkpoint, model, optimizer, scaler=scaler
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            optimizer,
            scaler=scaler,
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    return start_epoch
