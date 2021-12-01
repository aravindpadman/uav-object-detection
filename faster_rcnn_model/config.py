"""
config file
"""

class Config:
    OUTPUT_DIR = "/data/uav-object-detection/outputs/exp-1"
    DATA_DIR = "/data/uav-object-detection/data/"
    NUM_CLASSES = 12
    SAVE_CHECKPOINT = True

    class TRAIN:
        BATCH_SIZE = 8

    class TEST:
        BATCH_SIZE = 1
        CHECKPOINT_FILE_PATH = ""

    class TENSORBOARD:
        VISUALIZE = True
        LOG_DIR = ""

cfg = Config()
