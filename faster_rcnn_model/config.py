"""
config file
"""

class Config:
    OUTPUT_DIR = ""
    DATA_DIR = ""
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
