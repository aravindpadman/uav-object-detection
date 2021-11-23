"""
config file
"""

class Config:
    OUTPUT_DIR = "/workspace/data/outputs/exp-1"
    DATA_DIR = '/workspace/data/visdrone-object-detection-dataset'
    NUM_CLASSES = 12
    NUM_EPOCHS = 50
    SAVE_CHECKPOINT = True

    class TRAIN:
        BATCH_SIZE = 16
        NUM_WORKERS = 2
        PIN_MEMORY = True

    class TEST:
        BATCH_SIZE = 1
        CHECKPOINT_FILE_PATH = ""
        NUM_WORKERS = 4
        PIN_MEMORY = True

    class TENSORBOARD:
        VISUALIZE = True
        LOG_DIR = ""

cfg = Config()
