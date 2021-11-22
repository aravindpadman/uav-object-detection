"""config file"""

class Config:
    OUTPUT_DIR = "/home/welcome/uav-object-detection/outputs/exp-2"
    DATA_DIR = '/home/welcome/uav-object-detection/data'
    NUM_CLASSES = 12
    SAVE_CHECKPOINT = False

    class TRAIN:
        BATCH_SIZE = 8

    class TEST:
        BATCH_SIZE = 1
        CHECKPOINT_FILE_PATH = ""
        #CHECKPOINT_FILE_PATH = "/home/welcome/uav-object-detection/outputs/exp-1/checkpoints/checkpoint_epoch_00003.pyth"

    class TENSORBOARD:
        VISUALIZE = False
        LOG_DIR = ""

cfg = Config()
