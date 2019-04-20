import os, time
import torch
import numpy as np
import SimpleITK as sitk
import torch.optim as optim

import paths
from utils import get_logger
from signal_reader import read_signal
from main_calculator import MainCalculator
from config import parse_config

def datestr():
    now = time.localtime()
    return '{:04}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
print(datestr())


def main():

    logger = get_logger("Perfusion Parameters Calculation")
    config = parse_config()
    logger.info(config)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read and preprocess (brain-region extraction, low-pass filtering) raw signal (size: (slice, row, column, time))
    # For MRP, convert raw signal <= 0 to = 1
    RawPerfImg, origin, spacing, direction = \
        read_signal(paths.FileName, config.image_type, ToTensor = config.to_tensor, Mask = config.mask) 

    # Calculate perfusino parameters
    calculator = MainCalculator(RawPerfImg, origin, spacing, direction, config, paths.SaveFolder, device)
    calculator.run()


########################################################################################################################

if __name__ == '__main__':
    main()
