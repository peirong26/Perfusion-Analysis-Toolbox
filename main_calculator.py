import os
import torch
import logging
import numpy as np
import SimpleITK as sitk
from tensorboardX import SummaryWriter

import utils
import ParamsCalculator.ctc as ctc
import ParamsCalculator.mask as mask
import ParamsCalculator.aif as aif


class MainCalculator:
    """Main calculator.
    Args:
    raw_perf: numpy array or torch tensor of size (slice, row, column, time)
    save_path: path/to/save/folder
    device: device currently working on
    logger: info logger
    """
    def __init__(self, raw_perf, origin, spacing, direction, config, save_path, device, logger = None):
        if logger is None:
            self.logger = utils.get_logger('MainCalculator', level = logging.DEBUG)
        else:
            self.logger = logger

        self.logger.info(f"Sending the raw perfusion image to '{device}'")
        self.raw_perf = raw_perf.to(device)
        
        self.config    = config
        self.sitkinfo  = [origin, spacing, direction, save_path]
        self.device    = device
        self.nS        = self.raw_perf.size(0)
        self.nR        = self.raw_perf.size(1)
        self.nC        = self.raw_perf.size(2)
        self.nT        = self.raw_perf.size(3)
        self.size      = [self.nS, self.nR, self.nC, self.nT]


    def run(self):
        self.main_cal()


    def main_cal(self):

        # Implement when need to exclude the scalp and zones from the image adjacent to the outside of the brain
        #Mask = mask.cal(self.raw_perf, self.device) 

        # Compute and save absolute CTC
        CTC = ctc.cal(self.raw_perf, self.sitkinfo, self.config, self.device) # dtype = torch.float

        # Clustering: obtain AIF, exclude out arteries
        #AIF = aif.cal(CTC, self.config)

         