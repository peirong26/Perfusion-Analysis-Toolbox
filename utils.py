import os
import sys
import torch
import shutil
import logging
import numpy as np


def get_logger(name, level = logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_n_learnable_parameters(model):

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    return sum([np.prod(p.size()) for p in model_parameters])


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best validation accuracy so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def cutoff_percentile(image, mask = None, percentile_lower = 0.2, percentile_upper = 99.8):
	
	if mask is None:
		mask = image != image[0, 0, 0]
	cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
	cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
	print('Clip within [%.3f, %.3f]' % (cut_off_lower, cut_off_upper))

	res = np.copy(image)
	res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
	res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper

	return res


def load_checkpoint(checkpoint_path, model, optimizer = None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied
    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    state = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(state['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return state
