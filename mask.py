import os
import torch
import SimpleITK as sitk


def brain_region(RawPerf, device, background = 0):
	
	mask = torch.zeros(RawPerf.size(), device = device)
	mask = torch.where(RawPerf != background, RawPerf, mask)

	return mask

