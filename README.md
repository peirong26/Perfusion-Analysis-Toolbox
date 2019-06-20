# Perfusion-Analysis-Toolbox (Matlab Version)
Compute various perfusion parameters given a 4D perfusion image. 


## 1. Functions
Compute various perfusion parameters given a 4D perfusion image, and save them in .nii.

## 2. Usage 
a) Prepare perfusion image: saved in .nii format, with sitk.Size() == (Width, Height, Depth), along with "#ofTotalTimePoints" components per voxel

b) Set correct parameters in utils/DSC_mri_getOptions.m

c) Set path/to/image in DSC_main_demo.m

d) Run DSC_main_demo.m


## References:
https://github.com/marcocastellaro/dsc-mri-toolbox
