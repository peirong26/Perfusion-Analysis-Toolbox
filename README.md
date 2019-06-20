# Perfusion-Analysis-Toolbox (Pytorch Version)
Compute various perfusion parameters given a 4D perfusion image. 

(Current progress: Extracting AIF...)

## 1. Functions
a) Start from main.py, set correct parameters in config.py, set correct file paths in paths.py;

b) main_calculator.py: aggregate the main parameters calculators in ./ParamsCalculator;

## 2. Usage 
Prepare perfusion image, e.g., save in image.nii, with sitk.size() == (Width, Height, Depth), and with "TotalTimePoints" components per voxel

```
cd path/to/this/folder
python main.py
```
