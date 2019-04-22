import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from dipy.core.onetime import auto_attr
# %matplotlib inline  # remove annotation symbol when works in Jupyter Notebook

'''
Transform DICOM files to NifTi files if needed
'''

# Scan through a directory
class ScanFile(object):   
    '''
    Scan dir and record all files of specific type (pre-fixed prefic, postfix)
    '''

    def __init__(self, directory, prefix = None, postfix = None):  
        self.directory = directory  
        self.prefix = prefix  
        self.postfix = postfix  
          
    def scan_files(self):    
        files_list=[]    
        for dirpath,dirnames,filenames in os.walk(self.directory):   
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''  
            for special_file in filenames:    
                if self.postfix:  
                    if  special_file.endswith(self.postfix):    
                        files_list.append(os.path.join(dirpath,special_file))    
                elif self.prefix:
                    if special_file.startswith(self.prefix):  
                        files_list.append(os.path.join(dirpath,special_file))    
                else:    
                    files_list.append(os.path.join(dirpath,special_file))                     
        return files_list    
    
    def direct_sub(self): 
        # Get direct sub-directories of origin directory
        direct_sub = [name for name in os.listdir(self.directory) 
                      if os.path.isdir(os.path.join(self.directory, name))]
        return direct_sub
              
    def scan_subdir(self):
        # Get all sub-directories
        subdir_list=[]  
        for dirpath,dirnames,files in os.walk(self.directory):  
            subdir_list.append(dirpath)  
        return subdir_list


class LoadImage(object):
    '''
    Load Image(s) via SimpleITK, images could be .nii, .nii,gz, .mha, .dcm, ...
    '''
    def __init__(self, filename, isDICOM = False):
        '''
        filename: /path/to/file(s)
        '''
        self.filename = filename
        self.isDICOM  = isDICOM
    
    @auto_attr
    def read_image(self):
        '''
        Read in images
        '''
        if self.isDICOM:
            print("Reading Dicom directory:", os.path.basename(self.filename))
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(self.filename)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        else:
            print('File name: %s\n' % os.path.basename(self.filename))
            image = sitk.ReadImage(self.filename)
        array = sitk.GetArrayFromImage(image) 
        return image, array
    
    def show_info(self):
        '''
        Show information of an image, with (image, array)
        PatientsList = [os.path.join()]y) as returns
        '''
        image, array = self.read_image
        image_size = image.GetSize()
        # Image Info
        print('Image Info：')
        print('  Size:       ', image_size, 'Width/Height/Depth:', image.GetWidth(), image.GetHeight(), image.GetDepth())
        print('  Pixel Type: ', image.GetPixelIDTypeAsString())
        print('  Origin:     ', image.GetOrigin())
        print('  Spacing:    ', image.GetSpacing())
        print('  Direction:  ', image.GetDirection())
        print('  #_Component:', image.GetNumberOfComponentsPerPixel())
    
        # Corresponding Array Info
        print('Array Info：')
        print('  shape: ', array.shape)
    
        # May add some other functions for more information of the input image
        # ...
        
        return image, array
    
    def show_slices(self, n_slices = 528, RGBcmap = 'jet'):  # Works in Jupyter Notebook
        '''
        Display an 3D Gray/RGB image in n_slices 2D slices
        
        n_slices: a list of slices to be viewed
        RGBcmap: RGB image color map type (Default: 'jet')
        '''
        if not n_slices > 0:
            raise NotImplementedError('Input # of slices (%d) should be positive' % n_slices)
        
        img, array = self.read_image
            
        cmap = 'Greys'
        imgtype = 'Gray Image'
        print(array.shape)
        length = array.shape[0]
        step = int(length/n_slices)
        
        RGB = img.GetNumberOfComponentsPerPixel() == 3
        if RGB: # If should be displayed as a RGB image
            array = array[...,0]
            cmap = RGBcmap
            imgtype = 'RGB Image'
    
        for z in [step*i for i in range(n_slices)]:
            if img.GetNumberOfComponentsPerPixel() > 1:
                plt.imshow(array[z,...,0], cmap = cmap)
            else:
                plt.imshow(array[z,:,:], cmap = cmap)
            title = '%s - Slice #%d' % (imgtype, z)
            plt.title(title)
            plt.axis('off')
            plt.show()
        return
    
    def save_array(self, save_dir):
        '''
        Save image as numpy array
        save_dir: directory for saving .npy files
        '''
        image, array = self.read_image
        basename = os.path.basename(self.filename)
        print('File name: %s' % basename)
        print('Save as:   %s.npy\n' % basename)
        np.save(os.path.join(save_dir, "%s" % basename), array)
        return image, array

    def dcm2nii(self, save_dir):
        '''
        Save image as NifTi file
        save_dir: directory for saving .nii files
        '''
        image, array = self.read_image
        basename = os.path.basename(self.filename)
        sitk.WriteImage(image, os.path.join(save_dir, '%s.nii' % basename))
        return image, array




################################################################################

'''
Example of usage: transform one DICOM series file to NifTi format
'''

if __name__ == '__main__':
    
    subj_dir = '/home/peirong/Documents/Stroke/StrokeData/KitwareData/2019-StrokeCollaterals/2019-ForUse/CTAT-002 (CT-CTA-CTP)'
    dcm_series_dir = os.path.join(subj_dir, '1.3.12.2.1107.5.1.4.64384.30000018041610115947200014165') # directory of DICOM series
    loader = LoadImage(dcm_series_dir, isDICOM = True) # Initialize Class LoadImage
    img, nda = loader.show_info() # Show some neede DICOM meta data
    loader.dcm2nii(subj_dir) # Save as NifTi file, with basename same as "DICOM Series", under the directory of "subj_dir"