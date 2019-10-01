import sklearn
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk


path_to_nifit_file = "/home/eline/Downloads/Data_Eline/Data_Eline/Oxytarget_24 PRE/an_tumour.nii"

path_to_image_file = "/home/eline/Downloads/Data_Eline/Data_Eline/Oxytarget_24 PRE/T2/Image#11.dcm"

mask = sitk.ReadImage(path_to_nifit_file)
image = sitk.ReadImage(path_to_image_file)


image = sitk.GetArrayFromImage(image)
mask = sitk.GetArrayFromImage(mask)
plt.imshow(image[0])
plt.show()
plt.imshow(mask[10])
plt.show()



print(np.shape(image))
print(image[0])