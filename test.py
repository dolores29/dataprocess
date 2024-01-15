from util.ReadAndWrite import myDataload
import SimpleITK as sitk
import numpy as np


dcm_path = r'F:\PosEstimation\my_data\dataLumbarPig\dicomData\mask234_new'
mask_path = r'F:\PosEstimation\my_data\dataLumbarPig\dicomData\mask234'


myDl = myDataload()
org_img = myDl.read_dcm_series(dcm_path)
org_arr = sitk.GetArrayFromImage(org_img)

msk_img = myDl.read_dcm_series(mask_path)
msk_arr = sitk.GetArrayFromImage(msk_img)

mask_i = np.where(np.logical_not(msk_arr != 400), msk_arr, 0)
indices = np.argwhere(mask_i == 400)
min_zyx = np.min(indices, axis=0)
max_zyx = np.max(indices, axis=0)

print(np.unique(msk_arr))
print(np.unique(org_arr))
print(org_img.GetOrigin())
print(msk_img.GetOrigin())
