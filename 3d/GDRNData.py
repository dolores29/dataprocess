import os
import SimpleITK as sitk
import numpy as np
from util.ReadAndWrite import myDataload


def read_dcm_series(path_series_dcm):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(path_series_dcm)
    dcm_series = reader.GetGDCMSeriesFileNames(path_series_dcm, seriesIDs[0])
    reader.SetFileNames(dcm_series)
    sitk_img = reader.Execute()
    return sitk_img


def crop_sitkImg0(org_img, min_zyx, max_zyx):
    # 使用这个直接crop可以获得正确的origin,不用自己重新计算，但如果direction方向不对，就会出错
    crop_img = org_img[min_zyx[2]:max_zyx[2], min_zyx[1]:max_zyx[1], min_zyx[0]:max_zyx[0]]
    # size_xyz = np.array([min_zyx[2]+max_zyx[2], min_zyx[1]+max_zyx[1], min_zyx[0]+max_zyx[0]])
    # shape_crop = np.array([max_zyx[2]-min_zyx[2], max_zyx[1]-min_zyx[1], max_zyx[0]-min_zyx[0]])
    # center_i = 0.5*size_xyz*np.array(org_img.GetSpacing())+np.array(org_img.GetOrigin())
    # origin_crop = center_i-0.5*shape_crop*np.array(org_img.GetSpacing())
    return crop_img


# 排除direction不同而引起的问题
def crop_sitkImg(org_img, min_zyx, max_zyx):
    crop_img = org_img[min_zyx[2]:max_zyx[2], min_zyx[1]:max_zyx[1], min_zyx[0]:max_zyx[0]]
    origin_crop = np.array([min_zyx[2], min_zyx[1], min_zyx[0]]) * np.array(org_img.GetSpacing()) + np.array(org_img.GetOrigin())
    crop_img.SetOrigin(origin_crop)
    return crop_img


class GDRNDataProcess:
    def __init__(self, dcm_path, mask_path, std_path=None):
        self.dcm_path = dcm_path
        self.mask_path = mask_path
        self.save_path_root = os.path.dirname(dcm_path)
        if std_path is None:
            self.center_std = np.array([0, 0, 0])
        else:
            std_img = read_dcm_series(std_path)
            self.center_std = np.array([std_img.GetOrigin()[i] + std_img.GetSize()[i] * std_img.GetSpacing()[i] / 2.0
                                       for i in range(len(std_img.GetSize()))])
        self.org_img = read_dcm_series(dcm_path)
        self.org_msk = read_dcm_series(mask_path)
        self.org_img_arr = sitk.GetArrayFromImage(self.org_img)
        self.org_msk_arr = sitk.GetArrayFromImage(self.org_msk)
        print()

    def make_spine_data(self):
        positions_to_keep = np.logical_or(np.logical_or(
            self.org_msk_arr == 200, self.org_msk_arr == 400), self.org_msk_arr == 300)
        mask234 = np.where(positions_to_keep, self.org_msk_arr, 0)
        indices_234 = np.argwhere(mask234 != 0)
        min_zyx = np.min(indices_234, axis=0)
        max_zyx = np.max(indices_234, axis=0)

        map234_crop = crop_sitkImg(self.org_img, min_zyx, max_zyx)
        mask234_crop = crop_sitkImg(self.org_msk, min_zyx, max_zyx)
        mask234_crop_arr = sitk.GetArrayFromImage(mask234_crop)
        np.putmask(mask234_crop_arr, mask234_crop_arr < 200, 0)
        np.putmask(mask234_crop_arr, mask234_crop_arr > 400, 0)
        map234_crop_arr = sitk.GetArrayFromImage(map234_crop) * (mask234_crop_arr > 0)

        map_list = []
        mask_list = []
        center_std_dis = np.array([0, 0, 0])
        temp_y = 0
        # 生成单个椎骨的mask和map,需要crop
        for i in [200, 300, 400]:
            # map_i = np.where(np.logical_not(self.org_msk_arr != i), self.org_img_arr, 0)
            mask_i = np.where(np.logical_not(self.org_msk_arr != i), self.org_msk_arr, 0)
            indices = np.argwhere(mask_i == i)
            min_zyx = np.min(indices, axis=0)
            max_zyx = np.max(indices, axis=0)

            map_crop_i = crop_sitkImg(self.org_img, min_zyx, max_zyx)
            mask_crop_i = crop_sitkImg(self.org_msk, min_zyx, max_zyx)

            mask_crop_arr_i = sitk.GetArrayFromImage(mask_crop_i)
            np.putmask(mask_crop_arr_i, mask_crop_arr_i != i, 0)
            map_crop_arr_i = sitk.GetArrayFromImage(map_crop_i) * (mask_crop_arr_i > 0)

            # 计算当前椎骨(300)与标准椎骨(300)的平移量,统一平移
            if i == 300:
                current_center_300 = [mask_crop_i.GetOrigin()[i] + mask_crop_i.GetSize()[i] * mask_crop_i.GetSpacing()[i] / 2.0
                                      for i in range(len(mask_crop_i.GetSize()))]
                center_std_dis = np.array(current_center_300) - self.center_std

            mask_crop_i_new = sitk.GetImageFromArray(mask_crop_arr_i)
            mask_crop_i_new.SetOrigin(mask_crop_i.GetOrigin())
            mask_crop_i_new.SetSpacing(mask_crop_i.GetSpacing())
            mask_crop_i_new.SetDirection(mask_crop_i.GetDirection())

            map_crop_i_new = sitk.GetImageFromArray(map_crop_arr_i)
            map_crop_i_new.SetOrigin(map_crop_i.GetOrigin())
            map_crop_i_new.SetSpacing(map_crop_i.GetSpacing())
            map_crop_i_new.SetDirection(map_crop_i.GetDirection())

            mask_list.append(mask_crop_i_new)
            map_list.append(map_crop_i_new)

        self.save_new_data(map234_crop_arr, mask234_crop, center_std_dis, 'map234_crop_zero')
        self.save_new_data(mask234_crop_arr, mask234_crop, center_std_dis, 'mask234_crop_zero')

        for i, (map_, mask) in enumerate(zip(map_list, mask_list)):
            self.save_new_data(sitk.GetArrayFromImage(map_), map_, center_std_dis, 'map_zero_'+str(i+2))
            self.save_new_data(sitk.GetArrayFromImage(mask), mask, center_std_dis, 'mask_zero_'+str(i+2))

    def make_skull_data(self):
        indices = np.argwhere(self.org_msk_arr > 0)
        min_zyx = np.min(indices, axis=0)
        max_zyx = np.max(indices, axis=0)
        img_crop = crop_sitkImg(self.org_img, min_zyx, max_zyx)
        msk_crop = crop_sitkImg(self.org_msk, min_zyx, max_zyx)
        mask_crop_arr = sitk.GetArrayFromImage(msk_crop)
        map_crop_arr = np.where(mask_crop_arr > 0, sitk.GetArrayFromImage(img_crop), 0)

        current_center = [msk_crop.GetOrigin()[i] + msk_crop.GetSize()[i] * msk_crop.GetSpacing()[i] / 2.0
                          for i in range(len(msk_crop.GetSize()))]

        center_std_dis = np.array(current_center) - self.center_std
        self.save_new_data(map_crop_arr, msk_crop, center_std_dis, 'map_skull_crop')
        self.save_new_data(mask_crop_arr, msk_crop, center_std_dis, 'mask_skull_crop')

    def save_new_data(self, crop_arr, sitk_crop, center_std_dis, save_name):
        save_path = os.path.join(self.save_path_root, 'trans_dcm')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        myDl = myDataload()
        dicts = myDl.dicts.copy()
        crop_arr[crop_arr < 0] = 0
        dicts['Hu'] = crop_arr
        dicts['Spacing'] = sitk_crop.GetSpacing()
        dicts['Origin'] = sitk_crop.GetOrigin() - np.array(center_std_dis)
        myDl.writeDicom(os.path.join(save_path, save_name), dicts)


if __name__ == '__main__':
    dcm_path = r'F:\PosEstimation\my_data\dataLumbarPig\dicomData\map234'
    msk_path = r'F:\PosEstimation\my_data\dataLumbarPig\dicomData\mask234'
    std_path = r'F:\PosEstimation\my_data\dcmData\std_dcm\trans_crop\map3_trans_crop'
    gdrnData = GDRNDataProcess(dcm_path, msk_path)
    # gdrnData.make_skull_data()
    gdrnData.make_spine_data()









