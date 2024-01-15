import os

import SimpleITK as sitk
import numpy as np
from util.ReadAndWrite import myDataload
from util.tools_3d import isotropic_resampler
'''
载入img3d 和 label
根据标签中每个椎骨的中心坐标，向三个方向随机偏移
设定固定尺寸64, 128, 128
裁剪得到样本
'''


class MakeDatasetSeg3D:

    def __init__(self, label_shape=None):
        if label_shape is None:
            label_shape = [64, 128, 128]
        self.img3d_path = r"E:\workdata\spine\new_spine\new_spine_samp\originData"
        self.mask_path = r"E:\workdata\spine\new_spine\new_spine_samp\labelData"
        self.output_path = r"E:\workdata\spine\new_spine\new_spine_crop1"
        self.DMP = myDataload()
        self.label_shape = label_shape

    def get_img3d_names(self):
        case_name_list = os.listdir(self.img3d_path)
        list_names = [f for f in case_name_list if '.mhd' in f and 'label' not in f or 'msk' not in f]

        return list_names

    # 得到mask中 腰椎的标签 ：[100, 200, 300, 400, 500]
    def get_lumbar_label_index_pass(self, mask):
        index_all = np.unique(mask)
        number_list = []
        for i, index_ in enumerate(index_all[1:]):
            number_i = len(np.where(mask == index_)[0])
            number_list.append(number_i)
        number_arr = np.array(number_list)
        index_2 = np.argsort(number_arr)  # 从小到大排序
        index_2 = index_2[::-1]  # 逆序
        # 至少7个骨头
        if len(index_2) > 6:
            index_top7 = index_2[0:7]
            index_7 = np.argsort(index_top7)
            sort_7 = index_2[index_7]
            lumbar_index = index_all[sort_7[1:6] + 1]
        # 6个骨头
        elif len(index_2) == 6:
            if index_all.max() > 600:
                # 还得判断是否有骶骨
                index_top6 = index_2[0:6]
                index_6 = np.argsort(index_top6)
                sort_6 = index_2[index_6]
                lumbar_index = index_all[sort_6[0:5] + 1]
            else:
                # 自制标签
                lumbar_index = index_all[1:5]

    def get_box3d_128(self, mask_shape, box3d_1):
        label_shape = self.label_shape
        x1, y1, z1, x2, y2, z2 = box3d_1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_z = (z1 + z2) / 2
        start_x = int(center_x - 0.5 * label_shape[0])
        if start_x < 0:
            start_x = 0
        end_x = start_x + label_shape[0]
        if end_x > mask_shape[0]:
            end_x = mask_shape[0]
            start_x = end_x - label_shape[0]

        start_y = int(center_y - 0.5 * label_shape[1])
        if start_y < 0:
            start_y = 0
        end_y = start_y + label_shape[1]
        if end_y > mask_shape[1]:
            end_y = mask_shape[1]
            start_y = end_y - label_shape[1]

        start_z = int(center_z - 0.5 * label_shape[2])
        if start_z < 0:
            start_z = 0
        end_z = start_z + label_shape[2]
        if end_z > mask_shape[2]:
            end_z = mask_shape[2]
            start_z = end_z - label_shape[2]

        box3d_128 = [start_x, start_y, start_z, end_x, end_y, end_z]

        return box3d_128

    def get_start_end_list(self, m1, m2, mask_shape_j, label_shape_j, x1_128, x2_128, extent_=1):
        d = m2 - m1
        step = (label_shape_j - d) / 2

        start_1 = int(x1_128 - 0.0 * step)
        # start_2 = int(x1_128 - 0.7 * step)
        # start_3 = int(x1_128 + 0.7 * step)

        end_1 = start_1 + label_shape_j
        # end_2 = start_2 + label_shape_j
        # end_3 = start_3 + label_shape_j
        start_end_list = []

        # 如果超出范围, 则说明该椎骨比较靠进原数据边缘
        if 0 <= start_1 and end_1 <= mask_shape_j:
            start_end_list.append(np.array([start_1, end_1]))
        elif 0 > start_1 and end_1 <= mask_shape_j:
            start_end_list.append(np.array([0, end_1]))
            print('start_1 < 0')
        # if 0 < start_2 and end_2 < mask_shape_j:
        #     start_end_list.append(np.array([start_2, end_2]))
        # if 0 < start_3 and end_3 < mask_shape_j:
        #     start_end_list.append(np.array([start_3, end_3]))

        while len(start_end_list) < extent_:
            lambda_ = 2 * (np.random.rand() - 0.5)
            start_i = int(x1_128 + lambda_ * step)
            end_i = start_i + label_shape_j
            if 0 < start_i and end_i <= mask_shape_j:
                start_end_list.append(np.array([start_i, end_i]))
        # print("len(start_end_list):", len(start_end_list))

        return start_end_list

    # 获得每个腰椎的中心点
    def get_box3d_list(self, mask):
        label_list = [100, 200, 300, 400, 500]
        mask_shape = mask.shape
        label_shape = self.label_shape

        box3d_all_list = []
        for i, label_i in enumerate(label_list):
            # S = np.zeros_like(mask)
            # S[mask == label_i] = 1
            # center = ndimage.measurements.center_of_mass(S)
            # center = [int(center[0]), int(center[1]), int(center[2])]
            index_xyz = np.where(mask == label_i)
            if len(index_xyz[0]) == 0:
                continue
            x1 = index_xyz[0].min()
            x2 = index_xyz[0].max()
            y1 = index_xyz[1].min()
            y2 = index_xyz[1].max()
            z1 = index_xyz[2].min()
            z2 = index_xyz[2].max()

            box3d_1 = [x1, y1, z1, x2, y2, z2]  # zyx
            box3d_128 = self.get_box3d_128(mask_shape, box3d_1)
            if len(box3d_128) == 0:
                print("尺寸不足128!")
                return []
            start_x = []
            start_y = []
            start_z = []
            end_x = []
            end_y = []
            end_z = []

            for j in range(0, 3):
                m1, m2 = box3d_1[j], box3d_1[j + 3]
                r = self.get_start_end_list(m1, m2, mask_shape[j], label_shape[j], box3d_128[j], box3d_128[j + 3])
                # r = r_list[j]
                if len(r) == 0:
                    print('')
                    continue
                for k, start_end in enumerate(r):
                    if j == 0:
                        start_x.append(start_end[0])
                        end_x.append(start_end[1])
                    if j == 1:
                        start_y.append(start_end[0])
                        end_y.append(start_end[1])
                    if j == 2:
                        start_z.append(start_end[0])
                        end_z.append(start_end[1])
            label_i_box3d = []
            for ii, temp_start_x in enumerate(start_x):
                for jj, temp_start_y in enumerate(start_y):
                    for kk, temp_start_z in enumerate(start_z):
                        temp_start_box3d = [temp_start_x, temp_start_y, temp_start_z]
                        # temp_end_box3d = np.array(temp_start_box3d) + np.array(label_shape)

                        temp_end_box3d = np.array([end_x[ii], end_y[jj], end_z[kk]])
                        temp_start_box3d.extend(temp_end_box3d)
                        label_i_box3d.append(temp_start_box3d)

            box3d_all_list.append(label_i_box3d)

        return box3d_all_list

    def crop_images(self, images, box):
        min_val = images.min()
        ww, hh, dd = self.label_shape
        x1, y1, z1, x2, y2, z2 = box
        w, h, d = x2 - x1, y2 - y1, z2 - z1  # box 的长宽高
        crop_img = min_val * np.ones((ww, hh, dd))

        temp_1 = images[max(0, round(min(x1, x2))):round(max(x1, x2)), :, :]  # 深度 2 5
        temp_2 = temp_1[:, max(0, round(min(y1, y2))):round(max(y1, y2)), :]  # 宽度 0 3
        image_ = temp_2[:, :, max(0, round(min(z1, z2))):round(max(z1, z2))]  # 高度 1 4

        start_x = int((ww - w) / 2)
        start_y = int((hh - h) / 2)
        start_z = int((dd - d) / 2)

        crop_img[start_x:start_x + w, start_y:start_y + h, start_z:start_z + d] = image_

        return crop_img

    def make_(self, dataset_save_path_, flip_flag=False):
        # label_list = [100, 200, 300, 400, 500]
        img3d_save_path = os.path.join(dataset_save_path_, 'CropImg3d')
        mask_save_path = os.path.join(dataset_save_path_, 'CropMask')
        if not os.path.exists(img3d_save_path):
            os.mkdir(img3d_save_path)
        if not os.path.exists(mask_save_path):
            os.mkdir(mask_save_path)
        crop_img3d_dicts = self.DMP.dicts.copy()
        crop_label_dicts = self.DMP.dicts.copy()
        list_names = self.get_img3d_names()
        for i, single_name in enumerate(list_names):
            if single_name.split('.')[-1] == "raw":
                continue
            single_img3d_path = os.path.join(self.img3d_path, single_name)
            single_mask_name = single_name.split('.')[0] + '_label.mhd'
            single_mask_path = os.path.join(self.mask_path, single_mask_name)

            if not os.path.exists(single_mask_path):
                single_mask_name = single_name.split('.')[0] + '_msk.mhd'
                single_mask_path = os.path.join(self.mask_path, single_mask_name)
                if not os.path.exists(single_mask_path):
                    print(single_mask_path, " 文件不存在!")
                    continue

            single_img3d = sitk.ReadImage(single_img3d_path)
            single_img3d = isotropic_resampler(single_img3d, is_label=False)
            single_mask = sitk.ReadImage(single_mask_path)
            single_mask = isotropic_resampler(single_mask, is_label=True)

            single_img3d_arr = sitk.GetArrayFromImage(single_img3d)
            single_mask_arr = sitk.GetArrayFromImage(single_mask)
            if single_img3d_arr.shape == single_mask_arr.shape:
                print('', i)
            else:
                print("shape 不一致:", single_img3d_path)
            # assert single_img3d_arr.shape == single_mask_arr.shape
            label_list = np.unique(single_mask_arr)
            label_list[label_list > 500] = 0
            label_arr = np.array(label_list)
            label_arr = label_arr[label_arr != 0]

            box3d_all_list = self.get_box3d_list(single_mask_arr)
            if len(box3d_all_list) == 0:
                print('len(box3d_all_list) == 0', single_mask_path)
            for j, spine_j_box_list in enumerate(box3d_all_list):
                spine_base_name = single_name.split('.')[0] + "_L" + str(j).zfill(2)

                for k, single_box3d in enumerate(spine_j_box_list):
                    crop_img3d_name = spine_base_name + "_" + str(k).zfill(2) + '.mhd'
                    crop_label_name = spine_base_name + "_" + str(k).zfill(2) + '_label.mhd'
                    crop_img3d = self.crop_images(single_img3d_arr, single_box3d)
                    crop_label = self.crop_images(single_mask_arr, single_box3d)
                    origin = [float(single_box3d[2]), single_box3d[1], single_box3d[0]]
                    middle_val = np.zeros_like(crop_label)

                    # middle_val[crop_label > 0] = 50
                    # middle_val[crop_label == label_list[j]] = label_list[j]
                    middle_val[crop_label == label_arr[j]] = 2
                    if j >= 1:
                        middle_val[crop_label == label_arr[j - 1]] = 1
                        if j >= 2:
                            middle_val[crop_label == label_arr[j - 2]] = 1
                    if j < len(label_arr) - 1:
                        middle_val[crop_label == label_arr[j + 1]] = 3
                        if j < len(label_arr) - 2:
                            middle_val[crop_label == label_arr[j + 2]] = 3

                    crop_img3d_dicts["Hu"] = crop_img3d
                    crop_label_dicts["Hu"] = middle_val
                    crop_img3d_dicts["Spacing"] = single_img3d.GetSpacing()
                    crop_label_dicts["Spacing"] = single_mask.GetSpacing()
                    crop_img3d_dicts["Origin"] = origin
                    crop_label_dicts["Origin"] = origin
                    self.DMP.writeMhd(img3d_save_path, crop_img3d_dicts, crop_img3d_name)
                    self.DMP.writeMhd(mask_save_path, crop_label_dicts, crop_label_name)
                    if flip_flag:
                        crop_img3d_name = spine_base_name + "_" + str(k).zfill(2) + '_flip.mhd'
                        crop_label_name = spine_base_name + "_" + str(k).zfill(2) + '_flip_label.mhd'
                        crop_img3d_dicts["Hu"] = crop_img3d[::-1, :, :]
                        crop_label_dicts["Hu"] = middle_val[::-1, :, :]
                        self.DMP.writeMhd(img3d_save_path, crop_img3d_dicts, crop_img3d_name)
                        self.DMP.writeMhd(mask_save_path, crop_label_dicts, crop_label_name)
            print('')


if __name__ == '__main__':
    MDS3D = MakeDatasetSeg3D()
    dataset_save_path = MDS3D.output_path
    MDS3D.make_(dataset_save_path, flip_flag=False)
    print(0)
