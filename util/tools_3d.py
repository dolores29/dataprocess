import os
import itk
import SimpleITK as sitk
import numpy as np
import pydicom
import torch

from util.ReadAndWrite import myDataload
from util.common import itk2sitk, crop_sitkImg
from skimage.measure import label
myDl = myDataload()


def get_dcm_list(path_):
    name_list = os.listdir(path_)
    path_list = []
    for i, name in enumerate(name_list):
        temp_path = os.path.join(path_, name)
        if os.path.isdir(temp_path):
            if len(os.listdir(temp_path)) < 100:
                continue
            path_list.append(temp_path)

    return path_list


def knee_dcm2mhd():
    for index_ in range(1):
        dcm_ori_path = r'E:\workdata\knee\image\ky00002166'
        path_mask = r'E:\workdata\knee\mask\ky00002166_mask'
        save_path = r'F:\KneeCropDataset\ky002166'

        # origin_list = get_dcm_list(dcm_ori_path)
        mask_list = get_dcm_list(path_mask)
        for mask in mask_list:
            # ori_name = origin_path.split('\\')[-1]
            # mas_name = mask.split('\\')[-1].split('_')[0]
            origin_path = os.path.join(dcm_ori_path, mask.split('\\')[-1].split('_')[0])
            origin_img = myDl.read_dcm_series(origin_path)
            origin_img_sam = isotropic_resampler(origin_img)

            mask_img = myDl.read_dcm_series(mask)
            mask_img_sam = isotropic_resampler(mask_img, is_label=True)
            print(origin_path, mask)
            print(origin_img_sam.GetOrigin(), mask_img_sam.GetOrigin())
            print(origin_img_sam.GetSize(), mask_img_sam.GetSize())
            print("_________________________")
            path_name = os.path.basename(origin_path)
            save_origin_name = os.path.join(save_path, path_name)
            save_mask_name = os.path.join(save_path, path_name + '_label')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            sitk.WriteImage(origin_img_sam, save_origin_name + '.mhd')
            sitk.WriteImage(mask_img_sam, save_mask_name + '.mhd')


def make_knee_txt(data_path, txt_path):
    data_dirs = os.listdir(data_path)
    for data_dir in data_dirs:
        txt_path_ = os.path.join(txt_path, data_dir + '.txt')
        if os.path.exists(txt_path_):
            os.remove(txt_path_)
        data_dir_path = os.path.join(data_path, data_dir)
        for mhd in os.listdir(data_dir_path):
            if '.mhd' in mhd and 'label.mhd' not in mhd:
                origin_path = os.path.join('./Data', data_dir, mhd)
                mask_path = os.path.join('./Data', data_dir, mhd.split('.')[0] + '_label.mhd')

                with open(txt_path_, 'a') as f:
                    str_ = origin_path + ';' + mask_path + '\n'
                    print(str_)
                    f.write(str_)
    f.close()


def make_map(img_dicm, msk_dicm, map_dcm):
    '''
    根据msk和原始数据生成map
    '''
    dicts = myDl.read_dcm_of_sitk(img_dicm)
    img_array = dicts["Hu"]

    dicts_msk = myDl.read_dcm_of_sitk(msk_dicm)
    msk_array = dicts_msk["Hu"]
    msk = (msk_array > 0) + 0.0
    result_array = img_array * msk
    dicts["Hu"] = result_array
    myDl.writeDicom(map_dcm, dicts)


def merge_msks(dir_path, mhd_path, dcm_path):
    '''
        合并多个msk文件
    :param dir_path: 含有多个文件夹
    :param mhd_path: 保存为mhd形式的路径
    :param dcm_path: 保存为dcm形式的路径
    :return:
    '''
    dir_list_new = os.listdir(dir_path)
    # dir_list_new = [f for f in dir_list if 'crop' not in f]
    dir0 = myDl.read_dcm_series(os.path.join(dir_path, dir_list_new[0]))
    dir0_array = sitk.GetArrayFromImage(dir0)
    dir0_array[dir0_array < 0] = 0
    length = len(dir_list_new)
    for index in range(1, length):
        temp_msk = myDl.read_dcm_series(os.path.join(dir_path, dir_list_new[index]))
        temp_array = sitk.GetArrayFromImage(temp_msk)
        temp_array[temp_array < 0] = 0
        dir0_array = dir0_array + temp_array
    # dir0_array[dir0_array == 0] = -1024
    # 保存为mhd
    # save_name = 'crop_' + dir_path.split('\\')[-2] + '_label.mhd'
    # save_name = dir_path.split('\\')[-1] + '_label.mhd'
    # msk = sitk.GetImageFromArray(dir0_array)
    # msk.SetSpacing(dir0.GetSpacing())
    # msk.SetOrigin(dir0.GetOrigin())
    # msk.SetDirection(dir0.GetDirection())
    # msk_sam = isotropic_resampler(msk, is_label=True)
    #
    # sitk.WriteImage(msk_sam, os.path.join(mhd_path, save_name))

    # 保存为dicom
    new_dicts = myDl.dicts.copy()
    new_dicts['Hu'] = dir0_array
    new_dicts['Spacing'] = dir0.GetSpacing()
    new_dicts['Origin'] = dir0.GetOrigin()
    myDl.writeDicom(dcm_path, new_dicts)


def merge_nrrds(dir_path, save_path):
    nrrd_list = os.listdir(dir_path)
    org_img = sitk.ReadImage(os.path.join(dir_path, nrrd_list[0]))
    merge_arr = np.zeros_like(sitk.GetArrayFromImage(org_img))
    origin = org_img.GetOrigin()
    # 排序
    seg_imgs = []
    for nrrd_file in os.listdir(dir_path):
        if '_seg_' in nrrd_file:
            temp_img = sitk.ReadImage(os.path.join(dir_path, nrrd_file))
            seg_imgs.append(temp_img)
    seg_imgs_sort = sorted(seg_imgs, key=sort_by_origin_z)  # 大到小

    for i, temp_img in enumerate(seg_imgs_sort):
        merge_arr_copy = np.zeros_like(sitk.GetArrayFromImage(org_img))
        # temp_img = sitk.ReadImage(os.path.join(dir_path, nrrd_file))
        temp_array = sitk.GetArrayFromImage(temp_img) * (i+1)*100 # zyx
        nx, ny, nz = temp_img.GetSize()
        temp_origin = temp_img.GetOrigin()
        dx, dy, dz = abs(np.array(temp_origin) - np.array(origin)).astype(int)
        merge_arr_copy[dz:dz + nz, dy:dy + ny, dx:dx + nx] = temp_array
        merge_arr += merge_arr_copy
    merge_img = sitk.GetImageFromArray(merge_arr)
    merge_img.SetOrigin(origin)
    merge_img.SetSpacing(org_img.GetSpacing())
    sitk.WriteImage(merge_img, r'E:\workdata\temp\nrrd\testnrrd.nrrd')


def sort_by_origin_z(img):
    return -img.GetOrigin()[2]


def isotropic_resampler(img_mhd, new_spacing=None, is_label=False):
    if new_spacing is None:
        new_spacing = [1, 1, 1]
    resampler = sitk.ResampleImageFilter()
    if is_label:  # 如果是mask图像，就选择sitkNearestNeighbor这种插值
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:  # 如果是普通图像，就采用sitkBSpline插值法
        resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputDirection(img_mhd.GetDirection())
    resampler.SetOutputOrigin(img_mhd.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)
    orig_size = np.array(img_mhd.GetSize(), dtype=np.int)
    orig_spacing = img_mhd.GetSpacing()
    new_size = np.array([x * (y / z) for x, y, z in zip(orig_size, orig_spacing, new_spacing)])
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]

    img_arr = sitk.GetArrayFromImage(img_mhd)
    # resampler.SetDefaultPixelValue(img_arr.min())
    resampler.SetSize(new_size)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    isotropic_img = resampler.Execute(img_mhd)
    return isotropic_img


def crop_data(path_origin, path_label, new_image_path, new_label_path):
    new_var = [100, 200, 300, 400, 500]
    for org_name in os.listdir(path_origin):
        if '.mhd' in org_name and 'case23' in org_name:
            org_path = os.path.join(path_origin, org_name)
            img_itk = itk.imread(org_path)
            img_sitk = itk2sitk(img_itk)
            img_array = sitk.GetArrayFromImage(img_sitk)
            if 'case' in org_name or 'Case' in org_name:
                label_mhd_name = org_name.split('.')[0] + '_label.mhd'
            elif 'image' in org_name:
                label_mhd_name = "mask" + org_name.split('.')[0][-3:] + '.mhd'
            else:
                label_mhd_name = org_name.split('.')[0] + '_msk.mhd'
            label_path = os.path.join(path_label, label_mhd_name)
            label_itk = itk.imread(label_path)
            label_sitk = itk2sitk(label_itk)
            label_array = sitk.GetArrayFromImage(label_sitk)
            index_list = np.unique(label_array)
            print(org_name, index_list)
            index_min = np.where(label_array == index_list[-5])
            index_max = np.where(label_array == index_list[-1])

            x_min = index_min[2].min() - 90 if index_min[2].min() - 90 > 0 else 0
            y_min = index_min[1].min() - 90 if index_min[1].min() - 90 > 0 else 0
            z_min = index_min[0].min() - 20 if index_min[0].min() - 20 > 0 else 0
            x_max = index_max[2].max() + 90
            y_max = index_max[1].max() + 90
            z_max = index_max[0].max() + 20

            new_img_array = img_array[z_min:z_max, y_min:y_max, x_min:x_max]
            new_label_array = label_array[z_min:z_max, y_min:y_max, x_min:x_max]

            # 将腰椎标签统一为100-500
            lst_nums = index_list[-5:]
            np.putmask(new_label_array, new_label_array < index_list[-5], 0)
            # lst_nums = np.unique(new_label_array)
            print(lst_nums)
            for v_num, v_new in zip(lst_nums[-5:], new_var):
                np.putmask(new_label_array, new_label_array == v_num, v_new)
            print(new_label_array.max())
            np.putmask(new_label_array, new_label_array > 600, 0)

            new_img = sitk.GetImageFromArray(new_img_array)
            new_label = sitk.GetImageFromArray(new_label_array)
            new_img.SetSpacing(img_sitk.GetSpacing())
            new_img.SetOrigin(img_sitk.GetOrigin())
            new_label.SetSpacing(label_sitk.GetSpacing())
            new_label.SetOrigin(label_sitk.GetOrigin())
            sitk.WriteImage(new_img, os.path.join(new_image_path, org_name))
            sitk.WriteImage(new_label, os.path.join(new_label_path, label_mhd_name))
            # print(org_name)


# 将腰椎标签统一为100-500
def normalize_label(path_label, new_label_path):
    for label_name in os.listdir(path_label):
        if '.mhd' in label_name:
            label_path = os.path.join(path_label, label_name)
            label_itk = itk.imread(label_path)
            label_sitk = itk2sitk(label_itk)
            print(label_name, label_sitk.GetSpacing())
            print(label_name, label_sitk.GetSize())
            label_array = sitk.GetArrayFromImage(label_sitk)
            label_array = np.uint16(label_array)
            index_list = np.unique(label_array)
            print(label_name, index_list)

            label_array[label_array > index_list[-1]] = 0
            label_array[label_array < index_list[-5]] = 0
            label_array[label_array == index_list[-5]] = 100
            label_array[label_array == index_list[-4]] = 200
            label_array[label_array == index_list[-3]] = 300
            label_array[label_array == index_list[-2]] = 400
            label_array[label_array == index_list[-1]] = 500

            new_label = sitk.GetImageFromArray(label_array)
            new_label.SetSpacing(label_sitk.GetSpacing())
            new_label.SetOrigin(label_sitk.GetOrigin())
            sitk.WriteImage(new_label, os.path.join(new_label_path, label_name))


def crop_coxa_data(label_path, crop_path, proj_label):
    proj_label_list = open(os.path.join(proj_label), encoding='utf-8').read().strip().split('\n')

    for temp_name in os.listdir(label_path):
        if '.mhd' in temp_name and 'case105_hip_59_R' in temp_name:
            org_name = temp_name if '_label' not in temp_name else temp_name.split('.')[0][:-6]+'.mhd'
            # 获取当前数据对应的正侧位框数据
            indexs = [i for i, s in enumerate(proj_label_list) if org_name.split('.')[0] in s]
            if '_cor' in proj_label_list[indexs[0]]:
                box_cor = [float(var) for var in proj_label_list[indexs[0]].split()[1].split(',')]
                box_sag = [float(var) for var in proj_label_list[indexs[1]].split()[1].split(',')]
            else:
                box_cor = [float(var) for var in proj_label_list[indexs[1]].split()[1].split(',')]
                box_sag = [float(var) for var in proj_label_list[indexs[0]].split()[1].split(',')]
            # box_3d = [box_cor[0], box_sag[0], box_cor[1], box_cor[2], box_sag[2], box_cor[3]]
            # print(box_3d[3] - box_3d[0], box_3d[4] - box_3d[1], box_3d[5] - box_3d[2])
            org_data = sitk.ReadImage(os.path.join(path_origin, org_name))
            label_data = sitk.ReadImage(os.path.join(label_path, temp_name))
            crop_org_data = get_crop_data(org_data, box_cor, box_sag)
            crop_label_data = get_crop_data(label_data, box_cor, box_sag)
            new_label_name = org_name.split('.')[0]+'_crop.mhd' if '_label' not in temp_name else org_name.split('.')[0]+'_crop_label.mhd'
            new_org_name = org_name.split('.')[0]+'_crop.mhd'
            sitk.WriteImage(crop_org_data, os.path.join(crop_path, new_org_name))
            sitk.WriteImage(crop_label_data, os.path.join(crop_path, new_label_name))


def get_crop_data(org_data, box_cor, box_sag):
    org_size = org_data.GetSize()
    crop_box_size = (112, 96, 128)
    temp_zmin = box_cor[1] if box_cor[1] < box_sag[1] else box_sag[1]
    temp_zmax = box_cor[3] if box_cor[3] > box_sag[3] else box_sag[3]
    label_box_3d = [box_cor[0], box_sag[0], org_size[2]-temp_zmax, box_cor[2], box_sag[2], org_size[2]-temp_zmin]

    xmin = (label_box_3d[3]+label_box_3d[0])/2 - crop_box_size[0]/2
    xmax = (label_box_3d[3]+label_box_3d[0])/2 + crop_box_size[0]/2
    ymin = (label_box_3d[4]+label_box_3d[1])/2 - crop_box_size[1]/2
    ymax = (label_box_3d[4]+label_box_3d[1])/2 + crop_box_size[1]/2
    zmin = (label_box_3d[5]+label_box_3d[2])/2 - crop_box_size[2]/2
    zmax = (label_box_3d[5]+label_box_3d[2])/2 + crop_box_size[2]/2

    crop_box = np.array([xmin, ymin, zmin, xmax, ymax, zmax])

    ids_min = np.argwhere(crop_box[:3] < 0)
    ids_max = np.argwhere(crop_box[3:] > np.array(org_size))
    # 进行了边界超出限制
    if ids_min is not []:
        for item in range(ids_min.shape[0]):
            crop_box[ids_min[item, 0]] = 0
            if ids_min[item, 0] < 1:
                crop_box[ids_min[item, 0] + 3] = crop_box_size[0]
            else:
                crop_box[ids_min[item, 0] + 3] = crop_box_size[2]
    if ids_max is not []:
        for item in range(ids_max.shape[0]):
            crop_box[ids_max[item, 0] + 3] = org_size[ids_max[item, 0]]
            if ids_max[item, 0] < 1:
                crop_box[ids_max[item, 0]] = \
                    org_size[ids_max[item, 0]] - crop_box_size[0]
            else:
                crop_box[ids_max[item, 0]] = \
                    org_size[ids_max[item, 0]] - crop_box_size[2]

    crop_box = list(map(int, crop_box))
    crop_org_data = org_data[crop_box[0]:crop_box[3], crop_box[1]:crop_box[4], crop_box[2]:crop_box[5]]

    x, y, z = crop_org_data.GetSize()
    if x < crop_box_size[0] or y < crop_box_size[1] or z < crop_box_size[2]:
        crop_temp_arr = np.ones(crop_box_size,dtype=np.int16).transpose(2, 1, 0)*(-1024)
        crop_temp_arr[0:z, 0:y, 0:x] = sitk.GetArrayFromImage(crop_org_data)
        crop_temp_data = sitk.GetImageFromArray(crop_temp_arr)
        crop_temp_data.SetOrigin(crop_org_data.GetOrigin())
        return crop_temp_data
    return crop_org_data


# 对二值化图像进行孔洞填充
def fill_hole(sitk_img):
    sitk_binary = sitk.BinaryThreshold(sitk_img, lowerThreshold=100, upperThreshold=3000, insideValue=1, outsideValue=0)
    sitk_fillhole = sitk.BinaryFillhole(sitk_binary)
    return sitk_fillhole


def fill_hole_arr(bw, hole_min, hole_max, fill_2d=True):
    bw = bw > 0
    if len(bw.shape) == 2:
        background_lab = label(~bw, connectivity=1)
        fill_out = np.copy(background_lab)
        component_sizes = np.bincount(background_lab.ravel())
        too_big = component_sizes > hole_max
        too_big_mask = too_big[background_lab]
        fill_out[too_big_mask] = 0
        too_small = component_sizes < hole_min
        too_small_mask = too_small[background_lab]
        fill_out[too_small_mask] = 0
    elif len(bw.shape) == 3:
        if fill_2d:
            fill_out = np.zeros_like(bw)
            for zz in range(bw.shape[1]):
                background_lab = label(~bw[:, zz, :], connectivity=1)   # 1表示4连通， ~bw[zz, :, :]1变为0， 0变为1
                # 标记背景和孔洞， target区域标记为0
                out = np.copy(background_lab)
                # plt.imshow(bw[:, :, 87])
                # plt.show()
                component_sizes = np.bincount(background_lab.ravel())
                # 求各个类别的个数
                too_big = component_sizes > hole_max
                too_big_mask = too_big[background_lab]

                out[too_big_mask] = 0
                too_small = component_sizes < hole_min
                too_small_mask = too_small[background_lab]
                out[too_small_mask] = 0
                # 大于最大孔洞和小于最小孔洞的都标记为0， 所以背景部分被标记为0了。只剩下符合规则的孔洞
                fill_out[:, zz, :] = out
                # 只有符合规则的孔洞区域是1， 背景及target都是0
        else:
            background_lab = label(~bw, connectivity=1)
            fill_out = np.copy(background_lab)
            component_sizes = np.bincount(background_lab.ravel())
            too_big = component_sizes > hole_max
            too_big_mask = too_big[background_lab]
            fill_out[too_big_mask] = 0
            too_small = component_sizes < hole_min
            too_small_mask = too_small[background_lab]
            fill_out[too_small_mask] = 0
    else:
        print('error')
        return
    return np.logical_or(bw, fill_out)*1  # 或运算，孔洞的地方是1，原来target的地方也是1


# 去杂
def get_max_area_region(image, number=1):
    image = image.detach().numpy()
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image.astype(np.uint16))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label + 1)]
    area_list = []
    for l in range(1, num_label + 1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x - 1])[::-1]
    output = sitk.GetArrayFromImage(output_ex)

    if len(num_list_sorted) == 0:
        output = np.ones_like(output)
        return torch.tensor(output.astype(np.float32))

    max_area_region = np.zeros_like(output)
    for ii in range(0, number):
        max_area_region[output == num_list_sorted[ii]] = 1

    max_area_region = torch.tensor(max_area_region.astype(np.float32))
    return max_area_region


def gen_coxaData():
    # ++++++++++++++首先分别将原始数据和mask数据采样存为mhd start+++++++++++++
    orgDcm_path = r'E:\workdata\HipJoint\coxaOrigData\orgDcm'
    maskDcm_path = r'E:\workdata\HipJoint\coxaOrigData\labelDcm'
    org_path = r'E:\workdata\HipJoint\coxaOrigData\originMhdSam1'
    msk_path = r'E:\workdata\HipJoint\coxaOrigData\labelMhdSam1'
    txt_path = r'E:\workdata\HipJoint\coxaProjData\txt\coxa_all.txt'
    crop_path = r'E:\workdata\HipJoint\coxaOrigData\CropSamplesAll'
    # for dirs in os.listdir(orgDcm_path):
    #     origin_path = os.path.join(orgDcm_path, dirs)
    #     origin_img = myDl.read_dcm_series(origin_path)
    #     origin_img_sam = isotropic_resampler(origin_img)
    #     print(origin_path, dirs)
    #     print(origin_img_sam.GetOrigin())
    #     print(origin_img_sam.GetSize())
    #     save_origin_name = os.path.join(org_path, dirs)
    #     if not os.path.exists(org_path):
    #         os.mkdir(org_path)
    #     sitk.WriteImage(origin_img_sam, save_origin_name + '.mhd')
    #
    # for mask_dir in os.listdir(r'E:\workdata\HipJoint\label'):
    #     merge_msks(os.path.join(r'E:\workdata\HipJoint\label', mask_dir), msk_path, 1)
    # ++++++++++++++首先分别将原始数据和mask数据采样存为mhd end+++++++++++++
    # ++++++++++++++++然后检测数据和标签是否对应 start++++++++++++
    # for mhd_name in os.listdir(crop_path):
    #     if 'label.mhd' in mhd_name:
    #         org_name = mhd_name.split('.')[0][:-6] + '.mhd'
    #         origin_path = os.path.join(crop_path, org_name)
    #         mask_path = os.path.join(crop_path, mhd_name)
    #
    #         data_org = sitk.ReadImage(origin_path)
    #         data_msk = sitk.ReadImage(mask_path)
    #         arr_msk = sitk.GetArrayFromImage(data_msk)
    #         print(mhd_name, np.unique(arr_msk))
    #         print(data_org.GetOrigin(), data_msk.GetOrigin())
    #         print(data_org.GetSize(), data_msk.GetSize())
    #         print()
    # ++++++++++++++++然后检测数据和标签是否对应 end++++++++++++

    # ++++++++++++++++最后裁剪髋关节数据+++++++++++++++
    crop_coxa_data(msk_path,crop_path, txt_path)


def crop_spine_3(path_org, path_mask):
    org_img = myDl.read_dcm_series(path_org)
    mask_img = myDl.read_dcm_series(path_mask)
    org_arr = sitk.GetArrayFromImage(org_img)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    index2 = np.where(mask_arr == 200)
    index3 = np.where(mask_arr == 300)
    index4 = np.where(mask_arr == 400)
    zmax = index2[0].max()
    zmin = index4[0].min()
    ymax = max(index2[1].max(), index3[1].max(), index4[1].max())
    ymin = min(index2[1].min(), index3[1].min(), index4[1].min())
    xmax = max(index2[2].max(), index3[2].max(), index4[2].max())
    xmin = min(index2[2].min(), index3[2].min(), index4[2].min())

    new_img = org_img[xmin:xmax, ymin:ymax, zmin:zmax]
    new_mask = mask_img[xmin:xmax, ymin:ymax, zmin:zmax]

    size = new_img.GetSize()
    origin = new_img.GetOrigin()
    spacing = new_img.GetSpacing()
    center = [origin[i] + size[i] * spacing[i] / 2.0 for i in range(len(size))]

    new_dicts = myDl.dicts.copy()
    new_dicts['Hu'] = sitk.GetArrayFromImage(new_img)
    new_dicts['Spacing'] = spacing
    new_dicts['Origin'] = np.array(origin) - np.array(center)
    myDl.writeDicom(r'F:\PosEstimation\my_data\dicomData\org234_crop', new_dicts)

    new_dicts_msk = myDl.dicts.copy()
    new_dicts_msk['Hu'] = sitk.GetArrayFromImage(new_mask)
    new_dicts_msk['Spacing'] = spacing
    new_dicts_msk['Origin'] = np.array(origin) - np.array(center)
    myDl.writeDicom(r'F:\PosEstimation\my_data\dicomData\mask234_crop', new_dicts_msk)

    print(new_img.GetSize(), new_mask.GetSize())


if __name__ == '__main__':
    path_origin = r'E:\workdata\HipJoint\coxaOrigData\originMhdSam1'
    path_label = r'E:\workdata\HipJoint\coxaOrigData\labelMhdSam1'
    new_image_path = r'F:\CoxaCropDataset\CropSamplesTrain'
    new_label_path = r'F:\CoxaCropDataset\output'
    txt_path = r'E:\workdata\HipJoint\coxaProjData\txt\coxa_all.txt'


    # crop_data(path_origin, path_label, new_image_path, new_label_path)
    # normalize_label(path_label, new_label_path)

    # ++++++++++++++++++++处理髋关节数据+++++++++++++++++++++++++++++++++++++
    # gen_coxaData()
    # img = sitk.ReadImage(os.path.join(r'E:\workdata\HipJoint\coxaOrigData\CropSamplesAll','case7_hip_08_L_crop.mhd'))
    # print()
    # ++++++++++++++++++++处理髋关节数据+++++++++++++++++++++++++++++++++++++

    # knee_dcm2mhd()
    # make_knee_txt(r'F:\KneeCropDataset\Data', r'F:\KneeCropDataset\Txt')

    # crop_spine_3(r'F:\PosEstimation\my_data\dicomData\CT wzy_328_ori', r'F:\PosEstimation\my_data\dicomData\SpineCT')

    # ++++++++++++++++++++处理颅骨数据+++++++++++++++++++++++++++++++++++++
    # map_lugu = make_map(r'G:\workData\Skull\DICOM_scan\model_dcm', r'G:\workData\Skull\DICOM_scan\model_dcm_mask',r'G:\workData\Skull\DICOM_scan\model_map')
    # ++++++++++++++++++++处理颅骨数据+++++++++++++++++++++++++++++++++++++

    # merge_nrrds(r'E:\workdata\temp\nrrd', 1)

    # merge_msks(r'E:\workdata\temp\dcm\segs',1,r'E:\workdata\temp\dcm\segs')

    map_path = r'E:\workdata\regData\testData\S119430\S119430_map_crop'
    mask_path = r'E:\workdata\regData\testData\S119430\S119430_msk'
    map_crop_path = r'E:\workdata\regData\testData\S119430\S119430_map_crop'
    for file in os.listdir(map_path):
        map_path_ = os.path.join(map_path, file)
        msk_path_ = os.path.join(mask_path, file)
        map_data = myDl.read_dcm_series(map_path_)
        mask_data = myDl.read_dcm_series(msk_path_)
        print(map_data.GetOrigin(),mask_data.GetOrigin())
        mask_arr = sitk.GetArrayFromImage(mask_data)

        # indices_234 = np.argwhere(mask_arr != 0)
        # min_zyx = np.min(indices_234, axis=0)
        # max_zyx = np.max(indices_234, axis=0)

        # crop_img = map_data[min_zyx[2]:max_zyx[2], min_zyx[1]:max_zyx[1], min_zyx[0]:max_zyx[0]]
        # crop_img_1 = crop_sitkImg(map_data, min_zyx, max_zyx)
        # myDl.save_img_dcm(crop_img, os.path.join(map_crop_path, file))
        print()

