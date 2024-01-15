import os

import itk
import nrrd
import numpy as np
import SimpleITK as sitk
import vtk

from util.ReadAndWrite import myDataload
from util.tools_3d import isotropic_resampler


def make_knee():
    for index_ in range(1):
        dcm_ori_path = r'E:\workdata\knee\image\ky00002229'
        path_mask = r'E:\workdata\knee\mask\ky00002229_mask'
        save_path = r'F:\KneeCropDataset\new_label\ky002166'

        origin_list = get_dcm_list(dcm_ori_path)
        mask_list = get_dcm_list(path_mask)
        # for origin_path, mask in zip(origin_list, mask_list):
        for mask in mask_list:
            # ori_name = origin_path.split('\\')[-1]
            # mas_name = mask.split('\\')[-1].split('_')[0]
            # if not ori_name == mas_name:
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


def make_txt(data_path, txt_path):
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


def move_center(path_series_dcm, crop_size):
    # 移动图像自身的中心点到坐标原点
    # 生成map
    # arr_map = arr_img*np.clip(arr_msk, 0, 1)

    dicts = {"Hu": [], "Spacing": [], "Origin": [0, 0, 0]}
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(path_series_dcm)
    dcm_series = reader.GetGDCMSeriesFileNames(path_series_dcm, seriesIDs[0])
    reader.SetFileNames(dcm_series)
    sitk_img = reader.Execute()

    arr_img = sitk.GetArrayFromImage(sitk_img)
    arr_img = arr_img[(crop_size[2]-10):(crop_size[5]+10), :, :]
    sitk_img_new = sitk.GetImageFromArray(arr_img)
    sitk_img_new.SetOrigin(sitk_img.GetOrigin())
    sitk_img_new.SetSpacing(sitk_img.GetSpacing())

    # 获取图像的大小和原点信息
    size = sitk_img_new.GetSize()
    origin = sitk_img_new.GetOrigin()
    spacing = sitk_img_new.GetSpacing()
    # 计算图像中心点坐标
    center = [origin[i] + size[i] * spacing[i] / 2.0 for i in range(len(size))]

    dicts['Hu'] = sitk.GetArrayFromImage(sitk_img_new)
    dicts['Spacing'] = sitk_img_new.GetSpacing()
    dicts['Origin'] = np.array(origin) - np.array(center)
    print(center)
    return dicts


if __name__ == '__main__':
    dcm_dir = r'G:\workData\Skull\GDRN_Data_1\dataImages\skull_samp_2.dcm'
    # mhd_dir = r'E:\workdata\spine\Spine_Dataset\originData'
    myDl = myDataload()
    # img_sitk = sitk.ReadImage(dcm_dir)
    # resample_img = isotropic_resampler(img_sitk, new_spacing=(0.2, 0.2, 0.2))[:,:,0]
    # print(resample_img.GetSpacing(), resample_img.GetSize())
    # sitk.WriteImage(resample_img, 'G:\workData\Skull\GDRN_Data_1\dataImages\skull.dcm')

    # myDl.writeDicom(r'E:\workdata\spine\temp\org_dcm\save_dcm',dicts)

    # resample_dicts = myDl.resample(dicts)
    # downsample_dicts = myDl.downsample(dicts,np.array([1, 1, 1]))
    # dcm_img = myDl.sitk_reader(dcm_dir)
    # dcm_arr = sitk.GetArrayFromImage(dcm_img)
    # dcm_arr = dcm_arr[::-1,:,:]
    # dicts_new = myDl.dicts.copy()
    # dicts_new['Hu'] = dcm_arr
    # dicts_new['Spacing'] = dcm_img.GetSpacing()
    # dicts_new['Origin'] = dcm_img.GetOrigin()
    # myDl.writeDicom(r'E:\workdata\temp\se2_1_up_down', dicts_new)
    # myDl.writeDicom(r'E:\mhdDcm', downsample_dicts)

    # mhd转dcm
    # myDl.mhd2dcm(r"E:\workdata\temp\cer\mask",r"E:\workdata\temp\cer\mask_dcm")

    # dcm转为jpg
    # myDl.dcm2jpg(r'G:\workData\Skull\2023_08_17_13_01_23\ce\1/1.dcm', r'G:\workData\Skull\2023_08_17_13_01_23\ce')

    # dcm转nrrd
    # dir_path = r'E:\workdata\temp\nrrd\S49190\MRI'
    # save_path = r'E:\workdata\temp\nrrd\S49190'
    # for dir_ in os.listdir(dir_path):
    #     dcm_path = os.path.join(dir_path, dir_)
    #     myDl.dcm2nrrd(dcm_path, save_path+'//'+dir_+'.nrrd')

    # 改变origin
    # mhd_dir_path = r'E:\workdata\spine\temp\test_mhd\crop_o\CropMask_changeorg'
    # sitk_mhd_org = sitk.ReadImage(r'E:\workdata\spine\temp\test_mhd\orgin_samp\hos_00_00ANMUBL.mhd')
    # origin_org = sitk_mhd_org.GetOrigin()
    # for mhd_ in os.listdir(mhd_dir_path):
    #     if '.mhd' in mhd_:
    #         mhd_path = os.path.join(mhd_dir_path, mhd_)
    #         sitk_mhd = sitk.ReadImage(mhd_path)
    #         # if 'label' not in mhd_:
    #         #     sitk_mhd_sam = isotropic_resampler(sitk_mhd, is_label=False)
    #         # else:
    #         #     sitk_mhd_sam = isotropic_resampler(sitk_mhd, is_label=True)
    #
    #         # org_temp = tuple(map(sum, zip(sitk_mhd.GetOrigin(),origin_org)))
    #         # sitk_mhd.SetOrigin(org_temp)
    #         print(sitk_mhd.GetSize(),sitk_mhd.GetSpacing(),sitk_mhd.GetOrigin())
    #         sitk.WriteImage(sitk_mhd, save_path+'//'+mhd_.split('.')[0]+'.nrrd')
    #         sitk.WriteImage(sitk_mhd, r'E:\workdata\spine\temp\test_mhd\crop_o\CropMask_changeorg/'+mhd_)

    # map 转为 msk
    # myDl.map2msk(r'E:\workdata\spine\temp\demo_test\dcm', r'E:\workdata\spine\temp\demo_test\msk_dcm')

    # make_knee()
    # t111_knee(r'D:\myproject\pycode\AAmycode\makeKneeTxt\Data\ky002857')

    # make_txt(r'F:\KneeCropDataset\Data', r'F:\KneeCropDataset\Txt')


#     dcm_dir = r'G:\workData\Skull\GDRN_Data\dcmData\lugu_map_zero'
#     msk_dir = r'G:\workData\Skull\GDRN_Data\dcmData\lugu_mask_zero'
#     sitk_img = isotropic_resampler(myDl.read_dcm_series(dcm_dir))
#     sitk_msk = isotropic_resampler(myDl.read_dcm_series(msk_dir), is_label=True)
#     # itkImage = itk.imread(dcm_dir)
#     arr_img = sitk.GetArrayFromImage(sitk_img)
#     arr_msk = sitk.GetArrayFromImage(sitk_msk)
#     # indexs = np.where(arr_msk > 0)
#     # crop_size = [xmin,ymin,zmin,xmax,ymax,zmax] =
    #     [indexs[1].min(),indexs[2].min(),indexs[0].min(),indexs[1].max(),indexs[2].max(),indexs[0].max()]
#     # dicts = move_center(dcm_dir, crop_size)
#     dicts_new = myDl.dicts.copy()
#     dicts_new['Hu'] = arr_msk
#     dicts_new['Spacing'] = sitk_img.GetSpacing()
#     dicts_new['Origin'] = sitk_img.GetOrigin()
#     print(sitk_img.GetSpacing())
#     myDl.writeDicom(os.path.join(r'G:\workData\Skull\GDRN_Data\dcmData', 'lugu_mask_zero_sam'), dicts_new)

    # myDl.nii2dcm(r'E:\workdata\temp/CT128.nii','out_put.dcm')

    # dcm_path = r'F:\PosEstimation\my_data\dataLumbarPig\dicomData\map234'
    # save_path = r'F:\PosEstimation\my_data\dataLumbarPig\dicomData\map234_new'
    # dcm_dict = myDl.read_dcm_of_sitk(save_path)
    # myDl.writeDicom(save_path, dcm_dict)
    # myDl.dcm2stl(r'G:\workData\Skull\GDRN_Data\dcmData\lugu_map_zero',r'G:\workData\Skull\GDRN_Data\dcmData\lugu_map_zero.stl')

    myDl.dcm2nrrd(r'F:\PosEstimation\my_data\dicomData\dcmData\trans_crop\map3_trans_crop', r'F:\PosEstimation\my_data\test_data\trans_data/test.nrrd')



