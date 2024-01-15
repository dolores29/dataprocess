import math
import os
import shutil
from shutil import copyfile
import SimpleITK as sitk
import itk
import numpy as np
import pydicom

from util.tools_3d import itk2sitk

import itk
import vtk
import plyfile
import pickle
import pandas as pd


# 获取所有文件的路径
def get_files(path, path_list=[]):
    # 显示当前目录所有文件和子文件夹，放入file_list数组里
    file_list = os.listdir(path)
    # 循环判断每个file_list里的元素是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量
        cur_path = os.path.normpath(os.path.join(path, file))
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            path_list.append(cur_path)
            # 递归
            get_files(cur_path, path_list)
        else:
            # 将file添加进all_files里
            if os.path.splitext(file)[-1] == ".mhd" or os.path.splitext(file)[-1] == ".nrrd":
                path_list.append(cur_path)
    return path_list


# 查找当前文件夹下所有文件的后缀
def find_postfix(list_):
    number = len(list_)
    postfix_lists = []
    flag = 0  # 是否包含无后缀文件 0表示无，1表示有
    for i in range(0, number):
        postfix_ = os.path.splitext(list_[i])[1]  # 后缀 postfix_可能为空
        if postfix_ == "":
            flag = 1
        if any(postfix_ in s for s in postfix_lists):
            continue
        else:
            postfix_lists.append(postfix_)
    return postfix_lists, flag


# rename_file(r'E:\workdata\knee\image\ky00001052',r'E:\workdata\temp\test\new_ky00001052')
def rename_file(path_, new_path_):
    # 加dcm后缀，os.rename(old,new)是修改文件夹的名字
    for dirs in os.listdir(path_):
        dirpath = os.path.join(path_, dirs)
        new_dirPath = os.path.join(new_path_, dirs)
        if not os.path.exists(new_dirPath):
            os.mkdir(new_dirPath)
        root_ = os.listdir(dirpath)
        for ii in range(0, len(root_)):
            print(ii)
            path_base = root_[ii]
            path_name = os.path.join(dirpath, path_base)
            new_path_name = new_dirPath + '\\IM' + str(ii) + '.dcm'
            copyfile(path_name, new_path_name)


# ********************************  准备训练数据  ********************************************
# 将所有的txt文件合成一个文件
def get_all_labels(txt_save, main_save):
    file_all_label = open(os.path.join(main_save, "spine_lines.txt"), "w", encoding='utf-8')
    for txt in os.listdir(txt_save):
        proj_label = open(os.path.join(txt_save, txt), encoding='utf-8').read().strip()
        file_all_label.write(proj_label)
        file_all_label.write('\n')


# 为训练生成txt,包含路径
def divide_dataset(main_save, split_rate=[8, 1, 1]):
    annotation_path = os.path.join(main_save, 'spine_lines.txt')
    train_rate = split_rate[0]
    valid_rate = split_rate[1]
    test_rate = split_rate[2]
    all_rate = train_rate + valid_rate + test_rate
    with open(annotation_path) as f:
        lines = f.readlines()

    # 将所有医院数据放在一个txt中
    fhospital = open(os.path.join(main_save, "hospital_spine_lines.txt"), "w", encoding='utf-8')
    for i, line in enumerate(lines):
        fhospital.write(os.path.join(os.path.dirname(os.path.dirname(annotation_path)), "SpineImage",
                                     line.split(' ')[0] + '.jpg '))
        fhospital.write(line.split(' ', 1)[1])

    np.random.seed(0)  # 固定顺序
    np.random.shuffle(lines)
    all_number = len(lines)

    train_number = int(np.round(all_number * train_rate / all_rate))
    valid_number = int(np.round(all_number * valid_rate / all_rate))
    test_number = int(np.round(all_number * test_rate / all_rate))

    train_lines = lines[0: train_number]
    if split_rate[2] == 0:
        valid_lines = lines[train_number:]
        test_lines = []
    else:
        valid_lines = lines[train_number: train_number + valid_number]
        test_lines = lines[train_number + valid_number:]

    ftrain = open(os.path.join(main_save, "train_lines.txt"), "w", encoding='utf-8')
    for i, line in enumerate(train_lines):
        ftrain.write(
            os.path.join(os.path.dirname(os.path.dirname(annotation_path)), "SpineImage", line.split(' ')[0] + '.jpg '))
        ftrain.write(line.split(' ', 1)[1])
    fvalid = open(os.path.join(main_save, "valid_lines.txt"), "w", encoding='utf-8')
    for i, line in enumerate(valid_lines):
        fvalid.write(
            os.path.join(os.path.dirname(os.path.dirname(annotation_path)), "SpineImage", line.split(' ')[0] + '.jpg '))
        fvalid.write(line.split(' ', 1)[1])
    ftest = open(os.path.join(main_save, "test_lines.txt"), "w")
    for i, line in enumerate(test_lines):
        ftest.write(
            os.path.join(os.path.dirname(os.path.dirname(annotation_path)), "SpineImage", line.split(' ')[0] + '.jpg '))
        ftest.write(line.split(' ', 1)[1])

    ftrain.close()
    fvalid.close()
    ftest.close()


# 修改数据集的名字
def rename_datasets(data_path, label_path, new_data_path, new_label_path):
    sum_img = 0
    for org_name in os.listdir(data_path):
        if '.mhd' in org_name:
            org_path = os.path.join(data_path, org_name)
            img_itk = itk.imread(org_path)
            img_sitk = itk2sitk(img_itk)
            if 'case' in org_name or 'Case' in org_name:
                label_mhd_name = org_name.split('.')[0] + '_label.mhd'
                new_img_name = org_name
                new_label_name = label_mhd_name
            elif 'image' in org_name:
                label_mhd_name = "mask" + org_name.split('.')[0][-3:] + '.mhd'
                new_img_name = org_name.split('.')[0] + '.mhd'
                new_label_name = org_name.split('.')[0] + '_label.mhd'
            else:
                label_mhd_name = org_name.split('.')[0] + '_msk.mhd'
                new_img_name = 'hos_' + org_name.split('.')[0] + '.mhd'
                new_label_name = 'hos_' + org_name.split('.')[0] + '_label.mhd'
            path_label = os.path.join(label_path, label_mhd_name)
            label_itk = itk.imread(path_label)
            label_sitk = itk2sitk(label_itk)

            # new_img_name = 'bsk_lumbar_case'+str(sum_img).zfill(4)+'.mhd'
            # new_label_name = 'bsk_lumbar_case'+str(sum_img).zfill(4)+'_label.mhd'
            sitk.WriteImage(img_sitk, os.path.join(new_data_path, new_img_name))
            sitk.WriteImage(label_sitk, os.path.join(new_label_path, new_label_name))
            sum_img += 1


# 移动dcm数据
def moveDCM(org_path):
    for dcm_dir in os.listdir(org_path):
        fenge_dir = os.path.join(org_path, dcm_dir)
        for dcm_dirs in os.listdir(fenge_dir):
            temp_dir = os.path.join(fenge_dir, dcm_dirs)
            for dcm_file in os.listdir(temp_dir):
                dcm_file_path = os.path.join(temp_dir, dcm_file)
                for dcm_ in os.listdir(dcm_file_path):
                    dcm_ppp = os.path.join(dcm_file_path, dcm_)
                    print(dcm_ppp)
                    print(temp_dir+'new')
                    if not os.path.exists(temp_dir+'new'):
                        os.mkdir(temp_dir+'new')
                    shutil.move(dcm_ppp, temp_dir+'new')
            print(dcm_dirs)


def moveFile(dir_origin, dir_new):
    sum_img = 0
    for dirs in os.listdir(dir_origin):
        dirs_path = os.path.join(dir_origin, dirs)
        print(dirs)
        for dir_ in os.listdir(dirs_path):
            dir_path = os.path.join(dirs_path, dir_)
            for dcm in os.listdir(dir_path):
                temp_dcm_path = os.path.join(dir_path, dcm)
                new_dir_name = 'case'+str(sum_img)+'_hip_'+dirs
                new_dir_path = os.path.join(dir_new, new_dir_name)
                if not os.path.exists(new_dir_path):
                    os.mkdir(new_dir_path)
                shutil.copy(temp_dcm_path, new_dir_path)
        sum_img += 1


def filter_dicom(dicom_folder, new_dicom_folder):
    # 过滤掉无效的或者重复的切片
    os.makedirs(new_dicom_folder, exist_ok=True)
    # 获取DICOM文件列表
    dicom_files = [os.path.join(dicom_folder, filename) for filename in os.listdir(dicom_folder)]
    # 用于存储已处理的UID的集合和有效的文件路径
    processed_uids = set()
    valid_files = []

    for dicom_file in dicom_files:
        try:
            ds = pydicom.dcmread(dicom_file)
            instance_uid = ds.get('SOPInstanceUID', None)

            if instance_uid and instance_uid not in processed_uids and ds.Rows > 0 and ds.Columns > 0:
                new_file_path = os.path.join(new_dicom_folder, os.path.basename(dicom_file))
                copyfile(dicom_file, new_file_path)
                processed_uids.add(instance_uid)
                valid_files.append(new_file_path)
        except:
            print(f"Error processing DICOM: {dicom_file}")


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    # os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def save_json():
    import json
    # 定义要写入的数据
    data = [
        {"loss": 0.5, "epoch": 1},
        {"loss": 0.3, "epoch": 2},
        {"loss": 0.2, "epoch": 3},
    ]
    # 打开文件并逐行写入数据
    # 打开文件并逐行写入数据
    with open('losses.json', 'a') as f:

        try:
            for item in data:
                # f.write(str(item))
                json_str = json.dumps(item) + '\n'
                f.write(json_str)
                f.flush()
        except Exception as e:
            print("写入数据时发生异常:", e)
        finally:
            f.close()


def select_dir(org_path, save_path):
    dir_name_list = os.listdir(save_path)
    for dir1 in os.listdir(org_path):
        if dir1 not in dir_name_list:
            dir1_path = os.path.join(org_path, dir1)
            for dir2 in os.listdir(dir1_path):
                if '_CTA' in dir2 and '_mask' not in dir2:
                    dir2_path = os.path.join(dir1_path, dir2)
                    if os.path.isdir(dir2_path):
                        files = os.listdir(dir2_path)
                        if len(files) > 100:
                            print(dir2_path)
                            save_path_new = os.path.join(save_path, dir2)
                            # if not os.path.exists(save_path_new):
                            #     os.mkdir(save_path_new)
                            shutil.copytree(dir2_path,save_path_new)
    # 删除配置文件
    # for dir_ in os.listdir(save_path):
    #     dir_path = os.path.join(save_path, dir_)
    #     for file_ in os.listdir(dir_path):
    #         if not any(chr.isdigit() for chr in file_):
    #             print(file_)
    #             os.remove(os.path.join(dir_path, file_))


def read_csv():
    # 读取 CSV 文件
    df = pd.read_csv('../data/input_2.csv', encoding='utf-8')
    # 插入两列空白列
    new_columns = ['糖尿病', '高血压']
    for i, new_col in enumerate(new_columns):
        df.insert(df.columns.get_loc('comorbidity') + i+1, new_col, '')

    # 遍历每一行数据
    for index, row in df.iterrows():
        for new_col in new_columns:
            # 判断是否为空
            if isinstance(row.comorbidity, float) and math.isnan(row.comorbidity):
                flag_str = '0'
            else:
                if new_col in row.comorbidity:
                    flag_str = '1'
                else:
                    flag_str = '0'
            df.at[index, new_col] = flag_str
    # 将修改后的数据保存到新的 CSV 文件
    df.to_csv('../data/output.csv', index=False)


if __name__ == '__main__':
    txt_save = r'E:\workdata\spine\multi_lumbar\spine_hospital\SpineLabel'
    main_save = r'E:\workdata\spine\multi_lumbar\spine_hospital\SpineMain'
    # 将所有的标签整合到一个文件中
    # get_all_labels(txt_save, main_save)

    # 分序训练集、验证集和测试集
    # divide_dataset(main_save)

    # dir_origin = r'E:\workdata\temp\test\New_segment'
    # dir_new = r'E:\workdata\temp\test\orginal'
    # moveFile(dir_origin, dir_new)

    # test_dict = {}
    # camera = vtk.vtkCamera()
    # test_dict['DirectionOfProjection'] = camera.GetDirectionOfProjection()
    # test_dict['ViewPlaneNormal'] = camera.GetViewPlaneNormal()
    # test_dict['ViewUp'] = camera.GetViewUp()
    # test_dict['Position'] = camera.GetPosition()
    # test_dict['FocalPoint'] = camera.GetFocalPoint()
    # test_dict['Distance'] = camera.GetDistance()
    # save_pickle(test_dict, 'test.pkl')
    #
    # read_dict = read_pickle('test.pkl')
    # print(read_dict['ViewUp'])
    # print(read_dict['Distance'])

    # rename_file(r'E:\workdata\spine\new_spine\origin_data',r'E:\workdata\spine\new_spine\origin_data_newname')

    # jpg_path = r'E:\workdata\HipJoint\projs'
    # label_path = r'E:\workdata\HipJoint\coxaProjData\label'
    # label_list = os.listdir(label_path)
    # jpg_list = os.listdir(jpg_path)
    # for file_ in os.listdir(jpg_path):
    #     if file_.split('.')[0]+'.xml' in label_list:
    #         continue
    #     else:
    #         print(file_)
    #         # os.remove(os.path.join(jpg_path, file_))

    # dir_path = r'E:\workdata\HipJoint\coxaOrigData\tempDir'
    # j = 45
    # for i in range(77, 124, 2):
    #     print(i)
    #
    #     dir_name = 'case'+str(i)+'_hip_'+str(j)+'_L'
    #     dir_name1 ='case'+str(i+1)+'_hip_'+str(j)+'_R'
    #     os.makedirs(os.path.join(dir_path, dir_name), exist_ok=True)
    #     os.makedirs(os.path.join(dir_path, dir_name1), exist_ok=True)
    #     i += 2
    #     j += 1

    # for dir_ in os.listdir(dir_path):
    #     temp_path = os.path.join(dir_path, dir_)
    #     os.makedirs(os.path.join(temp_path, 'mask_gugu'), exist_ok=True)
    #     os.makedirs(os.path.join(temp_path, 'mask_kuan'), exist_ok=True)
    # print(os.path.abspath(r'data/DCM_tool_file/tool_dicom.dcm'))
    save_path =r'G:\workData\Spine\DataSet3'
    # save_path1 =r'E:\workdata\spine\CT_select'
    select_dir(r'E:\workdata\spine\DataSet3',save_path)
    # ct_list = os.listdir(save_path1)
    # for dir_ in os.listdir(save_path):
    #    if dir_ in ct_list:
    #        print(dir_)
    # save_json()

    # read_csv()



