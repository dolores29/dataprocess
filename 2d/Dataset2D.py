import os

import itk
import numpy as np
import SimpleITK as sitk
from PIL import Image

from util import file_util
from util.common import itk2sitk
from util.tools_2d import img_sitk_projs
from util.tools_3d import isotropic_resampler
from util.tools_process import rotate_spine_z

'''
制作2d图像数据集腰椎
根据3D图像数据和投影生成jpg图像腰椎，调整label文件生成txt标签文件
'''


class Dataset2D:
    def __init__(self):
        pass

    # 直接从3维数据中投影
    def load3dData(self, path_origin, path_label, jpg_save, txt_save):
        originImage_list = []
        labelImage_list = []
        path_origin_list = file_util.get_files(path_origin, [])
        path_origin_list = list(set(path_origin_list))
        path_origin_list.sort()
        sum_img = 20

        for org_name in os.listdir(path_origin):
            if '.mhd' in org_name:
                org_path = os.path.join(path_origin, org_name)
                img_itk = itk.imread(org_path)
                img_sitk = itk2sitk(img_itk)
                if 'case' in org_name or 'Case' in org_name:
                    label_mhd_name = org_name.split('.')[0] + '_label.mhd'
                elif 'image' in org_name:
                    label_mhd_name = "mask" + org_name.split('.')[0][-3:] + '.mhd'
                else:
                    label_mhd_name = org_name.split('.')[0] + '_msk.mhd'
                label_path = os.path.join(path_label, label_mhd_name)
                label_itk = itk.imread(label_path)
                label_sitk = itk2sitk(label_itk)
                img_sitk_sam = isotropic_resampler(img_sitk)
                label_sitk_sam = isotropic_resampler(label_sitk,is_label=True)

                img_rotate, label_rotate = rotate_spine_z(img_sitk_sam, label_sitk_sam, 0)
                # 1 投影
                proj_f, proj_l = img_sitk_projs(img_rotate)
                proj_f_name = "spine_" + str(sum_img).zfill(4) + '_' + org_name.split('.')[0] + '_f' + ".jpg"
                proj_l_name = "spine_" + str(sum_img + 1).zfill(4) + '_' + org_name.split('.')[0] + '_l' + ".jpg"
                proj_f.save(os.path.join(jpg_save, proj_f_name))
                proj_l.save(os.path.join(jpg_save, proj_l_name))
                # 2 生成txt
                self.makeLabel(label_rotate, proj_f_name, proj_l_name, txt_save)
                sum_img += 2
        return originImage_list, labelImage_list

    def makeLabel(self, img_label, proj_f_name, proj_l_name, txt_save):
        # if img_label.GetSpacing() != (1,1,1):
        #     img_label_sam = util.isotropic_resampler(img_label)
        # else:
        #     img_label_sam = img_label
        arr_img = sitk.GetArrayFromImage(img_label)
        lst_nums = np.unique(arr_img)
        # 找出非0标签的box
        box_list = []
        proj_f_file = open(os.path.join(txt_save, proj_f_name.split('.')[0] + ".txt"), "w", encoding='utf-8')
        proj_l_file = open(os.path.join(txt_save, proj_l_name.split('.')[0] + ".txt"), "w", encoding='utf-8')
        proj_f_file.write("%s" % proj_f_name.split('.')[0])
        proj_l_file.write("%s" % proj_l_name.split('.')[0])
        if 600 in lst_nums:
            lst_num_new = lst_nums[-6:-1]
        else:
            lst_num_new = lst_nums[-5:]
        for v_num in lst_num_new:
            lst_coordinate = np.where(arr_img == v_num)
            zmin = lst_coordinate[0].min()
            zmax = lst_coordinate[0].max()
            ymin = lst_coordinate[1].min()
            ymax = lst_coordinate[1].max()
            xmin = lst_coordinate[2].min()
            xmax = lst_coordinate[2].max()
            box = [xmin, ymin, zmin, xmax, ymax, zmax]  # 可能因为图像上下翻转 需要调整
            box_list.append(box)
            proj_f_file.write(" %s,%s,%s,%s,1" % (box[0], box[2], box[3], box[5]))
            proj_l_file.write(" %s,%s,%s,%s,2" % (box[1], box[2], box[4], box[5]))
        proj_f_file.write('\n')
        proj_l_file.write('\n')

    # 绕Z坐标轴旋转扩充数据集
    def extandDataByZ(self, path_origin, path_label, jpg_save, txt_save):
        angles = list(np.array([0, 5, 10, 15,  -5, -10, -15])*np.pi/180)
        sum_img = 210
        for org_name in os.listdir(path_origin):
            if '.mhd' in org_name:
                org_path = os.path.join(path_origin, org_name)
                img_itk = itk.imread(org_path)
                img_sitk = itk2sitk(img_itk)
                if 'case' in org_name or 'Case' in org_name:
                    label_mhd_name = org_name.split('.')[0] + '_label.mhd'
                # elif 'image' in org_name:
                #     label_mhd_name = "mask" + org_name.split('.')[0][-3:] + '.mhd'
                else:
                    label_mhd_name = org_name.split('.')[0] + '_msk.mhd'
                label_path = os.path.join(path_label, label_mhd_name)
                label_itk = itk.imread(label_path)
                label_sitk = itk2sitk(label_itk)
                # label_sitk_sam = isotropic_resampler(label_sitk, img_sitk.GetSpacing())

                # 投影img并生成label
                for angle in angles:
                    img_rotate, label_rotate = rotate_spine_z(img_sitk, label_sitk, angle)
                    print("size", img_rotate.GetSize())
                    # 1 投影
                    proj_f, proj_l = img_sitk_projs(img_rotate)
                    proj_f_name = "spine_" + str(sum_img).zfill(4) + '_' + org_name.split('.')[0] + '_f' + ".jpg"
                    proj_l_name = "spine_" + str(sum_img + 1).zfill(4) + '_' + org_name.split('.')[0] + '_l' + ".jpg"
                    proj_f.save(os.path.join(jpg_save, proj_f_name))
                    proj_l.save(os.path.join(jpg_save, proj_l_name))
                    # 2 生成txt
                    self.makeLabel(label_rotate, proj_f_name, proj_l_name, txt_save)
                    sum_img += 2

    # 翻转2维图像扩充数据集
    def image_flip(self):
        img_path = r'E:\workdata\spine\multi_lumbar\spine_hos_case\SpineImage'
        txt_path = r'E:\workdata\spine\multi_lumbar\spine_hos_case\SpineLabel'
        new_img_path = r'E:\workdata\spine\multi_lumbar\spine_hos_case\flip_image'
        new_txt_path = r'E:\workdata\spine\multi_lumbar\spine_hos_case\flip_label'
        sum_img = 980
        for img_name in os.listdir(img_path):

            image_org = Image.open(os.path.join(img_path, img_name))
            image_up_out = image_org.transpose(Image.FLIP_TOP_BOTTOM)  # 上下翻转
            image_left_out = image_org.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转
            w, h = image_org.size

            img_name_new = img_name.split('.')[0].split('_')
            img_name_new[1] = str(sum_img).zfill(4)
            img_name_new = '_'.join(img_name_new)

            new_file_txt = open(os.path.join(new_txt_path, img_name_new + ".txt"), "w", encoding='utf-8')
            new_file_txt.write("%s" % img_name_new)

            txt_org = open(os.path.join(txt_path, img_name.split('.')[0] + '.txt'),
                           encoding='utf-8').read().strip().split(' ')
            for txt in txt_org[1:]:
                x1, y1, x2, y2 = list(map(int, txt.split(',')[:-1]))
                # 上下翻转
                # x1_new = x1
                # y1_new = h - y2
                # x2_new = x2
                # y2_new = h - y1
                # 左右翻转
                x1_new = w - x1
                y1_new = y1
                x2_new = w - x2
                y2_new = y2
                new_file_txt.write(" %s,%s,%s,%s,%s" % (x1_new, y1_new, x2_new, y2_new, txt.split(',')[4]))

            image_left_out.save(os.path.join(new_img_path, img_name_new+'.jpg'))
            sum_img += 1


if __name__ == '__main__':
    data2d = Dataset2D()
    path_origin = r'E:\workdata\spine\spine3d_dataset\origin_newname'
    path_label = r'E:\workdata\spine\spine3d_dataset\label_newname'
    jpg_save = r'E:\workdata\spine\multi_lumbar\spine_case\SpineImage'
    txt_save = r'E:\workdata\spine\multi_lumbar\spine_case\SpineLabel'

    data2d.load3dData(path_origin, path_label, jpg_save, txt_save)

    # data2d.extandDataByZ(path_origin, path_label, jpg_save, txt_save)
    # data2d.image_flip()

    # 测试 显示
    # txt_list = os.listdir(txt_save)
    # txt_sam_list = random.sample(txt_list, 10)
    # for txt in txt_sam_list:
    #     # if 'case' in txt:
    #     proj_label = open(os.path.join(txt_save, txt), encoding='utf-8').read().strip().split()
    #     proj_jpg = Image.open(os.path.join(jpg_save, txt.split('.')[0]+'.jpg'))
    #
    #     draw_jpg = ImageDraw.Draw(proj_jpg)
    #     for item in proj_label[1:]:
    #         box = [float(var) for var in item.split(',')]
    #         pred_box = box[0], box[1], box[2], box[3]
    #         draw_jpg.rectangle(pred_box)
    #     proj_jpg.show()

    # 测试单张
    # proj_f_label = open(os.path.join(txt_save, 'spine_1158_case7_f.txt'), encoding='utf-8').read().strip().split()
    # proj_l_label = open(os.path.join(txt_save, 'spine_1159_case7_l.txt'), encoding='utf-8').read().strip().split()
    # proj_f = Image.open(os.path.join(jpg_save, 'spine_1158_case7_f.jpg'))
    # proj_l = Image.open(os.path.join(jpg_save, 'spine_1159_case7_l.jpg'))
    #
    # draw_f = ImageDraw.Draw(proj_f)
    # draw_l = ImageDraw.Draw(proj_l)
    # for item_f, item_l in zip(proj_f_label[1:],proj_l_label[1:]):
    #
    #     box_f = [float(var) for var in item_f.split(',')]
    #     box_l = [float(var) for var in item_l.split(',')]
    #     pred_box_f = box_f[0], box_f[1], box_f[2], box_f[3]
    #     pred_box_l = box_l[0], box_l[1], box_l[2], box_l[3]
    #     draw_f.rectangle(pred_box_f)
    #     draw_l.rectangle(pred_box_l)
    # proj_f.show()
    # proj_l.show()
