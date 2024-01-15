import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import SimpleITK as sitk
from util.ReadAndWrite import myDataload
myDl = myDataload()


# 查看检测框在正侧位投影图像上的位置
def show_frontal_lateral_preds(img_f, img_l, preds_3d):
    num_preds = preds_3d.shape[0]
    plt.subplot(121)
    plt.imshow(img_f)
    currentAxis = plt.gca()
    for i in range(num_preds):
        rect = patches.Rectangle((preds_3d[i, 0], preds_3d[i, 2]),
                                 preds_3d[i, 3] - preds_3d[i, 0],
                                 preds_3d[i, 5] - preds_3d[i, 2],
                                 fc='none', ec='g', lw=2)
        currentAxis.add_patch(rect)
    plt.subplot(122)
    plt.imshow(img_l)
    currentAxis = plt.gca()
    for i in range(num_preds):
        rect = patches.Rectangle((preds_3d[i, 1], preds_3d[i, 2]),
                                 preds_3d[i, 4] - preds_3d[i, 1],
                                 preds_3d[i, 5] - preds_3d[i, 2],
                                 fc='none', ec='g', lw=2)
        currentAxis.add_patch(rect)
    plt.show()


# 图像归一化
def set_normalization(arr_2D):
    V_min = arr_2D.min()
    V_max = arr_2D.max()
    return (arr_2D - V_min) / (V_max - V_min)


def arr2img(arr_2D):
    arr_2D = arr_2D.astype(np.float)
    arr_2D = set_normalization(arr_2D) * 255
    arr_2D = arr_2D.astype(np.uint8)
    img_2D = Image.fromarray(arr_2D)
    return img_2D


# 进行投影
def img_sitk_projs(img_sitk):
    proj_f = np.squeeze(sitk.GetArrayFromImage(sitk.MaximumProjection(img_sitk, projectionDimension=1)))
    proj_l = np.squeeze(sitk.GetArrayFromImage(sitk.MaximumProjection(img_sitk, projectionDimension=0)))
    proj_f_flip = np.flip(proj_f, 0)  # 上下翻转
    proj_l_flip = np.flip(proj_l, 0)
    proj_f_img = arr2img(proj_f_flip)
    proj_l_img = arr2img(proj_l_flip)

    plt.figure()
    plt.subplot(211)
    plt.imshow(proj_f_img, 'gray')
    plt.subplot(212)
    plt.imshow(proj_l_img, 'gray')
    plt.show()
    return proj_f_img, proj_l_img


# 测试txt中的框在jpg 显示
def show_labels(jpg_path, txt_path):
    proj_label_list = open(os.path.join(txt_path), encoding='utf-8').read().strip().split('\n')
    # for txt_name in proj_label_list:
    txt_name = r'./coxaData/projs/case20_hip_15_R_cor.jpg 0,0,196.0,209.0,1'
    jpg_name = jpg_path.split()[0].split('/')[-1]
    proj_jpg = Image.open(jpg_path)
    draw_jpg = ImageDraw.Draw(proj_jpg)
    box = [float(var) for var in txt_name.split()[1].split(',')]
    pred_box = box[0], box[1], box[2], box[3]  # xyxy
    draw_jpg.rectangle(pred_box,outline='red')
    draw_jpg.text((10, 10), jpg_name, fill='red')
    proj_jpg.show()


if __name__ == '__main__':
    jpg_path = r'E:\workdata\HipJoint\coxaProjData\projs'
    txt_path = r'E:\workdata\HipJoint\coxaProjData\txt\coxa_all.txt'
    show_labels(r'F:\PosEstimation\test_jpg/1_sam.png', txt_path)
    # mhd_path = r'E:\workdata\HipJoint\coxaOrigData\originMhdSam1/case105_hip_59_R.mhd'
    # proj_f_img, proj_l_img = img_sitk_projs(sitk.ReadImage(mhd_path))
    # proj_f_img.save(r'E:\workdata\HipJoint\jpg\case105_hip_59_R_cor.jpg')
    # proj_l_img.save(r'E:\workdata\HipJoint\jpg\case105_hip_59_R_sag.jpg')
