'''
对数据进行三线性插值采样、旋转等操作

'''
import math
import os

import cv2
import numpy as np
import time
import SimpleITK as sitk
import itertools

from util.ReadAndWrite import myDataload



def resize_demo(src, new_size):
    # 目标图像宽高
    dst_h, dst_w = new_size
    # 源图像宽高
    src_h, src_w = src.shape[:2]

    # 如果图像大小一致，直接复制返回即可
    if src_h == dst_h and src_w == dst_w:
        return src.copy()

    # 计算缩放比例
    scale_x = float(src_w) / dst_w
    scale_y = float(src_h) / dst_h

    # 遍历目标图像
    dst = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    # return dst
    # 对通道进行循环
    # for n in range(3):
    # 对 height 循环
    for dst_y in range(dst_h):
        # 对 width 循环
        for dst_x in range(dst_w):
            # 目标点在源上的坐标
            src_x = dst_x * scale_x
            src_y = dst_y * scale_y
            # 计算在源图上 4 个近邻点的位置
            # i,j
            i = int(np.floor(src_x))
            j = int(np.floor(src_y))

            u = src_x - i
            v = src_y - j
            if j == src_h - 1:
                j = src_h - 2
            if i == src_w - 1:
                i = src_h - 2
            dst[dst_y, dst_x] = (1 - u) * (1 - v) * src[j, i] + u * (1 - v) * \
                                src[j + 1, i] + (1 - u) * v * src[j, i + 1] + u * v * src[j + 1, i + 1]
    return dst


# 三线性插值,
def Trilinear(self, dicts, new_spacing=None):
    # org_spacing = dicts['Spacing']
    # org_array = dicts['Hu']
    # new_dicts = self.dicts.copy()
    # if new_spacing is None:
    #     new_spacing = np.array([1, 1, 1])
    #
    # # 原始图像的尺寸
    # org_x, org_y, org_z = org_array.shape
    # # 缩放比例
    # resize_factor = org_spacing / new_spacing  # x,y,z
    # factor_x, factor_y, factor_z = resize_factor
    # # 缩放后图像的尺寸
    # new_real_shape = org_array.shape * resize_factor
    # new_shape = np.round(new_real_shape).astype(int)
    #
    # dst_img = np.zeros(new_shape,dtype = np.int16)

    # 测试数据
    org_array = np.ones((200, 200, 100))
    dst_img = np.zeros((100, 100, 50))
    old_shape = [200, 200, 100]
    new_shape = [100, 100, 50]
    factor_x, factor_y, factor_z = 2, 2, 2
    org_x, org_y, org_z = 200, 200, 100
    # 测试数据

    for z in range(new_shape[2]):
        for y in range(new_shape[1]):
            for x in range(new_shape[0]):
                # 目标插值点在原始图像中的实际坐标
                src_x = x * factor_x
                src_y = y * factor_y
                src_z = z * factor_z

                src_x_int = math.floor(x * factor_x)
                src_y_int = math.floor(y * factor_y)
                src_z_int = math.floor(z * factor_z)
                w = src_z - src_z_int
                u = src_y - src_y_int
                v = src_x - src_x_int

                # 判断是否查出边界
                if src_x_int + 1 == org_x or src_y_int + 1 == org_y or src_z_int + 1 == org_z:
                    dst_img[x, y, z] = org_array[src_x_int, src_y_int, src_z_int]

                else:
                    # 计算在源图上4个近邻点的位置
                    C000 = org_array[src_x_int, src_y_int, src_z_int]
                    C001 = org_array[src_x_int + 1, src_y_int, src_z_int]
                    C011 = org_array[src_x_int + 1, src_y_int + 1, src_z_int]
                    C010 = org_array[src_x_int, src_y_int + 1, src_z_int]
                    C100 = org_array[src_x_int, src_y_int, src_z_int + 1]
                    C101 = org_array[src_x_int + 1, src_y_int, src_z_int + 1]
                    C111 = org_array[src_x_int + 1, src_y_int + 1, src_z_int + 1]
                    C110 = org_array[src_x_int, src_y_int + 1, src_z_int + 1]

                    dst_img[x, y, z] = C000 * (1 - v) * (1 - u) * (1 - w) + \
                                       C100 * v * (1 - u) * (1 - w) + \
                                       C010 * (1 - v) * u * (1 - w) + \
                                       C001 * (1 - v) * (1 - u) * w + \
                                       C101 * v * (1 - u) * w + \
                                       C011 * (1 - v) * u * w + \
                                       C110 * v * u * (1 - w) + \
                                       C111 * v * u * w

    return dst_img


# orginal_xyz:起始点的坐标，默认为（0，0，0）
def img3D2Point(imgArray, spacing, orginal_xyz=(0, 0, 0)):
    img_x, img_y, img_z = imgArray.shape
    dx, dy, dz = spacing
    org_x, org_y, org_z = orginal_xyz
    Coordinate_x = []
    for i in range(img_x):
        temp_x = (org_x + i * dx) * np.ones((img_y, img_z))
        Coordinate_x.append(temp_x)
    Coordinate_x = np.array(Coordinate_x)
    Coordinate_x = np.expand_dims(Coordinate_x, axis=0)

    Coordinate_y = []
    for i in range(img_y):
        temp_y = (org_y + i * dy) * np.ones((img_x, img_z))
        Coordinate_y.append(temp_y)
    Coordinate_y = np.array(Coordinate_y)
    Coordinate_y = np.transpose(Coordinate_y, (1, 0, 2))
    Coordinate_y = np.expand_dims(Coordinate_y, axis=0)

    Coordinate_z = []
    for i in range(img_z):
        temp_z = (org_z + i * dz) * np.ones((img_x, img_y))
        Coordinate_z.append(temp_z)
    Coordinate_z = np.array(Coordinate_z)
    Coordinate_z = np.transpose(Coordinate_z, (1, 2, 0))
    Coordinate_z = np.expand_dims(Coordinate_z, axis=0)

    img4d = np.expand_dims(imgArray, axis=0)
    # 将坐标和像素值全部放入一个一维矩阵中，也就是点云
    points = np.concatenate((Coordinate_x, Coordinate_y, Coordinate_z, img4d), axis=0)
    points = np.transpose(points, (1, 2, 3, 0))
    points = np.reshape(points, (img_x * img_y * img_z, 4))
    return points


def linear3(imgArray, points, default_value=0):
    img_x, img_y, img_z = imgArray.shape
    x0 = points[:, 0]
    y0 = points[:, 1]
    z0 = points[:, 2]

    x1 = np.floor(x0)
    x1 = x1.asType(np.int16)
    x2 = x1 + 1

    y1 = np.floor(y0)
    y1 = y1.asType(np.int16)
    y2 = y1 + 1

    z1 = np.floor(z0)
    z1 = z1.asType(np.int16)
    z2 = z1 + 1

    # 矩阵图像的边界点
    index_111 = np.where((x1 < 0) | (x1 > img_x) | (y1 < 0) | (y1 > img_y) | (z1 < 0) | (z1 > img_z))
    index_112 = np.where((x1 < 0) | (x1 > img_x) | (y1 < 0) | (y1 > img_y) | (z2 < 0) | (z2 > img_z))
    index_121 = np.where((x1 < 0) | (x1 > img_x) | (y2 < 0) | (y2 > img_y) | (z1 < 0) | (z1 > img_z))
    index_122 = np.where((x1 < 0) | (x1 > img_x) | (y2 < 0) | (y2 > img_y) | (z2 < 0) | (z2 > img_z))
    index_211 = np.where((x2 < 0) | (x2 > img_x) | (y1 < 0) | (y1 > img_y) | (z1 < 0) | (z1 > img_z))
    index_212 = np.where((x2 < 0) | (x2 > img_x) | (y1 < 0) | (y1 > img_y) | (z1 < 0) | (z2 > img_z))
    index_221 = np.where((x2 < 0) | (x2 > img_x) | (y2 < 0) | (y2 > img_y) | (z1 < 0) | (z1 > img_z))
    index_222 = np.where((x2 < 0) | (x2 > img_x) | (y2 < 0) | (y2 > img_y) | (z1 < 0) | (z2 > img_z))

    x1 = np.floor(x0)
    x1 = np.clip(x1, 0, img_x - 2)
    x1 = x1.astype(np.int16)
    x2 = x1 + 1

    y1 = np.floor(y0)
    y1 = np.clip(y1, 0, img_y - 2)
    y1 = y1.astype(np.int16)
    y2 = y1 + 1

    z1 = np.floor(z0)
    z1 = np.clip(z1, 0, img_z - 2)
    z1 = z1.astype(np.int16)
    z2 = z1 + 1

    p111 = imgArray[x1, y1, z1]
    p112 = imgArray[x1, y1, z2]
    p121 = imgArray[x1, y2, z1]
    p122 = imgArray[x1, y2, z2]
    p211 = imgArray[x2, y1, z1]
    p212 = imgArray[x2, y1, z2]
    p221 = imgArray[x2, y2, z1]
    p222 = imgArray[x2, y2, z2]

    p111[index_111] = default_value
    p112[index_112] = default_value
    p121[index_121] = default_value
    p122[index_122] = default_value
    p211[index_211] = default_value
    p212[index_212] = default_value
    p221[index_221] = default_value
    p222[index_222] = default_value

    v1 = (x2 - x0) * p111 + (x0 - x1) * p211
    v2 = (x2 - x0) * p121 + (x0 - x1) * p221
    w1 = (y2 - y0) * v1 + (y0 - y1) * v2

    v1 = (x2 - x0) * p112 + (x0 - x1) * p212
    v2 = (x2 - x0) * p122 + (x0 - x1) * p222
    w2 = (y2 - y0) * v1 + (y0 - y1) * v2

    new_pixel = (z2 - z0) * w1 + (z0 - z1) * w2
    new_pixel = np.clip(new_pixel, imgArray.min(), imgArray.max())
    new_pixel = new_pixel.astype(np.int16)

    x0 = np.expand_dims(x0, axis=1)
    y0 = np.expand_dims(y0, axis=1)
    z0 = np.expand_dims(z0, axis=1)
    new_pixel = np.expand_dims(new_pixel, axis=1)

    point_fill = np.concatenate((x0, y0, z0, new_pixel), axis=1)

    return point_fill


# 旋转数据,只能绕坐标轴旋转
class SpineRotation(object):
    """将样本中的图像旋转给定角度。
        Args:
            rot_angles（tuple或float）：所需的输出大小。 如果是元组，则输出为
            与rot_angles匹配。 如果是float，则匹配较小的图像边缘到rot_angles保持纵横比相同。
    """

    def __init__(self, rot_angles):
        assert isinstance(rot_angles, (float, tuple))
        self.rot_angles = getAngles(rot_angles)
        self.axis_X = [1, 0, 0]
        self.axis_Y = [0, 1, 0]
        self.axis_Z = [0, 0, 1]
        self.img = None
        self.img_mhd = None

    def __call__(self, sample):
        return self.transformRotation(sample)

    def transformRotation(self, sample_mhd):
        if 'label' in sample_mhd:
            self.img_mhd = sample_mhd['label']
            self.img_mhd = self.rotationMhd(self.img_mhd, isimg=False)
        else:
            self.img_mhd = sample_mhd['image']
            self.img_mhd = self.rotationMhd(self.img_mhd, isimg=True)
        return self.img_mhd

    def rotationMhd(self, img, isimg=True):
        self.img = img
        if isimg:
            interpolator = sitk.sitkLinear
        else:
            interpolator = sitk.sitkNearestNeighbor
        return self.spineResample(interpolator)

    def spineResample(self, interpolator):
        return sitk.Resample(image1=self.img,
                             size=self.setSize,
                             transform=self.setTransform,
                             interpolator=interpolator,
                             outputOrigin=self.setBoundsMin,
                             outputSpacing=self.setSpacing,
                             outputDirection=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                             defaultPixelValue=int(sitk.GetArrayFromImage(self.img).min()),
                             # HU unit for air in CT, possibly set to 0 in other cases
                             outputPixelType=self.img.GetPixelID())

    @property
    def setTransform(self):
        rotation_center = self.img.TransformContinuousIndexToPhysicalPoint(
            [(index - 1) / 2.0 for index in self.img.GetSize()])
        rotation_transform = sitk.VersorRigid3DTransform()
        rotation_transform.SetCenter(rotation_center)
        # rotation_transform.SetRotation(self.axis_X, self.rot_angles[0])
        # rotation_transform.SetRotation(self.axis_Y, self.rot_angles[1])
        rotation_transform.SetRotation(self.axis_Z, self.rot_angles[2])
        return rotation_transform

    @property
    def setBounds(self):
        image_bounds = []
        img_idxes = list(zip([0, 0, 0], [sz - 1 for sz in self.img.GetSize()]))
        for idx in itertools.product(img_idxes[0], img_idxes[1], img_idxes[2]):
            image_bounds.append(self.img.TransformIndexToPhysicalPoint([idx[0], idx[1], idx[2]]))
        return image_bounds

    @property
    def setPoints(self):
        all_points = []
        all_points.extend([self.setTransform.TransformPoint(pnt) for pnt in self.setBounds])
        return np.array(all_points)

    @property
    def setSpacing(self):
        return [np.min(self.img.GetSpacing())] * 3
        # return [0.5] * 3

    @property
    def setBoundsMin(self):
        return self.setPoints.min(0)

    @property
    def setSize(self):
        return self.img.GetSize()
        # return [int(sz / spc + 0.5) for spc, sz in zip(self.setSpacing,
        #                                                self.setPoints.max(0) - self.setPoints.min(0))]


def getAngles(angles):
    # print(angles)
    if isinstance(angles, float):
        angles_X, angles_Y, angles_Z = (angles, angles, angles)
    else:
        angles_X, angles_Y, angles_Z = angles
    rot_angles = (float(angles_X), float(angles_Y), float(angles_Z))
    return rot_angles


def rotate_dcm(dcm_path, save_path, angle):
    myDl = myDataload()
    image = myDl.sitk_reader(dcm_path)
    # angle = np.pi / 2
    fileRotate = SpineRotation(rot_angles=angle)
    image_ = fileRotate.transformRotation(image)

    Hu = sitk.GetArrayFromImage(image_)
    new_dicts = myDl.dicts.copy()
    new_dicts["Hu"] = Hu
    new_dicts["Origin"] = image_.GetOrigin()
    new_dicts["Spacing"] = image_.GetSpacing()
    myDl.writeDicom(save_path, new_dicts)


def rotate_spine_z(img_mhd, msk_mhd, rot_ang):
    angle_x, angle_y, angle_z = 0.0, 0.0, rot_ang
    spineTransform = SpineRotation(rot_angles=(angle_x, angle_y, angle_z))
    sample_1 = {'image': img_mhd}
    sample_2 = {'label': msk_mhd}
    sample1 = spineTransform(sample_1)
    sample2 = spineTransform(sample_2)
    return sample1, sample2


if __name__ == '__main__':
    src = cv2.imread('./t.png')
    dst = resize_demo(src, (500, 600))

    #  img_rotate, label_rotate = rotate_spine_z(img_sitk, label_sitk, 0)
    # rotate_dcm(dcm_path, save_path, angle)



