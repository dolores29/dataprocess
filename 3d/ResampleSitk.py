import os
import numpy as np
import SimpleITK as sitk

from util.ReadAndWrite import myDataload

'''
可以绕 过原点的任意轴进行旋转采样
'''


class ResampleSitk:
    def __init__(self):
        self.DMP = myDataload()
        self.dx, self.dy, self.dz = [0, 0, 0]

    def createResamplePar(self, itk_img, Matrix4X4, new_spacing=None, new_size=None):

        origin = itk_img.GetOrigin()
        spacing = itk_img.GetSpacing()
        size = itk_img.GetSize()
        center = origin + 0.5*np.array(size)*spacing
        center_4X1 = np.reshape(np.array([center[0], center[1], center[2], 1]), (4, 1))
        new_center = np.dot(Matrix4X4, center_4X1)
        if new_size is None:
            new_size = size
        if new_spacing is None:
            new_spacing = spacing
        length_x, length_y, length_z = np.array(new_size)*new_spacing
        length_xyz = tuple([length_x, length_y, length_z])
        new_origin = tuple(new_center[0:3, 0]) - 0.5*np.array(length_xyz)
        return new_origin

    def resample_direction(self, itk_img, Matrix4X4):
        """
            将体数据重采样的指定的spacing大小\n
            paras：
            outpacing：指定的spacing，例如[1,1,1]
            vol：sitk读取的image信息，这里是体数据\n
            return：重采样后的数据
        """
        Matrix4X4_inv = np.linalg.inv(Matrix4X4)
        matrix, translation = getTupleMatrix_Translate(Matrix4X4_inv)

        outsize = [0, 0, 0]
        # 读取文件的size和spacing信息
        img_size = itk_img.GetSize()  # x y z
        spacing = itk_img.GetSpacing()
        direction = itk_img.GetDirection()

        combined_affine_transform = sitk.AffineTransform(3)  # 三维
        # combined_affine_transform.PreMultiply()
        if matrix is None:
            matrix = tuple([1.0, 0, 0,
                            0, 1, 0,
                            0, 0, 1])

        combined_affine_transform.SetMatrix(matrix=matrix)
        if translation is None:
            translation = tuple([-10, -30, -50])  # 采样器的平移量
        combined_affine_transform.SetTranslation(translation)

        transform = sitk.Transform()
        transform.SetIdentity()

        # 计算改变spacing后的size，用物理尺寸/体素的大小
        outsize[0] = round(img_size[0])
        outsize[1] = round(img_size[1])
        outsize[2] = round(img_size[2])

        new_origin = self.createResamplePar(itk_img, Matrix4X4)
        new_direction = direction

        # 设定重采样的一些参数  采样器resampler的位置
        resampler = sitk.ResampleImageFilter()
        # resampler.SetReferenceImage(itk_img)
        # resampler.SetTransform(transform)
        resampler.SetTransform(combined_affine_transform)  # 主要是在这里输入旋转矩阵
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetSize(outsize)
        resampler.SetOutputOrigin(new_origin)
        resampler.SetOutputDirection(new_direction)
        resampler.SetOutputSpacing(spacing)
        new_sitk_img = resampler.Execute(itk_img)

        out_arr = sitk.GetArrayFromImage(new_sitk_img)

        return new_sitk_img

    def main(self, path_series_dcm, save_path_):
        # itk_img = self.DMP.read_dcm_series(path_series_dcm)
        dicts = self.DMP.read_dcm_of_sitk(path_series_dcm)
        itk_img = dicts["img_itk"]
        point_1 = [0, 600, 0]  # center
        Matrix4X4 = getRotateMatrixWXYZ(point_1, 270, "z")
        self.createResamplePar(itk_img, Matrix4X4)

        # Matrix4X4_inv = np.linalg.inv(Matrix4X4)  # 逆矩阵
        # tupleMatrix, tupleTranslate = getTupleMatrix_Translate(Matrix4X4_inv)
        new_itk_img = self.resample_direction(itk_img, Matrix4X4)
        new_arr_img = sitk.GetArrayFromImage(new_itk_img)

        dicts["Hu"] = new_arr_img
        dicts["Origin"] = new_itk_img.GetOrigin()
        self.DMP.writeDicom(save_path_, dicts)
        self.DMP.writeMhd(save_path_, dicts, '001.mhd')


def create_rotation_matrix(point_1, point_2, alpha):
    vector = np.array(point_2) - np.array(point_1)
    mod = np.linalg.norm(vector, axis=0, keepdims=True)
    unit_vector = vector / mod  # 旋转轴(单位向量)
    radian_alpha = alpha * np.pi / 180  # 旋转角（弧度制）
    # rotation_vector = unit_vector*radian_alpha
    rotation_vector = unit_vector
    rotation_vector = np.reshape(rotation_vector, (3, 1))
    E3 = np.eye(3)
    N = np.array([[0, - 1 * rotation_vector[2][0], rotation_vector[1][0]],
                  [rotation_vector[2][0], 0, - 1 * rotation_vector[0][0]],
                  [- 1 * rotation_vector[1][0], rotation_vector[0][0], 0]])
    nnT = np.dot(rotation_vector, rotation_vector.transpose())

    R3 = np.cos(radian_alpha) * E3 + (1 - np.cos(radian_alpha)) * nnT + np.sin(radian_alpha) * N
    R4 = np.array([[R3[0][0], R3[0][1], R3[0][2], 0],
                   [R3[1][0], R3[1][1], R3[1][2], 0],
                   [R3[2][0], R3[2][1], R3[2][2], 0],
                   [0, 0, 0, 1]])
    return R4


# 左乘
def getRotateMatrixWXYZ(point_1, alpha, axis="X"):
    # point_1 = [0, 0, 0]
    point_2 = point_1.copy()

    if axis == "X" or axis == "x":
        point_2[0] = point_2[0] + 10
    if axis == "Y" or axis == "y":
        point_2[1] = point_2[1] + 10
    if axis == "Z" or axis == "z":
        point_2[2] = point_2[2] + 10

    T = create_rotation_matrix(point_1, point_2, alpha)
    R1 = np.array([[1, 0, 0, -1 * point_1[0]],
                   [0, 1, 0, -1 * point_1[1]],
                   [0, 0, 1, -1 * point_1[2]],
                   [0, 0, 0, 1]])
    R2 = np.array([[1, 0, 0, point_1[0]],
                   [0, 1, 0, point_1[1]],
                   [0, 0, 1, point_1[2]],
                   [0, 0, 0, 1]])
    R = np.dot(T, R1)
    R = np.dot(R2, R)

    return R


def getTupleMatrix_Translate(Matrix4X4):
    Matrix3X3 = [Matrix4X4[0][0], Matrix4X4[0][1], Matrix4X4[0][2],
                 Matrix4X4[1][0], Matrix4X4[1][1], Matrix4X4[1][2],
                 Matrix4X4[2][0], Matrix4X4[2][1], Matrix4X4[2][2]]
    tupleMatrix = tuple(Matrix3X3)
    tupleTranslate = tuple([Matrix4X4[0][3], Matrix4X4[1][3], Matrix4X4[2][3]])

    return tupleMatrix, tupleTranslate

if __name__ == '__main__':
    RS = ResampleSitk()
    path = r"D:\360MoveData\Users\JJ\Desktop\case19"
    save_path = r"D:\360MoveData\Users\JJ\Desktop\temp_save_path"
    RS.main(path, save_path)
