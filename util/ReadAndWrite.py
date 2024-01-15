import math
import os

import itk
import nibabel
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pydicom
import shutil
import scipy
import vtk
from scipy import ndimage
import cv2


'''
使用sitk读取dcm文件，获得的是真实像素值，用pydicom读取获得的是Hu值（原始像素值加上了1024）
pydicom存储数据时也是按照Hu存的，使用了公式Hu = pixel*slope+intercept，像素 乘以 斜率1 加上 -1024
'''


class myDataload():
    def __init__(self):
        # self.tool_dcm_path = r'../data/DCM_tool_file/tool_dicom.dcm'
        self.tool_dcm_path = r'D:\myproject\pycode\AAmycode\DataProcess\data\DCM_tool_file\tool_dicom.dcm'
        # self.tool_dcm_path = r'F:\PosEstimation\my_data\dataLumbarPig\dicomData\orgData\slicer_234\yao2/IMG0004.dcm'
        self.dicts = {"Hu": [], "Spacing": [], "Origin":[0, 0, 0]}

    def read_mhd_of_sitk(self, path_str):
        if not os.path.isdir(path_str):
            if path_str.find('.mhd') >= 0:
                dicts_ = self.read_mhd(path_str)
                return dicts_
            else:
                print("没有 mhd 文件")
                return
        path_list = os.listdir(path_str)
        for temp_path in path_list:
            if temp_path.find('.mhd') >= 0:
                path_mhd = os.path.join(path_str, temp_path)
                dicts_ = self.read_mhd(path_mhd)

                return dicts_

    def read_mhd(self, path_mhd):
        dicts_ = self.dicts.copy()
        data = sitk.ReadImage(path_mhd)  # 路径名不能有中文
        img_array = sitk.GetArrayFromImage(data)  # 像素图片
        spacing = data.GetSpacing()  # 相邻体素间距 (0.3125,0.3125,1.0)
        origin = data.GetOrigin()

        dicts_["Hu"] = np.int16(img_array)
        dicts_["Spacing"] = [spacing[0], spacing[1], spacing[2]]
        dicts_["Origin"] = [origin[0], origin[1], origin[2]]
        return dicts_

    def read_dcm_of_sitk(self, path_series_dcm):
        # sitk读取的图像数据的坐标顺序为zyx
        dicts = self.dicts.copy()
        reader = sitk.ImageSeriesReader()
        seriesIDs = reader.GetGDCMSeriesIDs(path_series_dcm)
        # file_reader = sitk.ImageFileReader()
        # if seriesIDs:
        #     for series in seriesIDs:
        #         series_file_names = reader.GetGDCMSeriesFileNames(path_series_dcm, series)
        #         # 根据一个单张的dcm文件，读取这个series的metedata，即可以获取这个序列的描述符
        #         file_reader.SetFileName(series_file_names[0])
        #         file_reader.ReadImageInformation()
        #         series_description = file_reader.GetMetaData("0008|103e")  # 序列的描述符
        #         dicom_itk = sitk.ReadImage(series_file_names)
        #         print("正在保存序列：%s" % series_description)
        dcm_series = reader.GetGDCMSeriesFileNames(path_series_dcm, seriesIDs[0])
        reader.SetFileNames(dcm_series)
        img = reader.Execute()

        img_array = sitk.GetArrayFromImage(img)  # z, y, x
        origin = img.GetOrigin()  # x, y, z
        spacing = img.GetSpacing()  # x, y, z

        dicts["Hu"] = img_array
        dicts["Spacing"] = spacing
        dicts["Origin"] = origin
        return dicts

    def read_dcm_series(self, path_series_dcm):
        reader = sitk.ImageSeriesReader()
        seriesIDs = reader.GetGDCMSeriesIDs(path_series_dcm)
        dcm_series = reader.GetGDCMSeriesFileNames(path_series_dcm, seriesIDs[0])
        reader.SetFileNames(dcm_series)
        sitk_img = reader.Execute()
        return sitk_img

    '''
    使用sitk进行dicom数据读取 多个序列文件
    这里只有最邻近插值和线性插值
    '''
    def sitk_reader(self, dcm_path):
        reader = sitk.ImageSeriesReader()
        img_names = reader.GetGDCMSeriesFileNames(dcm_path)
        reader.SetFileNames(img_names)
        image3D = reader.Execute()
        return image3D

    def vtk_mhd_reader(self, mhd_path):
        path_list = os.listdir(mhd_path)
        for path in path_list:
            if ".mhd" in path:
                mhd_path = os.path.join(mhd_path, path)
        reader = sitk.ImageFileReader()
        # reader = vtk.vtkMetaImageReader()
        reader.SetFileName(mhd_path)
        image3D = reader.Execute()
        return image3D

    def save_img_dcm(self, sitk_img, save_path):
        dicts_new = self.dicts.copy()
        dicts_new['Hu'] = sitk.GetArrayFromImage(sitk_img)
        dicts_new['Spacing'] = sitk_img.GetSpacing()
        dicts_new['Origin'] = sitk_img.GetOrigin()
        self.writeDicom(save_path, dicts_new)

    def writeDicom(self, save_path, dicts, save_name=None):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        tool_ds = pydicom.dcmread(self.tool_dcm_path)
        Hu = dicts['Hu']
        origin = dicts["Origin"]
        if type(Hu[0, 0, 0]) == np.float32 and ((Hu.max() - Hu.min()) < 10):
            Hu = Hu * (2 ** 12)
        Hu = Hu.astype(np.int16)
        range_ = (Hu.min(), Hu.max())
        d = range_[1] - range_[0]
        # tool_ds[0X0028, 0X1050].value = range_[0] + d / 2  # 窗位
        # tool_ds[0X0028, 0X1051].value = d / 8  # 窗宽

        Spacing = dicts['Spacing']
        height_interval = Spacing[0]
        width_interval = Spacing[1]
        depth_interval = Spacing[2]
        number = Hu.shape[0]
        rows = Hu.shape[1]
        columns = Hu.shape[2]

        # 批量保存
        for i in range(0, number):
            if save_name is None:
                name_str = ('%05d.dcm' % (i))
                img_name = name_str

            image = Hu[i, :, :] + 1024
            # image = Hu[i, :, :]
            tool_ds.PixelData = image.tobytes()  # 修改该dicom文件的像素矩阵
            # tool0 = np.frombuffer(tool_ds.PixelData, dtype=np.int16)

            # tool_ds.PixelData = image
            tool_ds[0X0028, 0X0010].value = rows  # 行数
            tool_ds[0X0028, 0X0011].value = columns  # 列数

            # tool_ds[0X0018, 0X9306].value = depth_interval
            try:
                tool_ds[0X0045, 0X1002].value = depth_interval
            except:
                print("tool_ds[0X0045, 0X1002].value error")

            tool_ds[0X0028, 0X0030]._value = [height_interval, width_interval]  # 修改该dicom文件的像素间隔
            tool_ds[0X0018, 0X0050]._value = depth_interval  # 修改该dicom文件的切片厚度
            tool_ds[0X0020, 0X0013]._value = i + 1  # 切片序号 # 不用修改
            # sop_uid = tool_ds[0X0008, 0X0018]._value          # SOP Instance UID # 不用修改
            sop_uid = tool_ds.SOPInstanceUID
            tool_ds[0X0008, 0X0018].value = self.modify_uid(sop_uid, ('.' + str(i + 1)))  # 不用修改
            media_sop_uid = tool_ds.file_meta.MediaStorageSOPInstanceUID  # [0X0002,0X0003]

            tool_ds.file_meta.MediaStorageSOPInstanceUID = self.modify_uid(media_sop_uid, ('.' + str(i + 1)))  # 需要修改
            # tool_ds[0X0043,0X101E].value                        # DeltaStartTime
            image_position = tool_ds[0X0020, 0X0032]._value
            image_position[0] = str(origin[0])  # 空间x
            image_position[1] = str(origin[1])  # 空间y
            image_position[2] = str(origin[2] + i * depth_interval)  # 空间切片位置 z
            tool_ds[0X0020, 0X0032]._value = image_position  # 修改image_position

            # tool_ds[0X0020, 0X1041].value = str(origin[2] + i * depth_interval)  #
            try:
                tool_ds[0X0027, 0X1044].value = origin[2] + i * depth_interval  #
                tool_ds[0X0018, 0X0088].value = depth_interval
            except:
                print("tool_ds[0X0027, 0X1044].value error")
                print("tool_ds[0X0018, 0X0088].value error")

            tool_ds.save_as(os.path.join(save_path, img_name))

    def modify_uid(self, uid, str_):
        temp = os.path.splitext(uid)
        number = len(temp[0])
        new_uid = uid[0:number] + str_

        return new_uid

    def writeMhd(self, save_path, dicts, save_name):
        Hu = dicts['Hu']
        spacing = dicts['Spacing']
        origin = np.array(dicts['Origin'])
        img = sitk.GetImageFromArray(Hu)  # 转为sitk格式的变量
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        sitk.WriteImage(img, os.path.join(save_path, save_name))

    def mhd2dcm(self, mhd_path, save_path):
        dicts = self.read_mhd_of_sitk(mhd_path)
        self.writeDicom(save_path, dicts)

    def dcm2nrrd(self,dcm_path, nrrd_path):
        dcms_series = self.read_dcm_series(dcm_path)
        sitk.WriteImage(dcms_series, nrrd_path)  # 保存

    def dcm2jpg(self,dcm_path,save_path):
        # single_dcm_path = os.path.join(dcm_path,dcm)
        single_dcm_path = dcm_path
        dcm_name = os.path.basename(single_dcm_path)
        ds_array = sitk.ReadImage(single_dcm_path)
        img_array = sitk.GetArrayFromImage(ds_array)  # 获取array
        # 单张切片的 img_array的shape 为 （1，height，width）的形式
        shape = img_array.shape
        img_array = np.reshape(img_array, (shape[1], shape[2]))  # 获取array中的height和width
        high = np.max(img_array)
        low = np.min(img_array)
        lungwin = np.array([low * 1., high * 1.])
        newimg = (img_array - lungwin[0]) / (lungwin[1] - lungwin[0])  # 归一化
        newimg = (newimg * 255).astype('uint8')  # 将像素值扩展到[0,255]

        rotated_image = cv2.rotate(255-newimg, cv2.ROTATE_90_CLOCKWISE)
        padding_size = 100
        expanded_image = np.zeros((shape[1] + 2 * padding_size, shape[2] + 2 * padding_size), dtype=np.uint8)
        expanded_image[padding_size:padding_size + shape[1], padding_size:padding_size + shape[2]] = rotated_image

        save_path_name = os.path.join(save_path,dcm_name.split('.')[0]+'_pad_100.png')
        cv2.imwrite(save_path_name, expanded_image, [int(cv2.IMWRITE_JPEG_QUALITY), 20])

    def jpg2dcm(self, jpg_path, dcm_path):
        img = cv2.imread(jpg_path)
        img_arr16 = np.array(img, dtype=np.uint16)
        img_arr16 = img_arr16.astype('int16')
        data_changed = img_arr16[:, :, 0]  # 灰度图像三通道数据相同读取一个通道即可
        pd = data_changed.tobytes()
        tool_ds = pydicom.dcmread(r'E:\workdata\temp\registration\00002.dcm')  # 需要一个已有的DCM文件 最好是转为JPG的原DCM文件
        tool_ds.PixelData = pd  # 将这个DCM文件的图像像素信息修改为JPG文件的数据
        tool_ds.save_as(dcm_path)  # 保存为新的DCM文件

    def nii2dcm(self,nii_path, dcm_path):
        nii_image = nibabel.load(nii_path)
        nii_data = nii_image.get_fdata()
        data_changed = nii_data[:, :, 79]  # 切片
        pd = data_changed.tobytes()
        tool_ds = pydicom.dcmread(r'E:\workdata\temp\registration\00002.dcm')  # 需要一个已有的DCM文件 最好是转为JPG的原DCM文件
        tool_ds.PixelData = pd  # 将这个DCM文件的图像像素信息修改为JPG文件的数据
        tool_ds.save_as(dcm_path)

    def map2msk(self,map_dcm,msk_dcm):
        dicts = self.read_dcm_of_sitk(map_dcm)
        dcm_array = dicts['Hu']
        dcm_array_msk = (dcm_array > 0) + 0
        dicts['Hu'] = dcm_array_msk
        self.writeDicom(msk_dcm, dicts)

    def dcm2stl(self, dcm_path, stl_path):
        # 1. 加载DICOM序列
        # reader = itk.ImageSeriesReader.New()
        # dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
        # reader.SetFileNames(dicom_names)
        # image = reader.Update()
        image = itk.imread(dcm_path, itk.Image[itk.UC,3])
        # 2. 图像平滑（可选）
        # 如果需要平滑图像，您可以使用ITK的平滑滤波器，例如高斯滤波器。
        # 3. 图像阈值化
        # 将图像转换为二值图像，以便创建3D表面模型。
        threshold_filter = itk.BinaryThresholdImageFilter.New(Input=image)
        threshold_filter.SetLowerThreshold(200)  # 设置阈值，根据图像的灰度值来调整
        threshold_filter.SetUpperThreshold(1000)
        threshold_filter.SetInsideValue(1)
        threshold_filter.SetOutsideValue(0)
        threshold_image = threshold_filter.Update()
        # 4. 图像平滑处理（可选）
        # 进行平滑处理以减少生成的3D模型中的噪声。
        # 5. 进行3D表面提取
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(threshold_image)
        marching_cubes.ComputeNormalsOn()
        marching_cubes.SetValue(0, 0.5)  # 根据阈值二值化图像的结果来设置提取等值面的值
        # 6. 进行数据转换
        # 将VTK的PolyData数据转换为vtkPolyDataMapper，用于将数据可视化。
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(marching_cubes.GetOutputPort())
        # # 7. 进行数据可视化
        # actor = vtk.vtkActor()
        # actor.SetMapper(mapper)
        # # 8. 创建渲染器和渲染窗口
        # renderer = vtk.vtkRenderer()
        # renderer.AddActor(actor)
        # render_window = vtk.vtkRenderWindow()
        # render_window.AddRenderer(renderer)
        # # 9. 创建交互式渲染窗口
        # render_window_interactor = vtk.vtkRenderWindowInteractor()
        # render_window_interactor.SetRenderWindow(render_window)
        # # 10. 进行渲染
        # render_window.Render()
        # render_window_interactor.Start()

        # 11. 导出为STL文件
        stl_writer = vtk.vtkSTLWriter()
        stl_writer.SetInputConnection(marching_cubes.GetOutputPort())
        stl_writer.SetFileName(stl_path)
        stl_writer.Write()

    # 网上的包进行采样
    def resample(self, dicts, new_spacing=None):
        # Determine current pixel spacing
        new_dicts = self.dicts.copy()
        image = dicts['Hu']
        spacing = dicts['Spacing']
        origin = dicts['Origin']
        if new_spacing is None:
            new_spacing = np.array([1, 1, 1])

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        new_dicts['Hu'] = image
        new_dicts['Spacing'] = new_spacing
        new_dicts['Origin'] = origin
        return new_dicts

    # 凌静写的采样，最邻近插值
    def downsample(self, dicts_, new_spacing_=None):
        if new_spacing_ is None:
            new_spacing_ = np.array([1, 1, 1.0])
        # spacing 第3个数是切片厚度，第一二个数是左右间隔
        # new_spacing = np.array([1,1,0.5])
        # new_dicts = {"Hu": [], "Spacing": [], "path_name": [], "Origin": [], "Tool": None}
        new_dicts_ = dicts_.copy()
        old_Hu = dicts_['Hu']
        old_Origin = dicts_['Origin']
        old_spacing = dicts_['Spacing']
        height_interval = old_spacing[0]
        width_interval = old_spacing[1]
        if old_spacing[2] is None:
            depth_interval = 1.0
        else:
            depth_interval = old_spacing[2]

        new_height_interval = new_spacing_[0]
        new_width_interval = new_spacing_[1]
        new_depth_interval = new_spacing_[2]

        oldshape = old_Hu.shape  # 单位1 行列数以及切片数
        depth = (oldshape[0] - 1) * depth_interval
        height = (oldshape[1] - 1) * height_interval  # 单位mm
        width = (oldshape[2] - 1) * width_interval

        x_store = []
        for ii in range(0, oldshape[1]):
            x = ii * height_interval
            x_store.append(x)

        y_store = []
        for ii in range(0, oldshape[2]):
            y = ii * width_interval
            y_store.append(y)

        z_store = []
        for ii in range(0, oldshape[0]):
            z = ii * depth_interval
            z_store.append(z)

        new_shape = [round(1 + depth / new_depth_interval), round(1 + height / new_height_interval),
                     round(1 + width / new_width_interval)]

        number_z = new_shape[0]
        number_x = new_shape[1]
        number_y = new_shape[2]
        new_Hu = np.zeros((number_z, number_x, number_y))

        # 采样
        if True:
            ii = np.array(list(range(0, number_x)))
            j = np.array(list(range(0, number_y)))
            k = np.array(list(range(0, number_z)))

            x = ii * new_height_interval  # 空间位置
            y = j * new_width_interval
            z = k * new_depth_interval
            xx_store = np.ones((len(x_store), len(x))) * np.expand_dims(x_store, axis=1)
            xx_store = xx_store.transpose(1, 0)
            yy_store = np.ones((len(y_store), len(y))) * np.expand_dims(y_store, axis=1)
            yy_store = yy_store.transpose(1, 0)
            zz_store = np.ones((len(z_store), len(z))) * np.expand_dims(z_store, axis=1)
            zz_store = zz_store.transpose(1, 0)
            m0 = abs(xx_store - np.expand_dims(x, axis=1))
            m1 = abs(yy_store - np.expand_dims(y, axis=1))
            m2 = abs(zz_store - np.expand_dims(z, axis=1))
            index_0 = m0.argsort(1)
            index_1 = m1.argsort(1)
            index_2 = m2.argsort(1)

            loc_x = index_0[:, 0]  # 找空间中最近的原始数据点的索引 以此得出新数据的Hu值
            loc_y = index_1[:, 0]
            loc_z = index_2[:, 0]
            # loc_x = np.expand_dims(loc_x,axis=1)
            # loc_y = np.expand_dims(loc_y,axis=1)
            # loc_z = np.expand_dims(loc_z,axis=1)

            tempz = old_Hu[loc_z, :, :]
            tempzx = tempz[:, loc_x, :]
            tempzxy = tempzx[:, :, loc_y]
            new_Hu = tempzxy

        new_dicts_['Hu'] = new_Hu
        new_dicts_['Spacing'] = np.array(new_spacing_)
        new_dicts_['Origin'] = old_Origin

        return new_dicts_

    def sitkSample(self, image3D, mode_interpolators='NearestNeighbor', new_spacing=None):
        # 首先对图像进行归一化减去最小值，防止插值时补0出现边界模糊问题
        image_unit = sitk.GetImageFromArray(sitk.GetArrayFromImage(image3D) -
                                            sitk.GetArrayFromImage(image3D).min())
        image_unit.SetOrigin(image3D.GetOrigin())
        image_unit.SetSpacing(image3D.GetSpacing())
        image_unit.SetDirection(image3D.GetDirection())
        # 对归一化之后的图像进行采样
        if new_spacing is None:
            new_spacing = [1, 1, 1]
        resampler = sitk.ResampleImageFilter()
        if mode_interpolators == 'NearestNeighbor':
            # 最近邻插值
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            # 线性插值
            resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputDirection(image_unit.GetDirection())
        resampler.SetOutputOrigin(image_unit.GetOrigin())
        resampler.SetOutputSpacing(new_spacing)
        orig_size = np.array(image_unit.GetSize(), dtype=int)
        orig_spacing = image_unit.GetSpacing()
        new_size = np.array([x*(y/z) for x,y,z in zip(orig_size, orig_spacing, new_spacing)])
        new_size = np.ceil(new_size).astype(np.uint32)
        resampler.SetSize(new_size.tolist())
        resampler.SetTransform(sitk.Transform(3,sitk.sitkIdentity))
        return resampler.Execute(image_unit)

    '''
    三个方向的投影图
    '''
    def imshowView(self, sitkImage):

        image3D = sitkImage
        image3D_X = np.squeeze(sitk.GetArrayFromImage(sitk.MaximumProjection(image3D, projectionDimension=0)))
        image3D_Y = np.squeeze(sitk.GetArrayFromImage(sitk.MaximumProjection(image3D, projectionDimension=1)))
        image3D_Z = np.squeeze(sitk.GetArrayFromImage(sitk.MaximumProjection(image3D, projectionDimension=2)))

        plt.figure()

        plt.subplot(131)
        plt.imshow(image3D_X)
        plt.xlim(0, image3D_X.shape[1])
        plt.ylim(0, image3D_X.shape[0])
        plt.subplot(132)
        plt.imshow(image3D_Y)
        plt.xlim(0, image3D_Y.shape[1])
        plt.ylim(0, image3D_Y.shape[0])
        plt.subplot(133)
        plt.imshow(image3D_Z)
        plt.xlim(0, image3D_Z.shape[1])
        plt.ylim(0, image3D_Z.shape[0])
        plt.show()


