import SimpleITK as sitk
import numpy as np
from util.ReadAndWrite import myDataload
from skimage.measure import label
def sitkImageProcess():
    image = sitk.ReadImage("srcdicom.mha")
    np_image = sitk.GetArrayFromImage(image)

    # 1.log transform
    np_log_image = np.log1p(np_image - np.min(np_image))
    log_image = sitk.GetImageFromArray(np_log_image)
    log_image.SetOrigin(image.GetOrigin())
    log_image.SetDirection(image.GetDirection())
    log_image.SetSpacing(image.GetSpacing())
    sitk.WriteImage(log_image, "log_image.mha")

    # 2.power transform
    np_image_clone1 = np_image.copy()
    np_image_clone1 = (np_image_clone1 - np_image.mean()) / np_image.std()
    np_power_image = np.power(np_image_clone1, 3)
    power_image = sitk.GetImageFromArray(np_power_image)
    power_image.SetOrigin(image.GetOrigin())
    power_image.SetDirection(image.GetDirection())
    power_image.SetSpacing(image.GetSpacing())
    sitk.WriteImage(power_image, "power_image.mha")

    # 3.exp transform
    np_image_clone = np_image.copy()
    np_image_clone = (np_image_clone - np_image.mean()) / np_image.std()
    np_exp_image = np.exp(np_image_clone)
    exp_image = sitk.GetImageFromArray(np_exp_image)
    exp_image.SetOrigin(image.GetOrigin())
    exp_image.SetDirection(image.GetDirection())
    exp_image.SetSpacing(image.GetSpacing())
    sitk.WriteImage(exp_image, "exp_image.mha")

    # 4.Histogram equalization
    sitk_hisequal = sitk.AdaptiveHistogramEqualizationImageFilter()
    sitk_hisequal.SetAlpha(0.9)
    sitk_hisequal.SetBeta(0.9)
    sitk_hisequal.SetRadius(3)
    sitk_hisequal = sitk_hisequal.Execute(image)
    sitk.WriteImage(sitk_hisequal, "sitk_hisequal.mha")

    # 5.mean filter
    sitk_mean = sitk.MeanImageFilter()
    sitk_mean.SetRadius(5)
    sitk_mean = sitk_mean.Execute(image)
    sitk.WriteImage(sitk_mean, 'sitk_mean.mha')

    # 6.median filter
    sitk_median = sitk.MedianImageFilter()
    sitk_median.SetRadius(5)
    sitk_median = sitk_median.Execute(image)
    sitk.WriteImage(sitk_median, 'sitk_median.mha')

    # 7.gassuian
    sitk_gassuian = sitk.SmoothingRecursiveGaussianImageFilter()
    sitk_gassuian.SetSigma(3.0)
    sitk_gassuian.NormalizeAcrossScaleOff()
    sitk_gassuian = sitk_gassuian.Execute(image)
    sitk.WriteImage(sitk_gassuian, 'sitk_gassuian.mha')

    # 8.sobel
    image_float = sitk.Cast(image, sitk.sitkFloat32)
    sobel_op = sitk.SobelEdgeDetectionImageFilter()
    sobel_sitk = sobel_op.Execute(image_float)
    sobel_sitk = sitk.Cast(sobel_sitk, sitk.sitkInt16)
    sitk.WriteImage(sobel_sitk, "sobel_sitk.mha")

    # 9.canny
    canny_op = sitk.CannyEdgeDetectionImageFilter()
    canny_op.SetLowerThreshold(40)
    canny_op.SetUpperThreshold(120)
    canny_op.SetVariance(3)
    canny_op.SetMaximumError(0.5)
    canny_sitk = canny_op.Execute(image_float)
    canny_sitk = sitk.Cast(canny_sitk, sitk.sitkInt16)
    sitk.WriteImage(canny_sitk, "canny_sitk.mha")


# 最大连通域提取
def GetLargestConnectedCompont(binarysitk_image):
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 255
    outmask[labelmaskimage != maxlabel] = 0
    outmask_sitk = sitk.GetImageFromArray(outmask)
    outmask_sitk.SetDirection(binarysitk_image.GetDirection())
    outmask_sitk.SetSpacing(binarysitk_image.GetSpacing())
    outmask_sitk.SetOrigin(binarysitk_image.GetOrigin())
    return outmask_sitk

# 逻辑与操作
def GetMaskImage(sitk_src, sitk_mask, replacevalue=0):
    array_src = sitk.GetArrayFromImage(sitk_src)
    array_mask = sitk.GetArrayFromImage(sitk_mask)
    array_out = array_src.copy()
    array_out[array_mask == 0] = replacevalue
    outmask_sitk = sitk.GetImageFromArray(array_out)
    outmask_sitk.SetDirection(sitk_src.GetDirection())
    outmask_sitk.SetSpacing(sitk_src.GetSpacing())
    outmask_sitk.SetOrigin(sitk_src.GetOrigin())
    return outmask_sitk


def RemoveSmallConnectedCompont(sitk_maskimg, rate=0.5):
    """
    remove small object
    :param sitk_maskimg:input binary image
    :param rate:size rate
    :return:binary image
    """
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    not_remove = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size > maxsize * rate:
            not_remove.append(l)
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage != maxlabel] = 0
    for i in range(len(not_remove)):
        outmask[labelmaskimage == not_remove[i]] = 255
    outmask[outmask!=0]=255
    outmask_sitk = sitk.GetImageFromArray(outmask)
    outmask_sitk.SetDirection(sitk_maskimg.GetDirection())
    outmask_sitk.SetSpacing(sitk_maskimg.GetSpacing())
    outmask_sitk.SetOrigin(sitk_maskimg.GetOrigin())
    return outmask_sitk


def hole_filling(bw, hole_min, hole_max, fill_2d=True):
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


if __name__ == '__main__':
    # 读取Dicom序列
    pathDicom = 'F:\SpineCropDataset\output\dcm_path\case22_L01_00_test'
    reader = sitk.ImageSeriesReader()
    filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom)
    reader.SetFileNames(filenamesDICOM)
    sitk_src = reader.Execute()
    myDl = myDataload()

    # step1.设置固定阈值为100，把骨骼和心脏及主动脉都分割出来
    sitk_seg = sitk.BinaryThreshold(sitk_src, lowerThreshold=100, upperThreshold=3000, insideValue=1, outsideValue=0)
    # sitk.WriteImage(sitk_src, r'F:\CoxaCropDataset\output\mhd_path\testmhd\step1.mhd')

    # step2.形态学开运算+最大连通域提取,粗略的心脏和主动脉图像
    sitk_open = sitk.BinaryMorphologicalOpeningImageFilter()
    sitk_open.SetKernelRadius(3)
    sitk_open_image = sitk_open.Execute(sitk_src)

    # sitk_erode = sitk.BinaryErodeImageFilter()
    # sitk_erode.SetKernelRadius(100)
    # sitk_erode_image = sitk_erode.Execute(sitk_src)

    # sitk_erode_image = sitk.BinaryErode(sitk_src != 0, (1, 1, 1))
    # sitk_erode_arr = sitk.GetArrayFromImage(sitk_erode_image)
    # sitk_erode_arr[sitk_erode_arr == 1] = 1000
    # sitk_erode_out = sitk.GetImageFromArray(sitk_erode_arr)
    # sitk_erode_out.SetDirection(sitk_erode_image.GetDirection())
    # sitk_erode_out.SetSpacing(sitk_erode_image.GetSpacing())
    # sitk_erode_out.SetOrigin(sitk_erode_image.GetOrigin())
    #
    # sitk_dilate_image = sitk.BinaryDilate(sitk_erode_out != 0, (1, 1, 1))
    sitk_arr = sitk.GetArrayFromImage(sitk_seg)
    filled = hole_filling(sitk_arr, 0, 50, fill_2d=True)
    filled[filled==1]=1000


    # sitk_fillhole = sitk.BinaryFillhole(sitk_seg)
    #
    # sitk_dilate_arr = sitk.GetArrayFromImage(sitk_fillhole).astype(np.uint16)
    # sitk_dilate_arr[sitk_dilate_arr == 1] = 1000
    # sitk_dilate_out = sitk.GetImageFromArray(sitk_dilate_arr)
    # sitk_dilate_out.SetDirection(sitk_fillhole.GetDirection())
    # sitk_dilate_out.SetSpacing(sitk_fillhole.GetSpacing())
    # sitk_dilate_out.SetOrigin(sitk_fillhole.GetOrigin())

    new_dict = myDl.dicts.copy()
    new_dict['Hu'] = filled
    new_dict['Spacing'] = sitk_seg.GetSpacing()
    new_dict['Origin'] = sitk_seg.GetOrigin()
    myDl.writeDicom(r'F:\SpineCropDataset\output\fillhole', new_dict)
    # sitk.WriteImage(sitk_fillhole, r'F:\SpineCropDataset\output\fillhole')

    # step3.再将step1的结果与step2的结果相减,得到骨骼部分
    # array_open = sitk.GetArrayFromImage(sitk_open_image)
    # array_seg = sitk.GetArrayFromImage(sitk_seg)
    # array_mask = array_seg - array_open
    # sitk_mask = sitk.GetImageFromArray(array_mask)
    # sitk_mask.SetDirection(sitk_seg.GetDirection())
    # sitk_mask.SetSpacing(sitk_seg.GetSpacing())
    # sitk_mask.SetOrigin(sitk_seg.GetOrigin())
    # sitk.WriteImage(sitk_mask, r'F:\CoxaCropDataset\output\mhd_path\testmhd\step3.mhd')
    #
    # # step4.最大连通域提取，去除小连接
    # skeleton_mask = GetLargestConnectedCompont(sitk_mask)
    # sitk.WriteImage(skeleton_mask, r'F:\CoxaCropDataset\output\mhd_path\testmhd\step4.mhd')
    #
    # # step5.将得到的图像与原始图像进行逻辑与操作
    # sitk_skeleton = GetMaskImage(sitk_src, skeleton_mask, replacevalue=-1500)
    # sitk.WriteImage(sitk_skeleton, r'F:\CoxaCropDataset\output\mhd_path\testmhd\step5.mhd')