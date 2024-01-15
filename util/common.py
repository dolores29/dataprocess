import itk
import SimpleITK as sitk
import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk


def getAngles(angles):
    # print(angles)
    if isinstance(angles, float):
        angles_X, angles_Y, angles_Z = (angles, angles, angles)
    else:
        angles_X, angles_Y, angles_Z = angles
    rot_angles = (float(angles_X), float(angles_Y), float(angles_Z))
    return rot_angles


def itk2sitk(itk_image):
    img_arr = itk.GetArrayFromImage(itk_image)
    sitk_image = sitk.GetImageFromArray(img_arr)
    sitk_image.SetOrigin(tuple(itk_image.GetOrigin()))
    sitk_image.SetSpacing(tuple(itk_image.GetSpacing()))
    return sitk_image


def vtk2sitk(vtkImage):
    itk_image = itk.image_from_vtk_image(vtkImage)
    sitk_image = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_image))
    sitk_image.SetOrigin(tuple(itk_image.GetOrigin()))
    sitk_image.SetSpacing(tuple(itk_image.GetSpacing()))
    return sitk_image


# 排除direction不同而引起的问题
def crop_sitkImg(org_img, min_zyx, max_zyx):
    crop_img = org_img[min_zyx[2]:max_zyx[2], min_zyx[1]:max_zyx[1], min_zyx[0]:max_zyx[0]]
    origin_crop = np.array([min_zyx[2], min_zyx[1], min_zyx[0]]) * np.array(org_img.GetSpacing()) + np.array(org_img.GetOrigin())
    crop_img.SetOrigin(origin_crop)
    return crop_img


def myconv2d(img, kernel, pad_mode='empty'):
    assert len(img.shape) == 2 and len(kernel.shape) == 2
    assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1 and kernel.shape[0] > 1
    size = kernel.shape[0]
    rad = size // 2
    img = np.pad(img.astype(np.int32), ((rad, rad), (rad, rad)), pad_mode)

    pts = [(idx // size - rad, idx % size - rad) for idx in range(size * size)]
    img_moved = np.array([np.roll(np.roll(img, x, axis=0), y, axis=1) for (x, y) in pts])
    kernel_spanned = np.expand_dims(np.reshape(kernel, -1), axis=(1, 2))
    img_conv = np.sum(img_moved * kernel_spanned, axis=0)
    return img_conv[rad:-rad, rad:-rad]


def myconv2d_1(img, in_channels, out_channels, kernels, stride=1, padding=0):
    N, C, H, W = img.shape
    assert C == in_channels
    kh, kw = kernels.shape
    if padding:
        img = np.pad(img, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')  # padding along with all axis
        print("#" * 50 + " After Padding Img: " + "#" * 50)
        print(img)

    # 卷积公式计算输出维度
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1

    outputs = np.zeros([N, out_channels, out_h, out_w])

    for n in range(N):
        for _out in range(out_channels):
            for h in range(out_h):
                for w in range(out_w):
                    for x in range(kh):
                        for y in range(kw):
                            outputs[n][_out][h][w] += img[n][_out][h * stride + x][w * stride + y] * kernels[x][y]
    return outputs


# 三维数组转换成reader(ImageImport)
def numpy2vtkImageImport(source_numpy, spacing=None, origin=[0, 0, 0], direction=None, as_uint8=False):
    """
    numpy转成vtkImageImport
    :param source_numpy: numpy格式的图片（z,y,x）(A,C,S)
    :param spacing: 像素的间隔
    :param origin: 图片的原点
    :param direction: 方向
    :param as_uint8: 转成uint8
    :return: vtkImageImport
    """
    importer = vtk.vtkImageImport()
    origin_type = source_numpy.dtype
    if as_uint8:
        origin_type = np.uint8
        source_numpy = source_numpy.astype('uint8')
    # else:
    #     source_numpy = source_numpy.astype('int32')
    img_string = source_numpy.tobytes()
    dim = source_numpy.shape  # (z,y,x),(A,C,S)
    importer.CopyImportVoidPointer(img_string, len(img_string))
    if as_uint8:
        importer.SetDataScalarTypeToUnsignedChar()
    elif np.int16 == origin_type:
        importer.SetDataScalarTypeToShort()
    else:
        importer.SetDataScalarTypeToInt()
    importer.SetNumberOfScalarComponents(1)

    extent = importer.GetDataExtent()  # 0,0,0,0,0,0
    # 图像的维度=(extent[1]-extent[0]+1) * (extent[3]-extent[2]+1) * (extent[5]-DataExtent[4]+1)=(x,y,z)=(S,C,A)
    importer.SetDataExtent(extent[0], extent[0] + dim[2] - 1,
                           extent[2], extent[2] + dim[1] - 1,
                           extent[4], extent[4] + dim[0] - 1)
    importer.SetWholeExtent(extent[0], extent[0] + dim[2] - 1,
                            extent[2], extent[2] + dim[1] - 1,
                            extent[4], extent[4] + dim[0] - 1)
    if spacing is not None:
        importer.SetDataSpacing(*spacing)
    if origin is not None:
        importer.SetDataOrigin(*origin)
    if direction is not None:
        importer.SetDataDirection(direction)
    return importer


# 2维数组转换成vtkImageData
def arr2vtkimagedata(arr_img, spacing, origin):
    arr = np.array(arr_img)
    vtk_data = numpy_to_vtk(
        arr.ravel(), array_type=vtk.VTK_UNSIGNED_CHAR)
    dim = arr_img.size
    if len(dim) == 2:
        dim = (dim[0], dim[1], 1)
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(dim)
    vtk_img.SetSpacing(spacing)
    vtk_img.SetOrigin(origin)
    vtk_img.GetPointData().SetScalars(vtk_data)
    return vtk_img

if __name__ == '__main__':
    N = 6
    # img = np.array([idx for idx in range(N*N)]).reshape((N,N))
    img = np.array([idx for idx in range(1 * 3 * N * N)]).reshape((1, 3, N, N))
    print(img)

    kernels = np.asarray(
        [
            [-1, -1, 1],
            [-1, 2, 1],
            [1, 1, 1],
        ]
    )
    outputs = myconv2d_1(img,3,3, kernels,padding=1)
    print("#" * 50 + " After Conv Outputs: " + "#" * 50)
    print(outputs)
