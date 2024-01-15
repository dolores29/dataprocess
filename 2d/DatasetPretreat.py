import numpy as np
import cv2
import matplotlib.pyplot as plt
from util.ReadAndWrite import myDataload

# 显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Pretreat:
    """ 数据集预处理
    Args:
        dir_path (src): 数据路径.
    """

    def __init__(self):
        print()

    def probability_to_histogram(self, image):
        """
        根据像素概率将原始图像直方图均衡化
        :param img:类型为 ndarray
        :param prob:累计概率
        :return: 直方图均衡化后的图像
        """
        prob = np.zeros(shape=(256))
        for rv in image:
            for cv in rv:
                prob[cv] += 1
        r, c = image.shape[0]

        prob = prob / (r * c)

        prob = np.cumsum(prob)
        img_map = [int(i * prob[i]) for i in range(256)]  # 像素值映射
        # 像素值替换
        for ri in range(r):
            for ci in range(c):
                image[ri, ci] = img_map[image[ri, ci]]
        return image

    def sobel_filter(self, image):
        """
        Sobel算子主要用于边缘检测，在技术上它是以离散型的差分算子，用来运算图像亮度函数的梯度的近似值。
        Sobel算子是典型的基于一阶导数的边缘检测算子，由于该算子中引入了类似局部平均的运算，因此对噪声具有平滑作用，能很好的消除噪声的影响。
        Sobel算子对于象素的位置的影响做了加权，与Prewitt算子、Roberts算子相比因此效果更好。
        Sobel算子包含两组3x3的矩阵，分别为横向及纵向模板，将之与图像作平面卷积，即可分别得出横向及纵向的亮度差分近似值。
        """
        x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
        # 将其转回原来的uint8形式
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        out = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return out

    def conbineGray(self, image):
        """
        基于灰度形态学的应用
        形态梯度：梯度用于刻画目标边界或边缘位于图像灰度级剧烈变化的区域、形态学梯度根据
            膨胀或者腐蚀与原图作差组合来实现增强结构元素领域中像素的强度，突出高亮区域的外围。
        形态平滑：图像平滑又被称为图像模糊，用于消除图片中的噪声。
        高帽：高帽运算是原图像和原图像开运算结果的差值。
        黑帽：黑帽运算是原图像和原图像闭运算的差值。
        Args:
            image (str): 图像
        """
        # image = cv2.imread(image)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 30))
        Gd_out = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)  # 形态梯度
        sm = cv2.boxFilter(gray, -1, (3, 3), normalize=True)  # 形态平滑
        hat_g_out = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)  # 高帽变换
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        hat_b_out = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)  # 黑帽变换
        return hat_g_out

    def noiseGauss(self, image, mean=0, sigma=0.005):
        '''
        添加高斯噪声
        mean : 均值
        sigma: 控制高斯噪声的比例
        '''
        temp_img = np.array(image / 255, dtype=float)
        noise = np.random.normal(mean, sigma ** 0.5, temp_img.shape)
        noisy_img = np.zeros(temp_img.shape, np.float64)
        if len(temp_img.shape) == 2:
            noisy_img = temp_img + noise
        else:
            noisy_img[:, :, 0] = temp_img[:, :, 0] + noise
            noisy_img[:, :, 1] = temp_img[:, :, 1] + noise
            noisy_img[:, :, 2] = temp_img[:, :, 2] + noise
        # 将值限制在(-1/0,1)间，然后乘255恢复
        if noisy_img.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.

        out = np.clip(noisy_img, low_clip, 1.0)
        out = np.uint8(out * 255)
        return out

    def fillHole_RGB(self, imgPath, SizeThreshold):
        '''
        根据颜色分类,多色填充,阈值SizeThreshold以内的被填充
        '''
        # blur = cv2.GaussianBlur(image, (5, 5), 0)
        # ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # binary = image < ret
        # edges = roberts(binary)
        # fillImage = ndimage.binary_fill_holes(edges)

        im_in_rgb = cv2.imread(imgPath).astype(np.uint32)
        # 将im_in_rgb的RGB颜色转换为 0xbbggrr
        im_in_lbl = im_in_rgb[:, :, 0] + (im_in_rgb[:, :, 1] << 8) + (im_in_rgb[:, :, 2] << 16)
        # 将0xbbggrr颜色转换为0,1,2,...
        colors, im_in_lbl_new = np.unique(im_in_lbl, return_inverse=True)
        # 将im_in_lbl_new数组reshape为2维
        im_in_lbl_new = np.reshape(im_in_lbl_new, im_in_lbl.shape)
        # 创建从32位im_in_lbl_new到8位colorize颜色的映射
        colorize = np.empty((len(colors), 3), np.uint8)
        colorize[:, 0] = (colors & 0x0000FF)
        colorize[:, 1] = (colors & 0x00FF00) >> 8
        colorize[:, 2] = (colors & 0xFF0000) >> 16
        # 有几种颜色就设置几层数组，每层数组均为各种颜色的二值化数组
        im_result = np.zeros((len(colors),) + im_in_lbl_new.shape, np.uint8)
        im_th = np.zeros(im_in_lbl_new.shape, np.uint8)
        for i in range(len(colors)):
            for j in range(im_th.shape[0]):
                for k in range(im_th.shape[1]):
                    if (im_in_lbl_new[j][k] == i):
                        im_th[j][k] = 255
                    else:
                        im_th[j][k] = 0
            # 复制 im_in 图像
            im_floodfill = im_th.copy()
            # Mask 用于 floodFill,mask多出来的2可以保证扫描的边界上的像素都会被处理.
            h, w = im_th.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            isbreak = False
            for m in range(im_floodfill.shape[0]):
                for n in range(im_floodfill.shape[1]):
                    if (im_floodfill[m][n] == 0):
                        seedPoint = (m, n)
                        isbreak = True
                        break
                if (isbreak):
                    break
            # 得到im_floodfill
            cv2.floodFill(im_floodfill, mask, seedPoint, 255, 4)
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)  # 包含所有孔洞
            im_floodfill_inv_copy = im_floodfill_inv.copy()
            # 函数findContours获取轮廓
            contours, hierarchy = cv2.findContours(im_floodfill_inv_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for num in range(len(contours)):
                if (cv2.contourArea(contours[num]) >= SizeThreshold):
                    cv2.fillConvexPoly(im_floodfill_inv, contours[num], 0)
            # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
            im_out = im_th | im_floodfill_inv
            im_result[i] = im_out

        im_fillhole = np.zeros((im_in_lbl_new.shape[0], im_in_lbl_new.shape[1], 3), np.uint8)
        # 之前的颜色映射起到了作用
        for i in range(im_result.shape[1]):
            for j in range(im_result.shape[2]):
                for k in range(im_result.shape[0]):
                    if (im_result[k][i][j] == 255):
                        im_fillhole[i][j] = colorize[k]
                        break

        return im_fillhole

    def fillHole(self, im_in, SizeThreshold):
        '''
        将彩色图像转为二值图像再填充,阈值SizeThreshold以内的被填充
        '''
        # im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        # ret, binary  = cv2.threshold(im_in, 1, im_in.max(), cv2.THRESH_BINARY)  # 固定阈值
        ret, binary = cv2.threshold(im_in, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # otsu阈值
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # 膨胀图像
        dilated = cv2.dilate(binary, kernel)
        # 腐蚀图像
        eroded = cv2.erode(dilated, kernel)

        # 复制 im_in 图像
        im_floodfill = eroded.copy()
        # Mask 用于 floodFill，官方要求长宽+2
        h, w = im_in.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # floodFill函数中的seedPoint对应像素必须是背景
        isbreak = False
        for i in range(im_floodfill.shape[0]):
            for j in range(im_floodfill.shape[1]):
                if (im_floodfill[i][j] == 0):
                    seedPoint = (i, j)
                    isbreak = True
                    break
            if (isbreak):
                break
        # 得到im_floodfill 255填充非孔洞值
        cv2.floodFill(im_floodfill, mask, seedPoint, int(im_in.max()))
        # 得到im_floodfill的逆im_floodfill_inv
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # 函数findContours获取轮廓
        contours, hierarchy = cv2.findContours(im_floodfill_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for num in range(len(contours)):
            # print('第',number,'张孔洞面积', cv2.contourArea(contours[num]))
            if (cv2.contourArea(contours[num]) >= SizeThreshold):
                cv2.fillConvexPoly(im_floodfill_inv, contours[num], 0)
        # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
        im_out = eroded | im_floodfill_inv
        return im_out

    def fillhole_dcm(self, dcm_path, save_path):
        dicts = myDl.read_dcm_of_sitk(dcm_path)
        dcm_array = dicts['Hu']
        new_dcm_array = np.zeros(dcm_array.shape)
        for i in range(dcm_array.shape[0]):
            high = np.max(dcm_array[i])
            low = np.min(dcm_array[i])
            if high != low:
                lungwin = np.array([low * 1., high * 1.])
                newimg = (dcm_array[i] - lungwin[0]) / (lungwin[1] - lungwin[0])  # 归一化
                newimg = (newimg * 255).astype('uint8')
                img_out = self.fillHole(newimg, 300)
                new_img_out = img_out / 255 * (lungwin[1] - lungwin[0]) + lungwin[0]
                new_dcm_array[i] = new_img_out
        dicts["Hu"] = new_dcm_array
        myDl.writeDicom(save_path, dicts)
        return new_dcm_array


if __name__ == '__main__':
    pretreat = Pretreat()
    myDl = myDataload()
    # final_file = r"E:\workdata\spine\temp\test_jpg"
    # image_path = r'E:\workdata\spine\temp\test_jpg\net.png'
    dcm_path = r'E:\workdata\spine\temp\yao_dicom_ming\yao'
    save_path = r'E:\workdata\spine\temp\yao_dicom_ming\yao_fill'

    pretreat.fillhole_dcm(dcm_path, save_path)

    # imge = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # retimg = pretreat.fillHole(imge,200)
    #
    # # ret, th = cv2.threshold(imge, 220, 255, cv2.THRESH_BINARY_INV)
    # # cv2.imwrite(r"E:\workdata\spine\temp\test_jpg\net_gray.png", th)
    # plt.figure(num=2, figsize=(20, 20))
    #
    # ax = plt.subplot(211)
    # ax.axis('off')
    # ax.set_title('原始')
    # ax.imshow(imge)
    #
    # ax = plt.subplot(212)
    # ax.axis('off')
    # ax.set_title('填充')
    # ax.imshow(retimg)
    # pylab.show()

    # for file in os.listdir(final_file):
    #     if file == "aaa.jpg":
    #         img_path = os.path.join(final_file, file)
    #         # 经过图像增强
    #         img_original = cv2.imread(img_path,0)
    #
    #         # 直方图均衡化
    #         img_enhance1 = pretreat.probability_to_histogram(img_original)
    #         # Sobel算子
    #         img_enhance2 = pretreat.sobel_filter(img_original)
    #         # 高帽变换
    #         img_enhance3 = pretreat.conbineGray(img_original)
    #         # 高斯噪声
    #         img_enhance4 = pretreat.noiseGauss(img_original,mean=0,sigma=0.01)
    #
    #         # 保存到输出路径
    #         cv2.imwrite(os.path.join(final_file, "aaa1.jpg"), img_enhance1)
    #         cv2.imwrite(os.path.join(final_file, "aaa2.jpg"), img_enhance2)
    #         cv2.imwrite(os.path.join(final_file, "aaa3.jpg"), img_enhance3)
    #         cv2.imwrite(os.path.join(final_file, "aaa4.jpg"), img_enhance4)
