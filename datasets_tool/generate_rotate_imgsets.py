from math import *
import cv2
import os
import glob
import numpy as np

# https://doc.itprojects.cn/0007.zhishi.raspberrypi/02.doc/index.html#/c02.createimg


def rotate_img(img, angle):
    '''
    img   --image
    angle --rotation angle
    return--rotated img
    '''
    h, w = img.shape[:2]
    rotate_center = (w / 2, h / 2)
    # 获取旋转矩阵
    # 参数1为旋转中心点;
    # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
    # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    # 计算图像新边界
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    # 调整旋转矩阵以考虑平移
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated_img


if __name__ == '__main__':

    output_dir = "result"
    image_names = glob.glob("source/b-zu.png")

    for image_name in image_names:
        image = cv2.imread(image_name, -1)
        for i in range(1, 361):
            rotated_img1 = rotate_img(image, i)
            basename = os.path.basename(image_name)
            tag, _ = os.path.splitext(basename)
            cv2.imwrite(os.path.join(output_dir, 'b-zu-%d.jpg' % i), rotated_img1)
