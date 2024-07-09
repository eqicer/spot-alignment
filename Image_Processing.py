"""
# -*- coding:utf-8 -*-
@File : Image_Processing.py
@Author : guo
@Time : 2024/1/5 13:07
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 大津阈值分割算法   返回分割后的图像和分割阈值
def otsu(gray_img):
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    N = h * w
    threshold_t = 0
    max_g = 0

    # 遍历每一个灰度级
    for t in range(256):
        # 使用numpy直接对数组进行运算
        n0 = gray_img[np.where(gray_img < t)]
        n1 = gray_img[np.where(gray_img >= t)]
        w0 = len(n0) / N
        w1 = len(n1) / N
        u0 = np.mean(n0) if len(n0) > 0 else 0.
        u1 = np.mean(n1) if len(n1) > 0 else 0.

        g = w0 * w1 * (u0 - u1) ** 2
        if g > max_g:
            max_g = g
            threshold_t = t
    # print('类间方差最大阈值：', threshold_t)
    gray_img[gray_img < threshold_t] = 0
    gray_img[gray_img >= threshold_t] = 255
    return gray_img,threshold_t

def circlefit(im):
    coords = np.nonzero(im)  # 查找非零像素的坐标
    n = len(coords[0])  # 非零像素数量

    # 获取坐标数组
    x = coords[0]
    y = coords[1]
    # 计算坐标平方和
    xx = x**2
    yy = y**2
    xy = x * y

    # 构建线性方程组的系数矩阵
    A = np.array([[np.sum(x), np.sum(y), n],
                  [np.sum(xy), np.sum(yy), np.sum(y)],
                  [np.sum(xx), np.sum(xy), np.sum(x)]])
    B = np.array([-np.sum(xx + yy), -np.sum(xx * y + yy * y), -np.sum(xx * x + xy * y)])

    # 求解线性方程组
    a = np.linalg.solve(A, B)

    xc = -0.5 * a[0]
    yc = -0.5 * a[1]
    R = np.sqrt(xc**2 + yc**2 - a[2])

    center = [xc, yc]
    return center, R

def draw_cross(image, center_coordinates, cross_radius):
    """
    在指定位置绘制十字
    :param image: 传入的需要绘制十字的图片
    :param center_coordinates: 十字的中心坐标 (x, y)
    :param cross_radius: 十字臂的长度
    :return: 返回绘制好十字的图像
    """
    center_x, center_y = center_coordinates
    image = cv.line(image, (center_x - cross_radius, center_y), (center_x + cross_radius, center_y), (0, 0, 255), 3)
    image = cv.line(image, (center_x, center_y - cross_radius), (center_x, center_y + cross_radius), (0, 0, 255), 3)
    return image

if __name__ == '__main__':
    path = 'D:\\D:\\Desktop\\FINNGER\\ggg\\wu0001.bmp'
    img_gray = cv.imread(path, 0)

    # img_Guassian = cv.GaussianBlur(img_gray, (5, 5), 0.5)
    # otsu_img, threshold_t = otsu(img_Guassian)
    # print(threshold_t)
    # kernel = np.ones((5, 5))  # 核
    # opening = cv.morphologyEx(otsu_img, cv.MORPH_OPEN, kernel)  # 开运算
    # edge_img = cv.Canny(opening, 128, 200)

