import numpy as np
import cv2 as cv
import math
from numpy.linalg import det

def ransac(edge_spot, k=500, r_error_th=4):
    y_edge, x_edge = np.where(edge_spot)  # Coordinates of edge points
    n_points = len(y_edge)  # Number of edge points
    # Step 1: Estimate initial parameters
    metric = np.zeros((k, 5))  # Initialize matrix to store circle fitting metrics
    for i in range(k):
        sample_indices = np.random.choice(n_points, 3, replace=False)  # Randomly select 3 points
        sample_points = np.array([[x_edge[idx], y_edge[idx]] for idx in sample_indices])
        try:
            pc, r = points2circle(sample_points[0], sample_points[1], sample_points[2])
            # print(pc)
            distance_center_edge = np.sqrt((x_edge - pc[0]) ** 2 + (y_edge - pc[1]) ** 2)
            # print(distance_center_edge)
            r_residual = np.abs(distance_center_edge - r)
            # print(r_residual)
            count = np.sum(r_residual <= r_error_th)
            accuracy = count / n_points
            var = np.sqrt(np.sum(r_residual ** 2) / (n_points - 1))
            # print(var)
            metric[i, :] = [r, pc[0], pc[1], accuracy, var]
        except:
            continue

    # Step 2: Denoise
    accuracy = metric[:, 3]
    row_accuracy_max = np.argmax(accuracy)
    r0 = metric[row_accuracy_max, 0]
    sigma_r = 5 * np.sqrt(2)
    x0, y0 = metric[row_accuracy_max, 1:3]
    mask_signal = np.abs(np.sqrt((x_edge - x0) ** 2 + (y_edge - y0) ** 2) - r0) <= 3 * sigma_r
    x_new_edge_points = x_edge[mask_signal]
    y_new_edge_points = y_edge[mask_signal]

    # print(x_new_edge_points)
    # print(y_new_edge_points)
    # Step 3: Circle fitting using least squares regression
    para = circle_fitting(np.column_stack((x_new_edge_points, y_new_edge_points)))
    xc, yc, radi = para[0][0], para[0][1], para[1]

    return xc, yc, radi

def circle_fitting(points):
    assert len(points) >= 3

    XiSum = YiSum = Xi2Sum = Yi2Sum = Xi3Sum = Yi3Sum = XiYiSum = Xi2YiSum = XiYi2Sum = WiSum = 0

    for p in points:
        Xi, Yi = p
        XiSum += Xi
        YiSum += Yi
        Xi2Sum += Xi * Xi
        Yi2Sum += Yi * Yi
        Xi3Sum += Xi * Xi * Xi
        Yi3Sum += Yi * Yi * Yi
        XiYiSum += Xi * Yi
        Xi2YiSum += Xi * Xi * Yi
        XiYi2Sum += Xi * Yi * Yi
        WiSum += 1

    A = np.zeros((3, 3), dtype=np.float64)
    B = np.zeros((3, 1), dtype=np.float64)

    A[0, 0] = Xi2Sum
    A[0, 1] = XiYiSum
    A[0, 2] = XiSum

    A[1, 0] = XiYiSum
    A[1, 1] = Yi2Sum
    A[1, 2] = YiSum

    A[2, 0] = XiSum
    A[2, 1] = YiSum
    A[2, 2] = WiSum

    B[0, 0] = -(Xi3Sum + XiYi2Sum)
    B[1, 0] = -(Xi2YiSum + Yi3Sum)
    B[2, 0] = -(Xi2Sum + Yi2Sum)

    X = np.linalg.solve(A, B)
    a, b, c = X[:, 0]

    center = (-0.5 * a, -0.5 * b)
    radius = 0.5 * math.sqrt(a * a + b * b - 4 * c)

    return center, radius

# 三点拟合圆
def points2circle(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    num1 = len(p1)
    num2 = len(p2)
    num3 = len(p3)

    # 输入检查
    if (num1 == num2) and (num2 == num3):
        if num1 == 2:
            p1 = np.append(p1, 0)
            p2 = np.append(p2, 0)
            p3 = np.append(p3, 0)
        elif num1 != 3:
            print('\t仅支持二维或三维坐标输入')
            return None
    else:
        print('\t输入坐标的维数不一致')
        return None

    # 共线检查
    temp01 = p1 - p2
    temp02 = p3 - p2
    temp03 = np.cross(temp01, temp02)
    # 计算两个向量（向量数组）的叉乘。叉乘返回的数组既垂直于a，又垂直于b。
    # 如果a,b是向量数组，则向量在最后一维定义。该维度可以为2，也可以为3. 为2的时候会自动将第三个分量视作0补充进去计算。
    temp = (temp03 @ temp03) / (temp01 @ temp01) / (temp02 @ temp02)  # @装饰器的格式来写的目的就是为了书写简单方便
    # temp03 @ temp03中的@ 含义是数组中每个元素的平方之和
    if temp < 10 ** -6:
        pass
        # print('\t三点共线, 无法确定圆')
        return None

    temp1 = np.vstack((p1, p2, p3))  # 行拼接
    temp2 = np.ones(3).reshape(3, 1)  # 以a行b列的数组形式显示
    mat1 = np.hstack((temp1, temp2))  # size = 3x4

    m = +det(mat1[:, 1:])
    n = -det(np.delete(mat1, 1, axis=1))  # axis=1相对于把每一行当做列来排列
    p = +det(np.delete(mat1, 2, axis=1))
    q = -det(temp1)

    temp3 = np.array([p1 @ p1, p2 @ p2, p3 @ p3]).reshape(3, 1)
    temp4 = np.hstack((temp3, mat1))
    # 使用 stack，可以将一个列表转换为一个numpy数组，当axis=0的时候，和 使用 np.array() 没有什么区别，
    # 但是当 axis=1的时候，那么就是对每一行进行在列方向上进行运算，也就是列方向结合，
    # 此时矩阵的维度也从（2,3）变成了（3,2）
    # hstack(tup) ，参数tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组
    temp5 = np.array([2 * q, -m, -n, -p, 0])
    mat2 = np.vstack((temp4, temp5))  # size = 4x5

    A = +det(mat2[:, 1:])
    B = -det(np.delete(mat2, 1, axis=1))
    C = +det(np.delete(mat2, 2, axis=1))
    D = -det(np.delete(mat2, 3, axis=1))
    E = +det(mat2[:, :-1])

    pc = -np.array([B, C, D]) / 2 / A
    r = np.sqrt(B * B + C * C + D * D - 4 * A * E) / 2 / abs(A)

    return pc, r

if __name__ == '__main__':

    # image_addr = 'D:\\Desktop\\FINNGER\\ggg\\you0001.bmp'
    image_addr = 'D:\\Desktop\\FINNGER\\ggg\\wu0001.bmp'
    img_gray = cv.imread(image_addr, 0)  # 读取图片并灰度化处理
    img_Guassian = cv.GaussianBlur(img_gray, (5, 5), 0.5)  # 高斯滤波
    # otsu_img,threshold_t = otsu(img_Guassian)
    # otsu阈值分割
    ret, otsu_img = cv.threshold(img_Guassian, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = np.ones((5, 5))  # 核
    opening = cv.morphologyEx(otsu_img, cv.MORPH_OPEN, kernel)  # 开运算
    edge_img = cv.Canny(opening, 128, 200)  # 边缘检测

    # contours, _ = cv.findContours(edge_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # cv.imshow("1",edge_img)
    # cv.waitKey(0)
    # np.set_printoptions(threshold=np.inf)
    # print(1 in otsu_img)
    # print(otsu_img)
    xc, yc, radi = ransac(edge_img)
    print("Circle center:", xc, yc)
    print("Radius:", radi)


    # 在原始图像上绘制圆
    output_img = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)  # 转换为彩色图像
    cv.circle(output_img,  (int(xc), int(yc)),  int(radi), (0, 255, 0), 2)

    # 显示结果
    cv.imshow("Circle", output_img)
    cv.waitKey(0)
    cv.destroyAllWindows()