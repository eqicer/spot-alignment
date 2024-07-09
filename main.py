"""
# -*- coding:utf-8 -*-
@File : main.py
@Author : guo
@Time : 2023/8/2 16:21
"""
import os
import sys
import threading
import time
import datetime


import cv2 as cv
import numpy as np

from Hex import ascii_to_hex

import serial
import serial.tools.list_ports
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer, QCoreApplication, QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog
from main_ui import Ui_MainWindow

from Image_Processing import otsu, circlefit, draw_cross
import RANSAC

image_count = 0  # 全局变量用于保存图像计数
image_addr = ''  # 全局变量用于存储图像地址

Camera_X = 1280
Camera_Y = 960


class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.cap = None
        self.location_thread = None
        self.running = False
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setFixedSize(self.width(), self.height())
        self.setWindowTitle("水下光斑对准系统")

        self.ser = serial.Serial()

        # 定时检测串口
        self.timer_check_port = QTimer()
        self.timer_check_port.timeout.connect(self.refresh_port)
        self.timer_check_port.start(1)

        self.ui.close_serial_button.setEnabled(False)
        self.ui.open_serial_button.clicked.connect(self.open_serial)
        self.ui.close_serial_button.clicked.connect(self.close_port)

        self.ui.clear_button.clicked.connect(self.clear)

        # 定时器让其定时读取显示图片
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.show_image)
        # 打开摄像头
        self.ui.open_camer_button.clicked.connect(self.open_camera)
        # 关闭摄像头
        self.ui.close_camer_button.clicked.connect(self.close_camera)
        # 采集图像
        self.ui.collect_photo_button.clicked.connect(self.taking_pictures2)
        # 连续采集
        self.ui.continuous_collect_button.clicked.connect(self.continuous_taking_pictures)

        # 定位
        self.ui.location_button.clicked.connect(self.location)
        self.ui.stop_location_button.clicked.connect(self.stop_location)

        # self.ui.open_image.triggered.connect(self.open_image)
        # self.ui.image_process_button.clicked.connect(self.image_process)
        self.ui.image_process_button.clicked.connect(self.image_process)

        self.ui.open_image_button.clicked.connect(self.open_image)

        self.ui.up.clicked.connect(self.up)
        self.ui.down.clicked.connect(self.down)
        self.ui.left.clicked.connect(self.left)
        self.ui.right.clicked.connect(self.right)
        self.ui.pause.clicked.connect(self.pause)

        self.ui.scan_button.clicked.connect(self.scan)
        self.ui.reset_button.clicked.connect(self.reset)

        # 打开一个文件，如果文件不存在会自动创建
        with open(r'data.txt', 'a+', encoding='utf-8') as test:
            # 将文件指针移动到文件末尾
            test.seek(0, 2)
            # 获取当前时间
            now = datetime.datetime.now()
            # 将当前时间格式化为字符串
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            # 将时间字符串写入文件
            test.write('\n' + current_time + '\n')
            # test.truncate(0)  # 清空文件

    # 自动识别并刷新串口
    def refresh_port(self):
        port_list = self.get_port_list()
        num = len(port_list)  # 识别出来的串口数量
        num_last = self.ui.serial_box.count()  # ui下拉框里面的串口数
        if num == 0:
            self.ui.open_serial_button.setEnabled(True)
        if num != num_last:
            self.ui.serial_box.clear()
            self.ui.serial_box.addItems(self.get_port_list())

    def get_port_list(self):
        com_list = []
        port_list = serial.tools.list_ports.comports()  # 识别串口
        for port in port_list:
            # print(port)
            com_list.append(port[0])  # 将串口号加到列表中
        # print(com_list)
        return com_list

    # 打开串口函数
    def open_serial(self):
        try:
            port_name = self.ui.serial_box.currentText()  # 获取串口号
            self.ser = serial.Serial(port=port_name, baudrate=115200, bytesize=8, stopbits=1,
                                     parity='N')
            self.ui.open_serial_button.setEnabled(False)
            self.ui.serial_box.setEnabled(False)
            self.ui.close_serial_button.setEnabled(True)
            self.ui.open_serial_button.setText('打开成功')
        except:
            QMessageBox.critical(self, '错误', '打开串口失败!!!\n请选择正确的串口或该串口被占用!')
            return None

        # 定时接收数据定时器
        self.data_receive_timer = QTimer()
        self.data_receive_timer.timeout.connect(self.data_receive)
        self.data_receive_timer.start(1)

    # 接收数据
    def data_receive(self):
        try:
            num = self.ser.inWaiting()  # 监测接收的字符串长度
            if num > 0:
                recv_data = self.ser.read(num)

                angle = recv_data.decode()[-6:-3]
                angle2 = recv_data.decode()[-3:]
                # temp=int(angle)
                # angle=str(temp)
                # temp2=int(angle2)
                # angle2=str(temp2)
                # print(angle)
                # print(angle2)
                angle = angle.lstrip('0')
                angle2 = angle2.lstrip('0')
                self.ui.x_angle.setText(angle2)
                self.ui.y_angle.setText(angle)

                # 判断是否以16进制显示
                if self.ui.hex_box.isChecked():
                    self.ui.serial_recieve.append(ascii_to_hex(recv_data.decode()))
                else:
                    self.ui.serial_recieve.append(recv_data.decode())

                # angle=recv_data.decode().split()
                # self.ui.serial_recieve.append(recv_data.decode())
                # self.ui.serial_recieve.insertPlainText(recv_data.decode())
                # 光标移到最后
                # textcursor = self.ui.serial_recieve.textCursor()
                # textcursor.movePosition(textcursor.End)
                # self.ui.serial_recieve.setTextCursor(textcursor)
                self.ui.serial_recieve.moveCursor(QTextCursor.End)
            else:
                pass
        except:
            # QMessageBox.critical(self, '串口异常', '串口接收数据异常，请重新连接设备！')

            self.ser.close()
            self.ui.open_serial_button.setEnabled(True)
            self.ui.close_serial_button.setEnabled(False)
            self.ui.serial_box.setEnabled(True)
            self.ui.open_serial_button.setText('打开串口')
            # return None
            pass

    # 关闭串口函数
    def close_port(self):
        try:
            self.ser.close()  # 关闭串口
            self.data_receive_timer.stop()
            self.ui.open_serial_button.setEnabled(True)
            self.ui.close_serial_button.setEnabled(False)
            self.ui.serial_box.setEnabled(True)
            self.ui.open_serial_button.setText('打开串口')
        except:
            QMessageBox.critical(self, '串口异常', '关闭串口失败，请重启程序！')
            return None

    # 打开摄像头
    def open_camera(self):
        self.CAM_NUM = 1  # 0：打开本地摄像头 1：打开外置摄像头
        self.cap = cv.VideoCapture(self.CAM_NUM, cv.CAP_DSHOW)  # 摄像头
        flag = self.cap.open(self.CAM_NUM)
        if flag is False:
            QMessageBox.warning(self, "警告", "该设备未正常连接", QMessageBox.Ok)
        else:
            self.camera_timer.start(40)  # 每40毫秒读取一次，即刷新率为25帧
            self.show_image()
            self.ui.open_camer_button.setEnabled(False)
            self.ui.open_camer_button.setText('打开成功')

    def show_image(self):
        flag, self.image = self.cap.read()  # 从视频流中读取图片
        image_show = cv.resize(self.image, (Camera_X, Camera_Y))  # 把读到的帧的大小重新设置
        # image_show = self.image
        width, height = image_show.shape[:2]  # 行:宽，列:高
        image_show = cv.cvtColor(image_show, cv.COLOR_BGR2RGB)  # opencv读的通道是BGR,要转成RGB
        # image_show = cv.flip(image_show, 1)  # 水平翻转，因为摄像头拍的是镜像的。

        # 在图像中心绘制一个红色的十字标记
        center_x = height // 2  # 960
        center_y = width // 2  # 540
        cv.line(image_show, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 3)
        cv.line(image_show, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 3)

        # 把读取到的视频数据变成QImage形式(图片数据、高、宽、RGB颜色空间，三个通道各有2**8=256种颜色)
        self.showImage = QtGui.QImage(image_show.data, height, width, QImage.Format_RGB888)
        self.ui.camer_show.setPixmap(QPixmap.fromImage(self.showImage))  # 往显示视频的Label里显示QImage
        self.ui.camer_show.setScaledContents(True)  # 图片自适应

        self.ui.image_x_center.setText(str(Camera_X / 2))
        self.ui.image_y_center.setText(str(Camera_Y / 2))

    # 采集图像
    def taking_pictures(self):
        if self.cap is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头", QMessageBox.Ok)
        else:
            global image_count, image_addr
            # 检查 'images' 文件夹是否存在，若不存在则创建
            if not os.path.exists('images'):
                os.makedirs('images')
            if not os.path.exists('images/origin_images'):
                os.makedirs('images/origin_images')
            # FName = fr"images/cap{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
            image_count += 1
            FName = fr"images/origin_images/{image_count:04d}"
            image_addr = FName + '.jpg'
            # print(FName)
            # self.ui.camer_show_2.setPixmap(QtGui.QPixmap.fromImage(self.showImage))
            self.showImage.save(FName + ".jpg", "JPG", 100)
            # self.ui.camer_show_2.setScaledContents(True)  # 图片自适应

    # 只点击采集图像按钮
    def taking_pictures2(self):
        if self.cap is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头", QMessageBox.Ok)
        else:
            global image_count, image_addr
            # 检查 'images' 文件夹是否存在，若不存在则创建
            if not os.path.exists('images'):
                os.makedirs('images')
            if not os.path.exists('images/origin_images'):
                os.makedirs('images/origin_images')
            # FName = fr"images/cap{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
            image_count += 1
            FName = fr"images/origin_images/{image_count:04d}"
            image_addr = FName + '.jpg'
            # print(FName)
            self.ui.camer_show_2.setPixmap(QtGui.QPixmap.fromImage(self.showImage))
            self.showImage.save(FName + ".jpg", "JPG", 100)
            self.ui.camer_show_2.setScaledContents(True)  # 图片自适应

    # 连续采集图像
    def continuous_taking_pictures(self):
        if self.cap is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头", QMessageBox.Ok)
        else:
            text = self.ui.frame_num.text()
            if text == '':
                QMessageBox.warning(self, "警告", "请输入要采集的帧数", QMessageBox.Ok)
            else:
                self.ui.serial_recieve.moveCursor(QTextCursor.End)
                self.ui.serial_recieve.append('正在采集图像')
                self.ui.collect_photo_button.setEnabled(False)
                self.ui.continuous_collect_button.setEnabled(False)
                self.ui.frame_num.setEnabled(False)
                self.ui.continuous_collect_button.setText('正在采集')
                self.taking_thread = threading.Thread(target=self.taking)
                self.taking_thread.start()

    def taking(self):
        text = int(self.ui.frame_num.text())
        if text == '':
            QMessageBox.warning(self, "警告", "请输入要采集的帧数", QMessageBox.Ok)
        while text > 0:
            self.taking_pictures()
            text -= 1
            time.sleep(0.1)
        self.ui.serial_recieve.insertPlainText('\t采集完成')
        self.ui.collect_photo_button.setEnabled(True)
        self.ui.continuous_collect_button.setEnabled(True)
        self.ui.frame_num.setEnabled(True)
        self.ui.continuous_collect_button.setText('连续采集')
        self.ui.serial_recieve.moveCursor(QTextCursor.End)

    # 关闭摄像头
    def close_camera(self):
        if self.cap is None:
            QMessageBox.warning(self, "警告", "摄像头未打开", QMessageBox.Ok)
        # elif self.pross_thread.is_alive():
        #     QMessageBox.warning(self, "警告", "正在处理图像中，请勿关闭！", QMessageBox.Ok)
        #     pass
        else:
            global image_count, image_addr
            image_count = 0
            self.camera_timer.stop()  # 停止读取
            self.cap.release()  # 释放摄像头
            self.cap = None
            image_addr = ''
            self.ui.camer_show.clear()  # 清除label组件上的图片
            self.ui.camer_show.setText('摄像头区域')
            self.ui.camer_show_2.clear()  # 清除label组件上的图片
            self.ui.camer_show_2.setText('图像显示区域')
            self.ui.open_camer_button.setEnabled(True)
            self.ui.open_camer_button.setText('打开摄像头')
            self.ui.image_x_center.clear()
            self.ui.image_y_center.clear()
            # self.cap = cv2.VideoCapture(self.CAM_NUM,cv2.CAP_DSHOW)

    # 打开图片
    def open_image(self):
        global image_addr
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "",
                                                       "*.jpg;;*.png;;*.bmp;;All Files(*)")
        if imgName == '':
            pass
        else:
            image_addr = imgName
            jpg = QtGui.QPixmap(imgName).scaled(self.ui.camer_show_2.width(), self.ui.camer_show_2.height())
            self.ui.camer_show_2.setPixmap(jpg)
            self.ui.camer_show_2.setScaledContents(True)  # 图片自适应

            self.ui.image_path.setText(imgName)

    # 图像处理
    def image_process(self):
        if image_addr == '':
            QMessageBox.warning(self, "警告", "请先导入图像", QMessageBox.Ok)
        else:
            self.ui.image_process_button.setText('正在处理')
            self.ui.image_process_button.setEnabled(False)
            self.ui.collect_photo_button.setEnabled(False)
            self.ui.continuous_collect_button.setEnabled(False)
            self.ui.frame_num.setEnabled(False)
            self.ui.open_image_button.setEnabled(False)

            self.thread = image_process_thread()
            # 连接线程的信号到槽函数
            self.thread.emitcenter.connect(self.show_circle)
            # 启动线程
            self.thread.start()

    def show_circle(self, center):
        jpg = QtGui.QPixmap(fr"./images/circle_result/circle_result_{image_count:04d}.png")
        self.ui.camer_show_2.setPixmap(jpg)  # 显示图片
        self.ui.camer_show_2.setScaledContents(True)  # 图片自适应

        self.ui.image_process_button.setText('图像处理')
        self.ui.image_process_button.setEnabled(True)
        self.ui.collect_photo_button.setEnabled(True)
        self.ui.continuous_collect_button.setEnabled(True)
        self.ui.frame_num.setEnabled(True)
        self.ui.open_image_button.setEnabled(True)

        if 0 <= center[0] <= Camera_X and 0 <= center[1] <= Camera_Y:

            self.ui.x_coordinate.setText(str('{:.4f}'.format(center[0])))
            self.ui.y_coordinate.setText(str('{:.4f}'.format(center[1])))

            errorx = Camera_X / 2 - center[0]
            errory = Camera_Y / 2 - center[1]

            self.ui.x_error.setText(str(('{:.4f}'.format(errorx))))
            self.ui.y_error.setText(str(('{:.4f}'.format(errory))))
            try:
                # 保存字符串到文本文件
                with open('data.txt', 'a') as file:
                    file.write(str(center[0]) + ',')
                    file.write(str(center[1]) + ',')
                    file.write(str(errorx) + ',')
                    file.write(str(errory) + '\n')

            except Exception as e:
                print(e)
        else:
            pass

        # 串口发送中心坐标
        if self.ser.isOpen():
            if 0 <= center[0] <= Camera_X and 0 <= center[1] <= Camera_Y:
                center = [str(center[0]), str(center[1])]
                print(center)
                # self.ser.write(center)
                x_coord = float(center[0])
                y_coord = float(center[1])

                x_coord = str(x_coord).encode()  # 字节流
                y_coord = str(y_coord).encode()

                send_xdata_hex = bytes.fromhex(x_coord.hex())
                send_ydata_hex = bytes.fromhex(y_coord.hex())

                head = '?'
                foot = '['
                split = ','
                head_bytes = head.encode('utf-8')
                foot_bytes = foot.encode('utf-8')
                split_bytes = split.encode('utf-8')
                # 将字节对象转换为十六进制字符串
                hex_head = head_bytes.hex()
                hex_foot = foot_bytes.hex()

                hex_split = split_bytes.hex()
                # 再次将十六进制字符串转换为字节对象
                head_data = bytes.fromhex(hex_head)
                foot_data = bytes.fromhex(hex_foot)
                split_data = bytes.fromhex(hex_split)

                self.ser.write(head_data)
                self.ser.write(send_xdata_hex)
                self.ser.write(split_data)
                self.ser.write(send_ydata_hex)
                self.ser.write(foot_data)
            else:
                pass

    # 定位时采集图像
    def location_taking(self):
        self.running = True
        if self.running is True:
            while True:
                self.taking_pictures()
                self.thread2 = image_process_thread()
                # 连接线程的信号到槽函数
                self.thread2.emitcenter.connect(self.show_circle)
                # 启动线程
                self.thread2.start()

                time.sleep(1)
                if self.running is False:
                    break

    # 自动定位
    def location(self):
        if self.cap is None:
            QMessageBox.warning(self, "警告", "摄像头未打开", QMessageBox.Ok)
        else:
            self.location_thread = threading.Thread(target=self.location_taking)
            self.location_thread.start()

    # 停止定位
    def stop_location(self):
        if self.location_thread and self.location_thread.is_alive():
            self.running = False
        self.ui.camer_show_2.clear()
        self.ui.camer_show_2.setText('图像显示区域')
        # self.location_thread.join()

    def up(self):
        if not self.ser.isOpen():
            QMessageBox.warning(self, "警告", "请打开串口", QMessageBox.Ok)
        else:
            self.ser.write(b'a')

    def down(self):
        if not self.ser.isOpen():
            QMessageBox.warning(self, "警告", "请打开串口", QMessageBox.Ok)
        else:
            self.ser.write(b'b')

    def left(self):
        if not self.ser.isOpen():
            QMessageBox.warning(self, "警告", "请打开串口", QMessageBox.Ok)
        else:
            self.ser.write(b'd')

    def right(self):
        if not self.ser.isOpen():
            QMessageBox.warning(self, "警告", "请打开串口", QMessageBox.Ok)
        else:
            self.ser.write(b'c')

    def pause(self):
        if not self.ser.isOpen():
            QMessageBox.warning(self, "警告", "请打开串口", QMessageBox.Ok)
        else:
            self.ser.write(b'p')

    def scan(self):
        if self.cap is None:
            QMessageBox.warning(self, "警告", "摄像头未打开", QMessageBox.Ok)
        elif not self.ser.isOpen():
            QMessageBox.warning(self, "警告", "请打开串口", QMessageBox.Ok)
        else:
            self.ser.write(b's')

    def reset(self):
        if not self.ser.isOpen():
            QMessageBox.warning(self, "警告", "请打开串口", QMessageBox.Ok)
        else:
            self.ser.write(b'r')

    def clear(self):
        self.ui.serial_recieve.clear()

    # 关闭程序
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '提示', "确认退出吗？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            # 用过sys.exit(0)和sys.exit(app.exec_())，但没起效果
            # os._exit(0)
            sys.exit(0)
        else:
            event.ignore()


# 图像处理线程
class image_process_thread(QThread):
    # 信号发送
    emitcenter = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        img = cv.imread(image_addr,1)  # 读取图片并灰度化处理
        img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_Guassian = cv.GaussianBlur(img_gray, (5, 5), 0.5)  # 高斯滤波
        # otsu_img,threshold_t = otsu(img_Guassian)                       # otsu阈值分割
        ret, otsu_img = cv.threshold(img_Guassian, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # otsu阈值分割
        kernel = np.ones((5, 5))  # 核
        opening = cv.morphologyEx(otsu_img, cv.MORPH_OPEN, kernel)  # 开运算
        edge_img = cv.Canny(opening, 128, 200)  # 边缘检测

        xc, yc, radi = RANSAC.ransac(edge_img)  # RANSAC算法

        # print("Circle center:", xc, yc)
        # print("Radius:", radi)

        # 在原始图像上绘制圆
        img = cv.imread(image_addr)
        cv.circle(img, (int(xc), int(yc)), int(radi), (0, 0, 255), 2)
        draw_cross(img, (int(xc), int(yc)), 15)

        if not os.path.exists('images/circle_result'):
            os.makedirs('images/circle_result')
        cv.imwrite(fr"./images/circle_result/circle_result_{image_count:04d}.png", img)

        self.emitcenter.emit([xc, yc])


if __name__ == '__main__':
    QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    myshow = Main()
    myshow.show()
    sys.exit(app.exec_())
