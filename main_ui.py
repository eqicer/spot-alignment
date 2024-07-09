# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1127, 776)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.title_label = QtWidgets.QLabel(self.centralwidget)
        self.title_label.setGeometry(QtCore.QRect(300, -10, 501, 101))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(28)
        self.title_label.setFont(font)
        self.title_label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setObjectName("title_label")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 60, 1111, 501))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.camer_show = QtWidgets.QLabel(self.groupBox_2)
        self.camer_show.setGeometry(QtCore.QRect(10, 20, 520, 380))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(20)
        self.camer_show.setFont(font)
        self.camer_show.setFrameShape(QtWidgets.QFrame.Box)
        self.camer_show.setAlignment(QtCore.Qt.AlignCenter)
        self.camer_show.setObjectName("camer_show")
        self.camer_show_2 = QtWidgets.QLabel(self.groupBox_2)
        self.camer_show_2.setGeometry(QtCore.QRect(580, 20, 520, 380))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(20)
        self.camer_show_2.setFont(font)
        self.camer_show_2.setFrameShape(QtWidgets.QFrame.Box)
        self.camer_show_2.setAlignment(QtCore.Qt.AlignCenter)
        self.camer_show_2.setObjectName("camer_show_2")
        self.x_coordinate = QtWidgets.QLineEdit(self.groupBox_2)
        self.x_coordinate.setEnabled(True)
        self.x_coordinate.setGeometry(QtCore.QRect(990, 410, 111, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.x_coordinate.setFont(font)
        self.x_coordinate.setText("")
        self.x_coordinate.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.x_coordinate.setAlignment(QtCore.Qt.AlignCenter)
        self.x_coordinate.setReadOnly(True)
        self.x_coordinate.setObjectName("x_coordinate")
        self.open_image_button = QtWidgets.QPushButton(self.groupBox_2)
        self.open_image_button.setGeometry(QtCore.QRect(580, 410, 91, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.open_image_button.setFont(font)
        self.open_image_button.setObjectName("open_image_button")
        self.collect_photo_button = QtWidgets.QPushButton(self.groupBox_2)
        self.collect_photo_button.setGeometry(QtCore.QRect(120, 410, 111, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.collect_photo_button.setFont(font)
        self.collect_photo_button.setObjectName("collect_photo_button")
        self.label_11 = QtWidgets.QLabel(self.groupBox_2)
        self.label_11.setGeometry(QtCore.QRect(880, 406, 141, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.image_process_button = QtWidgets.QPushButton(self.groupBox_2)
        self.image_process_button.setGeometry(QtCore.QRect(760, 410, 91, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.image_process_button.setFont(font)
        self.image_process_button.setObjectName("image_process_button")
        self.close_camer_button = QtWidgets.QPushButton(self.groupBox_2)
        self.close_camer_button.setGeometry(QtCore.QRect(430, 410, 101, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.close_camer_button.setFont(font)
        self.close_camer_button.setObjectName("close_camer_button")
        self.continuous_collect_button = QtWidgets.QPushButton(self.groupBox_2)
        self.continuous_collect_button.setGeometry(QtCore.QRect(250, 410, 91, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.continuous_collect_button.setFont(font)
        self.continuous_collect_button.setObjectName("continuous_collect_button")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(400, 430, 30, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_12 = QtWidgets.QLabel(self.groupBox_2)
        self.label_12.setGeometry(QtCore.QRect(880, 446, 151, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.y_coordinate = QtWidgets.QLineEdit(self.groupBox_2)
        self.y_coordinate.setEnabled(True)
        self.y_coordinate.setGeometry(QtCore.QRect(990, 450, 111, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.y_coordinate.setFont(font)
        self.y_coordinate.setAlignment(QtCore.Qt.AlignCenter)
        self.y_coordinate.setReadOnly(True)
        self.y_coordinate.setObjectName("y_coordinate")
        self.open_camer_button = QtWidgets.QPushButton(self.groupBox_2)
        self.open_camer_button.setGeometry(QtCore.QRect(10, 410, 101, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.open_camer_button.setFont(font)
        self.open_camer_button.setObjectName("open_camer_button")
        self.frame_num = QtWidgets.QLineEdit(self.groupBox_2)
        self.frame_num.setGeometry(QtCore.QRect(340, 430, 61, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.frame_num.setFont(font)
        self.frame_num.setObjectName("frame_num")
        self.image_path = QtWidgets.QLineEdit(self.groupBox_2)
        self.image_path.setEnabled(True)
        self.image_path.setGeometry(QtCore.QRect(580, 465, 271, 25))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.image_path.setFont(font)
        self.image_path.setText("")
        self.image_path.setFrame(False)
        self.image_path.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.image_path.setReadOnly(True)
        self.image_path.setObjectName("image_path")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(470, 560, 651, 211))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.groupBox_3.setObjectName("groupBox_3")
        self.down = QtWidgets.QPushButton(self.groupBox_3)
        self.down.setGeometry(QtCore.QRect(70, 110, 61, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.down.setFont(font)
        self.down.setObjectName("down")
        self.left = QtWidgets.QPushButton(self.groupBox_3)
        self.left.setGeometry(QtCore.QRect(10, 70, 61, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.left.setFont(font)
        self.left.setObjectName("left")
        self.right = QtWidgets.QPushButton(self.groupBox_3)
        self.right.setGeometry(QtCore.QRect(130, 70, 61, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.right.setFont(font)
        self.right.setObjectName("right")
        self.up = QtWidgets.QPushButton(self.groupBox_3)
        self.up.setGeometry(QtCore.QRect(70, 30, 61, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.up.setFont(font)
        self.up.setObjectName("up")
        self.reset_button = QtWidgets.QPushButton(self.groupBox_3)
        self.reset_button.setGeometry(QtCore.QRect(350, 160, 71, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.reset_button.setFont(font)
        self.reset_button.setStyleSheet("background-color: rgb(255, 85, 0);\n"
"background-color: rgb(255, 0, 127);")
        self.reset_button.setObjectName("reset_button")
        self.label_17 = QtWidgets.QLabel(self.groupBox_3)
        self.label_17.setGeometry(QtCore.QRect(220, 10, 151, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.groupBox_3)
        self.label_18.setGeometry(QtCore.QRect(240, 60, 131, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.groupBox_3)
        self.label_19.setGeometry(QtCore.QRect(460, 60, 121, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.groupBox_3)
        self.label_20.setGeometry(QtCore.QRect(440, 10, 141, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.x_error = QtWidgets.QLineEdit(self.groupBox_3)
        self.x_error.setEnabled(True)
        self.x_error.setGeometry(QtCore.QRect(330, 70, 91, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.x_error.setFont(font)
        self.x_error.setText("")
        self.x_error.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.x_error.setAlignment(QtCore.Qt.AlignCenter)
        self.x_error.setReadOnly(True)
        self.x_error.setObjectName("x_error")
        self.y_error = QtWidgets.QLineEdit(self.groupBox_3)
        self.y_error.setEnabled(True)
        self.y_error.setGeometry(QtCore.QRect(550, 70, 81, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.y_error.setFont(font)
        self.y_error.setText("")
        self.y_error.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.y_error.setAlignment(QtCore.Qt.AlignCenter)
        self.y_error.setReadOnly(True)
        self.y_error.setObjectName("y_error")
        self.image_x_center = QtWidgets.QLineEdit(self.groupBox_3)
        self.image_x_center.setEnabled(True)
        self.image_x_center.setGeometry(QtCore.QRect(330, 20, 91, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.image_x_center.setFont(font)
        self.image_x_center.setText("")
        self.image_x_center.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.image_x_center.setAlignment(QtCore.Qt.AlignCenter)
        self.image_x_center.setReadOnly(True)
        self.image_x_center.setObjectName("image_x_center")
        self.image_y_center = QtWidgets.QLineEdit(self.groupBox_3)
        self.image_y_center.setEnabled(True)
        self.image_y_center.setGeometry(QtCore.QRect(550, 20, 81, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.image_y_center.setFont(font)
        self.image_y_center.setText("")
        self.image_y_center.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.image_y_center.setAlignment(QtCore.Qt.AlignCenter)
        self.image_y_center.setReadOnly(True)
        self.image_y_center.setObjectName("image_y_center")
        self.location_button = QtWidgets.QPushButton(self.groupBox_3)
        self.location_button.setGeometry(QtCore.QRect(130, 160, 81, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.location_button.setFont(font)
        self.location_button.setObjectName("location_button")
        self.pause = QtWidgets.QPushButton(self.groupBox_3)
        self.pause.setGeometry(QtCore.QRect(70, 70, 61, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.pause.setFont(font)
        self.pause.setObjectName("pause")
        self.stop_location_button = QtWidgets.QPushButton(self.groupBox_3)
        self.stop_location_button.setGeometry(QtCore.QRect(240, 160, 81, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.stop_location_button.setFont(font)
        self.stop_location_button.setObjectName("stop_location_button")
        self.label_21 = QtWidgets.QLabel(self.groupBox_3)
        self.label_21.setGeometry(QtCore.QRect(480, 110, 51, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.y_angle = QtWidgets.QLineEdit(self.groupBox_3)
        self.y_angle.setEnabled(True)
        self.y_angle.setGeometry(QtCore.QRect(550, 120, 81, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.y_angle.setFont(font)
        self.y_angle.setText("")
        self.y_angle.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.y_angle.setAlignment(QtCore.Qt.AlignCenter)
        self.y_angle.setReadOnly(True)
        self.y_angle.setObjectName("y_angle")
        self.x_angle = QtWidgets.QLineEdit(self.groupBox_3)
        self.x_angle.setEnabled(True)
        self.x_angle.setGeometry(QtCore.QRect(330, 120, 91, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.x_angle.setFont(font)
        self.x_angle.setText("")
        self.x_angle.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.x_angle.setAlignment(QtCore.Qt.AlignCenter)
        self.x_angle.setReadOnly(True)
        self.x_angle.setObjectName("x_angle")
        self.label_22 = QtWidgets.QLabel(self.groupBox_3)
        self.label_22.setGeometry(QtCore.QRect(260, 110, 50, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.groupBox_3)
        self.label_23.setGeometry(QtCore.QRect(630, 110, 41, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(20)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.groupBox_3)
        self.label_24.setGeometry(QtCore.QRect(420, 110, 41, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(20)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.scan_button = QtWidgets.QPushButton(self.groupBox_3)
        self.scan_button.setGeometry(QtCore.QRect(10, 160, 81, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.scan_button.setFont(font)
        self.scan_button.setObjectName("scan_button")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 560, 451, 211))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.close_serial_button = QtWidgets.QPushButton(self.groupBox)
        self.close_serial_button.setGeometry(QtCore.QRect(10, 110, 81, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.close_serial_button.setFont(font)
        self.close_serial_button.setObjectName("close_serial_button")
        self.serial_box = QtWidgets.QComboBox(self.groupBox)
        self.serial_box.setGeometry(QtCore.QRect(10, 20, 81, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.serial_box.setFont(font)
        self.serial_box.setObjectName("serial_box")
        self.open_serial_button = QtWidgets.QPushButton(self.groupBox)
        self.open_serial_button.setGeometry(QtCore.QRect(10, 60, 81, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.open_serial_button.setFont(font)
        self.open_serial_button.setObjectName("open_serial_button")
        self.serial_recieve = QtWidgets.QTextEdit(self.groupBox)
        self.serial_recieve.setGeometry(QtCore.QRect(100, 20, 341, 161))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.serial_recieve.setFont(font)
        self.serial_recieve.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        self.serial_recieve.setReadOnly(True)
        self.serial_recieve.setObjectName("serial_recieve")
        self.clear_button = QtWidgets.QPushButton(self.groupBox)
        self.clear_button.setGeometry(QtCore.QRect(10, 160, 81, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.clear_button.setFont(font)
        self.clear_button.setObjectName("clear_button")
        self.hex_box = QtWidgets.QCheckBox(self.groupBox)
        self.hex_box.setGeometry(QtCore.QRect(110, 180, 131, 31))
        self.hex_box.setObjectName("hex_box")
        MainWindow.setCentralWidget(self.centralwidget)
        self.open_image = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.open_image.setFont(font)
        self.open_image.setObjectName("open_image")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.title_label.setText(_translate("MainWindow", "水下无线激光对准系统"))
        self.groupBox_2.setTitle(_translate("MainWindow", "摄像头"))
        self.camer_show.setText(_translate("MainWindow", "摄像头"))
        self.camer_show_2.setText(_translate("MainWindow", "图像显示"))
        self.open_image_button.setText(_translate("MainWindow", "打开图像"))
        self.collect_photo_button.setText(_translate("MainWindow", "采集光斑图像"))
        self.label_11.setText(_translate("MainWindow", "光斑中心X坐标"))
        self.image_process_button.setText(_translate("MainWindow", "图像处理"))
        self.close_camer_button.setText(_translate("MainWindow", "关闭摄像头"))
        self.continuous_collect_button.setText(_translate("MainWindow", "连续采集"))
        self.label.setText(_translate("MainWindow", "帧"))
        self.label_12.setText(_translate("MainWindow", "光斑中心Y坐标"))
        self.open_camer_button.setText(_translate("MainWindow", "打开摄像头"))
        self.groupBox_3.setTitle(_translate("MainWindow", "云台控制"))
        self.down.setText(_translate("MainWindow", "下"))
        self.left.setText(_translate("MainWindow", "左"))
        self.right.setText(_translate("MainWindow", "右"))
        self.up.setText(_translate("MainWindow", "上"))
        self.reset_button.setText(_translate("MainWindow", "复位"))
        self.label_17.setText(_translate("MainWindow", "相机中心X坐标"))
        self.label_18.setText(_translate("MainWindow", "方位脱靶量"))
        self.label_19.setText(_translate("MainWindow", "俯仰脱靶量"))
        self.label_20.setText(_translate("MainWindow", "相机中心Y坐标"))
        self.location_button.setText(_translate("MainWindow", "自动对准"))
        self.pause.setText(_translate("MainWindow", "暂停"))
        self.stop_location_button.setText(_translate("MainWindow", "停止对准"))
        self.label_21.setText(_translate("MainWindow", "俯仰角"))
        self.label_22.setText(_translate("MainWindow", "方位角"))
        self.label_23.setText(_translate("MainWindow", "°"))
        self.label_24.setText(_translate("MainWindow", "°"))
        self.scan_button.setText(_translate("MainWindow", "开始扫描"))
        self.groupBox.setTitle(_translate("MainWindow", "串口"))
        self.close_serial_button.setText(_translate("MainWindow", "关闭串口"))
        self.open_serial_button.setText(_translate("MainWindow", "打开串口"))
        self.clear_button.setText(_translate("MainWindow", "清除"))
        self.hex_box.setText(_translate("MainWindow", "16进制显示"))
        self.open_image.setText(_translate("MainWindow", "打开图像"))