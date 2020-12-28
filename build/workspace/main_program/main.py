# -*- coding: utf-8 -*-
# @Time : 2020/12/27 19:27
# @Author :荆东东
# @Site : 河南科技大学
# @File : 1.py
# @Software: PyCharm

import os
import subprocess
import sys

import cv2
from PyQt5.QtCore import QTimer, QCoreApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame

# ==================== import neural network =================================
from build.workspace.neural_network.classification23_taichi_eigenvalue import train_net
from build.workspace.neural_network.data_process import pointDistance, pointAngle
from build.workspace.neural_network.predict_eigenvalue import predict_result

sys.path.append('../neural_network/')
print(sys.path)
try:
    from classification23_taichi_eigenvalue import *
    from mainWindow import *
    from data_process import *
    from predict_eigenvalue import *
except ImportError as e:
    raise e

# ====================import openpose=========================================
dir_path = os.path.dirname(os.path.realpath(__file__))
#print(dir_path)
try:
    # 一定要注意是 build目录下的python而不是openpose根目录下的
    # 如果一直报错可以将绝对路径加入 path环境变量中去。
    # 或者将绝对路径引进来 F:\\OPENPOSE\\openpose\\build\\python\\openpose\\Release
    # 或是如下添加绝对路径
    # sys.path.append('/home/wfnian/OPENPOSE/openpose/build/python')
    # sys.path.append('D:/openpose-master/build/python/openpose/Release');
    sys.path.append(dir_path + '/../../python/openpose/Release');
    os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
    import pyopenpose as op
    # from openpose import pyopenpose as op
    # 此句和上句同理 两者只要一者起效便可


except ImportError as e:
    print('Did you enable `BUILD_PYTHON`')
    raise e
# =============================参数args 设置====================================
path = 'D:/openpose-master/build/workspace/main_program/1.jpg'
# 详细参考flags.hpp 文件
params = dict()
# params["model_folder"] = "/home/wfnian/OPENPOSE/openpose/models"
params["model_folder"] = "D:/openpose-master/models"
# 根据自己的实际情况选择 model路径
params["number_people_max"] = 1  # 只检测一个人
params["camera_resolution"] = "640x480"   # 相机分辨率
params["render_threshold"] = 0.01
# ============================= 启动openPose ==================================
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

pos = ["预备势", "起势", "左右野马分鬃", "白鹤亮翅", "左右搂膝拗步", "手挥琵琶",
       "左右倒卷肱", "左揽雀尾", "右拦雀尾", "单鞭", "云手", "高探马", "右蹬脚",
       "双峰贯耳", "转身左蹬脚", "左下式独立", "右下式独立", "左右穿梭", "海底针",
       "闪通臂", "转身搬拦捶", "如封似闭", "十字手", "太极拳"]


class Video:
    def __init__(self, capture):
        self.capture = capture
        self.currentFrame = None
        self.previousFrame = None

    def captureFrame(self):
        """
        capture frame and return captured frame
        """
        ret, readFrame = self.capture.read()
        return readFrame

    def captureNextFrame(self):
        """
        capture frame and reverse RBG BGR and return opencv image
        """
        ret, readFrame = self.capture.read()
        if ret:
            self.currentFrame = cv2.cvtColor(readFrame, cv2.COLOR_BGR2RGB)

    def convertFrame(self):
        # converts frame to format suitable for QtGui
        try:
            height, width = self.currentFrame.shape[:2]
            img = QImage(self.currentFrame, width, height, QtGui.QImage.Format_RGB888)
            img = QPixmap.fromImage(img)
            self.previousFrame = self.currentFrame
            return img
        except cv2.Error:
            return None


def connectTensorboard():
    import webbrowser

    url = "http://localhost:6006"
    try:
        # from subprocess import call
        # subprocess.Popen("tensorboard --logdir /home/wfnian/project/posture_recognition/workspace/neural_network/runs/")
        # call("tensorboard --logdir /home/wfnian/project/posture_recognition/workspace/neural_network/runs/")
        # tensorboard --logdir /home/wfnian/project/posture_recognition/workspace/neural_network/runs/
        webbrowser.get('chrome').open_new_tab(url)
    except Exception as e:
        webbrowser.open_new_tab(url)


class mWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mWindow, self).__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.train_network)

        self._timer1 = QTimer(self)
        # self._timer1.timeout.connect(self.setProcessBarValue)
        self._timer1.start(30)  # 每隔多长时间

        self._timer2 = QTimer(self)
        self._timer2.timeout.connect(self.showCapture)
        # self.video = Video(cv2.VideoCapture(0))
        self._timer2.start(10)  # 每隔多长时间

        self.pushButton_3.clicked.connect(connectTensorboard)
        self.pushButton_4.clicked.connect(connectTensorboard)

        # ================ 关闭窗口的美化 =========================
        self.pushButton_5.setFixedSize(15, 15)
        self.pushButton_6.setFixedSize(15, 15)
        self.pushButton_7.setFixedSize(15, 15)
        self.pushButton_5.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')
        self.pushButton_6.setStyleSheet(
            '''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.pushButton_7.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.pushButton_7.clicked.connect(QCoreApplication.instance().quit)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

    def train_network(self):

        train_net()
        self.label_2.setPixmap(QPixmap("../sundry/train_loss_acc_eigenvalue.png"))

    def showCapture(self):
        try:
            frame = self.video.captureFrame()
            # path = 'D:/openpose-master/build/workspace/main_program/1.jpg'
            #imageToProcess = cv2.imread(path)
            print("aaa")
            datum = op.Datum()
            print("bbb")
            datum.cvInputData = frame
            #datum.cvInputData = imageToProcess
            print("ccc")
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            print("ddd")
            resPic = datum.cvOutputData
            # cv2.putText(resPic, "OpenPose", (25, 25),
            #             cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 222))
            print("eee")
            pic = cv2.cvtColor(resPic, cv2.COLOR_BGR2RGB)  # 通道转化
            print("fff")
            #加了两个QtGui
            pic = QtGui.QImage(pic, pic.shape[1], pic.shape[0], QtGui.QImage.Format_RGB888)   # 将图片转化成Qt可读格式
            print("ggg")
            self.label_3.setPixmap(QtGui.QPixmap.fromImage(pic))  # 加载图片,并自定义图片展示尺寸,显示图片
            print("hhh")
            keyPoints = datum.poseKeypoints.tolist()
            print(keyPoints[0])
            print("iii")
            self.label_4.setText(pos[predict_result(pointDistance(keyPoints[0]) +
                                                    pointAngle(keyPoints[0]))])

        except TypeError:
            print("No frame")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywin = mWindow()
    mywin.setStyleSheet("#MainWindow{border-image:url(../sundry/back5.png);}")
    mywin.show()
    sys.exit(app.exec_())
