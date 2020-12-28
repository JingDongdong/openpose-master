# -*- coding: utf-8 -*-
# @Time : 2020/12/27 19:27
# @Author :荆东东
# @Site : 河南科技大学
# @File : 1.py
# @Software: PyCharm

import sys
import cv2
import os
from sys import platform
import time

# Import Openpose (Windows)
from build.workspace.neural_network.data_process import pointDistance, pointAngle
from build.workspace.neural_network.predict_eigenvalue import predict_result

dir_path = os.path.dirname(os.path.realpath(__file__))

# Change these variables to point to the correct folder (Release/x64 etc.)
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
    print("13")
    # from openpose import pyopenpose as op
    # 此句和上句同理 两者只要一者起效便可
except ImportError as e:
    print('Did you enable `BUILD_PYTHON`')
    raise e

path = 'D:/openpose-master/build/workspace/main_program/4.jpg'
# path ='D:/openpose-master/build/workspace/dataset/taichi/marked_pic/p_175_22.jpg'
# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "D:/openpose-master/models"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

pos = ["预备势", "起势", "左右野马分鬃", "白鹤亮翅", "左右搂膝拗步", "手挥琵琶",
       "左右倒卷肱", "左揽雀尾", "右拦雀尾", "单鞭", "云手", "高探马", "右蹬脚",
       "双峰贯耳", "转身左蹬脚", "左下式独立", "右下式独立", "左右穿梭", "海底针",
       "闪通臂", "转身搬拦捶", "如封似闭", "十字手", "太极拳"]
# Read image and face rectangle locations
imageToProcess = cv2.imread(path)

# Create new datum
datum = op.Datum()
datum.cvInputData = imageToProcess

# Process and display image
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
keyPoints = datum.poseKeypoints.tolist()
# print("123\n")
# print(keyPoints[0])
# print("Pose keypoints: \n" + str(datum.poseKeypoints.tolist()))
print(pos[predict_result(pointDistance(keyPoints[0]) + pointAngle(keyPoints[0]))])
cv2.imshow("OpenPose 1.7.1 - Tutorial Python API", datum.cvOutputData)
cv2.waitKey(0)