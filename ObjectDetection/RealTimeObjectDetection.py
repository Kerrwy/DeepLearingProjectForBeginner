# Write a real time object detection using deeplearning and opencv
# 基于Opencv深度学习的目标检测器，图片从本地导入

# USAGE: In Pycharm Terminal, you can input follow:
# python RealTimeObjectDetection.py
# --prototxt MobileNetSSD_deploy.prtototxt.txt
# --model MobileNetSSD_deploy.caffemodel
# --image  .\DADA\ILSVRC2017_test_00005476.JPEG


# 1. 导入包
from imutils.video import VideoStream  # VideoStream用来读取本地摄像头视频流
from imutils.video import FPS  # VideoStream用来计算本地摄像头视频帧数
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# 2. 构建参数解析器，解析参数
ap = argparse.ArgumentParser()  # ArgumentParser: The main entry point for command-line parsing

# ap.add_argument("-i","--image", required=True,
#                 help="path to input image")         # 输入图片的路径
# ap.add_argument("-p","--prototxt", required=True,
#                 help="path to Caffe 'deploy' prototxt file")  # caffe prototxt文件路径
# ap.add_argument("-m", "--model", required=True,
#                 help="path to Caffe pre-train model")         # 预训练模型路径
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")  # --confidence：过滤弱检测的最小概率阈值，默认值为 20%。

# the parse_args() method is invoked to convert the args at thecommand-line into an object with attributes.
args = vars(ap.parse_args())     # vars() 函数返回对象object的属性和属性值的字典对象。

# ADDDDDDD 我直接在这里加入参数运行
path = r"F:\python offer works\Project"
args["prototxt"] = os.path.join(path, 'MobileNetSSD_deploy.prototxt.txt')
args["model"] = os.path.join(path, 'MobileNetSSD_deploy.caffemodel')
# args["image"] = os.path.join(path, 'DATA\Car4\img\\0001.jpg')

# 3. 初始化类标签的列表，MobileNet SSD被训练，然后生成每一个类别的一系列边界框
CLASS = ["background", "aeroplane", "bicycle", "bird", "boat", " bottle",
         "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
         "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASS),3))   # numpy.random.uniform(low,high,size)均匀分布


# 4. 导入模型
print("[INFO] loading model....")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# ADDDDDDD 这里，我加入多帧以视频的形式检测
for r, pt, dirs  in os.walk(os.path.join(path, 'DATA\Basketball\img')):
    for dr in dirs:
        args["image"] = os.path.join(r, dr)


        # 5. 加载图像。准备blob
        image = cv2.imread(args["image"])
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 0.007843, (300,300), 127.5)
        '''
        Blob是一个四维的数组，用于存储数据，包括输入数据、输出数据、权值等；
        Blob是Caffe中处理和传递实际数据的数据封装包，并且在CPU与GPU之间具有同步处理能力。
        从数学意义上说，blob是按C风格连续存储的N维数组。
        '''
        # 6. 输入blob到网络中，获取检测结果
        net.setInput(blob)
        detections = net.forward()  # shape:(1,1,3,7), 第三维是框的个数。第四维是每一个框的类别(1)、置信度(2)，坐标(3:7)

        # 7. 对检测的目标查看置信度，绘制边界框和显示标签
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # 一个图像检测了多少目标
            if confidence > args["confidence"]:
                # 提取类标签的索引
                idx = int(detections[0, 0, i ,1])
                # 计算边界框的x,y坐标
                box = detections[0, 0, i, 3:7]*np.array([w,h,w,h])   # detections[0, 0, i, 3:7]在0到1之间
                (startX, startY, endX, endY) = box.astype("int")

                # 将预测结果在帧上显示
                label = "{}:{:.2f}%".format(CLASS[idx],
                                            confidence*100)
                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY-15 if startY-15>15 else startY+15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # 8. 显示帧，更新fps计数器
        cv2.imshow("Frame", image)
        key = cv2.waitKey(25)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break