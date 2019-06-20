import cv2
import numpy as np
import glob
import os
import types
import shutil
from matplotlib import pyplot as plt

PATH = 'C:/Users/cfq/Desktop/algorithms/faces/'            #存放图片的绝对路径

def GetPictrueMatrix():
    Matrix = np.zeros((10000, 1))
    paths = glob.glob(os.path.join(PATH, '*.png'))          #得到全部的png图片
    for filename in paths:
        img = cv2.imread(filename, 0)           #读取灰度图象
        img = cv2.resize(img, (100, 100))       #将图片像素调成100x100
        img = img.reshape(10000, 1)
        Matrix = np.hstack((Matrix, img))       #矩阵横向拼接

    Matrix = np.delete(Matrix, 0, axis=1)       #删去第一列全零元素
    return Matrix           #10000x1899


def PCA(Matrix):
    Covariance = np.cov(Matrix)         #得到协方差矩阵
    Eigenvalues, Eigenvector = np.linalg.eig(Covariance)     #得到特征向量
    sorted_Eigenvalues = np.argsort(Eigenvalues)  #特征值排序
    top_eig = Eigenvector[:, sorted_Eigenvalues[:-51:-1]]         #根据特征值得到前50个特征向量
    top_eig = top_eig.T         #50x10000
    return top_eig


def SampleProjectoin(top_eig):
    e1 = cv2.imread('C:/Users/cfq/Desktop/algorithms/typical_samples/1.png', 0)        #读入三个测试文件
    e1 = cv2.resize(e1, (100, 100))
    e1 = e1.reshape(10000, 1)
    e2 = cv2.imread('C:/Users/cfq/Desktop/algorithms/typical_samples/2.png', 0)
    e2 = cv2.resize(e2, (100, 100))
    e2 = e2.reshape(10000, 1)
    e3 = cv2.imread('C:/Users/cfq/Desktop/algorithms/typical_samples/3.png', 0)
    e3 = cv2.resize(e3, (100, 100))
    e3 = e3.reshape(10000, 1)

    s1 = np.dot(top_eig, e1)            #三个测试文件的矩阵x特征矩阵 50x10000
    s2 = np.dot(top_eig, e2)
    s3 = np.dot(top_eig, e3)

    return s1, s2, s3


def FaceId(top_eig, s1, s2, s3):
    paths = glob.glob(os.path.join(PATH, '*.png'))  # 得到全部的png图片
    for filename in paths:          #依次读取每个距离
        img = cv2.imread(filename, 0)  # 读取灰度图象
        img = cv2.resize(img, (100, 100))  # 将图片像素调成100x100
        img = img.reshape(10000, 1)
        img = np.dot(top_eig, img)
        dist1 = np.linalg.norm(s1 - img)    #计算要分类图片与三个样例图片的欧式距离
        dist2 = np.linalg.norm(s2 - img)
        dist3 = np.linalg.norm(s3 - img)
        if dist1<dist2 and dist1<dist3:         #距离最小则进行分类
            shutil.copy(filename, 'C:/Users/cfq/Desktop/algorithms/pattern1/')
        if dist2<dist1 and dist2<dist3:
            shutil.copy(filename, 'C:/Users/cfq/Desktop/algorithms/pattern2/')
        if dist3<dist1 and dist3<dist2:
            shutil.copy(filename, 'C:/Users/cfq/Desktop/algorithms/pattern3/')



if __name__ == '__main__':
    Matrix = GetPictrueMatrix()
    top_eig = PCA(Matrix)
    s1, s2, s3 = SampleProjectoin(top_eig)
    FaceId(top_eig, s1, s2, s3)

