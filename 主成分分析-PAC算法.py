# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def meanX(dataX):
    return np.mean(dataX, axis=0)


def pca(XMat, k):
    average = meanX(XMat)
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)  # 计算协方差矩阵
    # print covX
    featValue, featVec = np.linalg.eig(covX)  # 求解协方差矩阵的特征值和特征向量
    # print featValue, featVec
    index = np.argsort(-featValue)  # 依照featValue进行从大到小排序
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        # 注意特征向量时列向量。而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]])  # 所以这里须要进行转置
        print(selectVec)
        print(selectVec.T)
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData


def loaddata(datafile):
    return np.array(pd.read_csv(datafile, sep="\t", header=-1)).astype(np.float)


def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i, 0])
        axis_y1.append(dataArr1[i, 1])
        axis_x2.append(dataArr2[i, 0])
        axis_y2.append(dataArr2[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    # plt.savefig("outfile.png")
    plt.show()


def main():
    datafile = "PACdata.txt"
    XMat = loaddata(datafile)
    k = 2
    return pca(XMat, k)


if __name__ == "__main__":
    finalData, reconMat = main()
    plotBestFit(finalData, reconMat)
    # print(finalData)
