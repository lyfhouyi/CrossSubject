# 从 txt 文件中提取数据，并构造 numpy 数组
import os
import numpy as np


# 从 dirName 文件夹中提取数据数组
# 返回值：数据数组(numpy 四维数组(实验时间顺序 * 3个样本(1 正 2 负)* 通道(16) * 采样点(110)))
def getDataFromTxtFile(dirName):
    ORIGINAL_CHANNELS = 64  # 原始通道数
    VALID_CHANNELS = 16  # 有效通道数
    WINDOW_WIDTH = 110  # 信号窗宽

    # 根据文件名排序
    def mySort(e):
        # return int(e.split('_')[1])
        return len(e)
        # end mySort

    # 从数据文件、时间文件中提取样本数组(numpy 三维数组(3个样本(1 正 2 负)* 通道(16) * 采样点(110)))
    def myGetSample(dataFile, timeFile, target):
        # 提取数据文件中的数据
        dataAll = np.empty((ORIGINAL_CHANNELS, 0))
        with open(dataFile, 'r') as f:
            lineList = []
            for line in f.readlines():
                lineList.append(float(line))
                if len(lineList) == ORIGINAL_CHANNELS:
                    # if max(lineList)==0 and min(lineList)==0:
                    #     print('============')
                    #     break
                    dataAll = np.c_[dataAll, np.array(lineList)]
                    lineList = []

        # 提取样本
        dataAll = dataAll[:VALID_CHANNELS, :]
        sampleArray = np.zeros(
            (3, VALID_CHANNELS,
             WINDOW_WIDTH))  # sampleArray[0]、[1]、[2] 分别为点亮 A、B、C 时的样本
        aCount = 0
        bCount = 0
        cCount = 0
        with open(timeFile, 'r') as f:
            for line in f.readlines():
                if line.split(' ')[0] == '10':
                    # 点亮 A 时的样本
                    sampleArray[
                        0] += dataAll[:,
                                      int((int(line.split(' ')[1]) + 1) /
                                          64):int((int(line.split(' ')[1]) +
                                                   1) / 64 + WINDOW_WIDTH)]
                    aCount += 1
                elif line.split(' ')[0] == '11':
                    # 点亮 B 时的样本
                    sampleArray[
                        1] += dataAll[:,
                                      int((int(line.split(' ')[1]) + 1) /
                                          64):int((int(line.split(' ')[1]) +
                                                   1) / 64 + WINDOW_WIDTH)]
                    bCount += 1
                else:
                    # 点亮 C 时的样本
                    sampleArray[
                        2] += dataAll[:,
                                      int((int(line.split(' ')[1]) + 1) /
                                          64):int((int(line.split(' ')[1]) +
                                                   1) / 64 + WINDOW_WIDTH)]
                    cCount += 1
        sampleArray[0] /= aCount
        sampleArray[1] /= bCount
        sampleArray[2] /= cCount

        # 将靶刺激样本（正样本）换至第一个信号数组
        sampleArray[[0, target], :, :] = sampleArray[[target, 0], :, :]

        print('已处理', dataFile, sampleArray.shape)
        return sampleArray
        # end myGetSample

    ret = np.empty((0, 3, VALID_CHANNELS, WINDOW_WIDTH))

    for _, dirs, _ in os.walk(dirName):

        dirs.sort(key=mySort)
        for dir in dirs:
            absdir = os.path.join(dirName, dir)
            print('正在处理文件夹',absdir)
            for _, _, files in os.walk(absdir):
                dataFiles = [f for f in files if f[0] == 'g']
                timeFiles = [f for f in files if f[0] == 'L']
                dataFiles.sort(key=mySort)
                timeFiles.sort(key=mySort)
                for dataFile, timeFile in zip(dataFiles, timeFiles):
                    sampleArray=myGetSample(os.path.join(absdir, dataFile),
                                os.path.join(absdir, timeFile),
                                (int(dir.split('_')[1]) - 1) % 3)
                    sampleArray=np.expand_dims(sampleArray,axis=0)
                    ret=np.r_[ret,sampleArray]
    return ret
    # end getDataFromTxtFile


def test():
    dirName = r'D:\硕士学习\毕业论文\data\20200914_CWM\singleMask'
    sampleArray=getDataFromTxtFile(dirName)
    print('sampleArray.shape = ',sampleArray.shape)


# test()