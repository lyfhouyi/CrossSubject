import numpy as np
import scipy.signal


# 信号分帧
# 必选参数：时域信号,帧长，帧移
# 命名关键字参数：窗函数（）
# 返回值：帧信号矩阵(numpy 数组(行：帧；列：信号点))
def enframe(xRow_TD, frameWidth, frameInc):
    N = len(xRow_TD)  #信号长度（采样次数）
    #计算帧数
    if (N <= frameWidth):
        frameCnt = 1
    else:
        frameCnt = int(np.ceil(
            (N - frameWidth) / frameInc) + 1)  # frameCnt-1个满帧，最后一个帧可能未填满
    # end if
    zerosForPad = np.ones(
        ((frameCnt - 1) * frameInc + frameWidth - N, )) * np.mean(
            xRow_TD)  #填充数组（使用时域信号平均值填充）
    xPad_TD = np.concatenate((xRow_TD, zerosForPad))  #填充后的信号数组
    indices = np.tile(np.arange(frameWidth), (frameCnt, 1)) + np.tile(
        np.arange(frameCnt) * frameInc, (frameWidth, 1)).T  #索引数组（时域信号按帧索引）
    xFrames_TD = xPad_TD[indices]  #帧信号矩阵(行：帧数；列：帧长)
    return xFrames_TD
    # end enframe


#计算功率谱密度(同步C++)
#输入参数：单通道信号，采样频率
#输出参数：频率数组，功率谱密度数组
def myPSD(X_TD, fs):
    Ns = len(X_TD)
    y = scipy.fft.fft(X_TD) * 2 / Ns  #傅里叶变换
    mag = np.abs(y)  #振幅谱
    shift_y = scipy.fft.fftshift(mag)  #重排后的振幅谱
    f = scipy.fft.fftfreq(Ns, d=1 / fs)  #频率序列
    shift_f = scipy.fft.fftshift(f)  #重排后的频率序列
    pos_shift_y = shift_y[shift_y.size // 2:]  #截取有效振幅谱(后半段)
    f = shift_f[shift_f.size // 2:]  #截取有效频率序列(后半段)
    Pxx = pos_shift_y**2
    return f, Pxx
    # end myPSD


# 巴特沃斯滤波器
# 必选参数：时域信号,采样频率,阶数
# 可变参数：截止频率-下频率，上频率
# 命名关键字参数：滤波方式（带通滤波-bandPass，带阻滤波-bandstop，高通滤波-highpass，低通滤波-lowpass）
# 返回值：滤波后的时域信号
def filter_Butterworth(xRow_TD, fs, order, *stopband, passBand):
    sos = scipy.signal.butter(order,
                              stopband,
                              btype=passBand,
                              fs=fs,
                              output='sos')
    xFiltered_TD = scipy.signal.sosfilt(sos, xRow_TD)
    return xFiltered_TD
    # end filter_Butterworth


# 二维数组分帧
def enFrame2D(data, frameCnt, frameWidth, frameInc):
    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert data.shape[1] >= frameInc * (frameCnt - 1) + frameWidth
    data_framed = []
    for i in range(frameCnt):
        frameI = data[:, i * frameInc:i * frameInc + frameWidth]
        data_framed.append(frameI)
    return np.array(data_framed)
    # end enFrame2D


# 三维数组分帧
def enFrame3D(data, frameCnt, frameWidth, frameInc):

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert data.shape[2] >= frameInc * (frameCnt - 1) + frameWidth
    data_framed = []
    for i in range(frameCnt):
        # print(i*frameInc,':',i*frameInc+frameWidth)
        frameI = data[:, :, i * frameInc:i * frameInc + frameWidth]
        data_framed.append(frameI)
    return np.array(data_framed)
    # end enFrame3D


# 数据集划分
def splitSet(set_X, set_y, ratio, shuffle=False):

    # 输入检查：输入数据集 X,y 的样本数量必须一致
    assert set_X.shape[0] == set_y.shape[0] and ratio >= 0
    X = set_X.copy()
    y = set_y.copy()
    if shuffle:
        shuffle_index = np.random.permutation(set_X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]
    if ratio < 1:
        ratio = ratio * set_X.shape[0]
    ratio=int(ratio)
    return X[:ratio], y[:ratio], X[ratio:], y[ratio:]


def test():
    x = np.arange(32 * 1280).reshape((32, 1280))
    ret = enFrame2D(x, 3, 640, 320)
    print(ret.shape)


# test()