# 提取源数据，数据预处理，得到 numpy 数组

import os
import pickle
import sys
import time

import numpy as np
import scipy.signal
import h5py
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FastICA
from matplotlib.pyplot import MultipleLocator

sys.path.append(r'D:\myCode\EMG_py\SignalPreprocessing')
import SignalPreprocessing as sp

# import mySignalPreprocessing as mysp
# import myFeatureExtraction as myfe


# 使用 ICA 查看信号成分-单个样本(若 ICA 算法收敛，则 A_ 与 W_ 互为逆矩阵且 S_=(X-M_)*W_.T , X=S_*A_.T+M_)
# 必选参数：多通道数据数组(numpy 二维数组(行：各数据通道，列：各采样点))，采样频率，去除成分的通道索引(list)
# 关键字参数：图像名称，子图标题(只在多图模式下有效)，显示的上频率
# 返回值：训练好的 ICA 转换器
def plot_ICA(dataArray, fs, channels, **kw):
    trans_ICA = FastICA(random_state=0, max_iter=5000)
    X = dataArray.T  #信号矩阵 X
    S_ = trans_ICA.fit_transform(X)  #成分矩阵 S_
    A_ = trans_ICA.mixing_  #混合矩阵 A_
    W_ = trans_ICA.components_  #解混矩阵 W_
    M_ = trans_ICA.mean_  #均值矩阵 M_
    X_r = trans_ICA.inverse_transform(S_)
    print('ICA分解完成！')
    print('迭代次数：', trans_ICA.n_iter_)
    print('ICA分解有损失吗？？？\t', ['有', '没有'][np.allclose(X, X_r)])

    ##########
    # 此处进行成分处理
    startTime = time.process_time()
    # ApEnArray = ApEn_channels_Mul(S_.T, 0.23, 2)  #计算各成分近似熵
    ApEnArray = []
    print(ApEnArray)
    endTime = time.process_time()
    print('完成：计算各成分近似熵，用时: %s 秒' % (endTime - startTime))
    plot_channels_Mul(
        S_.T,
        name='原始信号成分',
        yticklabels=['IC %d ' % i for i in range(dataArray.shape[0])],
        yticklabels2=['-->%f' % i for i in ApEnArray])  #绘制各成分曲线
    S_[:, channels] = 0  #将伪迹成分置零
    plot_channels_Mul(
        S_.T,
        name='处理后信号成分',
        yticklabels=['IC %d ' % i for i in range(dataArray.shape[0])],
        yticklabels2=['-->%f' % i for i in ApEnArray])  #绘制各成分曲线
    ##########
    X_recover = trans_ICA.inverse_transform(S_)  #恢复信号各通道
    plot_PSD_Mul(S_.T, fs, subplots=True, name='信号成分功率谱密度', **kw)
    plot_channels_Mul(dataArray, X_recover.T, name='ICA对比：第 30 组实验')
    return trans_ICA
    # end plot_ICA


# 绘制多通道功率谱密度曲线
# 必选参数：多通道数据数组(numpy 二维数组(行：各数据通道，列：各采样点))，采样频率
# 命名关键字参数：是否为多图模式(多图模式下，每个通道一个子图)=否
# 关键字参数：图像名称，子图标题(只在多图模式下有效)，显示的上频率，图例-通道名称
def plot_PSD_Mul(dataArray, fs, *, subplots=False, **kw):
    fig = plt.figure(kw.get('name', 1))
    if subplots is False:
        ax = plt.subplot()
        for i, xRaw_TD in enumerate(dataArray):
            #遍历各通道
            # f, Pxx_den = sp.myPSD(xRaw_TD, fs)
            f, Pxx_den = scipy.signal.welch(xRaw_TD, fs,
                                            nperseg=1024)  #计算功率谱密度
            Pxx_den_db = 10 * np.log10(Pxx_den)
            plotX_index = kw.get('plot_upperFreq', f.max())
            plotX = f[np.where(f <= plotX_index + 0.3)]
            plotY = Pxx_den_db[np.where(f <= plotX_index + 0.3)]
            plt.plot(plotX[1:],
                     plotY[1:],
                     label=kw.get(
                         'labels',
                         ['CH%d' % (i + 1)
                          for _ in range(dataArray.shape[0])])[i])  #显示功率谱密度
        # end for
        plt.xlabel('频率 (Hz)')
        plt.ylabel('dB')
        # ax.set_xticks(np.linspace(0, plotX_index, plotX_index/10+1))  #设置 x 轴标签
        ax.grid(linestyle='--')
        plt.legend()
    else:
        channel_cnt = dataArray.shape[0]  #通道数
        COLS = 6
        ROWS = int((channel_cnt + COLS - 1) / COLS)
        for i, xRaw_TD in zip(range(channel_cnt), dataArray):
            #遍历各通道
            ax = fig.add_subplot(ROWS, COLS, i + 1)
            f, Pxx_den = sp.myPSD(xRaw_TD, fs)
            Pxx_den_db = 10 * np.log10(Pxx_den)
            plotX_index = kw.get('plot_upperFreq', f.max())
            plotX = f[np.where(f <= plotX_index + 0.3)]
            plotY = Pxx_den_db[np.where(f <= plotX_index + 0.3)]
            ax.plot(plotX[3:], plotY[3:])  #显示功率谱密度
            ax.set_title(
                kw.get('sub_title',
                       ['IC %d ' % ic for ic in range(channel_cnt)])[i])
        # end for
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=None,
                            hspace=0.8)
    # end if
    plt.show()
    # end plot_PSD_Mul


# 时序数据按频带分通道
# 必选参数：时域信号(numpy 二维数组(行：各数据通道，列：各采样点))，采样频率，频带范围列表
# 返回值：频带分通道后的时域信号(numpy 三维数组(各频带 * 各数据通道 * 各采样点))
def getOriginalPicture(dataArray, fs, bandRange):
    channelCnt = len(bandRange)  #通道数
    ret = np.zeros(
        (channelCnt, dataArray.shape[0], dataArray.shape[1]))  #构造返回值
    for dataNo, rawData in enumerate(dataArray):
        for channelNo in range(channelCnt):
            ret[channelNo][dataNo] = sp.filter_Butterworth(
                rawData,
                fs,
                8,
                bandRange[channelNo][0],
                bandRange[channelNo][1],
                passBand='bandPass')
        # end for
    # end for
    return ret
    # end getOriginalPicture


# 绘制多通道幅值-时间曲线
# 必选参数：多通道数据数组(numpy 二维数组(行：各数据通道，列：各采样点))
# 可变参数：对比数组(行维度与多通道数据数组相同)
# 关键字参数：图像名称，y 轴标签，第二个 y 轴标签
def plot_channels_Mul(dataArray, *comparisonArray, **kw):
    #输入检查
    if (not len(comparisonArray)
            == 0) and (dataArray.shape[0] != comparisonArray[0].shape[0]):
        raise ValueError('plot_rawData_Mul：对比数组维度错误！')
    # end if
    plt.figure(kw.get('name', 1))
    ax = plt.subplot()
    channel_cnt = dataArray.shape[0]  #通道数
    scope = dataArray.max() - dataArray.min()  #绘制的通道在 y 轴上的幅度间距
    ypos = np.linspace(0, channel_cnt * scope * 0.5,
                       channel_cnt)  #各通道在 y 轴上的绘制位置
    #构造绘图数组
    plotArray = np.zeros(dataArray.shape)
    ax.set_xticks(np.linspace(0, dataArray.shape[1], 11))  #设置 x 轴标签
    #设置 y 轴标签
    for i in range(channel_cnt):
        #遍历各通道
        plotArray[i, :] = dataArray[i, :] + ypos[i]  #设置通道值
        plt.plot(plotArray[i, :], linewidth=0.5, color='b')  #绘制通道折线图
        if not len(comparisonArray) == 0:
            #绘制对比数组通道折线图
            plt.plot(comparisonArray[0][i, :] + ypos[i],
                     linewidth=1,
                     color='r')
        # end if
    # end for
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        kw.get('yticklabels', ['ch %d ' % i for i in range(channel_cnt)]))
    plt.xlabel('时间 (ms)')
    plt.ylabel('通道')
    if not kw.get('yticklabels2') is None:
        #绘制双 y 轴标签
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ypos)
        ax2.set_yticklabels(kw.get('yticklabels2'))
        plt.ylabel('近似熵')
    # end if
    ax.grid(linestyle='--')
    plt.show()
    # end plot_channels_Mul


# 计算多通道近似熵
# 必选参数：多通道成分数组(numpy 二维数组(行：各成分通道，列：各采样点))，相似性容限，子区间粒度
# 返回值：各成分近似熵数组(numpy 一维数组(元素为各成分近似熵，顺序与成分通道顺序一致))
def ApEn_channels_Mul(dataArray, c, m_init):

    # 计算区间粒度为 m 的 phi 统计量
    # 必选参数：单通道成分数组(numpy 一维数组(元素为各采样点))，相似性容限，子区间粒度
    # 返回值：phi_m 统计量
    def calcPhi_m(sourceArray, c, m):
        sd = np.std(sourceArray)
        xFrames_source = sp.enframe(sourceArray, m, 1)  #成分帧数组
        nArray = []  #初始化各子区间满足准则的数量数组
        for xi in xFrames_source:
            #遍历每一个子区间
            di = abs(xFrames_source - xi).max(axis=1)
            ni = np.count_nonzero(di <= c * sd)  #计算该子区间满足准则的距离数量
            nArray.append(ni)
            # print(len(nArray))
        # end for
        tmp = [np.log(ni / (len(sourceArray) - m)) for ni in nArray]
        phi_m = np.mean(tmp)
        return phi_m
        # end calcPhi_m

    ApEn = [
        calcPhi_m(xi, c, m_init) - calcPhi_m(xi, c, m_init + 1)
        for xi in dataArray
    ]  #遍历各通道计算近似熵
    return np.array(ApEn)
    # end ApEn_channels_Mul


# 提取一个位置信号标签(确定目标直线+计算实际轨迹与目标直线的偏离程度)
# 必选参数：实际轨迹数组(numpy 二维数组(行：各坐标轴；列：采样点))，目标方向
# 关键字参数：阈值(若给出阈值则将标签进行二值化：True-偏离程度较低；False-偏离程度较高)
# 返回值：该样本的标签值，活动开始时刻索引，活动结束时刻索引
def getLabel_POS_single(posArray, direction, **kw):
    FS = 60  # 设备采样频率
    TIME_START = 3  # 实验开始时刻
    TIME_CUE = 6  #任务开始时刻
    THRESHOLD_N = 20  #连续 THRESHOLD_N 点大于阈值即判定为有意识移动的开始

    posArray_valid = posArray[:, FS * TIME_CUE:]  #实验开始时刻后的数据是有效数据
    # 输入检查
    if not direction in ['up', 'down', 'left', 'right']:
        raise ValueError('getLabel_POS_single：方向输入错误！')
    # end if

    axis_move = [0, 1][direction in ['up', 'down']]  #移动方向坐标轴
    axis_still = [0, 1][direction not in ['up', 'down']]  #静止方向坐标轴
    dir_change = posArray_valid[axis_move]  #移动方向坐标

    # 确定正确移动阈值(移动速度>阈值，代表有意识的移动；移动速度<阈值或移动速度为负，均视为无意识抖动)
    if direction in ['down', 'right']:
        dir_increase = dir_change[:-1] - dir_change[1:]
    else:
        dir_increase = dir_change[1:] - dir_change[:-1]
    # end if
    threshold = 5 * dir_increase.mean()  #正确移动阈值

    # 判定运动开始、结束时刻
    meetThreshold = dir_increase > threshold
    #活动开始时刻的索引(连续 THRESHOLD_N 点大于阈值即判定为有意识移动的开始)，活动结束时刻的索引(有意识移动开始后，首个小于阈值的点即判定为有意识移动的结束)
    actList = [
        xi for xi in range(len(meetThreshold) - THRESHOLD_N + 1)
        if sum(meetThreshold[xi:xi + THRESHOLD_N]) == THRESHOLD_N
    ]
    if len(actList) == 0:
        actBegin = 60
        actEnd = actBegin + 60
    else:
        actBegin = actList[0]
        actEnd = actList[-1]
    # actBegin = np.where(meetThreshold)[0][0]  #活动开始时刻的索引(首个大于阈值的点即判定为有意识移动的开始)

    if False:  # 关闭异常活动可视化功能
        # 异常活动可视化(异常活动：抖动造成的，在目标方向上，连续 THRESHOLD_N 点移动速度大于阈值)
        if actBegin < FS * (TIME_CUE - TIME_START):
            print('actBegin=', actBegin)
            print('actEnd=', actEnd)
            # end if
            fig = plt.figure()
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            ax1 = fig.add_subplot(3, 2, 1)
            ax1.plot(posArray[axis_move], posArray[axis_still])
            ax1.set_title('原始数据：运动轨迹')

            ax2 = fig.add_subplot(3, 2, 2)
            ax2.plot(posArray_valid[axis_move], posArray_valid[axis_still])
            ax2.set_title('start 时刻后的数据：运动轨迹')

            ax3 = fig.add_subplot(3, 2, 3)
            ax3.plot(posArray_valid[axis_still])
            ax3.set_title('start 时刻后的数据：时间-静止坐标')

            ax4 = fig.add_subplot(3, 2, 4)
            ax4.plot(posArray_valid[axis_move])
            ax4.axvline(actBegin, c='r')
            ax4.axvline(actEnd, c='r')
            ax4.set_title('start 时刻后的数据：时间-运动坐标')

            ax5 = fig.add_subplot(3, 2, 5)
            ax5.plot(dir_increase)
            ax5.axhline(threshold, c='r')
            ax5.axvline(actBegin, c='r')
            ax5.axvline(actEnd, c='r')
            ax5.set_title('start 时刻后的数据：时间-目标方向运动速度')

            # ax6 = fig.add_subplot(3, 2,6)
            # ax6.plot(posArray_valid[axis_move])
            plt.subplots_adjust(left=None,
                                bottom=None,
                                right=None,
                                top=None,
                                wspace=None,
                                hspace=0.8)
            plt.show()
    # end if

    # 计算样本标签
    coordinate_still = posArray_valid[
        axis_still, :actBegin -
        2 * FS].mean()  #实验开始时刻至活动开始之前两秒的时间段，作为静止方向坐标的标定阶段
    posArray_calc = posArray_valid[:, :actEnd +
                                   FS]  #实验开始时刻至活动结束之后一秒的时间段，作为运动轨迹偏离量的计算部分
    deviation = (np.abs(posArray_calc[axis_still] - coordinate_still)).sum()
    # print('该样本的标签值', deviation)
    # print('活动开始时刻索引', actBegin + FS * TIME_START)
    # print('活动结束时刻索引', actEnd + FS * TIME_START)

    if kw.get('threshold') is None:
        #偏离程度无需二值化
        return deviation, actBegin + FS * TIME_CUE, actEnd + FS * TIME_CUE
    else:
        #偏离程度二值化
        return deviation < kw.get(
            'threshold'), actBegin + FS * TIME_CUE, actEnd + FS * TIME_CUE
    # end if
    # end getLabel_POS_single

    # print(posArray_valid.shape)
    # plt.figure()
    # ax = plt.subplot()
    # # plt.plot(posArray_valid[axis_move], posArray_valid[axis_still])
    # # plt.axhline(coordinate_still,c='r')

    # # plt.plot(range(posArray_valid.shape[1]-1),dir_increase)
    # # plt.axhline(threshold,c='r')

    # plt.plot(range(posArray_valid.shape[1]), dir_change)
    # # plt.plot(range(posArray_valid.shape[1]), posArray_valid[axis_still]-coordinate_still)
    # plt.axvline(actBegin, c='r')
    # plt.axvline(actEnd, c='r')

    # print(dir_increase.mean())
    # ax.set_xticks(range(0, 1000, 50))
    # plt.grid()
    # plt.show()


# EEG 特征提取-单个样本
# 必选参数：信号数组(numpy 二维数组(行：通道；列：采样点))，采样频率
# 关键字参数：信号分段位置(若给出，则各通道逐段提取时域特征并级联)
# 返回值：特征数组(numpy 一维数组(元素为各特征值))
def EEG_featureExtraction(dataArray, fs, **kw):

    # EEG 时域特征提取-单个样本
    # 必选参数：信号数组(numpy 二维数组(行：通道；列：采样点))
    # 关键字参数：信号分段位置(若给出，则各通道逐段提取时域特征并级联)
    # 返回值：特征数组(numpy 一维数组(元素为各特征值))
    def EEG_featureExtraction_TD(dataArray, **kw):

        # 使用 PCA 对时域信号降维
        # 必选参数：信号数组(numpy 二维数组(行：通道；列：采样点))
        # 返回值：特征数组(numpy 一维数组(元素为各特征值))
        def Use_PCA(dataArray):
            # trans_PCA=PCA(n_components=0.99,svd_solver='full')
            trans_PCA = PCA(n_components=4, svd_solver='full')
            dataArray_reduced = trans_PCA.fit_transform(dataArray)
            # print('时域特征提取完毕！')
            # print('主成分占比：', trans_PCA.explained_variance_ratio_.sum())
            # print('主成分数量：', trans_PCA.n_components_)
            # print(trans_PCA.explained_variance_ratio_)
            return dataArray_reduced.ravel('F')  #按列展开，同步C++
            # end Use_PCA

        if kw.get('segment_TD') is None:
            #不需要对信号数组分段
            return Use_PCA(dataArray)
        else:
            #需要对信号数组分段
            index = [0] + [
                int(np.ceil(xi * dataArray.shape[1]))
                for xi in kw.get('segment_TD')
            ]  #信号数组分段索引
            segList = [
                dataArray[:, index[i]:index[i + 1]]
                for i in range(len(index) - 1)
            ] + [dataArray[:, index[len(index) - 1]:]]  #对信号数组分段
            feature = []  #初始化时域特征列表
            for seg_i in segList:
                # 遍历各段
                feature += list(Use_PCA(seg_i))  #计算该段时域特征向量并压入时域特征列表
            # end for
            return np.array(feature)
        # end if
        # end EEG_featureExtraction_TD

    # EEG 频域特征提取-单个样本
    # 必选参数：信号数组(numpy 二维数组(行：通道；列：采样点))，采样频率
    # 返回值：特征数组(numpy 一维数组(元素为各特征值))
    def EEG_featureExtraction_FD(dataArray, fs):
        featureList = []  #初始化特征值列表
        for xRaw_TD in dataArray:
            #遍历各通道
            f, Pxx_den = sp.myPSD(xRaw_TD, fs)
            Pxx_den_all = Pxx_den[np.where((f >= 0.53)
                                           & (f <= 60))].sum()  #计算有效功率密度总和
            Pxx_den_theta = Pxx_den[np.where(
                (f >= 4) & (f <= 8))].sum()  #计算 theta 节律的功率密度和
            Pxx_den_alpha = Pxx_den[np.where(
                (f > 8) & (f <= 13))].sum()  #计算 alpha 节律的功率密度和
            Pxx_den_beta = Pxx_den[np.where(
                (f > 13) & (f <= 30))].sum()  #计算 beta 节律的功率密度和
            featureList += [
                Pxx_den_theta / Pxx_den_all, Pxx_den_alpha / Pxx_den_all,
                Pxx_den_beta / Pxx_den_all
            ]
        # end for
        return np.array(featureList)
        # end EEG_featureExtraction_FD

    feature_TD = EEG_featureExtraction_TD(dataArray,
                                          **kw)  #提取时域特征(numpy 一维数组(元素为各特征值))
    feature_FD = EEG_featureExtraction_FD(dataArray,
                                          fs)  #提取频域特征(numpy 一维数组(元素为各特征值))
    feature_sample = np.r_[feature_TD,
                           feature_FD]  #生成单样本频域数组(numpy 一维数组(元素为各特征值))
    # print('时域特征：', feature_TD.shape)
    # print('频域特征：', feature_FD.shape)
    # print('样本特征：', feature_sample.shape)
    return feature_sample
    # end EEG_featureExtraction


# 使用训练完成的 ICA 转换器去除样本伪迹-单个样本(若 ICA 算法收敛，则 A_ 与 W_ 互为逆矩阵且 S_=(X-M_)*W_.T , X=S_*A_.T+M_)
# 必选参数：多通道数据数组(numpy 二维数组(行：各数据通道，列：各采样点))，ICA 转换器
# 关键字参数：去除的成分通道索引
# 返回值：去伪迹后的多通道数据数组(numpy 二维数组(行：各数据通道，列：各采样点))
def use_ICA_removeArtifacts(dataArray, trans_ICA, **kw):
    S_ = trans_ICA.transform(dataArray.T)
    if not kw.get('channels') is None:
        #去伪迹
        S_[:, kw.get('channels')] = 0  #将伪迹成分置零
    # end if
    X_recover = trans_ICA.inverse_transform(S_)  #恢复信号各通道
    return X_recover.T
    # end use_ICA_removeArtifacts


# 预处理一个 EEG 信号
# 必选参数：原始信号数组(numpy 二维数组(行：通道；列：采样点))，采样频率
# 关键字参数：训练好的 ICA 转换器，去除的成分通道索引
# 返回值：预处理后的 EEG 信号(numpy 二维数组(行：通道；列：采样点))
def preprocessing_EEG_single(dataArray_raw, fs, **kw):
    # print('preprocessing_EEG_single：,dataArray_raw.shape=',
    #       dataArray_raw.shape)
    #去除直流分量
    dataArray_NoDC = dataArray_raw - dataArray_raw.mean(axis=1).reshape(
        -1, 1)  #各通道去直流

    dataArray_NoDC = [
        sp.filter_Butterworth(xRow_TD, fs, 5, 49, passBand='lowpass')
        for xRow_TD in dataArray_NoDC
    ]  #各通道信号滤波
    dataArray_NoDC = np.array(dataArray_NoDC)
    dataArray_NoDC = [
        sp.filter_Butterworth(xRow_TD, fs, 5, 0.5, passBand='highpass')
        for xRow_TD in dataArray_NoDC
    ]  #各通道信号滤波
    dataArray_NoDC_filtered = np.array(dataArray_NoDC)

    # #带阻滤波-截止频率：49~51Hz
    # dataList_NoDC_filtered = [
    #     sp.filter_Butterworth(xRow_TD, fs, 5, 49, 51, passBand='bandstop')
    #     for xRow_TD in dataArray_NoDC
    # ]  #各通道信号滤波
    # dataArray_NoDC_filtered = np.array(dataList_NoDC_filtered)

    # ICA 去伪迹
    if not kw.get('trans_ICA') is None:
        #需要使用 ICA 去伪迹
        dataArray_preprocessed = use_ICA_removeArtifacts(
            dataArray_NoDC_filtered, **kw)
    else:
        dataArray_preprocessed = dataArray_NoDC_filtered
    # end if
    return dataArray_preprocessed


#训练并保存 ICA 转换器
def save_ICA():
    YTICKLABELS = [
        'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 'FC3',
        'FC4', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP3',
        'CP4', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz', 'Oz', 'O1',
        'O2', 'HEOR', 'HEOL'
    ]  #数据格式-通道号
    FS = 1000

    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    ICACHANNEl_DICT = {
        'cw': [14, 21],
        'cwm': [21, 20],
        'kx': [4, 5],
        'pbl': [2],
        'wrd': [7],
        'wxc': [8],
        'xsc': [34],
        'yzg': []
    }
    for name in NAME_LIST:
        path = 'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\右\\fwj_poseeg43.txt'
        # 提取 EEG 特征值
        file_EEG = open(path, 'r')
        lines = file_EEG.readlines()
        dataArray = np.zeros((64, len(lines)))
        for eeg_j, line in zip(range(len(lines)), lines):
            wordList = line.split('\t')
            floatList = [float(xi) for xi in wordList[:-2]
                         ]  #不转换行尾的换行符(原始数据中，最后一个是 '\n' ，倒数第二个是 '0' )
            dataArray[:, eeg_j] = np.array(floatList)
        # end for
        dataArray_use = dataArray[[
            7, 8, 9, 10, 11, 34, 13, 14, 15, 16, 19, 20, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 36, 37, 42, 43, 44, 45, 46, 47, 48, 49, 56, 57, 58,
            60, 61
        ], 1000:12000]  #筛选有效的通道
        dataArray_use_NoDC = dataArray_use - dataArray_use.mean(
            axis=1).reshape((dataArray_use.shape[0], -1))

        dataArray_use_NoDC = [
            sp.filter_Butterworth(xRow_TD, FS, 5, 80, passBand='lowpass')
            for xRow_TD in dataArray_use_NoDC
        ]  #各通道信号滤波
        dataArray_use_NoDC = np.array(dataArray_use_NoDC)
        dataArray_use_NoDC = [
            sp.filter_Butterworth(xRow_TD, FS, 5, 0.5, passBand='highpass')
            for xRow_TD in dataArray_use_NoDC
        ]  #各通道信号滤波
        dataArray_use_NoDC = np.array(dataArray_use_NoDC)

        #带阻滤波-截止频率：49~51Hz
        dataArray_right_0_filtered = [
            sp.filter_Butterworth(xRow_TD, FS, 5, 49, 51, passBand='bandstop')
            for xRow_TD in dataArray_use_NoDC
        ]  #各通道信号滤波
        dataArray = np.array(dataArray_right_0_filtered)

        plot_channels_Mul(dataArray,
                          name='工频滤波后，各信号通道：第 34 组实验',
                          yticklabels=YTICKLABELS)
        trans_ICA = plot_ICA(dataArray,
                             FS,
                             ICACHANNEl_DICT.get(name),
                             plot_upperFreq=60)  #训练 ICA 转换器

        # 保存 ICA 转换器(S_=(X-M_)*W_.T , X=S_*A_.T+M_)
        path_tmp = os.getcwd()
        fileD = open(
            path_tmp + r'\\..\data\\' + name + '\\单任务运动\\' + r'\\ica.pickle',
            'wb')
        pickle.dump(trans_ICA, fileD)  #保存 ICA 转换器
        fileD.close()
    # end for
    # end save_ICA


# 从 single_move_eeg_with_label_beginTimeFile.pickle 中提取特征及标签，预处理，特征提取，并保存（单任务运动，人工特征，标签 0123 分别对应上下左右）
def get_EEG_features():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    FS_EEG = 1000  #EEG采样频率

    for name in NAME_LIST:
        path = os.getcwd()
        fileD = open(
            path + r'\\..\data\\' + name + '\\四分类\\' +
            r'\\single_move_eeg_with_label_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        labelList = y[:, 0]  #标签向量列表
        featureList = []  #初始化特征向量列表
        for dataArray, yI in zip(X, y):
            actBeginTime = yI[1]
            dataArray_preprocessed = preprocessing_EEG_single(
                dataArray, FS_EEG)  # 数据预处理
            feature_i = EEG_featureExtraction(
                dataArray_preprocessed[:,
                                       actBeginTime - 640:actBeginTime + 640],
                FS_EEG)  # 特征提取(motion_start 前后 640ms 段)
            featureList.append(feature_i)
        # end for

        featureArray = np.array(featureList)  #样本集特征矩阵(numpy 二维数组(行：各样本；列：特征值))
        labelArray = np.array(labelList).reshape(
            (len(labelList), -1))  #样本集标签矩阵(numpy 二维数组(行：各样本；列：标签))
        print('get_EEG_features：' + name + ' 样本矩阵提取完成')
        print('get_EEG_features：featureArray.shape = ', featureArray.shape)
        print('get_EEG_features：labelArray.shape = ', labelArray.shape)

        path = os.getcwd()
        fileD = open(
            path + r'\\..\data\\' + name + '\\四分类\\' +
            r'\\single_move_motion_start_motion_end_beginTimeFile.pickle',
            'wb')
        pickle.dump([featureArray, labelArray], fileD)
        fileD.close()
    # end for
    # end get_EEG_features


# 从 single_move_eeg_with_label_beginTimeFile.pickle 中选取多个时段提取特征及标签，预处理，特征提取，并保存（单任务运动，人工特征，标签 0123 分别对应上下左右）
def get_EEG_features_multiSegment():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    NAME_LIST = ['xsc', 'yzg']
    # NAME_LIST = ['wxc']
    FS_EEG = 1000  #EEG采样频率
    SEG_START=np.linspace(-2080, 1120, 41).astype(int) # 窗移 80ms

    for name in NAME_LIST:
        print('name = ',name)
        path = os.getcwd()
        fileD = open(
            path + r'\\..\data\\' + name + '\\四分类\\' +
            r'\\single_move_eeg_with_label_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        for timeBeginStart in SEG_START:
            print('timeBeginStart = ',timeBeginStart)
            labelList = y[:, 0]  #标签向量列表
            featureList = []  #初始化特征向量列表
            for dataArray, yI in zip(X, y):
                actBeginTime = yI[1]
                dataArray_preprocessed = preprocessing_EEG_single(
                    dataArray, FS_EEG)  # 数据预处理
                feature_i = EEG_featureExtraction(
                    dataArray_preprocessed[:, actBeginTime -
                                           timeBeginStart:actBeginTime -
                                           timeBeginStart + 1280],
                    FS_EEG)  # 特征提取(motion_start 前后 640ms 段)
                featureList.append(feature_i)
            # end for

            featureArray = np.array(
                featureList)  #样本集特征矩阵(numpy 二维数组(行：各样本；列：特征值))
            labelArray = np.array(labelList).reshape(
                (len(labelList), -1))  #样本集标签矩阵(numpy 二维数组(行：各样本；列：标签))
            print('get_EEG_features_multiSegment：' + name + ' 样本矩阵提取完成')
            print('get_EEG_features_multiSegment：featureArray.shape = ',
                  featureArray.shape)
            print('get_EEG_features_multiSegment：labelArray.shape = ',
                  labelArray.shape)

            fileName = 'single_move_motion_start%d.pickle' % timeBeginStart
            path = os.getcwd()
            fileD = open(path + r'\\..\data\\' + name + '\\四分类\\' + fileName,
                         'wb')
            pickle.dump([featureArray, labelArray], fileD)
            fileD.close()
        # end for
    # end for
    # end get_EEG_features_multiSegment


# 从 single_move_eeg_with_label_beginTimeFile.pickle 中分帧并提取特征及标签，预处理，特征提取，并保存（单任务运动，人工特征，标签 0123 分别对应上下左右）
def get_EEG_features_multiFrame():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    FS_EEG = 1000  #EEG采样频率

    # 适用于 Net46
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 320  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net46

    for name in NAME_LIST:
        path = os.getcwd()
        fileD = open(
            path + r'\\..\data\\' + name + '\\单任务运动\\' +
            r'\\single_move_eeg_with_label_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        labelList = y[:, 0]  #标签向量列表
        featureList = []  #初始化特征向量列表
        for dataArray, yI in zip(X, y):
            actBeginTime = yI[1]
            dataArray_preprocessed = preprocessing_EEG_single(
                dataArray, FS_EEG)  # 数据预处理
            dataUse = dataArray_preprocessed[:,
                                             actBeginTime - 640:actBeginTime +
                                             640]  # 使用数据 motion_start 前后 640ms 段
            dataUse_framed = sp.enFrame2D(dataUse, FRAME_CNT, FRAME_WIDTH,
                                          FRAME_INC)
            feature_framed = []
            for frame in dataUse_framed:
                feature_framed.append(EEG_featureExtraction(frame, FS_EEG))
            featureList.append(feature_framed)
        # end for

        featureArray = np.array(
            featureList)  #样本集特征矩阵(numpy 三维数组(各样本 * 各帧 * 特征值))
        labelArray = np.array(labelList).reshape(
            (len(labelList), -1))  #样本集标签矩阵(numpy 二维数组(行：各样本；列：标签))
        print('get_EEG_features：' + name + ' 样本矩阵提取完成')
        print('get_EEG_features：featureArray.shape = ', featureArray.shape)
        print('get_EEG_features：labelArray.shape = ', labelArray.shape)

        path = os.getcwd()
        fileD = open(
            path + r'\\..\data\\' + name + '\\单任务运动\\' +
            r'\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'wb')
        pickle.dump([featureArray, labelArray], fileD)
        fileD.close()
    # end for
    # end get_EEG_features_multiFrame


# 从原始 EEG 数据中提取特征及标签，预处理，特征提取，并保存（单任务运动，人工特征，标签 0123 分别对应上下左右）
def get_EEG_single_move_artificial_features():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    # NAME_LIST = ['cwm']
    ICACHANNEl_DICT = {
        'cw': [],
        'cwm': [],
        'kx': [],
        'pbl': [],
        'wrd': [],
        'wxc': [],
        'xsc': [],
        'yzg': []
    }
    for name in NAME_LIST:
        pathList = [
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\上\\',
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\下\\',
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\左\\',
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\右\\'
        ]

        FS_EEG = 1000  #EEG采样频率
        FS_POS = 60  #POS采样频率
        TIME_START = 3  #实验开始时刻
        TIME_CUE = 6  #任务开始时刻

        featureList = []  #初始化特征向量列表
        labelList = []  #初始化标签向量列表

        # 提取原始数据，预处理 + 特征提取
        for pi in range(len(pathList)):
            print(
                '\n\nget_EEG_single_move_artificial_features：++++++++++++++++\n\n',
                pi)
            #遍历各动作集
            path = pathList[pi]  #动作文件路径

            for trial_i in range(1, 51):
                # 遍历各组实验

                # 提取标签值，活动开始时刻索引，活动结束时刻索引
                file_POS = open(path + 'posxyz_%d.txt' % trial_i, 'r')
                lines = file_POS.readlines()
                dataArray = np.zeros((12, len(lines)))
                for pos_j, line in zip(range(len(lines)), lines):
                    wordList = line.split('\t')
                    floatList = [float(xi) for xi in wordList[:-1]
                                 ]  #不转换行尾的换行符(原始数据中，最后一个是 '\n' )
                    dataArray[:, pos_j] = np.array(floatList)
                # end for
                # label_i = getLabel_POS_single(dataArray[[0, 2], :],
                #                               ['up', 'down', 'left',
                #                                'right'][pi])  #计算标签值
                labelList.append(pi)
                # end for

                # 提取 EEG 特征值
                file_EEG = open(path + 'fwj_poseeg%d.txt' % trial_i, 'r')
                lines = file_EEG.readlines()
                dataArray = np.zeros((64, len(lines)))
                for eeg_j, line in zip(range(len(lines)), lines):
                    wordList = line.split('\t')
                    floatList = [float(xi) for xi in wordList[:-2]
                                 ]  #不转换行尾的换行符(原始数据中，最后一个是 '\n' ，倒数第二个是 '0' )
                    dataArray[:, eeg_j] = np.array(floatList)
                # end for
                file_EEG.close()
                dataArray_use = dataArray[[
                    7,
                    8,
                    9,
                    10,
                    11,
                    34,
                    13,
                    14,
                    15,
                    16,
                    19,
                    20,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    36,
                    37,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    56,
                    57,
                    58,
                ], :]  #筛选有效的通道
                ##################################
                # 在此处使用 ICA
                ##################################

                dataArray_preprocessed = preprocessing_EEG_single(
                    dataArray_use, FS_EEG)  # 数据预处理
                # print('使用的数据段：',int(FS_EEG * label_i[1] /FS_POS)-500,'   ',int(FS_EEG * label_i[1] /FS_POS)+500)
                # feature_i = EEG_featureExtraction(
                #     dataArray_preprocessed[:,
                #                            int(FS_EEG * label_i[1] /FS_POS)-500:int(FS_EEG * label_i[1] /FS_POS)+500],
                #     FS_EEG)  # 特征提取(motion_start 前后 0.5 秒段)

                feature_i = EEG_featureExtraction(
                    dataArray_preprocessed[:, 9000:10000],
                    FS_EEG)  # 特征提取(motion_start 前后 0.5 秒段)

                featureList.append(feature_i)
                # print(
                #     'get_EEG_single_move_artificial_features：feature_i.shape=',
                #     feature_i.shape)

                # print(
                #     'get_EEG_single_move_artificial_features：len(featureList)=',
                #     len(featureList))
                # print(
                #     'get_EEG_single_move_artificial_features：len(labelList)=',
                #     len(labelList))
                # print(
                #     'get_EEG_single_move_artificial_features：++++++++++++++++')
            # end for
        # end for

        featureArray = np.array(featureList)  #样本集特征矩阵(numpy 二维数组(行：各样本；列：特征值))
        labelArray = np.array(labelList).reshape(
            (len(labelList),
             -1))  #样本集标签矩阵(numpy 二维数组(行：各样本；列：标签值，活动开始时刻索引，活动结束时刻索引))
        print('get_EEG_single_move_artificial_features：' + name + ' 样本矩阵提取完成')
        print('get_EEG_single_move_artificial_features：featureArray.shape = ',
              featureArray.shape)
        print('get_EEG_single_move_artificial_features：labelArray.shape = ',
              labelArray.shape)

        path = os.getcwd()
        fileD = open(
            path + r'\\..\data\\' + name + '\\单任务运动\\' +
            r'\\single_move_motion_start_motion_end.pickle', 'wb')
        pickle.dump([featureArray, labelArray], fileD)
        fileD.close()
    # end for
    # end get_EEG_single_move_artificial_features


# 录入一次活动的活动开始时刻
# 必选参数：实际轨迹数组(numpy 二维数组(行：各坐标轴；列：采样点))，目标方向
# 返回值：活动开始时刻索引
def getAction(posArray, direction):
    FS = 60  # 设备采样频率
    TIME_START = 3  #实验开始时刻
    THRESHOLD_N = 5

    posArray_valid = posArray[:, FS * TIME_START:]  #实验开始时刻后的数据是有效数据
    # 输入检查
    if not direction in ['up', 'down', 'left', 'right']:
        raise ValueError('getLabel_POS_single：方向输入错误！')
    # end if

    axis_move = [0, 1][direction in ['up', 'down']]  #移动方向坐标轴
    axis_still = [0, 1][direction not in ['up', 'down']]  #静止方向坐标轴

    actBegin = 320

    fig = plt.figure(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # ax1 = fig.add_subplot(2, 2, 1)
    # ax1.plot(posArray[axis_move], posArray[axis_still])
    # ax1.set_title('原始数据：运动轨迹')

    # ax2 = fig.add_subplot(2, 2, 2)
    # ax2.plot(posArray_valid[axis_move], posArray_valid[axis_still])
    # ax2.set_title('start 时刻后的数据：运动轨迹')

    x_major_locator = MultipleLocator(20)
    # x_minor_locator=MultipleLocator(20)

    ax3 = fig.add_subplot(2, 1, 1)
    ax3.plot(posArray_valid[axis_still])
    ax3.set_title('start 时刻后的数据：时间-静止坐标')
    ax3.axvline(actBegin, c='r')
    ax3.xaxis.set_major_locator(x_major_locator)
    # ax3.xaxis.set_minor_locator(x_minor_locator)
    ax3.grid(axis='x', which='major')
    # ax3.grid()

    ax4 = fig.add_subplot(2, 1, 2)
    ax4.plot(posArray_valid[axis_move])
    ax4.set_title('start 时刻后的数据：时间-运动坐标')
    ax4.axvline(actBegin, c='r')
    ax4.xaxis.set_major_locator(x_major_locator)
    # ax4.xaxis.set_minor_locator(x_minor_locator)
    ax4.grid(axis='x', which='major')
    # ax4.grid()

    plt.subplots_adjust(left=None,
                        bottom=None,
                        right=None,
                        top=None,
                        wspace=None,
                        hspace=0)

    plt.show()
    actBeginInput = input('输入运动开始时刻：')
    if (actBeginInput != ''):
        actBegin = actBeginInput
    print('返回值：', float(TIME_START + float(actBegin) / FS))
    return float(TIME_START + float(actBegin) / FS)
    # end getAction


# 提取运动轨迹，手动标记运动开始时刻，并保存
def getActionBegin():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    # NAME_LIST = ['cwm']
    for name in NAME_LIST:
        pathList = [
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\上\\',
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\下\\',
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\左\\',
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\右\\'
        ]

        FS_EEG = 1000  #EEG采样频率
        FS_POS = 60  #POS采样频率

        # 提取原始数据，预处理 + 特征提取
        for pi in range(len(pathList)):
            print('\n\ngetActiongBegin：++++++++++++++++\n\n', pi)
            #遍历各动作集
            path = pathList[pi]  #动作文件路径

            for trial_i in range(1, 51):
                # 遍历各组实验
                if 'actBeginTime_%d.txt' % trial_i in os.listdir(path):
                    print('+')
                    continue
                # 提取标签值，活动开始时刻索引，活动结束时刻索引
                print(path + 'posxyz_%d.txt' % trial_i)
                file_POS = open(path + 'posxyz_%d.txt' % trial_i, 'r')
                lines = file_POS.readlines()
                file_POS.close()
                dataArray = np.zeros((12, len(lines)))
                for pos_j, line in zip(range(len(lines)), lines):
                    wordList = line.split('\t')
                    floatList = [float(xi) for xi in wordList[:-1]
                                 ]  #不转换行尾的换行符(原始数据中，最后一个是 '\n' )
                    dataArray[:, pos_j] = np.array(floatList)
                # end for

                actBeginTime = getAction(dataArray[[0, 2], :],
                                         ['up', 'down', 'left',
                                          'right'][pi])  #计算标签值
                print('运动开始时刻（毫秒）：', actBeginTime * FS_EEG)

                file = open(path + 'actBeginTime_%d.txt' % trial_i, 'w')
                file.write('%d\n' % int(actBeginTime * FS_EEG))
                file.close()

    # end getActionBegin


# 从 EEG txt文件中提取格式化的 EEG 数据，预处理，并保存（单任务运动，EEG 信号格式：numpy 三维数组(各样本 * 各通道 * 采样点))）
def get_EEG_single_move_beginTimeFile():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    # NAME_LIST = ['cwm']
    ICACHANNEl_DICT = {
        'cw': [],
        'cwm': [],
        'kx': [],
        'pbl': [],
        'wrd': [],
        'wxc': [],
        'xsc': [],
        'yzg': []
    }
    for name in NAME_LIST:
        pathList = [
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\上\\',
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\下\\',
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\左\\',
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\右\\'
        ]

        FS_EEG = 1000  #EEG采样频率
        FS_POS = 60  #POS采样频率
        TIME_START = 3  #实验开始时刻
        TIME_CUE = 6  #任务开始时刻

        dataArrayList = []  #初始化 EEG 信号列表
        labelList = []  #初始化标签向量列表

        # 提取原始数据，预处理 + 特征提取
        for pi in range(len(pathList)):
            print('\n\nget_EEG_single_move_beginTimeFile：++++++++++++++++\n\n',
                  pi)
            #遍历各动作集
            path = pathList[pi]  #动作文件路径

            for trial_i in range(1, 51):
                # 遍历各组实验

                # 提取标签值，活动开始时刻索引，活动结束时刻索引
                file_POS = open(path + 'actBeginTime_%d.txt' % trial_i, 'r')
                lines = file_POS.readlines()

                beginTime = int(lines[0])

                label_i = [pi, beginTime]
                labelList.append(label_i)
                # end for

                # EEG 信号预处理
                file_EEG = open(path + 'fwj_poseeg%d.txt' % trial_i, 'r')
                lines = file_EEG.readlines()
                dataArray = np.zeros((64, len(lines)))
                for eeg_j, line in zip(range(len(lines)), lines):
                    wordList = line.split('\t')
                    floatList = [float(xi) for xi in wordList[:-2]
                                 ]  #不转换行尾的换行符(原始数据中，最后一个是 '\n' ，倒数第二个是 '0' )
                    dataArray[:, eeg_j] = np.array(floatList)
                # end for
                file_EEG.close()
                dataArray_use = dataArray[[
                    14, 12, 10, 8, 7, 9, 11, 13, 15, 20, 16, 19, 32, 30, 28,
                    26, 25, 27, 29, 31, 33, 37, 36, 47, 45, 43, 42, 44, 46, 48,
                    49, 56
                ], :]  #筛选有效的通道
                ##################################
                # 在此处使用 ICA
                ##################################

                dataArray_preprocessed = preprocessing_EEG_single(
                    dataArray_use, FS_EEG)  # 数据预处理
                # print('使用的数据段：',int(FS_EEG * label_i[1] /FS_POS)-500,'   ',int(FS_EEG * label_i[1] /FS_POS)+500)
                # feature_i = EEG_featureExtraction(
                #     dataArray_preprocessed[:,
                #                            int(FS_EEG * label_i[1] /FS_POS)-500:int(FS_EEG * label_i[1] /FS_POS)+500],
                #     FS_EEG)  # 特征提取(motion_start 前后 0.5 秒段)

                dataArrayList.append(dataArray_preprocessed[:, :18000])

                # print(
                #     'get_EEG_single_move_artificial_features：feature_i.shape=',
                #     feature_i.shape)

                # print(
                #     'get_EEG_single_move_artificial_features：len(featureList)=',
                #     len(featureList))
                # print(
                #     'get_EEG_single_move_artificial_features：len(labelList)=',
                #     len(labelList))
                # print(
                #     'get_EEG_single_move_artificial_features：++++++++++++++++')
            # end for
        # end for

        dataArraySet = np.array(
            dataArrayList)  # EEG 信号(numpy 三维数组(各样本 * 各通道 * 采样点))
        labelArray = np.array(labelList).reshape(
            (len(labelList),
             -1))  #样本集标签矩阵(numpy 二维数组(行：各样本；列：标签值，活动开始时刻索引，活动结束时刻索引))
        print('get_EEG_single_move_beginTimeFile：' + name + ' 样本矩阵提取完成')
        print('get_EEG_single_move_beginTimeFile：labelArray.shape = ',
              labelArray.shape)
        print('get_EEG_single_move_beginTimeFile：dataArraySet.shape = ',
              dataArraySet.shape)
        path = os.getcwd()
        fileD = open(
            path + r'\\..\data\\' + name + '\\单任务运动\\' +
            r'\\single_move_eeg_with_label_beginTimeFile.pickle', 'wb')
        pickle.dump([dataArraySet, labelArray], fileD)
        fileD.close()
    # end for
    # end get_EEG_single_move_beginTimeFile

# 从 EEG txt文件中提取不同通道数的（通道按上下二分类排序）格式化的 EEG 数据，预处理，并保存（单任务运动，EEG 信号格式：numpy 三维数组(各样本 * 各通道 * 采样点))）
def get_EEG_single_move_beginTimeFile_UD2class_ChannelSelect():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    CHANNEL_SIZES = np.linspace(4, 32, 15).astype(int)
    # NAME_LIST = ['cwm']
    ICACHANNEl_DICT = {
        'cw': [],
        'cwm': [],
        'kx': [],
        'pbl': [],
        'wrd': [],
        'wxc': [],
        'xsc': [],
        'yzg': []
    }
    for channel_size in CHANNEL_SIZES:
        for name in NAME_LIST:
            pathList = [
                'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\上\\',
                'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\下\\',
                'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\左\\',
                'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\右\\'
            ]

            FS_EEG = 1000  #EEG采样频率
            FS_POS = 60  #POS采样频率
            TIME_START = 3  #实验开始时刻
            TIME_CUE = 6  #任务开始时刻

            dataArrayList = []  #初始化 EEG 信号列表
            labelList = []  #初始化标签向量列表

            # 提取原始数据，预处理 + 特征提取
            for pi in range(len(pathList)):
                print('\n\nget_EEG_single_move_beginTimeFile：++++++++++++++++\n\n',
                    pi)
                #遍历各动作集
                path = pathList[pi]  #动作文件路径

                for trial_i in range(1, 51):
                    # 遍历各组实验

                    # 提取标签值，活动开始时刻索引，活动结束时刻索引
                    file_POS = open(path + 'actBeginTime_%d.txt' % trial_i, 'r')
                    lines = file_POS.readlines()

                    beginTime = int(lines[0])

                    label_i = [pi, beginTime]
                    labelList.append(label_i)
                    # end for

                    # EEG 信号预处理
                    file_EEG = open(path + 'fwj_poseeg%d.txt' % trial_i, 'r')
                    lines = file_EEG.readlines()
                    dataArray = np.zeros((64, len(lines)))
                    for eeg_j, line in zip(range(len(lines)), lines):
                        wordList = line.split('\t')
                        floatList = [float(xi) for xi in wordList[:-2]
                                    ]  #不转换行尾的换行符(原始数据中，最后一个是 '\n' ，倒数第二个是 '0' )
                        dataArray[:, eeg_j] = np.array(floatList)
                    # end for
                    file_EEG.close()
                    # dataArray_use = dataArray[[37, 12, 28, 29, 26, 30, 44, 42, 56, 48, 16, 49, 15, 10, 14, 8, 19, 25, 32, 11, 43, 27, 20, 7, 9, 13, 45, 36, 31, 46, 47, 33], :]  #筛选有效的通道
                    dataArray_use = dataArray[[
                        14, 12, 10, 8, 7, 9, 11, 13, 15, 20, 16, 19, 32, 30, 28,
                        26, 25, 27, 29, 31, 33, 37, 36, 47, 45, 43, 42, 44, 46, 48,
                        49, 56
                    ], :] 
                    ##################################
                    # 在此处使用 ICA
                    ##################################

                    dataArray_preprocessed = preprocessing_EEG_single(
                        dataArray_use, FS_EEG)  # 数据预处理
                    dataArrayList.append(dataArray_preprocessed[:, :18000])
                # end for
            # end for

            dataArraySet = np.array(
                dataArrayList)  # EEG 信号(numpy 三维数组(各样本 * 各通道 * 采样点))
            labelArray = np.array(labelList).reshape(
                (len(labelList),
                -1))  #样本集标签矩阵(numpy 二维数组(行：各样本；列：标签值，活动开始时刻索引，活动结束时刻索引))
            print('get_EEG_single_move_beginTimeFile：' + name + ' 样本矩阵提取完成')
            print('get_EEG_single_move_beginTimeFile：labelArray.shape = ',
                labelArray.shape)
            print('get_EEG_single_move_beginTimeFile：dataArraySet.shape = ',
                dataArraySet.shape)


            labelList = labelArray[:, 0]  #标签向量列表
            featureList = []  #初始化特征向量列表
            for dataArray, yI in zip(dataArraySet, labelArray):
                actBeginTime = yI[1]
                dataArray_preprocessed = preprocessing_EEG_single(
                    dataArray[:channel_size,:], FS_EEG)  # 数据预处理
                feature_i = EEG_featureExtraction(
                    dataArray_preprocessed[:,
                                        actBeginTime - 640:actBeginTime + 640],
                    FS_EEG)  # 特征提取(motion_start 前后 640ms 段)
                featureList.append(feature_i)
            # end for

            featureArray = np.array(featureList)  #样本集特征矩阵(numpy 二维数组(行：各样本；列：特征值))
            labelArray = np.array(labelList).reshape(
                (len(labelList), -1))  #样本集标签矩阵(numpy 二维数组(行：各样本；列：标签))
            print('get_EEG_features：' + name + ' 样本矩阵提取完成')
            print('get_EEG_features：featureArray.shape = ', featureArray.shape)
            print('get_EEG_features：labelArray.shape = ', labelArray.shape)

            fileName = 'single_move_ChannelSelect_UD2class_%d.pickle' % channel_size
            path = os.getcwd()
            fileD = open(
                path + r'\\..\data\\' + name + '\\四分类\\' +fileName,
                'wb')
            pickle.dump([featureArray, labelArray], fileD)
            fileD.close()        
        # end for
    # end for
    # end get_EEG_single_move_beginTimeFile_UD2class_ChannelSelect


# 从 EEG txt文件中提取不同通道数的（通道按左右二分类排序）格式化的 EEG 数据，预处理，并保存（单任务运动，EEG 信号格式：numpy 三维数组(各样本 * 各通道 * 采样点))）
def get_EEG_single_move_beginTimeFile_LR2class_ChannelSelect():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    CHANNEL_SIZES = np.linspace(4, 32, 15).astype(int)
    # NAME_LIST = ['cwm']
    ICACHANNEl_DICT = {
        'cw': [],
        'cwm': [],
        'kx': [],
        'pbl': [],
        'wrd': [],
        'wxc': [],
        'xsc': [],
        'yzg': []
    }
    for channel_size in CHANNEL_SIZES:
        for name in NAME_LIST:
            pathList = [
                'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\上\\',
                'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\下\\',
                'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\左\\',
                'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\右\\'
            ]

            FS_EEG = 1000  #EEG采样频率
            FS_POS = 60  #POS采样频率
            TIME_START = 3  #实验开始时刻
            TIME_CUE = 6  #任务开始时刻

            dataArrayList = []  #初始化 EEG 信号列表
            labelList = []  #初始化标签向量列表

            # 提取原始数据，预处理 + 特征提取
            for pi in range(len(pathList)):
                print('\n\nget_EEG_single_move_beginTimeFile：++++++++++++++++\n\n',
                    pi)
                #遍历各动作集
                path = pathList[pi]  #动作文件路径

                for trial_i in range(1, 51):
                    # 遍历各组实验

                    # 提取标签值，活动开始时刻索引，活动结束时刻索引
                    file_POS = open(path + 'actBeginTime_%d.txt' % trial_i, 'r')
                    lines = file_POS.readlines()

                    beginTime = int(lines[0])

                    label_i = [pi, beginTime]
                    labelList.append(label_i)
                    # end for

                    # EEG 信号预处理
                    file_EEG = open(path + 'fwj_poseeg%d.txt' % trial_i, 'r')
                    lines = file_EEG.readlines()
                    dataArray = np.zeros((64, len(lines)))
                    for eeg_j, line in zip(range(len(lines)), lines):
                        wordList = line.split('\t')
                        floatList = [float(xi) for xi in wordList[:-2]
                                    ]  #不转换行尾的换行符(原始数据中，最后一个是 '\n' ，倒数第二个是 '0' )
                        dataArray[:, eeg_j] = np.array(floatList)
                    # end for
                    file_EEG.close()
                    # dataArray_use = dataArray[[45, 49, 25, 43, 19, 30, 20, 29, 32, 26, 47, 16, 9, 13, 33, 10, 44, 31, 7, 15, 14, 11, 46, 56, 36, 8, 48, 37, 27, 28, 42, 12], :]  #筛选有效的通道
                    dataArray_use = dataArray[[
                        14, 12, 10, 8, 7, 9, 11, 13, 15, 20, 16, 19, 32, 30, 28,
                        26, 25, 27, 29, 31, 33, 37, 36, 47, 45, 43, 42, 44, 46, 48,
                        49, 56
                    ], :]                     
                    ##################################
                    # 在此处使用 ICA
                    ##################################

                    dataArray_preprocessed = preprocessing_EEG_single(
                        dataArray_use, FS_EEG)  # 数据预处理
                    # print('使用的数据段：',int(FS_EEG * label_i[1] /FS_POS)-500,'   ',int(FS_EEG * label_i[1] /FS_POS)+500)
                    # feature_i = EEG_featureExtraction(
                    #     dataArray_preprocessed[:,
                    #                            int(FS_EEG * label_i[1] /FS_POS)-500:int(FS_EEG * label_i[1] /FS_POS)+500],
                    #     FS_EEG)  # 特征提取(motion_start 前后 0.5 秒段)

                    dataArrayList.append(dataArray_preprocessed[:, :18000])

                    # print(
                    #     'get_EEG_single_move_artificial_features：feature_i.shape=',
                    #     feature_i.shape)

                    # print(
                    #     'get_EEG_single_move_artificial_features：len(featureList)=',
                    #     len(featureList))
                    # print(
                    #     'get_EEG_single_move_artificial_features：len(labelList)=',
                    #     len(labelList))
                    # print(
                    #     'get_EEG_single_move_artificial_features：++++++++++++++++')
                # end for
            # end for

            dataArraySet = np.array(
                dataArrayList)  # EEG 信号(numpy 三维数组(各样本 * 各通道 * 采样点))
            labelArray = np.array(labelList).reshape(
                (len(labelList),
                -1))  #样本集标签矩阵(numpy 二维数组(行：各样本；列：标签值，活动开始时刻索引，活动结束时刻索引))
            print('get_EEG_single_move_beginTimeFile：' + name + ' 样本矩阵提取完成')
            print('get_EEG_single_move_beginTimeFile：labelArray.shape = ',
                labelArray.shape)
            print('get_EEG_single_move_beginTimeFile：dataArraySet.shape = ',
                dataArraySet.shape)
            # path = os.getcwd()
            # fileD = open(
            #     path + r'\\..\data\\' + name + '\\四分类\\%d\\'%size +
            #     r'\\single_move_eeg_with_label_beginTimeFile.pickle', 'wb')
            # pickle.dump([dataArraySet, labelArray], fileD)
            # fileD.close()

            labelList = labelArray[:, 0]  #标签向量列表
            featureList = []  #初始化特征向量列表
            for dataArray, yI in zip(dataArraySet, labelArray):
                actBeginTime = yI[1]
                dataArray_preprocessed = preprocessing_EEG_single(
                    dataArray[:channel_size,:], FS_EEG)  # 数据预处理
                feature_i = EEG_featureExtraction(
                    dataArray_preprocessed[:,
                                        actBeginTime - 640:actBeginTime + 640],
                    FS_EEG)  # 特征提取(motion_start 前后 640ms 段)
                featureList.append(feature_i)
            # end for

            featureArray = np.array(featureList)  #样本集特征矩阵(numpy 二维数组(行：各样本；列：特征值))
            labelArray = np.array(labelList).reshape(
                (len(labelList), -1))  #样本集标签矩阵(numpy 二维数组(行：各样本；列：标签))
            print('get_EEG_features：' + name + ' 样本矩阵提取完成')
            print('get_EEG_features：featureArray.shape = ', featureArray.shape)
            print('get_EEG_features：labelArray.shape = ', labelArray.shape)

            fileName = 'single_move_ChannelSelect_LR2class_%d.pickle' % channel_size
            path = os.getcwd()
            fileD = open(
                path + r'\\..\data\\' + name + '\\四分类\\' +fileName,
                'wb')
            pickle.dump([featureArray, labelArray], fileD)
            fileD.close()        
        # end for
    # end for
    # end get_EEG_single_move_beginTimeFile_LR2class_ChannelSelect

# 从 EEG txt文件中提取格式化的 EEG 数据，预处理，并保存（单任务运动，EEG 信号格式：numpy 三维数组(各样本 * 各通道 * 采样点))）
def get_EEG_single_move():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    # NAME_LIST = ['cwm']
    ICACHANNEl_DICT = {
        'cw': [],
        'cwm': [],
        'kx': [],
        'pbl': [],
        'wrd': [],
        'wxc': [],
        'xsc': [],
        'yzg': []
    }
    for name in NAME_LIST:
        pathList = [
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\上\\',
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\下\\',
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\左\\',
            'D:\\实验数据\\费炜杰的实验\\' + name + '\\单任务运动\\右\\'
        ]

        FS_EEG = 1000  #EEG采样频率
        FS_POS = 60  #POS采样频率
        TIME_START = 3  #实验开始时刻
        TIME_CUE = 6  #任务开始时刻

        dataArrayList = []  #初始化 EEG 信号列表
        labelList = []  #初始化标签向量列表

        # 提取原始数据，预处理 + 特征提取
        for pi in range(len(pathList)):
            print('\n\nget_EEG_single_move：++++++++++++++++\n\n', pi)
            #遍历各动作集
            path = pathList[pi]  #动作文件路径

            for trial_i in range(1, 51):
                # 遍历各组实验

                # 提取标签值，活动开始时刻索引，活动结束时刻索引
                file_POS = open(path + 'posxyz_%d.txt' % trial_i, 'r')
                lines = file_POS.readlines()
                dataArray = np.zeros((12, len(lines)))
                for pos_j, line in zip(range(len(lines)), lines):
                    wordList = line.split('\t')
                    floatList = [float(xi) for xi in wordList[:-1]
                                 ]  #不转换行尾的换行符(原始数据中，最后一个是 '\n' )
                    dataArray[:, pos_j] = np.array(floatList)
                # end for
                label_i = getLabel_POS_single(dataArray[[0, 2], :],
                                              ['up', 'down', 'left',
                                               'right'][pi])  #计算标签值
                label_i = [pi, label_i[1], label_i[2]]
                labelList.append(label_i)
                # end for

                # EEG 信号预处理
                file_EEG = open(path + 'fwj_poseeg%d.txt' % trial_i, 'r')
                lines = file_EEG.readlines()
                dataArray = np.zeros((64, len(lines)))
                for eeg_j, line in zip(range(len(lines)), lines):
                    wordList = line.split('\t')
                    floatList = [float(xi) for xi in wordList[:-2]
                                 ]  #不转换行尾的换行符(原始数据中，最后一个是 '\n' ，倒数第二个是 '0' )
                    dataArray[:, eeg_j] = np.array(floatList)
                # end for
                file_EEG.close()
                dataArray_use = dataArray[[
                    7, 8, 9, 10, 11, 34, 13, 14, 15, 16, 19, 20, 25, 26, 27,
                    28, 29, 30, 31, 32, 33, 36, 37, 42, 43, 44, 45, 46, 47, 48,
                    49, 56
                ], :]  #筛选有效的通道
                ##################################
                # 在此处使用 ICA
                ##################################

                dataArray_preprocessed = preprocessing_EEG_single(
                    dataArray_use, FS_EEG)  # 数据预处理
                # print('使用的数据段：',int(FS_EEG * label_i[1] /FS_POS)-500,'   ',int(FS_EEG * label_i[1] /FS_POS)+500)
                # feature_i = EEG_featureExtraction(
                #     dataArray_preprocessed[:,
                #                            int(FS_EEG * label_i[1] /FS_POS)-500:int(FS_EEG * label_i[1] /FS_POS)+500],
                #     FS_EEG)  # 特征提取(motion_start 前后 0.5 秒段)

                dataArrayList.append(dataArray_preprocessed[:, :18000])

                # print(
                #     'get_EEG_single_move_artificial_features：feature_i.shape=',
                #     feature_i.shape)

                # print(
                #     'get_EEG_single_move_artificial_features：len(featureList)=',
                #     len(featureList))
                # print(
                #     'get_EEG_single_move_artificial_features：len(labelList)=',
                #     len(labelList))
                # print(
                #     'get_EEG_single_move_artificial_features：++++++++++++++++')
            # end for
        # end for

        dataArraySet = np.array(
            dataArrayList)  # EEG 信号(numpy 三维数组(各样本 * 各通道 * 采样点))
        labelArray = np.array(labelList).reshape(
            (len(labelList),
             -1))  #样本集标签矩阵(numpy 二维数组(行：各样本；列：标签值，活动开始时刻索引，活动结束时刻索引))
        print('get_EEG_single_move：' + name + ' 样本矩阵提取完成')
        print('get_EEG_single_move：labelArray.shape = ', labelArray.shape)
        print('get_EEG_single_move：dataArraySet.shape = ', dataArraySet.shape)
        path = os.getcwd()
        fileD = open(
            path + r'\\..\data\\' + name + '\\单任务运动\\' +
            r'\\single_move_eeg_with_label.pickle', 'wb')
        pickle.dump([dataArraySet, labelArray], fileD)
        fileD.close()
    # end for
    # end get_EEG_single_move


# 提取多被试样本集，并保存
# 数据格式：
#   样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def getSampleSetMulti_beginTimeFile():
    print('getSampleSetMulti_beginTimeFile 四分类（上、下、左、右）跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    FS = 1000  # 采样频率
    BANDRANGE = [[4, 8], [8, 13], [13, 30]]  # EEG 图片高度维度上的划分
    path = os.getcwd()
    X_sampleSetMulti = []
    Y_sampleSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_eeg_with_label_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))
        dataSet = []
        for dataArray, yI in zip(X, y):
            # 提取 频带分通道后的时域信号(numpy 三维数组(各频带 * 各数据通道 * 各采样点))
            actBeginTime = yI[1]
            pictureI = getOriginalPicture(
                dataArray[:, actBeginTime - 640:actBeginTime + 1000], FS,
                BANDRANGE)
            dataSet.append(pictureI)
        dataSet = np.array(dataSet)
        # 提取 样本集_Y(numpy 一维数组)
        y = y[:, 0]

        # 构造样本集
        X_sampleSetMulti.append(dataSet)
        Y_sampleSetMulti.append(y)
        print('getSampleSetMulti_beginTimeFile：样本数量', sum(y == 0), sum(y == 1),
              sum(y == 2), sum(y == 3))
    # end for

    path = os.getcwd()
    fileD = open(
        path + r'\\..\data\\单任务运动\\' +
        r'\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'wb')
    pickle.dump([X_sampleSetMulti, Y_sampleSetMulti], fileD)
    fileD.close()
    # end getSampleSetMulti_beginTimeFile


# 提取多被试样本集，并保存
# 数据格式：
#   样本集_X(numpy 四维数组(各样本 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def getSampleSet_beginTimeFile():
    print('getSampleSet_beginTimeFile 四分类（上、下、左、右）跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    path = os.getcwd()
    X_sampleSetMulti = []
    Y_sampleSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_eeg_with_label_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))
        dataSet = []
        for dataArray, yI in zip(X, y):
            # 提取 频带分通道后的时域信号(numpy 三维数组(各频带 * 各数据通道 * 各采样点))
            actBeginTime = yI[1]
            pictureI = dataArray[:, actBeginTime - 640:actBeginTime + 1000]
            dataSet.append(pictureI)
        dataSet = np.array(dataSet)
        
        # 提取 样本集_Y(numpy 一维数组)
        y = y[:, 0]

        # 构造样本集
        X_sampleSetMulti.append(dataSet)
        Y_sampleSetMulti.append(y)
        print('getSampleSet_beginTimeFile：样本数量', sum(y == 0), sum(y == 1),
              sum(y == 2), sum(y == 3))
    # end for

    path = os.getcwd()
    fileD = open(
        path + r'\\..\data\\四分类\\' +
        r'\\single_move_crossSubject_sampleSet_X_y.pickle', 'wb')
    pickle.dump([X_sampleSetMulti, Y_sampleSetMulti], fileD)
    fileD.close()
    # end getSampleSet_beginTimeFile


# 提取多被试样本集，并保存
# 数据格式：
#   样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def getSampleSetMulti():
    print('getSampleSetMulti 四分类（上、下、左、右）跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    FS = 1000  # 采样频率
    BANDRANGE = [[4, 8], [8, 13], [13, 30]]  # EEG 图片高度维度上的划分
    path = os.getcwd()
    X_sampleSetMulti = []
    Y_sampleSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\单任务运动' +
            '\\single_move_eeg_with_label.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 截断 EEG 信号
        X = X[:, :, 6000:8000]

        # 提取 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))
        dataSet = []
        for dataArray in X:
            # 提取 频带分通道后的时域信号(numpy 三维数组(各频带 * 各数据通道 * 各采样点))
            pictureI = getOriginalPicture(dataArray, FS, BANDRANGE)
            dataSet.append(pictureI)
        dataSet = np.array(dataSet)
        # 提取 样本集_Y(numpy 一维数组)
        y = y[:, 0]

        # 构造样本集
        X_sampleSetMulti.append(dataSet)
        Y_sampleSetMulti.append(y)
        print('getSampleSetMulti：样本数量', sum(y == 0), sum(y == 1), sum(y == 2),
              sum(y == 3))
    # end for

    path = os.getcwd()
    fileD = open(
        path + r'\\..\data\\单任务运动\\' +
        r'\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'wb')
    pickle.dump([X_sampleSetMulti, Y_sampleSetMulti], fileD)
    fileD.close()
    # end getSampleSetMulti


# 从原始 EEG 数据中提取特征及标签，预处理，特征提取，并保存（多任务运动，人工特征，标签 0123 分别对应上下左右）
def get_EEG_features_multi_task():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    FS_EEG=1000
    dataPath='D:\实验数据\费炜杰的实验\多任务\EEG_raw_for_2_back.mat'
    file=h5py.File(dataPath,'r')
    for name in NAME_LIST:
        dataRow=np.array(file['EEG_raw_%s'%name])
        featureList=[] # 初始化特征向量列表    
        labelList = [] # 初始化标签向量列表    
        # 提取“上”
        for i in range(50):
            dataArray=dataRow[i,0,:,:]
            dataArray_use = dataArray[[
                14, 12, 10, 8, 7, 9, 11, 13, 15, 20, 16, 19, 32, 30, 28,
                26, 25, 27, 29, 31, 33, 37, 36, 47, 45, 43, 42, 44, 46, 48,
                49, 56
            ], :]  #筛选有效的通道   
            dataArray_preprocessed = preprocessing_EEG_single(
                dataArray_use, FS_EEG)    
            
            feature_i = EEG_featureExtraction(
                dataArray_preprocessed[:,2360:2360+1280],FS_EEG)  # 特征提取(运动开始时刻前后 640ms 段)            

            featureList.append(feature_i)
            labelList.append(0) 
        # 提取“下”
        for i in range(50):
            dataArray=dataRow[i,1,:,:]
            dataArray_use = dataArray[[
                14, 12, 10, 8, 7, 9, 11, 13, 15, 20, 16, 19, 32, 30, 28,
                26, 25, 27, 29, 31, 33, 37, 36, 47, 45, 43, 42, 44, 46, 48,
                49, 56
            ], :]  #筛选有效的通道   
            dataArray_preprocessed = preprocessing_EEG_single(
                dataArray_use, FS_EEG)    
            
            feature_i = EEG_featureExtraction(
                dataArray_preprocessed[:,2360:2360+1280],FS_EEG)  # 特征提取(运动开始时刻前后 640ms 段)            

            featureList.append(feature_i)
            labelList.append(1)         
        # 提取“左”
        for i in range(50):
            dataArray=dataRow[i,2,:,:]
            dataArray_use = dataArray[[
                14, 12, 10, 8, 7, 9, 11, 13, 15, 20, 16, 19, 32, 30, 28,
                26, 25, 27, 29, 31, 33, 37, 36, 47, 45, 43, 42, 44, 46, 48,
                49, 56
            ], :]  #筛选有效的通道   
            dataArray_preprocessed = preprocessing_EEG_single(
                dataArray_use, FS_EEG)    
            
            feature_i = EEG_featureExtraction(
                dataArray_preprocessed[:,2360:2360+1280],FS_EEG)  # 特征提取(运动开始时刻前后 640ms 段)            

            featureList.append(feature_i)
            labelList.append(2)     
        # 提取“右”
        for i in range(50):
            dataArray=dataRow[i,3,:,:]
            dataArray_use = dataArray[[
                14, 12, 10, 8, 7, 9, 11, 13, 15, 20, 16, 19, 32, 30, 28,
                26, 25, 27, 29, 31, 33, 37, 36, 47, 45, 43, 42, 44, 46, 48,
                49, 56
            ], :]  #筛选有效的通道   
            dataArray_preprocessed = preprocessing_EEG_single(
                dataArray_use, FS_EEG)    
            
            feature_i = EEG_featureExtraction(
                dataArray_preprocessed[:,2360:2360+1280],FS_EEG)  # 特征提取(运动开始时刻前后 640ms 段)            

            featureList.append(feature_i)
            labelList.append(3)     
        featureArray = np.array(
            featureList)  # EEG 信号(numpy 三维数组(各样本 * 各通道 * 采样点))
        labelArray = np.array(labelList)  # 样本集标签矩阵(numpy 二维数组(行：各样本；列：标签值))
    
        print('get_EEG_multi_task：' + name + ' 样本矩阵提取完成')
        print('get_EEG_multi_task：labelArray.shape = ',
                labelArray.shape)
        print('get_EEG_multi_task：featureArray.shape = ',
                featureArray.shape)
        path = os.getcwd()
        fileD = open(
            path + r'\\..\data\\' + name + '\\多任务\\' +
            r'\\multi_task_motion_start_motion_end.pickle', 'wb')
        pickle.dump([featureArray, labelArray], fileD)
        fileD.close()   
    # end for
    # end get_EEG_features_multi_task

# getActionBegin()

# get_EEG_single_move_artificial_features()
save_ICA()

##### 使用的数据：自动判断开始运动时刻，基本没用
# get_EEG_single_move()
# getSampleSetMulti()
##### 使用的数据：自动判断开始运动时刻，基本没用

##### 使用的数据：手动判断开始运动时刻
# get_EEG_features()
# get_EEG_features_multiSegment()
# get_EEG_features_multiFrame()
# get_EEG_single_move_beginTimeFile()
# getSampleSet_beginTimeFile() 
# getSampleSetMulti_beginTimeFile()
# get_EEG_single_move_beginTimeFile_UD2class_ChannelSelect()
# get_EEG_single_move_beginTimeFile_LR2class_ChannelSelect()

# get_EEG_features_multi_task()