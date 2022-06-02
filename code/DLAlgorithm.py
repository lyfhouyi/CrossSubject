# 深度学习算法

import os
import math
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair, _single, _triple
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset

import Plot
import SignalPreprocessing as sp

NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']


class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)


def conv2d_same_padding(input,
                        weight,
                        bias=None,
                        stride=1,
                        padding=1,
                        dilation=1,
                        groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input,
                    weight,
                    bias,
                    stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation,
                    groups=groups)
    # end conv2d_same_padding


class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels // groups,
                             *kernel_size))
        else:
            self.weight = Parameter(
                torch.Tensor(out_channels, in_channels // groups,
                             *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # end __init__

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        # end for
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # end if
        # end reset_parameters

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0, ) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
        # end __repr__

    # end _ConvNd


class Conv2d(_ConvNd):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, False, _pair(0), groups, bias)
        # end __init__

    # 修改这里的实现函数
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
        # end forward

    # end Conv2d

def print_Net_parameters():
    netDict={'Net2':Net2(),'Net2g':Net2g(),'Net22':Net22(),'Net25':Net25(),'Net26':Net26(),'Net27':Net27(),'Net28':Net28(),'Net29':Net29()}
    for netName in netDict:
        net_tmp = netDict[netName]
        print("\n{}：参数个数 {}  \n明细：".format(netName,sum(x.numel() for x in net_tmp.parameters())))
       
        for x in net_tmp.parameters():
            print(x.data.shape, len(x.data.reshape(-1)))

    
    # end print_Net_parameters

# 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def CNN_cross_subject_4class_pretraining_Net4():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\四分类\\CNN_Net4_pretraining' +
        r'\\learning_curve_CNN_Net4_pretraining_4class.txt', 'w')
    fileLog.write('CNN 四分类器_预训练（上、下、左、右）跨被试\n')
    print('CNN 四分类器_预训练（上、下、左、右）跨被试')

    # 适用于 Net4、Net49
    FRAME_CNT = 4  # 帧数
    FRAME_WIDTH = 480  # 帧宽
    FRAME_INC = 240  # 帧移
    # 适用于 Net4、Net49

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 30)
    classes = ('上', '下', '左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net4()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("CNN_cross_subject_4class_pretraining_Net4：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "CNN_cross_subject_4class_pretraining_Net4：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y
        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        # 信号分帧
        X_train_framed = []
        for X_train_I in X_train:
            X_train_framed.append(
                sp.enFrame3D(X_train_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_train = np.array(X_train_framed)
        X_test_framed = []
        for X_test_I in X_test:
            X_test_framed.append(
                sp.enFrame3D(X_test_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
            pass
        X_test = np.array(X_test_framed)

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,135]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net4()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.0005,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(300):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.9):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\四分类\\CNN_Net4_pretraining' +
                            '\\%s_%s_Net4.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'CNN_cross_subject_4class_pretraining_Net4：%s 训练集容量 %d 训练完成' %
                (NAME_LIST[subjectNo], size))
            fileLog.write(
                'CNN_cross_subject_4class_pretraining_Net4：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})

        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_cross_subject_4class_pretraining_Net4：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_cross_subject_4class_pretraining_Net4：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\四分类\\CNN_Net4_pretraining' +
        r'\\learning_curve_CNN_Net4_pretraining_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\四分类\\CNN_Net4_pretraining' +
        r'\\confusion_matrix_CNN_Net4_pretraining_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_Net4_pretraining_4class')
    # end CNN_cross_subject_4class_pretraining_Net4


# 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def CNN_cross_subject_4class_pretraining_Net42():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\四分类\\CNN_Net42_pretraining' +
        r'\\learning_curve_CNN_Net42_pretraining_4class.txt', 'w')
    fileLog.write('CNN 四分类器_预训练（上、下、左、右）跨被试\n')
    print('CNN 四分类器_预训练（上、下、左、右）跨被试')

    # 适用于 Net42、Net48
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 160  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net42、Net48

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 30)
    classes = ('上', '下', '左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net42()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("CNN_cross_subject_4class_pretraining_Net42：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "CNN_cross_subject_4class_pretraining_Net42：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y
        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        # 信号分帧
        X_train_framed = []
        for X_train_I in X_train:
            X_train_framed.append(
                sp.enFrame3D(X_train_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_train = np.array(X_train_framed)
        X_test_framed = []
        for X_test_I in X_test:
            X_test_framed.append(
                sp.enFrame3D(X_test_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
            pass
        X_test = np.array(X_test_framed)

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net42()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.0001,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(500):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.9):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\四分类\\CNN_Net42_pretraining' +
                            '\\%s_%s_Net42.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'CNN_cross_subject_4class_pretraining_Net42：%s 训练集容量 %d 训练完成' %
                (NAME_LIST[subjectNo], size))
            fileLog.write(
                'CNN_cross_subject_4class_pretraining_Net42：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_cross_subject_4class_pretraining_Net42：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_cross_subject_4class_pretraining_Net42：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\四分类\\CNN_Net42_pretraining' +
        r'\\learning_curve_CNN_Net42_pretraining_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\四分类\\CNN_Net42_pretraining' +
        r'\\confusion_matrix_CNN_Net42_pretraining_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_Net42_pretraining_4class')
    # end CNN_cross_subject_4class_pretraining_Net42


# 样本集_X(numpy 四维数组(各样本 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def LSTM_cross_subject_4class_pretraining_Net44():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\四分类\\LSTM_Net44_pretraining' +
        r'\\learning_curve_LSTM_Net44_pretraining_4class.txt', 'w')
    fileLog.write('LSTM 四分类器_预训练（上、下、左、右）跨被试\n')
    print('LSTM 四分类器_预训练（上、下、左、右）跨被试')


    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    classes = ('上', '下', '左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSet_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()


    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[2],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 对数据样本转置，以匹配 LSTM 输入要求
    X_sampleSetMulti = [
        np.swapaxes(X_sampleSet,1,2) for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net44()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("LSTM_cross_subject_4class_pretraining_Net44：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "LSTM_cross_subject_4class_pretraining_Net44：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[:,:256,:] #取前 1280ms 的数据

        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][:,:256,:]] #取前 1280ms 的数据
                y_train = np.r_[y_train, Y_sampleSetMulti[j]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size,:], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net44()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.005,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(400):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\四分类\\LSTM_Net44_pretraining' +
                            '\\%s_%s_Net44.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'LSTM_cross_subject_4class_pretraining_Net44：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'LSTM_cross_subject_4class_pretraining_Net44：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_4class_pretraining_Net44：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_4class_pretraining_Net44：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\四分类\\LSTM_Net44_pretraining' +
        r'\\learning_curve_LSTM_Net44_pretraining_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\四分类\\LSTM_Net44_pretraining' +
        r'\\confusion_matrix_LSTM_Net44_pretraining_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_4class('LSTM_Net44_pretraining')
    fileD = open(
        path + r'\\..\model_all\\四分类\\LSTM_Net44_pretraining' +
        r'\\learning_curve_LSTM_Net44_pretraining_4class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net44_pretraining_4class')
    # end LSTM_cross_subject_4class_pretraining_Net44


# 样本集_X(numpy 二维数组(各样本 * 各特征))，样本集_Y(numpy 一维数组)
def NN_cross_subject_4class_pretraining_Net45():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\四分类\\NN_Net45_pretraining' +
        r'\\learning_curve_NN_Net45_pretraining_4class.txt', 'w')
    fileLog.write('NN 四分类器（上、下、左、右）跨被试\n')
    print('NN 四分类器（上、下、左、右）跨被试')
    TRAIN_SIZES = np.linspace(0.1, 0.5, 30)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = np.c_[X, y]
        print('NN_cross_subject_4class_pretraining_Net45：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    net_tmp = Net45()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("NN_cross_subject_4class_pretraining_Net45：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "NN_cross_subject_4class_pretraining_Net45：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        dataSet_test = dataSetMulti[subjectNo]  # 留下的用于测试的被试样本

        # 提取用于训练的被试集样本
        dataSet_train = np.empty((0, dataSet.shape[1]))
        for j in range(n_subjects):
            if j != subjectNo:
                dataSet_train = np.r_[dataSet_train, dataSetMulti[j]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(dataSet_train.shape[0])
        dataSet_train = dataSet_train[shuffle_index]

        # 训练集特征缩放
        scaler = StandardScaler()
        X_train_scalered = scaler.fit_transform(dataSet_train[:, :-1])
        dataSet_train_scalered = np.c_[X_train_scalered,
                                       dataSet_train[:, -1]]  #特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        # 记录训练集特征缩放参数，测试时会用到
        mean_ = scaler.mean_  #均值
        scale_ = scaler.scale_  #标准差(=np.std(axis=0),注：无ddof=1)

        # 对测试样本集进行特征缩放
        X_test_scalered = (dataSet_test[:, :-1] - mean_) / scale_
        dataSet_test_scalered = np.c_[X_test_scalered,
                                      dataSet_test[:, -1]]  #特征缩放后的测试样本集(X,y)
        dataSet_test_scalered[np.isnan(dataSet_test_scalered)] = 0  # 处理异常特征

        t_sizes = dataSet_train_scalered.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [561,410,560]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(dataSet_train_scalered[:size, :-1],
                                        dataSet_train_scalered[:size,
                                                               -1], transform)
            test_dataset = data_loader(dataSet_test_scalered[:, :-1],
                                       dataSet_test_scalered[:, -1], transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net45()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.002,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            epoch = 0
            while epoch < 20000:
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                # print('test_accu_best = ',test_accu_best)

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.95):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\四分类\\NN_Net45_pretraining' +
                            '\\%s_%s_Net45.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
                epoch += 1
                if running_loss / running_loss_initial < 0.2:
                    break
            # end while
            print(
                'NN_cross_subject_4class_pretraining_Net45：%s 训练集容量 %d 训练完成' %
                (NAME_LIST[subjectNo], size))
            fileLog.write(
                'NN_cross_subject_4class_pretraining_Net45：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'NN_cross_subject_4class_pretraining_Net45：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\四分类\\NN_Net45_pretraining' +
        r'\\learning_curve_NN_Net45_pretraining_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\四分类\\NN_Net45_pretraining' +
        r'\\confusion_matrix_NN_Net45_pretraining_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'NN_Net45_pretraining_4class')
    # end NN_cross_subject_4class_pretraining_Net45


# 样本集_X(numpy 三维数组(各样本 * 各帧 * 各特征))，样本集_Y(numpy 一维数组)
def LSTM_cross_subject_4class_pretraining_Net46():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\四分类\\LSTM_Net46_pretraining' +
        r'\\learning_curve_LSTM_Net46_pretraining_4class.txt', 'w')
    fileLog.write('LSTM 四分类器（上、下、左、右）跨被试\n')
    print('LSTM 四分类器（上、下、左、右）跨被试')
    TRAIN_SIZES = np.linspace(0.1, 0.5, 30)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = [X, y]
        print('LSTM_cross_subject_4class_pretraining_Net46：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    net_tmp = Net46()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("LSTM_cross_subject_4class_pretraining_Net46：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "LSTM_cross_subject_4class_pretraining_Net46：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        # 留下的用于测试的被试样本
        dataSet_test_X = dataSetMulti[subjectNo][0]
        dataSet_test_Y = dataSetMulti[subjectNo][1]
        # 提取用于训练的被试集样本
        dataSet_train_X = np.empty(
            (0, dataSet[0].shape[1], dataSet[0].shape[2]))
        dataSet_train_Y = np.empty((0, dataSet[1].shape[1]))
        for j in range(n_subjects):
            if j != subjectNo:
                dataSet_train_X = np.r_[dataSet_train_X, dataSetMulti[j][0]]
                dataSet_train_Y = np.r_[dataSet_train_Y, dataSetMulti[j][1]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(dataSet_train_X.shape[0])
        dataSet_train_X = dataSet_train_X[shuffle_index]
        dataSet_train_Y = dataSet_train_Y[shuffle_index]

        dataSet_train_scalered_X = dataSet_train_X
        # 训练集特征缩放
        # scaler = StandardScaler()
        # dataSet_train_scalered_X = scaler.fit_transform(dataSet_train_X[:, :,:])
        dataSet_train_scalered_X[np.isnan(
            dataSet_train_scalered_X)] = 0  # 处理异常特征

        # 记录训练集特征缩放参数，测试时会用到
        # mean_ = scaler.mean_  #均值
        # scale_ = scaler.scale_  #标准差(=np.std(axis=0),注：无ddof=1)

        dataSet_test_scalered = dataSet_test_X
        # 对测试样本集进行特征缩放
        # dataSet_test_scalered = (dataSet_test_X[:,:,:] - mean_) / scale_
        dataSet_test_scalered[np.isnan(dataSet_test_scalered)] = 0  # 处理异常特征

        t_sizes = dataSet_train_scalered_X.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [140,410,560]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(dataSet_train_scalered_X[:size, :],
                                        dataSet_train_Y[:size, :], transform)
            test_dataset = data_loader(dataSet_test_scalered, dataSet_test_Y,
                                       transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=64,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net46()
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.0001)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(800):  # loop over the dataset multiple times
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long().view(-1))
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                # print('test_accu_best = ',test_accu_best)

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels.view(-1)).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels.view(-1)).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    print('\t\ttrain_accu_cur = ', train_accu_cur)
                    if (epoch == 0) or (test_accu_best < test_accu_cur):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\四分类\\LSTM_Net46_pretraining' +
                            '\\%s_%s_Net46.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            # end for
            print(
                'LSTM_cross_subject_4class_pretraining_Net46：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'LSTM_cross_subject_4class_pretraining_Net46：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_4class_pretraining_Net46：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\四分类\\LSTM_Net46_pretraining' +
        r'\\learning_curve_LSTM_Net46_pretraining_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\四分类\\LSTM_Net46_pretraining' +
        r'\\confusion_matrix_LSTM_Net46_pretraining_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net46_pretraining_4class')
    # end LSTM_cross_subject_4class_pretraining_Net46


# 样本集_X(numpy 三维数组(各样本 * 各帧 * 各特征))，样本集_Y(numpy 一维数组)
def LSTM_cross_subject_4class_pretraining_Net47():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\四分类\\LSTM_Net47_pretraining' +
        r'\\learning_curve_LSTM_Net47_pretraining_4class.txt', 'w')
    fileLog.write('LSTM 四分类器（上、下、左、右）跨被试\n')
    print('LSTM 四分类器（上、下、左、右）跨被试')
    TRAIN_SIZES = np.linspace(0.1, 0.5, 30)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = [X, y]
        print('LSTM_cross_subject_4class_pretraining_Net47：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    net_tmp = Net47()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("LSTM_cross_subject_4class_pretraining_Net47：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "LSTM_cross_subject_4class_pretraining_Net47：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        # 留下的用于测试的被试样本
        dataSet_test_X = dataSetMulti[subjectNo][0]
        dataSet_test_Y = dataSetMulti[subjectNo][1]
        # 提取用于训练的被试集样本
        dataSet_train_X = np.empty(
            (0, dataSet[0].shape[1], dataSet[0].shape[2]))
        dataSet_train_Y = np.empty((0, dataSet[1].shape[1]))
        for j in range(n_subjects):
            if j != subjectNo:
                dataSet_train_X = np.r_[dataSet_train_X, dataSetMulti[j][0]]
                dataSet_train_Y = np.r_[dataSet_train_Y, dataSetMulti[j][1]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(dataSet_train_X.shape[0])
        dataSet_train_X = dataSet_train_X[shuffle_index]
        dataSet_train_Y = dataSet_train_Y[shuffle_index]

        dataSet_train_scalered_X = dataSet_train_X
        # 训练集特征缩放
        # scaler = StandardScaler()
        # dataSet_train_scalered_X = scaler.fit_transform(dataSet_train_X[:, :,:])
        dataSet_train_scalered_X[np.isnan(
            dataSet_train_scalered_X)] = 0  # 处理异常特征

        # 记录训练集特征缩放参数，测试时会用到
        # mean_ = scaler.mean_  #均值
        # scale_ = scaler.scale_  #标准差(=np.std(axis=0),注：无ddof=1)

        dataSet_test_scalered = dataSet_test_X
        # 对测试样本集进行特征缩放
        # dataSet_test_scalered = (dataSet_test_X[:,:,:] - mean_) / scale_
        dataSet_test_scalered[np.isnan(dataSet_test_scalered)] = 0  # 处理异常特征

        t_sizes = dataSet_train_scalered_X.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [140,410,560]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(dataSet_train_scalered_X[:size, :],
                                        dataSet_train_Y[:size, :], transform)
            test_dataset = data_loader(dataSet_test_scalered, dataSet_test_Y,
                                       transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=64,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net47()
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.0001)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(800):  # loop over the dataset multiple times
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long().view(-1))
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                # print('test_accu_best = ',test_accu_best)

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels.view(-1)).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels.view(-1)).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    print('\t\ttrain_accu_cur = ', train_accu_cur)
                    if (epoch == 0) or (test_accu_best < test_accu_cur):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\四分类\\LSTM_Net47_pretraining' +
                            '\\%s_%s_Net47.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            # end for
            print(
                'LSTM_cross_subject_4class_pretraining_Net47：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'LSTM_cross_subject_4class_pretraining_Net47：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_4class_pretraining_Net47：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\四分类\\LSTM_Net47_pretraining' +
        r'\\learning_curve_LSTM_Net47_pretraining_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\四分类\\LSTM_Net47_pretraining' +
        r'\\confusion_matrix_LSTM_Net47_pretraining_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net47_pretraining_4class')
    # end LSTM_cross_subject_4class_pretraining_Net47


# 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def CNN_LSTM_cross_subject_4class_pretraining_Net48():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\四分类\\CNN_LSTM_Net48_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net48_pretraining_4class.txt', 'w')
    fileLog.write('CNN_LSTM 四分类器_预训练（上、下、左、右）跨被试\n')
    print('CNN_LSTM 四分类器_预训练（上、下、左、右）跨被试')

    # 适用于 Net42、Net48
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 160  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net42、Net48

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 30)
    classes = ('上', '下', '左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net48()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("CNN_LSTM_cross_subject_4class_pretraining_Net48：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "CNN_LSTM_cross_subject_4class_pretraining_Net48：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y
        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        # 信号分帧
        X_train_framed = []
        for X_train_I in X_train:
            X_train_framed.append(
                sp.enFrame3D(X_train_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_train = np.array(X_train_framed)
        X_test_framed = []
        for X_test_I in X_test:
            X_test_framed.append(
                sp.enFrame3D(X_test_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
            pass
        X_test = np.array(X_test_framed)

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net48()
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.0002)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(500):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    print('\t\ttrain_accu_cur = ', train_accu_cur)
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\四分类\\CNN_LSTM_Net48_pretraining'
                            + '\\%s_%s_Net48.pkl' %
                            (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'CNN_LSTM_cross_subject_4class_pretraining_Net48：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'CNN_LSTM_cross_subject_4class_pretraining_Net48：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_LSTM_cross_subject_4class_pretraining_Net48：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_LSTM_cross_subject_4class_pretraining_Net48：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\四分类\\CNN_LSTM_Net48_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net48_pretraining_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\四分类\\CNN_LSTM_Net48_pretraining' +
        r'\\confusion_matrix_CNN_LSTM_Net48_pretraining_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_LSTM_Net48_pretraining_4class')
    # end CNN_LSTM_cross_subject_4class_pretraining_Net48


# 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def CNN_LSTM_cross_subject_4class_pretraining_Net49():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\四分类\\CNN_LSTM_Net49_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net49_pretraining_4class.txt', 'w')
    fileLog.write('CNN_LSTM 四分类器_预训练（上、下、左、右）跨被试\n')
    print('CNN_LSTM 四分类器_预训练（上、下、左、右）跨被试')

    # 适用于 Net4、Net49
    FRAME_CNT = 4  # 帧数
    FRAME_WIDTH = 480  # 帧宽
    FRAME_INC = 240  # 帧移
    # 适用于 Net4、Net49

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 30)
    classes = ('上', '下', '左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net49()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("CNN_LSTM_cross_subject_4class_pretraining_Net49：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "CNN_LSTM_cross_subject_4class_pretraining_Net49：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y
        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        # 信号分帧
        X_train_framed = []
        for X_train_I in X_train:
            X_train_framed.append(
                sp.enFrame3D(X_train_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_train = np.array(X_train_framed)
        X_test_framed = []
        for X_test_I in X_test:
            X_test_framed.append(
                sp.enFrame3D(X_test_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
            pass
        X_test = np.array(X_test_framed)

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net49()
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.0001)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(300):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    print('\t\ttrain_accu_cur = ', train_accu_cur)
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\四分类\\CNN_LSTM_Net49_pretraining'
                            + '\\%s_%s_Net49.pkl' %
                            (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'CNN_LSTM_cross_subject_4class_pretraining_Net49：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'CNN_LSTM_cross_subject_4class_pretraining_Net49：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_LSTM_cross_subject_4class_pretraining_Net49：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_LSTM_cross_subject_4class_pretraining_Net49：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\四分类\\CNN_LSTM_Net49_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net49_pretraining_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\四分类\\CNN_LSTM_Net49_pretraining' +
        r'\\confusion_matrix_CNN_LSTM_Net49_pretraining_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_LSTM_Net49_pretraining_4class')
    # end CNN_LSTM_cross_subject_4class_pretraining_Net49


def CNN_cross_subject_4class_fine_Net4():
    fileLog = open(
        os.getcwd() + r'\\..\model\\四分类\\CNN_Net4_fine' +
        r'\\learning_curve_CNN_Net4_fine_4class.txt', 'w')
    fileLog.write('CNN 四分类器_微调（上、下、左、右）跨被试\n')
    print('CNN 四分类器_微调（上、下、左、右）跨被试')

    # 适用于 Net4、Net49
    FRAME_CNT = 4  # 帧数
    FRAME_WIDTH = 480  # 帧宽
    FRAME_INC = 240  # 帧移
    # 适用于 Net4、Net49

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 50)
    classes = ('上', '下', '左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\四分类\\CNN_Net4_pretraining' + '\\%s_622_Net4.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        print('CNN_cross_subject_4class_fine_Net4：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 信号分帧
        X_framed = []
        for X_I in X:
            X_framed.append(
                sp.enFrame3D(X_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X = np.array(X_framed)

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.9:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print('CNN_cross_subject_4class_fine_Net4：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'CNN_cross_subject_4class_fine_Net4：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_cross_subject_4class_fine_Net4：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_cross_subject_4class_fine_Net4：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\CNN_Net4_fine' +
        r'\\learning_curve_CNN_Net4_fine_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\四分类\\CNN_Net4_fine' +
        r'\\confusion_matrix_CNN_Net4_fine_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_Net4_fine_4class')

    # end CNN_cross_subject_4class_fine_Net4


def CNN_cross_subject_4class_fine_Net42():
    fileLog = open(
        os.getcwd() + r'\\..\model\\四分类\\CNN_Net42_fine' +
        r'\\learning_curve_CNN_Net42_fine_4class.txt', 'w')
    fileLog.write('CNN 四分类器_微调（上、下、左、右）跨被试\n')
    print('CNN 四分类器_微调（上、下、左、右）跨被试')

    # 适用于 Net42、Net48
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 160  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net42、Net48

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 50)
    classes = ('上', '下', '左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\四分类\\CNN_Net42_pretraining' + '\\%s_661_Net42.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        print('CNN_cross_subject_4class_fine_Net42：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 信号分帧
        X_framed = []
        for X_I in X:
            X_framed.append(
                sp.enFrame3D(X_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X = np.array(X_framed)

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.9:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print('CNN_cross_subject_4class_fine_Net42：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'CNN_cross_subject_4class_fine_Net42：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_cross_subject_4class_fine_Net42：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_cross_subject_4class_fine_Net42：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\CNN_Net42_fine' +
        r'\\learning_curve_CNN_Net42_fine_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\四分类\\CNN_Net42_fine' +
        r'\\confusion_matrix_CNN_Net42_fine_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_Net42_fine_4class')
    # end CNN_cross_subject_4class_fine_Net42


def NN_cross_subject_4class_fine_Net45():
    fileLog = open(
        os.getcwd() + r'\\..\model\\四分类\\NN_Net45_fine' +
        r'\\learning_curve_NN_Net45_fine_4class.txt', 'w')
    fileLog.write('NN 四分类器_微调（上、下、左、右）跨被试\n')
    print('NN 四分类器_微调（上、下、左、右）跨被试')

    FINE_SIZES = np.linspace(0, 1.0, 50)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = np.c_[X, y]
        print('NN_cross_subject_4class_fine_Net45：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\四分类\\NN_Net45_pretraining' + '\\%s_700_Net45.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        dataSet = dataSetMulti[subjectNo]  # 取出留下的被试样本集

        # 样本集特征缩放
        scaler = StandardScaler()
        X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_train_scalered = np.c_[X_train_scalered,
                                       dataSet[:, -1]]  #特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        print('NN_cross_subject_4class_fine_Net45：样本数量',
              sum(dataSet_train_scalered[:, -1] == 0),
              sum(dataSet_train_scalered[:, -1] == 1),
              sum(dataSet_train_scalered[:, -1] == 2),
              sum(dataSet_train_scalered[:, -1] == 3))

        fine_set, test_set = train_test_split(
            dataSet_train_scalered, test_size=0.1)  #划分微调训练集、测试集(默认洗牌)
        X_fine = fine_set[:, :-1]  #微调训练集特征矩阵
        y_fine = fine_set[:, -1]  #微调训练集标签数组
        X_test = test_set[:, :-1]  #测试集特征矩阵
        y_test = test_set[:, -1]  #测试集标签数组

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                epoch = 0
                while epoch < 20000:
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    # print('[%d] loss: %.3f ,%.3f%%' %
                    #       (epoch + 1, running_loss,
                    #        100*running_loss / running_loss_initial))
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))

                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #       (epoch + 1, running_loss,
                        #        100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)

                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.9:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                    epoch += 1
                    if running_loss / running_loss_initial < 0.2:
                        break
                # end while
                print('NN_cross_subject_4class_fine_Net45：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'NN_cross_subject_4class_fine_Net45：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'NN_cross_subject_4class_fine_Net45：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'NN_cross_subject_4class_fine_Net45：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\NN_Net45_fine' +
        r'\\learning_curve_NN_Net45_fine_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\四分类\\NN_Net45_fine' +
        r'\\confusion_matrix_NN_Net45_fine_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'NN_Net45_fine_4class')
    # end NN_cross_subject_4class_fine_Net45


def LSTM_cross_subject_4class_fine_Net46():
    fileLog = open(
        os.getcwd() + r'\\..\model\\四分类\\LSTM_Net46_fine' +
        r'\\learning_curve_LSTM_Net46_fine_4class.txt', 'w')
    fileLog.write('LSTM 四分类器_微调（上、下、左、右）跨被试\n')
    print('LSTM 四分类器_微调（上、下、左、右）跨被试')

    FINE_SIZES = np.linspace(0, 1.0, 50)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = [X, y]
        print('LSTM_cross_subject_4class_fine_Net46：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\四分类\\LSTM_Net46_pretraining' + '\\%s_487_Net46.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        # 取出留下的被试样本集
        dataSet_test_X = dataSetMulti[subjectNo][0]
        dataSet_test_Y = dataSetMulti[subjectNo][1]
        dataSet_train_scalered = dataSet_test_X
        # 样本集特征缩放
        # scaler = StandardScaler()
        # X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
        # dataSet_train_scalered = np.c_[X_train_scalered,
        #                                dataSet[:, -1]]  # 特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        print('LSTM_cross_subject_4class_fine_Net46：样本数量',
              sum(dataSet_test_Y == 0), sum(dataSet_test_Y == 1),
              sum(dataSet_test_Y == 2), sum(dataSet_test_Y == 3))

        X_test, y_test, X_fine, y_fine = sp.splitSet(
            dataSet_train_scalered, dataSet_test_Y, ratio=0.1,
            shuffle=True)  # 划分微调训练集、测试集(洗牌)

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [180]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.0001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels.view(-1)).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        800):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long().view(-1))
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    # print('[%d] loss: %.3f ,%.3f%%' %
                    #       (epoch + 1, running_loss,
                    #        100*running_loss / running_loss_initial))
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels.view(-1)).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels.view(-1)).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))

                        print('\t\t[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('\t\ttest_accu_best = ', test_accu_best)
                        print('\t\ttest_accu_cur = ', test_accu_cur)
                        print('\t\ttrain_accu_cur = ', train_accu_cur)

                        if test_accu_best < test_accu_cur:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                # end for
                print('LSTM_cross_subject_4class_fine_Net46：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'LSTM_cross_subject_4class_fine_Net46：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_4class_fine_Net46：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_4class_fine_Net46：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\LSTM_Net46_fine' +
        r'\\learning_curve_LSTM_Net46_fine_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\四分类\\LSTM_Net46_fine' +
        r'\\confusion_matrix_LSTM_Net46_fine_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net46_fine_4class')
    # end LSTM_cross_subject_4class_fine_Net46


def LSTM_cross_subject_4class_fine_Net47():
    fileLog = open(
        os.getcwd() + r'\\..\model\\四分类\\LSTM_Net47_fine' +
        r'\\learning_curve_LSTM_Net47_fine_4class.txt', 'w')
    fileLog.write('LSTM 四分类器_微调（上、下、左、右）跨被试\n')
    print('LSTM 四分类器_微调（上、下、左、右）跨被试')

    FINE_SIZES = np.linspace(0, 1.0, 50)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = [X, y]
        print('LSTM_cross_subject_4class_pretraining_Net47：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\四分类\\LSTM_Net47_pretraining' + '\\%s_680_Net47.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        # 取出留下的被试样本集
        dataSet_test_X = dataSetMulti[subjectNo][0]
        dataSet_test_Y = dataSetMulti[subjectNo][1]
        dataSet_train_scalered = dataSet_test_X
        # 样本集特征缩放
        # scaler = StandardScaler()
        # X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
        # dataSet_train_scalered = np.c_[X_train_scalered,
        #                                dataSet[:, -1]]  # 特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        print('LSTM_cross_subject_4class_fine_Net47：样本数量',
              sum(dataSet_test_Y == 0), sum(dataSet_test_Y == 1),
              sum(dataSet_test_Y == 2), sum(dataSet_test_Y == 3))

        X_test, y_test, X_fine, y_fine = sp.splitSet(
            dataSet_train_scalered, dataSet_test_Y, ratio=0.1,
            shuffle=True)  # 划分微调训练集、测试集(洗牌)

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [180]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.0001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels.view(-1)).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        800):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long().view(-1))
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    # print('[%d] loss: %.3f ,%.3f%%' %
                    #       (epoch + 1, running_loss,
                    #        100*running_loss / running_loss_initial))
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels.view(-1)).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels.view(-1)).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))

                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #       (epoch + 1, running_loss,
                        #        100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)

                        if test_accu_best < test_accu_cur:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                # end for
                print('LSTM_cross_subject_4class_fine_Net47：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'LSTM_cross_subject_4class_fine_Net47：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_4class_fine_Net47：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_4class_fine_Net47：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\LSTM_Net47_fine' +
        r'\\learning_curve_LSTM_Net47_fine_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\四分类\\LSTM_Net47_fine' +
        r'\\confusion_matrix_LSTM_Net47_fine_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net47_fine_4class')
    # end LSTM_cross_subject_4class_fine_Net47


def CNN_LSTM_cross_subject_4class_fine_Net48():
    fileLog = open(
        os.getcwd() + r'\\..\model\\四分类\\CNN_LSTM_Net48_fine' +
        r'\\learning_curve_CNN_LSTM_Net48_fine_4class.txt', 'w')
    fileLog.write('CNN_LSTM 四分类器_微调（上、下、左、右）跨被试\n')
    print('CNN_LSTM 四分类器_微调（上、下、左、右）跨被试')

    # 适用于 Net42、Net48
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 160  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net42、Net48

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 50)
    classes = ('上', '下', '左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\四分类\\CNN_LSTM_Net48_pretraining' + '\\%s_603_Net48.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        print('CNN_LSTM_cross_subject_4class_fine_Net48：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 信号分帧
        X_framed = []
        for X_I in X:
            X_framed.append(
                sp.enFrame3D(X_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X = np.array(X_framed)

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.0001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #     (epoch + 1, running_loss,
                        #     100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.97:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print(
                    'CNN_LSTM_cross_subject_4class_fine_Net48：%s 训练集容量 %d 训练完成'
                    % (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'CNN_LSTM_cross_subject_4class_fine_Net48：%s 训练集容量 %d 训练完成\n'
                    % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_LSTM_cross_subject_4class_fine_Net48：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_LSTM_cross_subject_4class_fine_Net48：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\CNN_LSTM_Net48_fine' +
        r'\\learning_curve_CNN_LSTM_Net48_fine_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\四分类\\CNN_LSTM_Net48_fine' +
        r'\\confusion_matrix_CNN_LSTM_Net48_fine_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.myLearning_curve_LIST2DICT_4class('CNN_LSTM_Net48_fine')
    fileD = open(
        path + r'\\..\model\\四分类\\CNN_LSTM_Net48_fine' +
        r'\\learning_curve_CNN_LSTM_Net48_fine_4class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_LSTM_Net48_fine_4class')
    # end CNN_LSTM_cross_subject_4class_fine_Net48


def CNN_LSTM_cross_subject_4class_fine_Net49():
    fileLog = open(
        os.getcwd() + r'\\..\model\\四分类\\CNN_LSTM_Net49_fine' +
        r'\\learning_curve_CNN_LSTM_Net49_fine_4class.txt', 'w')
    fileLog.write('CNN_LSTM 四分类器_微调（上、下、左、右）跨被试\n')
    print('CNN_LSTM 四分类器_微调（上、下、左、右）跨被试')

    # 适用于 Net4、Net49
    FRAME_CNT = 4  # 帧数
    FRAME_WIDTH = 480  # 帧宽
    FRAME_INC = 240  # 帧移
    # 适用于 Net4、Net49

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 50)
    classes = ('上', '下', '左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\四分类\\CNN_LSTM_Net49_pretraining' + '\\%s_680_Net49.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        print('CNN_LSTM_cross_subject_4class_fine_Net49：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 信号分帧
        X_framed = []
        for X_I in X:
            X_framed.append(
                sp.enFrame3D(X_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X = np.array(X_framed)

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.00005)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #     (epoch + 1, running_loss,
                        #     100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.98:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print(
                    'CNN_LSTM_cross_subject_4class_fine_Net49：%s 训练集容量 %d 训练完成'
                    % (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'CNN_LSTM_cross_subject_4class_fine_Net49：%s 训练集容量 %d 训练完成\n'
                    % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_LSTM_cross_subject_4class_fine_Net49：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_LSTM_cross_subject_4class_fine_Net49：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\CNN_LSTM_Net49_fine' +
        r'\\learning_curve_CNN_LSTM_Net49_fine_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\四分类\\CNN_LSTM_Net49_fine' +
        r'\\confusion_matrix_CNN_LSTM_Net49_fine_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.myLearning_curve_LIST2DICT_4class('CNN_LSTM_Net49_fine')
    fileD = open(
        path + r'\\..\model\\四分类\\CNN_LSTM_Net49_fine' +
        r'\\learning_curve_CNN_LSTM_Net49_fine_4class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_LSTM_Net49_fine_4class')
    # end CNN_LSTM_cross_subject_4class_fine_Net49


class Net4(nn.Module):
    # 输入数据：
    #   4 帧数据，帧宽 480ms，帧移 160ms，采样频率 200 Hz
    #   各帧：3 频率通道 * 32 通道 * 采样点
    def __init__(self):
        super(Net4, self).__init__()
        self.pool1 = nn.MaxPool2d((2, 4), 2)
        self.pool2 = nn.MaxPool2d((3, 6), 2)
        self.conv1 = nn.Conv2d(3, 6, (5, 15))
        self.conv2 = nn.Conv2d(6, 8, (5, 15))
        self.conv3 = Conv2d(8, 8, (3, 3))

        self.fc1 = nn.Linear(4 * 8 * 2 * 4, 32)
        self.fc2 = nn.Linear(32, 4)

        self.dropout = nn.Dropout(p=0.4)

        self.seq1 = nn.Sequential(self.conv1, nn.ReLU(), self.pool1,
                                  self.conv2, nn.ReLU(), self.pool1,
                                  self.conv3, nn.ReLU(), self.pool2)
        self.seq2 = nn.Sequential(self.fc1, self.dropout, nn.ReLU(), self.fc2)
        # end __init__

    def forward(self, x):
        # CNN 模块
        output1 = torch.empty((x.shape[0], 0)).double()
        for step in range(4):
            xStep = self.seq1(x[:, step, :, :, :])
            xStep = xStep.view(-1, 8 * 2 * 4)
            output1 = torch.cat((output1, xStep), 1)

        # 全连接模块
        output2 = self.seq2(output1)

        return output2
        # end forward


class Net42(nn.Module):
    # 输入数据：
    #   7 帧数据，帧宽 160ms，帧移 160ms，采样频率 200 Hz
    #   各帧：3 频率通道 * 32 通道 * 采样点
    def __init__(self):
        super(Net42, self).__init__()
        self.pool = nn.AvgPool2d((2, 2), 2)
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 8, (5, 5))

        self.fc1 = nn.Linear(7 * 8 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 4)

        self.dropout = nn.Dropout(p=0.4)

        self.seq1 = nn.Sequential(self.conv1, nn.ReLU(), self.pool, self.conv2,
                                  nn.ReLU(), self.pool)
        self.seq2 = nn.Sequential(self.fc1, self.dropout, nn.ReLU(), self.fc2)
        # end __init__

    def forward(self, x):
        # CNN 模块
        output1 = torch.empty((x.shape[0], 0)).double()
        for step in range(7):
            xStep = self.seq1(x[:, step, :, :, :])
            xStep = xStep.view(-1, 8 * 5 * 5)
            output1 = torch.cat((output1, xStep), 1)

        # 全连接模块
        output2 = self.seq2(output1)

        return output2
        # end forward


# class Net43(nn.Module):
#     # 输入数据：
#     #   3 帧数据，帧宽 640ms，帧移 320ms，采样频率 200 Hz
#     #   各帧：3 频率通道 * 32 通道 * 采样点
#     def __init__(self):
#         super(Net43, self).__init__()
#         self.pool = nn.AvgPool2d((2, 2), 2)
#         self.conv1 = nn.Conv2d(3, 6, (5, 5))
#         self.conv2 = nn.Conv2d(6, 8, (5, 5))

#         self.fc1 = nn.Linear(7 * 8 * 5 * 5, 64)
#         self.fc2 = nn.Linear(64, 4)

#         self.seq1 = nn.Sequential(self.conv1, nn.ReLU(), self.pool, self.conv2,
#                                   nn.ReLU(), self.pool)
#         self.seq2 = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)
#         # end __init__

#     def forward(self, x):
#         # CNN 模块
#         output1 = torch.empty((x.shape[0], 0))
#         for step in range(7):
#             xStep = self.seq1(x[:, step, :, :, :])
#             xStep = xStep.view(-1, 8 * 5 * 5)
#             output1 = torch.cat((output1, xStep), 1)

#         # 全连接模块
#         output2 = self.seq2(output1)

#         return output2
#         # end forward


# # 训练时间过长，弃用
# class Net44(nn.Module):
#     # 输入数据：
#     #   4 帧数据，帧宽 480ms，帧移 160ms，采样频率 200 Hz
#     #   各帧：3 频率通道 * 32 通道 * 采样点
#     def __init__(self):
#         super(Net44, self).__init__()
#         self.pool1 = nn.MaxPool2d((2, 4), 2)
#         self.pool2 = nn.MaxPool2d((3, 6), 2)
#         self.conv1 = nn.Conv2d(3, 6, (5, 15))
#         self.conv2 = nn.Conv2d(6, 16, (5, 15))
#         self.conv3 = Conv2d(16, 32, (3, 3))
#         self.conv4 = Conv2d(32, 32, (3, 3))
#         self.conv5 = Conv2d(32, 8, (3, 3))

#         self.fc1 = nn.Linear(4 * 8 * 2 * 4, 32)
#         self.fc2 = nn.Linear(32, 4)

#         self.dropout = nn.Dropout(p=0.4)

#         self.seq1 = nn.Sequential(self.conv1, nn.ReLU(), self.pool1,
#                                   self.conv2, nn.ReLU(), self.pool1,
#                                   self.conv3, nn.ReLU(), self.conv4, nn.ReLU(),
#                                   self.conv5, nn.ReLU(), self.pool2)
#         self.seq2 = nn.Sequential(self.fc1, self.dropout, nn.ReLU(), self.fc2)
#         # end __init__

#     def forward(self, x):
#         # CNN 模块
#         output1 = torch.empty((x.shape[0], 0))
#         for step in range(4):
#             xStep = self.seq1(x[:, step, :, :, :])
#             xStep = xStep.view(-1, 8 * 2 * 4)
#             output1 = torch.cat((output1, xStep), 1)

#         # 全连接模块
#         output2 = self.seq2(output1)

#         return output2
#         # end forward

class Net44(nn.Module):
    # 输入数据：
    #   256 时间步，每个时间步 32 通道脑电信号值
    #   各帧：原始信号
    def __init__(self):
        super(Net44, self).__init__()
        self.lstm = nn.LSTM(32, 16, 3, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(256 * 16, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 4)
        self.dropout = nn.Dropout(p=0.4)

        self.seq1 = nn.Sequential(self.lstm)
        self.seq2 = nn.Sequential(self.fc1, self.dropout, nn.ReLU(), self.fc2,self.dropout, nn.ReLU(), self.fc3)
        # end __init__

    def forward(self, x):
        # LSTM 模块
        output1, _ = self.seq1(x)
        output1 = output1.reshape(output1.shape[0], -1)

        # 全连接模块
        output2 = self.seq2(output1)

        return output2
        # end forward


class Net45(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net45, self).__init__()
        self.fc1 = nn.Linear(224, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 4)

        self.dropout = nn.Dropout(p=0.4)
        self.seq1 = nn.Sequential(self.fc1, self.dropout, nn.ReLU(), self.fc2,
                                  self.dropout, nn.ReLU(), self.fc3,
                                  self.dropout, nn.ReLU(), self.fc4,
                                  self.dropout, nn.ReLU())
        self.seq2 = nn.Sequential(self.fc5, self.dropout, nn.ReLU(), self.fc6)
        # end __init__

    def forward(self, x):
        # 全连接预训练模块
        output1 = self.seq1(x)

        # 全连接微调模块
        output2 = self.seq2(output1)

        return output2
        # end forward


class Net46(nn.Module):
    # 输入数据：
    #   7 帧数据，帧宽 320ms，帧移 160ms
    #   各帧：人工特征
    def __init__(self):
        super(Net46, self).__init__()
        self.lstm = nn.LSTM(224, 16, 3, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(16, 4)

        self.seq1 = nn.Sequential(self.lstm)
        self.seq2 = nn.Sequential(self.fc)
        # end __init__

    def forward(self, x):
        # LSTM 模块
        output1, _ = self.seq1(x)
        output1 = output1[:, -1, :]

        # 全连接模块
        output2 = self.seq2(output1)

        return output2
        # end forward


class Net47(nn.Module):
    # 输入数据：
    #   7 帧数据，帧宽 320ms，帧移 160ms
    #   各帧：人工特征
    def __init__(self):
        super(Net47, self).__init__()
        self.lstm = nn.LSTM(224, 16, 3, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(7 * 16, 4)

        self.seq1 = nn.Sequential(self.lstm)
        self.seq2 = nn.Sequential(self.fc)
        # end __init__

    def forward(self, x):
        # LSTM 模块
        output1, _ = self.seq1(x)
        output1 = output1.reshape(output1.shape[0], -1)

        # 全连接模块
        output2 = self.seq2(output1)

        return output2
        # end forward


class Net48(nn.Module):
    # 输入数据：
    #   7 帧数据，帧宽 160ms，帧移 160ms，采样频率 200 Hz
    #   各帧：3 频率通道 * 32 通道 * 采样点
    def __init__(self):
        super(Net48, self).__init__()
        self.pool = nn.AvgPool2d((2, 2), 2)
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 8, (5, 5))
        self.lstm = nn.LSTM(8 * 5 * 5, 16, 3, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(16, 4)

        self.seq1 = nn.Sequential(self.conv1, nn.ReLU(), self.pool, self.conv2,
                                  nn.ReLU(), self.pool)
        self.seq2 = nn.Sequential(self.lstm)
        self.seq3 = nn.Sequential(self.fc)

        # end __init__

    def forward(self, x):
        # CNN 模块
        output1 = torch.empty((x.shape[0], 0, 8 * 5 * 5)).double()
        for step in range(x.shape[1]):
            xStep = self.seq1(x[:, step, :, :, :])
            xStep = xStep.view(-1, 8 * 5 * 5).unsqueeze(dim=1)
            output1 = torch.cat((output1, xStep), 1)

        # LSTM 模块
        output2, _ = self.seq2(output1)
        output2 = output2[:, -1, :]

        # 全连接模块
        output3 = self.seq3(output2)

        return output3
        # end forward


class Net49(nn.Module):
    # 输入数据：
    #   4 帧数据，帧宽 480ms，帧移 160ms，采样频率 200 Hz
    #   各帧：3 频率通道 * 32 通道 * 采样点
    def __init__(self):
        super(Net49, self).__init__()
        self.pool1 = nn.MaxPool2d((2, 4), 2)
        self.pool2 = nn.MaxPool2d((3, 6), 2)
        self.conv1 = nn.Conv2d(3, 6, (5, 15))
        self.conv2 = nn.Conv2d(6, 8, (5, 15))
        self.conv3 = Conv2d(8, 8, (3, 3))
        self.lstm = nn.LSTM(8 * 2 * 4, 8, 3, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(32, 4)

        self.seq1 = nn.Sequential(self.conv1, nn.ReLU(), self.pool1,
                                  self.conv2, nn.ReLU(), self.pool1,
                                  self.conv3, nn.ReLU(), self.pool2)
        self.seq2 = nn.Sequential(self.lstm)
        self.seq3 = nn.Sequential(self.fc)

        # end __init__

    def forward(self, x):
        # CNN 模块
        output1 = torch.empty((x.shape[0], 0, 8 * 2 * 4)).double()
        for step in range(x.shape[1]):
            xStep = self.seq1(x[:, step, :, :, :])
            xStep = xStep.view(-1, 8 * 2 * 4).unsqueeze(dim=1)
            output1 = torch.cat((output1, xStep), 1)

        # LSTM 模块
        output2, _ = self.seq2(output1)
        output2 = output2.reshape(output2.shape[0], -1)

        # 全连接模块
        output3 = self.seq3(output2)

        return output3
        # end forward


# 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def CNN_cross_subject_UD2class_pretraining_Net2():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\上下二分类\\CNN_Net2_pretraining' +
        r'\\learning_curve_CNN_Net2_pretraining_UD2class.txt', 'w')
    fileLog.write('CNN 上下二分类器_预训练（上、下）跨被试\n')
    print('CNN 上下二分类器_预训练（上、下）跨被试')

    # 适用于 Net2、Net29
    FRAME_CNT = 4  # 帧数
    FRAME_WIDTH = 480  # 帧宽
    FRAME_INC = 240  # 帧移
    # 适用于 Net2、Net29

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    classes = ('上', '下')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net2()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("CNN_cross_subject_UD2class_pretraining_Net2：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "CNN_cross_subject_UD2class_pretraining_Net2：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[:100]
        y_test = y_test[:100]
        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][:100]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j][:100]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        # 信号分帧
        X_train_framed = []
        for X_train_I in X_train:
            X_train_framed.append(
                sp.enFrame3D(X_train_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_train = np.array(X_train_framed)
        X_test_framed = []
        for X_test_I in X_test:
            X_test_framed.append(
                sp.enFrame3D(X_test_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
            pass
        X_test = np.array(X_test_framed)

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,135]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net2()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.0005,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(300):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.95):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\上下二分类\\CNN_Net2_pretraining' +
                            '\\%s_%s_Net2.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'CNN_cross_subject_UD2class_pretraining_Net2：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'CNN_cross_subject_UD2class_pretraining_Net2：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})

        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_cross_subject_UD2class_pretraining_Net2：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_cross_subject_UD2class_pretraining_Net2：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\CNN_Net2_pretraining' +
        r'\\learning_curve_CNN_Net2_pretraining_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\上下二分类\\CNN_Net2_pretraining' +
        r'\\confusion_matrix_CNN_Net2_pretraining_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('CNN_Net2_pretraining')
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\CNN_Net2_pretraining' +
        r'\\learning_curve_CNN_Net2_pretraining_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_Net2_pretraining_UD2class')
    # end CNN_cross_subject_UD2class_pretraining_Net2


# 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def CNN_cross_subject_UD2class_pretraining_Net22():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\上下二分类\\CNN_Net22_pretraining' +
        r'\\learning_curve_CNN_Net22_pretraining_UD2class.txt', 'w')
    fileLog.write('CNN 上下二分类器_预训练（上、下）跨被试\n')
    print('CNN 上下二分类器_预训练（上、下）跨被试')

    # 适用于 Net22、Net28
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 160  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net22、Net28

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    classes = ('上', '下')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net22()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("CNN_cross_subject_UD2class_pretraining_Net22：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "CNN_cross_subject_UD2class_pretraining_Net22：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[:100]
        y_test = y_test[:100]
        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][:100]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j][:100]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        # 信号分帧
        X_train_framed = []
        for X_train_I in X_train:
            X_train_framed.append(
                sp.enFrame3D(X_train_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_train = np.array(X_train_framed)
        X_test_framed = []
        for X_test_I in X_test:
            X_test_framed.append(
                sp.enFrame3D(X_test_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_test = np.array(X_test_framed)

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net22()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.0001,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(500):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\上下二分类\\CNN_Net22_pretraining' +
                            '\\%s_%s_Net22.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'CNN_cross_subject_UD2class_pretraining_Net22：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'CNN_cross_subject_UD2class_pretraining_Net22：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_cross_subject_UD2class_pretraining_Net22：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_cross_subject_UD2class_pretraining_Net22：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\CNN_Net22_pretraining' +
        r'\\learning_curve_CNN_Net22_pretraining_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\上下二分类\\CNN_Net22_pretraining' +
        r'\\confusion_matrix_CNN_Net22_pretraining_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('CNN_Net22_pretraining')
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\CNN_Net22_pretraining' +
        r'\\learning_curve_CNN_Net22_pretraining_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_Net22_pretraining_UD2class')
    # end CNN_cross_subject_UD2class_pretraining_Net22


# 样本集_X(numpy 四维数组(各样本 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def LSTM_cross_subject_UD2class_pretraining_Net23():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\上下二分类\\LSTM_Net23_pretraining' +
        r'\\learning_curve_LSTM_Net23_pretraining_UD2class.txt', 'w')
    fileLog.write('LSTM 上下二分类器_预训练（上、下）跨被试\n')
    print('LSTM 上下二分类器_预训练（上、下）跨被试')

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    classes = ('上', '下')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSet_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()


    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[2],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 对数据样本转置，以匹配 LSTM 输入要求
    X_sampleSetMulti = [
        np.swapaxes(X_sampleSet,1,2) for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net23()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("LSTM_cross_subject_UD2class_pretraining_Net23：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "LSTM_cross_subject_UD2class_pretraining_Net23：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[:100]
        y_test = y_test[:100]

        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][:100]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j][:100]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net23()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.005,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(400):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\上下二分类\\LSTM_Net23_pretraining' +
                            '\\%s_%s_Net23.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'LSTM_cross_subject_UD2class_pretraining_Net23：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'LSTM_cross_subject_UD2class_pretraining_Net23：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_UD2class_pretraining_Net23：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_UD2class_pretraining_Net23：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\LSTM_Net23_pretraining' +
        r'\\learning_curve_LSTM_Net23_pretraining_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\上下二分类\\LSTM_Net23_pretraining' +
        r'\\confusion_matrix_LSTM_Net23_pretraining_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('LSTM_Net23_pretraining')
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\LSTM_Net23_pretraining' +
        r'\\learning_curve_LSTM_Net23_pretraining_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net23_pretraining_UD2class')
    # end LSTM_cross_subject_UD2class_pretraining_Net23


# 样本集_X(numpy 四维数组(各样本 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def LSTM_cross_subject_UD2class_pretraining_Net24():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\上下二分类\\LSTM_Net24_pretraining' +
        r'\\learning_curve_LSTM_Net24_pretraining_UD2class.txt', 'w')
    fileLog.write('LSTM 上下二分类器_预训练（上、下）跨被试\n')
    print('LSTM 上下二分类器_预训练（上、下）跨被试')


    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    classes = ('上', '下')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSet_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()


    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[2],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 对数据样本转置，以匹配 LSTM 输入要求
    X_sampleSetMulti = [
        np.swapaxes(X_sampleSet,1,2) for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net24()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("LSTM_cross_subject_UD2class_pretraining_Net24：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "LSTM_cross_subject_UD2class_pretraining_Net24：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[:100,:256,:] #取前 1280ms 的数据
        y_test = y_test[:100] #取前 1280ms 的数据

        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][:100,:256,:]] #取前 1280ms 的数据
                y_train = np.r_[y_train, Y_sampleSetMulti[j][:100]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size,:], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net24()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.005,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(400):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\上下二分类\\LSTM_Net24_pretraining' +
                            '\\%s_%s_Net24.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'LSTM_cross_subject_UD2class_pretraining_Net24：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'LSTM_cross_subject_UD2class_pretraining_Net24：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_UD2class_pretraining_Net24：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_UD2class_pretraining_Net24：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\LSTM_Net24_pretraining' +
        r'\\learning_curve_LSTM_Net24_pretraining_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\上下二分类\\LSTM_Net24_pretraining' +
        r'\\confusion_matrix_LSTM_Net24_pretraining_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('LSTM_Net24_pretraining')
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\LSTM_Net24_pretraining' +
        r'\\learning_curve_LSTM_Net24_pretraining_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net24_pretraining_UD2class')
    # end LSTM_cross_subject_UD2class_pretraining_Net24

# 样本集_X(numpy 二维数组(各样本 * 各特征))，样本集_Y(numpy 一维数组)
def NN_cross_subject_UD2class_pretraining_Net25():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\上下二分类\\NN_Net25_pretraining' +
        r'\\learning_curve_NN_Net25_pretraining_UD2class.txt', 'w')
    fileLog.write('NN 上下二分类器_预训练（上、下）跨被试\n')
    print('NN 上下二分类器_预训练（上、下）跨被试')
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[:100]
        y = y[:100]

        # 构造样本集
        dataSet = np.c_[X, y]
        print('NN_cross_subject_UD2class_pretraining_Net25：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    net_tmp = Net25()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("NN_cross_subject_UD2class_pretraining_Net25：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "NN_cross_subject_UD2class_pretraining_Net25：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        dataSet_test = dataSetMulti[subjectNo]  # 留下的用于测试的被试样本

        # 提取用于训练的被试集样本
        dataSet_train = np.empty((0, dataSet.shape[1]))
        for j in range(n_subjects):
            if j != subjectNo:
                dataSet_train = np.r_[dataSet_train, dataSetMulti[j]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(dataSet_train.shape[0])
        dataSet_train = dataSet_train[shuffle_index]

        # 训练集特征缩放
        scaler = StandardScaler()
        X_train_scalered = scaler.fit_transform(dataSet_train[:, :-1])
        dataSet_train_scalered = np.c_[X_train_scalered,
                                       dataSet_train[:, -1]]  #特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        # 记录训练集特征缩放参数，测试时会用到
        mean_ = scaler.mean_  #均值
        scale_ = scaler.scale_  #标准差(=np.std(axis=0),注：无ddof=1)

        # 对测试样本集进行特征缩放
        X_test_scalered = (dataSet_test[:, :-1] - mean_) / scale_
        dataSet_test_scalered = np.c_[X_test_scalered,
                                      dataSet_test[:, -1]]  #特征缩放后的测试样本集(X,y)
        dataSet_test_scalered[np.isnan(dataSet_test_scalered)] = 0  # 处理异常特征

        t_sizes = dataSet_train_scalered.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [561,410,560]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(dataSet_train_scalered[:size, :-1],
                                        dataSet_train_scalered[:size,
                                                               -1], transform)
            test_dataset = data_loader(dataSet_test_scalered[:, :-1],
                                       dataSet_test_scalered[:, -1], transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net25()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.002,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            epoch = 0
            while epoch < 20000:
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                # print('test_accu_best = ',test_accu_best)

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.95):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\上下二分类\\NN_Net25_pretraining' +
                            '\\%s_%s_Net25.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
                epoch += 1
                if running_loss / running_loss_initial < 0.2:
                    break
            # end while
            print(
                'NN_cross_subject_UD2class_pretraining_Net25：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'NN_cross_subject_UD2class_pretraining_Net25：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'NN_cross_subject_UD2class_pretraining_Net25：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\NN_Net25_pretraining' +
        r'\\learning_curve_NN_Net25_pretraining_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\上下二分类\\NN_Net25_pretraining' +
        r'\\confusion_matrix_NN_Net25_pretraining_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('NN_Net25_pretraining')
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\NN_Net25_pretraining' +
        r'\\learning_curve_NN_Net25_pretraining_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'NN_Net25_pretraining_UD2class')
    # end NN_cross_subject_UD2class_pretraining_Net25


# 样本集_X(numpy 三维数组(各样本 * 各帧 * 各特征))，样本集_Y(numpy 一维数组)
def LSTM_cross_subject_UD2class_pretraining_Net26():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\上下二分类\\LSTM_Net26_pretraining' +
        r'\\learning_curve_LSTM_Net26_pretraining_UD2class.txt', 'w')
    fileLog.write('LSTM 上下二分类器_预训练（上、下）跨被试\n')
    print('LSTM 上下二分类器_预训练（上、下）跨被试')
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[:100]
        y = y[:100]

        # 构造样本集
        dataSet = [X, y]
        print('LSTM_cross_subject_UD2class_pretraining_Net26：样本数量',
              sum(y == 0), sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    net_tmp = Net26()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("LSTM_cross_subject_UD2class_pretraining_Net26：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "LSTM_cross_subject_UD2class_pretraining_Net26：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        # 留下的用于测试的被试样本
        dataSet_test_X = dataSetMulti[subjectNo][0]
        dataSet_test_Y = dataSetMulti[subjectNo][1]
        # 提取用于训练的被试集样本
        dataSet_train_X = np.empty(
            (0, dataSet[0].shape[1], dataSet[0].shape[2]))
        dataSet_train_Y = np.empty((0, dataSet[1].shape[1]))
        for j in range(n_subjects):
            if j != subjectNo:
                dataSet_train_X = np.r_[dataSet_train_X, dataSetMulti[j][0]]
                dataSet_train_Y = np.r_[dataSet_train_Y, dataSetMulti[j][1]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(dataSet_train_X.shape[0])
        dataSet_train_X = dataSet_train_X[shuffle_index]
        dataSet_train_Y = dataSet_train_Y[shuffle_index]

        dataSet_train_scalered_X = dataSet_train_X
        # 训练集特征缩放
        # scaler = StandardScaler()
        # dataSet_train_scalered_X = scaler.fit_transform(dataSet_train_X[:, :,:])
        dataSet_train_scalered_X[np.isnan(
            dataSet_train_scalered_X)] = 0  # 处理异常特征

        # 记录训练集特征缩放参数，测试时会用到
        # mean_ = scaler.mean_  #均值
        # scale_ = scaler.scale_  #标准差(=np.std(axis=0),注：无ddof=1)

        dataSet_test_scalered = dataSet_test_X
        # 对测试样本集进行特征缩放
        # dataSet_test_scalered = (dataSet_test_X[:,:,:] - mean_) / scale_
        dataSet_test_scalered[np.isnan(dataSet_test_scalered)] = 0  # 处理异常特征

        t_sizes = dataSet_train_scalered_X.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [140,410,560]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(dataSet_train_scalered_X[:size, :],
                                        dataSet_train_Y[:size, :], transform)
            test_dataset = data_loader(dataSet_test_scalered, dataSet_test_Y,
                                       transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=64,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net26()
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.0001)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(800):  # loop over the dataset multiple times
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long().view(-1))
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                # print('test_accu_best = ',test_accu_best)

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels.view(-1)).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels.view(-1)).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    print('\t\ttrain_accu_cur = ', train_accu_cur)
                    if (epoch == 0) or (test_accu_best < test_accu_cur):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\上下二分类\\LSTM_Net26_pretraining' +
                            '\\%s_%s_Net26.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            # end for
            print(
                'LSTM_cross_subject_UD2class_pretraining_Net26：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'LSTM_cross_subject_UD2class_pretraining_Net26：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_UD2class_pretraining_Net26：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\LSTM_Net26_pretraining' +
        r'\\learning_curve_LSTM_Net26_pretraining_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\上下二分类\\LSTM_Net26_pretraining' +
        r'\\confusion_matrix_LSTM_Net26_pretraining_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('LSTM_Net26_pretraining')
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\LSTM_Net26_pretraining' +
        r'\\learning_curve_LSTM_Net26_pretraining_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net26_pretraining_UD2class')
    # end LSTM_cross_subject_UD2class_pretraining_Net26


# 样本集_X(numpy 三维数组(各样本 * 各帧 * 各特征))，样本集_Y(numpy 一维数组)
def LSTM_cross_subject_UD2class_pretraining_Net27():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\上下二分类\\LSTM_Net27_pretraining' +
        r'\\learning_curve_LSTM_Net27_pretraining_UD2class.txt', 'w')
    fileLog.write('LSTM 上下二分类器_预训练（上、下）跨被试\n')
    print('LSTM 上下二分类器_预训练（上、下）跨被试')
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[:100]
        y = y[:100]

        # 构造样本集
        dataSet = [X, y]
        print('LSTM_cross_subject_UD2class_pretraining_Net27：样本数量',
              sum(y == 0), sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    net_tmp = Net27()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("LSTM_cross_subject_UD2class_pretraining_Net27：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "LSTM_cross_subject_UD2class_pretraining_Net27：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        # 留下的用于测试的被试样本
        dataSet_test_X = dataSetMulti[subjectNo][0]
        dataSet_test_Y = dataSetMulti[subjectNo][1]
        # 提取用于训练的被试集样本
        dataSet_train_X = np.empty(
            (0, dataSet[0].shape[1], dataSet[0].shape[2]))
        dataSet_train_Y = np.empty((0, dataSet[1].shape[1]))
        for j in range(n_subjects):
            if j != subjectNo:
                dataSet_train_X = np.r_[dataSet_train_X, dataSetMulti[j][0]]
                dataSet_train_Y = np.r_[dataSet_train_Y, dataSetMulti[j][1]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(dataSet_train_X.shape[0])
        dataSet_train_X = dataSet_train_X[shuffle_index]
        dataSet_train_Y = dataSet_train_Y[shuffle_index]

        dataSet_train_scalered_X = dataSet_train_X
        # 训练集特征缩放
        # scaler = StandardScaler()
        # dataSet_train_scalered_X = scaler.fit_transform(dataSet_train_X[:, :,:])
        dataSet_train_scalered_X[np.isnan(
            dataSet_train_scalered_X)] = 0  # 处理异常特征

        # 记录训练集特征缩放参数，测试时会用到
        # mean_ = scaler.mean_  #均值
        # scale_ = scaler.scale_  #标准差(=np.std(axis=0),注：无ddof=1)

        dataSet_test_scalered = dataSet_test_X
        # 对测试样本集进行特征缩放
        # dataSet_test_scalered = (dataSet_test_X[:,:,:] - mean_) / scale_
        dataSet_test_scalered[np.isnan(dataSet_test_scalered)] = 0  # 处理异常特征

        t_sizes = dataSet_train_scalered_X.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [140,410,560]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(dataSet_train_scalered_X[:size, :],
                                        dataSet_train_Y[:size, :], transform)
            test_dataset = data_loader(dataSet_test_scalered, dataSet_test_Y,
                                       transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=64,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net27()
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.0001)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(800):  # loop over the dataset multiple times
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long().view(-1))
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                # print('test_accu_best = ',test_accu_best)

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels.view(-1)).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels.view(-1)).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    print('\t\ttrain_accu_cur = ', train_accu_cur)
                    if (epoch == 0) or (test_accu_best < test_accu_cur):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\上下二分类\\LSTM_Net27_pretraining' +
                            '\\%s_%s_Net27.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            # end for
            print(
                'LSTM_cross_subject_UD2class_pretraining_Net27：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'LSTM_cross_subject_UD2class_pretraining_Net27：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_UD2class_pretraining_Net27：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\LSTM_Net27_pretraining' +
        r'\\learning_curve_LSTM_Net27_pretraining_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\上下二分类\\LSTM_Net27_pretraining' +
        r'\\confusion_matrix_LSTM_Net27_pretraining_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('LSTM_Net27_pretraining')
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\LSTM_Net27_pretraining' +
        r'\\learning_curve_LSTM_Net27_pretraining_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net27_pretraining_UD2class')
    # end LSTM_cross_subject_UD2class_pretraining_Net27


# 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def CNN_LSTM_cross_subject_UD2class_pretraining_Net28():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\上下二分类\\CNN_LSTM_Net28_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net28_pretraining_UD2class.txt', 'w')
    fileLog.write('CNN_LSTM 上下二分类器_预训练（上、下）跨被试\n')
    print('CNN_LSTM 上下二分类器_预训练（上、下）跨被试')

    # 适用于 Net22、Net28
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 160  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net22、Net28

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    classes = ('上', '下')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net28()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("CNN_LSTM_cross_subject_UD2class_pretraining_Net28：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "CNN_LSTM_cross_subject_UD2class_pretraining_Net28：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[:100]
        y_test = y_test[:100]

        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][:100]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j][:100]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        # 信号分帧
        X_train_framed = []
        for X_train_I in X_train:
            X_train_framed.append(
                sp.enFrame3D(X_train_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_train = np.array(X_train_framed)
        X_test_framed = []
        for X_test_I in X_test:
            X_test_framed.append(
                sp.enFrame3D(X_test_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
            pass
        X_test = np.array(X_test_framed)

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net28()
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.0002)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(500):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    print('\t\ttrain_accu_cur = ', train_accu_cur)
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\上下二分类\\CNN_LSTM_Net28_pretraining'
                            + '\\%s_%s_Net28.pkl' %
                            (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'CNN_LSTM_cross_subject_UD2class_pretraining_Net28：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'CNN_LSTM_cross_subject_UD2class_pretraining_Net28：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_LSTM_cross_subject_UD2class_pretraining_Net28：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_LSTM_cross_subject_UD2class_pretraining_Net28：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\CNN_LSTM_Net28_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net28_pretraining_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\CNN_LSTM_Net28_pretraining' +
        r'\\confusion_matrix_CNN_LSTM_Net28_pretraining_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('CNN_LSTM_Net28_pretraining')
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\CNN_LSTM_Net28_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net28_pretraining_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_LSTM_Net28_pretraining_UD2class')
    # end CNN_LSTM_cross_subject_UD2class_pretraining_Net28


# 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def CNN_LSTM_cross_subject_UD2class_pretraining_Net29():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\上下二分类\\CNN_LSTM_Net29_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net29_pretraining_UD2class.txt', 'w')
    fileLog.write('CNN_LSTM 上下二分类器_预训练（上、下）跨被试\n')
    print('CNN_LSTM 上下二分类器_预训练（上、下）跨被试')

    # 适用于 Net2、Net29
    FRAME_CNT = 4  # 帧数
    FRAME_WIDTH = 480  # 帧宽
    FRAME_INC = 240  # 帧移
    # 适用于 Net2、Net29

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    classes = ('上', '下')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net29()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("CNN_LSTM_cross_subject_UD2class_pretraining_Net29：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "CNN_LSTM_cross_subject_UD2class_pretraining_Net29：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[:100]
        y_test = y_test[:100]

        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][:100]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j][:100]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        # 信号分帧
        X_train_framed = []
        for X_train_I in X_train:
            X_train_framed.append(
                sp.enFrame3D(X_train_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_train = np.array(X_train_framed)
        X_test_framed = []
        for X_test_I in X_test:
            X_test_framed.append(
                sp.enFrame3D(X_test_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
            pass
        X_test = np.array(X_test_framed)

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net29()
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.0002)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(450):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    print('\t\ttrain_accu_cur = ', train_accu_cur)
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\上下二分类\\CNN_LSTM_Net29_pretraining'
                            + '\\%s_%s_Net29.pkl' %
                            (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'CNN_LSTM_cross_subject_UD2class_pretraining_Net29：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'CNN_LSTM_cross_subject_UD2class_pretraining_Net29：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_LSTM_cross_subject_UD2class_pretraining_Net29：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_LSTM_cross_subject_UD2class_pretraining_Net29：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\CNN_LSTM_Net29_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net29_pretraining_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\CNN_LSTM_Net29_pretraining' +
        r'\\confusion_matrix_CNN_LSTM_Net29_pretraining_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('CNN_LSTM_Net29_pretraining')
    fileD = open(
        path + r'\\..\model_all\\上下二分类\\CNN_LSTM_Net29_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net29_pretraining_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_LSTM_Net29_pretraining_UD2class')
    # end CNN_LSTM_cross_subject_UD2class_pretraining_Net29


def CNN_cross_subject_UD2class_fine_Net2():
    fileLog = open(
        os.getcwd() + r'\\..\model\\上下二分类\\CNN_Net2_fine' +
        r'\\learning_curve_CNN_Net2_fine_UD2class.txt', 'w')
    fileLog.write('CNN 上下二分类器_微调（上、下）跨被试\n')
    print('CNN 上下二分类器_微调（上、下）跨被试')

    # 适用于 Net2、Net29
    FRAME_CNT = 4  # 帧数
    FRAME_WIDTH = 480  # 帧宽
    FRAME_INC = 240  # 帧移
    # 适用于 Net2、Net29

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 25)
    classes = ('上', '下')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\上下二分类\\CNN_Net2_pretraining' + '\\%s_290_Net2.pkl' % (
            NAME_LIST[subjectNo])
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        # 提取二分类数据
        X = X[:100]
        y = y[:100]
        print('CNN_cross_subject_UD2class_fine_Net2：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 信号分帧
        X_framed = []
        for X_I in X:
            X_framed.append(
                sp.enFrame3D(X_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X = np.array(X_framed)

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.95:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print('CNN_cross_subject_UD2class_fine_Net2：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'CNN_cross_subject_UD2class_fine_Net2：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_cross_subject_UD2class_fine_Net2：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_cross_subject_UD2class_fine_Net2：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\上下二分类\\CNN_Net2_fine' +
        r'\\learning_curve_CNN_Net2_fine_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\上下二分类\\CNN_Net2_fine' +
        r'\\confusion_matrix_CNN_Net2_fine_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('CNN_Net2_fine')
    fileD = open(
        path + r'\\..\model\\上下二分类\\CNN_Net2_fine' +
        r'\\learning_curve_CNN_Net2_fine_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_Net2_fine_UD2class')

    # end CNN_cross_subject_UD2class_fine_Net2


def CNN_cross_subject_UD2class_fine_Net22():
    fileLog = open(
        os.getcwd() + r'\\..\model\\上下二分类\\CNN_Net22_fine' +
        r'\\learning_curve_CNN_Net22_fine_UD2class.txt', 'w')
    fileLog.write('CNN 上下二分类器_微调（上、下）跨被试\n')
    print('CNN 上下二分类器_微调（上、下）跨被试')

    # 适用于 Net22、Net28
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 160  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net22、Net28

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 25)
    classes = ('上', '下')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\上下二分类\\CNN_Net22_pretraining' + '\\%s_330_Net22.pkl' % (
            NAME_LIST[subjectNo])
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        # 提取二分类数据
        X = X[:100]
        y = y[:100]
        print('CNN_cross_subject_UD2class_fine_Net22：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 信号分帧
        X_framed = []
        for X_I in X:
            X_framed.append(
                sp.enFrame3D(X_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X = np.array(X_framed)

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.95:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print(
                    'CNN_cross_subject_UD2class_fine_Net22：%s 训练集容量 %d 训练完成' %
                    (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'CNN_cross_subject_UD2class_fine_Net22：%s 训练集容量 %d 训练完成\n'
                    % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_cross_subject_UD2class_fine_Net22：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_cross_subject_UD2class_fine_Net22：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\上下二分类\\CNN_Net22_fine' +
        r'\\learning_curve_CNN_Net22_fine_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\上下二分类\\CNN_Net22_fine' +
        r'\\confusion_matrix_CNN_Net22_fine_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('CNN_Net22_fine')
    fileD = open(
        path + r'\\..\model\\上下二分类\\CNN_Net22_fine' +
        r'\\learning_curve_CNN_Net22_fine_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_Net22_fine_UD2class')

    # end CNN_cross_subject_UD2class_fine_Net22


def LSTM_cross_subject_UD2class_fine_Net23():
    fileLog = open(
        os.getcwd() + r'\\..\model\\上下二分类\\LSTM_Net23_fine' +
        r'\\learning_curve_LSTM_Net23_fine_UD2class.txt', 'w')
    fileLog.write('LSTM 上下二分类器_微调（上、下）跨被试\n')
    print('LSTM 上下二分类器_微调（上、下）跨被试')

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 25)
    classes = ('上', '下')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSet_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()


    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[2],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 对数据样本转置，以匹配 LSTM 输入要求
    X_sampleSetMulti = [
        np.swapaxes(X_sampleSet,1,2) for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\上下二分类\\LSTM_Net23_pretraining' + '\\%s_330_Net23.pkl' % (
            NAME_LIST[subjectNo])
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        # 提取二分类数据
        X = X[:100]
        y = y[:100]
        print('LSTM_cross_subject_UD2class_fine_Net23：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=10,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.002)#0.005

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=64,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.95:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print('LSTM_cross_subject_UD2class_fine_Net23：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'LSTM_cross_subject_UD2class_fine_Net23：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_UD2class_fine_Net23：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_UD2class_fine_Net23：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\上下二分类\\LSTM_Net23_fine' +
        r'\\learning_curve_LSTM_Net23_fine_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\上下二分类\\LSTM_Net23_fine' +
        r'\\confusion_matrix_LSTM_Net23_fine_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('LSTM_Net23_fine')
    fileD = open(
        path + r'\\..\model\\上下二分类\\LSTM_Net23_fine' +
        r'\\learning_curve_LSTM_Net23_fine_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net23_fine_UD2class')

    # end LSTM_cross_subject_UD2class_fine_Net23


def LSTM_cross_subject_UD2class_fine_Net24():
    fileLog = open(
        os.getcwd() + r'\\..\model\\上下二分类\\LSTM_Net24_fine' +
        r'\\learning_curve_LSTM_Net24_fine_UD2class.txt', 'w')
    fileLog.write('LSTM 上下二分类器_微调（上、下）跨被试\n')
    print('LSTM 上下二分类器_微调（上、下）跨被试')

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 25)
    classes = ('上', '下')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSet_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()


    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[2],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 对数据样本转置，以匹配 LSTM 输入要求
    X_sampleSetMulti = [
        np.swapaxes(X_sampleSet,1,2) for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\上下二分类\\LSTM_Net24_pretraining' + '\\%s_330_Net24.pkl' % (
            NAME_LIST[subjectNo])
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        # 提取二分类数据
        X = X[:100,:256,:]
        y = y[:100]
        print('LSTM_cross_subject_UD2class_fine_Net24：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=10,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.002)#0.005

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=64,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.95:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print('LSTM_cross_subject_UD2class_fine_Net24：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'LSTM_cross_subject_UD2class_fine_Net24：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_UD2class_fine_Net24：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_UD2class_fine_Net24：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\上下二分类\\LSTM_Net24_fine' +
        r'\\learning_curve_LSTM_Net24_fine_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\上下二分类\\LSTM_Net24_fine' +
        r'\\confusion_matrix_LSTM_Net24_fine_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('LSTM_Net24_fine')
    fileD = open(
        path + r'\\..\model\\上下二分类\\LSTM_Net24_fine' +
        r'\\learning_curve_LSTM_Net24_fine_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net24_fine_UD2class')

    # end LSTM_cross_subject_UD2class_fine_Net24

def NN_cross_subject_UD2class_fine_Net25():
    fileLog = open(
        os.getcwd() + r'\\..\model\\上下二分类\\NN_Net25_fine' +
        r'\\learning_curve_NN_Net25_fine_UD2class.txt', 'w')
    fileLog.write('NN 上下二分类器_微调（上、下）跨被试\n')
    print('NN 上下二分类器_微调（上、下）跨被试')

    FINE_SIZES = np.linspace(0, 1.0, 25)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[:100]
        y = y[:100]
        # 构造样本集
        dataSet = np.c_[X, y]
        print('NN_cross_subject_UD2class_fine_Net25：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\上下二分类\\NN_Net25_pretraining' + '\\%s_330_Net25.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        dataSet = dataSetMulti[subjectNo]  # 取出留下的被试样本集

        # 样本集特征缩放
        scaler = StandardScaler()
        X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_train_scalered = np.c_[X_train_scalered,
                                       dataSet[:, -1]]  #特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        print('NN_cross_subject_UD2class_fine_Net25：样本数量',
              sum(dataSet_train_scalered[:, -1] == 0),
              sum(dataSet_train_scalered[:, -1] == 1),
              sum(dataSet_train_scalered[:, -1] == 2),
              sum(dataSet_train_scalered[:, -1] == 3))

        fine_set, test_set = train_test_split(
            dataSet_train_scalered, test_size=0.1)  #划分微调训练集、测试集(默认洗牌)
        X_fine = fine_set[:, :-1]  #微调训练集特征矩阵
        y_fine = fine_set[:, -1]  #微调训练集标签数组
        X_test = test_set[:, :-1]  #测试集特征矩阵
        y_test = test_set[:, -1]  #测试集标签数组

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                epoch = 0
                while epoch < 20000:
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    # print('[%d] loss: %.3f ,%.3f%%' %
                    #       (epoch + 1, running_loss,
                    #        100*running_loss / running_loss_initial))
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))

                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #       (epoch + 1, running_loss,
                        #        100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)

                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.9:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                    epoch += 1
                    if running_loss / running_loss_initial < 0.2:
                        break
                # end while
                print('NN_cross_subject_UD2class_fine_Net25：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'NN_cross_subject_UD2class_fine_Net25：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'NN_cross_subject_UD2class_fine_Net25：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'NN_cross_subject_UD2class_fine_Net25：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\上下二分类\\NN_Net25_fine' +
        r'\\learning_curve_NN_Net25_fine_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\上下二分类\\NN_Net25_fine' +
        r'\\confusion_matrix_NN_Net25_fine_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('NN_Net25_fine')
    fileD = open(
        path + r'\\..\model\\上下二分类\\NN_Net25_fine' +
        r'\\learning_curve_NN_Net25_fine_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'NN_Net25_fine_UD2class')
    # end NN_cross_subject_UD2class_fine_Net25


def LSTM_cross_subject_UD2class_fine_Net26():
    fileLog = open(
        os.getcwd() + r'\\..\model\\上下二分类\\LSTM_Net26_fine' +
        r'\\learning_curve_LSTM_Net26_fine_UD2class.txt', 'w')
    fileLog.write('LSTM 上下二分类器_微调（上、下）跨被试\n')
    print('LSTM 上下二分类器_微调（上、下）跨被试')

    FINE_SIZES = np.linspace(0, 1.0, 25)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[:100]
        y = y[:100]
        # 构造样本集
        dataSet = [X, y]
        print('LSTM_cross_subject_UD2class_fine_Net26：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\上下二分类\\LSTM_Net26_pretraining' + '\\%s_290_Net26.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        # 取出留下的被试样本集
        dataSet_test_X = dataSetMulti[subjectNo][0]
        dataSet_test_Y = dataSetMulti[subjectNo][1]
        dataSet_train_scalered = dataSet_test_X
        # 样本集特征缩放
        # scaler = StandardScaler()
        # X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
        # dataSet_train_scalered = np.c_[X_train_scalered,
        #                                dataSet[:, -1]]  # 特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        print('LSTM_cross_subject_UD2class_fine_Net26：样本数量',
              sum(dataSet_test_Y == 0), sum(dataSet_test_Y == 1),
              sum(dataSet_test_Y == 2), sum(dataSet_test_Y == 3))

        X_test, y_test, X_fine, y_fine = sp.splitSet(
            dataSet_train_scalered, dataSet_test_Y, ratio=0.1,
            shuffle=True)  # 划分微调训练集、测试集(洗牌)

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [180]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.0001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels.view(-1)).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        800):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long().view(-1))
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    # print('[%d] loss: %.3f ,%.3f%%' %
                    #       (epoch + 1, running_loss,
                    #        100*running_loss / running_loss_initial))
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels.view(-1)).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels.view(-1)).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))

                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #       (epoch + 1, running_loss,
                        #        100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)

                        if test_accu_best < test_accu_cur:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                # end for
                print(
                    'LSTM_cross_subject_UD2class_fine_Net26：%s 训练集容量 %d 训练完成' %
                    (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'LSTM_cross_subject_UD2class_fine_Net26：%s 训练集容量 %d 训练完成\n'
                    % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_UD2class_fine_Net26：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_UD2class_fine_Net26：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\上下二分类\\LSTM_Net26_fine' +
        r'\\learning_curve_LSTM_Net26_fine_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\上下二分类\\LSTM_Net26_fine' +
        r'\\confusion_matrix_LSTM_Net26_fine_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('LSTM_Net26_fine')
    fileD = open(
        path + r'\\..\model\\上下二分类\\LSTM_Net26_fine' +
        r'\\learning_curve_LSTM_Net26_fine_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net26_fine_UD2class')
    # end LSTM_cross_subject_UD2class_fine_Net26


def LSTM_cross_subject_UD2class_fine_Net27():
    fileLog = open(
        os.getcwd() + r'\\..\model\\上下二分类\\LSTM_Net27_fine' +
        r'\\learning_curve_LSTM_Net27_fine_UD2class.txt', 'w')
    fileLog.write('LSTM 上下二分类器_微调（上、下）跨被试\n')
    print('LSTM 上下二分类器_微调（上、下）跨被试')

    FINE_SIZES = np.linspace(0, 1.0, 25)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[:100]
        y = y[:100]
        # 构造样本集
        dataSet = [X, y]
        print('LSTM_cross_subject_UD2class_fine_Net27：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\上下二分类\\LSTM_Net27_pretraining' + '\\%s_330_Net27.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        # 取出留下的被试样本集
        dataSet_test_X = dataSetMulti[subjectNo][0]
        dataSet_test_Y = dataSetMulti[subjectNo][1]
        dataSet_train_scalered = dataSet_test_X
        # 样本集特征缩放
        # scaler = StandardScaler()
        # X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
        # dataSet_train_scalered = np.c_[X_train_scalered,
        #                                dataSet[:, -1]]  # 特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        print('LSTM_cross_subject_UD2class_fine_Net27：样本数量',
              sum(dataSet_test_Y == 0), sum(dataSet_test_Y == 1),
              sum(dataSet_test_Y == 2), sum(dataSet_test_Y == 3))

        X_test, y_test, X_fine, y_fine = sp.splitSet(
            dataSet_train_scalered, dataSet_test_Y, ratio=0.1,
            shuffle=True)  # 划分微调训练集、测试集(洗牌)

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [180]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.0001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels.view(-1)).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        800):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long().view(-1))
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    # print('[%d] loss: %.3f ,%.3f%%' %
                    #       (epoch + 1, running_loss,
                    #        100*running_loss / running_loss_initial))
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels.view(-1)).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels.view(-1)).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))

                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #       (epoch + 1, running_loss,
                        #        100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)

                        if test_accu_best < test_accu_cur:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                # end for
                print(
                    'LSTM_cross_subject_UD2class_fine_Net27：%s 训练集容量 %d 训练完成' %
                    (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'LSTM_cross_subject_UD2class_fine_Net27：%s 训练集容量 %d 训练完成\n'
                    % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_UD2class_fine_Net27：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_UD2class_fine_Net27：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\上下二分类\\LSTM_Net27_fine' +
        r'\\learning_curve_LSTM_Net27_fine_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\上下二分类\\LSTM_Net27_fine' +
        r'\\confusion_matrix_LSTM_Net27_fine_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('LSTM_Net27_fine')
    fileD = open(
        path + r'\\..\model\\上下二分类\\LSTM_Net27_fine' +
        r'\\learning_curve_LSTM_Net27_fine_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net27_fine_UD2class')
    # end LSTM_cross_subject_UD2class_fine_Net27


def CNN_LSTM_cross_subject_UD2class_fine_Net28():
    fileLog = open(
        os.getcwd() + r'\\..\model\\上下二分类\\CNN_LSTM_Net28_fine' +
        r'\\learning_curve_CNN_LSTM_Net28_fine_UD2class.txt', 'w')
    fileLog.write('CNN_LSTM 上下二分类器_微调（上、下）跨被试\n')
    print('CNN_LSTM 上下二分类器_微调（上、下）跨被试')

    # 适用于 Net22、Net28
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 160  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net22、Net28

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 25)
    classes = ('上', '下')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\上下二分类\\CNN_LSTM_Net28_pretraining' + '\\%s_350_Net28.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        # 提取二分类数据
        X = X[:100]
        y = y[:100]
        print('CNN_LSTM_cross_subject_UD2class_fine_Net28：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 信号分帧
        X_framed = []
        for X_I in X:
            X_framed.append(
                sp.enFrame3D(X_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X = np.array(X_framed)

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.0001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #     (epoch + 1, running_loss,
                        #     100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.97:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print(
                    'CNN_LSTM_cross_subject_UD2class_fine_Net28：%s 训练集容量 %d 训练完成'
                    % (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'CNN_LSTM_cross_subject_UD2class_fine_Net28：%s 训练集容量 %d 训练完成\n'
                    % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_LSTM_cross_subject_UD2class_fine_Net28：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_LSTM_cross_subject_UD2class_fine_Net28：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\上下二分类\\CNN_LSTM_Net28_fine' +
        r'\\learning_curve_CNN_LSTM_Net28_fine_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\上下二分类\\CNN_LSTM_Net28_fine' +
        r'\\confusion_matrix_CNN_LSTM_Net28_fine_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.myLearning_curve_LIST2DICT_UD2class('CNN_LSTM_Net28_fine')
    fileD = open(
        path + r'\\..\model\\上下二分类\\CNN_LSTM_Net28_fine' +
        r'\\learning_curve_CNN_LSTM_Net28_fine_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_LSTM_Net28_fine_UD2class')
    # end CNN_LSTM_cross_subject_UD2class_fine_Net28


def CNN_LSTM_cross_subject_UD2class_fine_Net29():
    fileLog = open(
        os.getcwd() + r'\\..\model\\上下二分类\\CNN_LSTM_Net29_fine' +
        r'\\learning_curve_CNN_LSTM_Net29_fine_UD2class.txt', 'w')
    fileLog.write('CNN_LSTM 上下二分类器_微调（上、下）跨被试\n')
    print('CNN_LSTM 上下二分类器_微调（上、下）跨被试')

    # 适用于 Net2、Net29
    FRAME_CNT = 4  # 帧数
    FRAME_WIDTH = 480  # 帧宽
    FRAME_INC = 240  # 帧移
    # 适用于 Net2、Net29

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 25)
    classes = ('上', '下')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\上下二分类\\CNN_LSTM_Net29_pretraining' + '\\%s_210_Net29.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        # 提取二分类数据
        X = X[:100]
        y = y[:100]
        print('CNN_LSTM_cross_subject_UD2class_fine_Net29：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 信号分帧
        X_framed = []
        for X_I in X:
            X_framed.append(
                sp.enFrame3D(X_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X = np.array(X_framed)

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.0001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #     (epoch + 1, running_loss,
                        #     100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.97:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print(
                    'CNN_LSTM_cross_subject_UD2class_fine_Net29：%s 训练集容量 %d 训练完成'
                    % (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'CNN_LSTM_cross_subject_UD2class_fine_Net29：%s 训练集容量 %d 训练完成\n'
                    % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_LSTM_cross_subject_UD2class_fine_Net29：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_LSTM_cross_subject_UD2class_fine_Net29：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\上下二分类\\CNN_LSTM_Net29_fine' +
        r'\\learning_curve_CNN_LSTM_Net29_fine_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\上下二分类\\CNN_LSTM_Net29_fine' +
        r'\\confusion_matrix_CNN_LSTM_Net29_fine_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.myLearning_curve_LIST2DICT_UD2class('CNN_LSTM_Net29_fine')
    fileD = open(
        path + r'\\..\model\\上下二分类\\CNN_LSTM_Net29_fine' +
        r'\\learning_curve_CNN_LSTM_Net29_fine_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_LSTM_Net29_fine_UD2class')
    # end CNN_LSTM_cross_subject_UD2class_fine_Net29


class Net2(nn.Module):
    # 输入数据：
    #   4 帧数据，帧宽 480ms，帧移 160ms，采样频率 200 Hz
    #   各帧：3 频率通道 * 32 通道 * 采样点
    def __init__(self):
        super(Net2, self).__init__()
        self.pool1 = nn.MaxPool2d((2, 4), 2)
        self.pool2 = nn.MaxPool2d((3, 6), 2)
        self.conv1 = nn.Conv2d(3, 6, (5, 15))
        self.conv2 = nn.Conv2d(6, 8, (5, 15))
        self.conv3 = Conv2d(8, 8, (3, 3))

        self.fc1 = nn.Linear(4 * 8 * 2 * 4, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)

        self.dropout = nn.Dropout(p=0.4)

        self.seq1 = nn.Sequential(self.conv1, nn.ReLU(), self.pool1,
                                  self.conv2, nn.ReLU(), self.pool1,
                                  self.conv3, nn.ReLU(), self.pool2)
        self.seq2 = nn.Sequential(self.fc1, self.dropout, nn.ReLU(), self.fc2,
                                  self.dropout, nn.ReLU(), self.fc3)
        # end __init__

    def forward(self, x):
        # CNN 模块
        output1 = torch.empty((x.shape[0], 0)).double()
        for step in range(4):
            xStep = self.seq1(x[:, step, :, :, :])
            xStep = xStep.view(-1, 8 * 2 * 4)
            output1 = torch.cat((output1, xStep), 1)

        # 全连接模块
        output2 = self.seq2(output1)

        return output2
        # end forward


class Net2g(nn.Module):# 未验证
    # 输入数据：
    #   6 帧数据，帧宽 480ms，帧移 160ms，采样频率 200 Hz
    #   各帧：3 频率通道 * 32 通道 * 采样点
    def __init__(self):
        super(Net2g, self).__init__()
        self.pool1 = nn.MaxPool2d((2, 4), 2)
        self.pool2 = nn.MaxPool2d((3, 6), 2)
        self.conv1 = nn.Conv2d(3, 6, (5, 15))
        self.conv2 = nn.Conv2d(6, 8, (5, 15))
        self.conv3 = Conv2d(8, 8, (3, 3))

        self.fc1 = nn.Linear(6 * 8 * 2 * 4, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)

        self.dropout = nn.Dropout(p=0.4)

        self.seq1 = nn.Sequential(self.conv1, nn.ReLU(), self.pool1,
                                  self.conv2, nn.ReLU(), self.pool1,
                                  self.conv3, nn.ReLU(), self.pool2)
        self.seq2 = nn.Sequential(self.fc1, self.dropout, nn.ReLU(), self.fc2,
                                  self.dropout, nn.ReLU(), self.fc3)
        # end __init__

    def forward(self, x):
        # CNN 模块
        output1 = torch.empty((x.shape[0], 0)).double()
        for step in range(6):
            xStep = self.seq1(x[:, step, :, :, :])
            xStep = xStep.view(-1, 8 * 2 * 4)
            output1 = torch.cat((output1, xStep), 1)

        # 全连接模块
        output2 = self.seq2(output1)

        return output2
        # end forward

class Net22(nn.Module):
    # 输入数据：
    #   7 帧数据，帧宽 160ms，帧移 160ms，采样频率 200 Hz
    #   各帧：3 频率通道 * 32 通道 * 采样点
    def __init__(self):
        super(Net22, self).__init__()
        self.pool = nn.AvgPool2d((2, 2), 2)
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 8, (5, 5))

        self.fc1 = nn.Linear(7 * 8 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

        self.dropout = nn.Dropout(p=0.4)

        self.seq1 = nn.Sequential(self.conv1, nn.ReLU(), self.pool, self.conv2,
                                  nn.ReLU(), self.pool)
        self.seq2 = nn.Sequential(self.fc1, self.dropout, nn.ReLU(), self.fc2,
                                  self.dropout, nn.ReLU(), self.fc3)
        # end __init__

    def forward(self, x):
        # CNN 模块
        output1 = torch.empty((x.shape[0], 0)).double()
        for step in range(7):
            xStep = self.seq1(x[:, step, :, :, :])
            xStep = xStep.view(-1, 8 * 5 * 5)
            output1 = torch.cat((output1, xStep), 1)

        # 全连接模块
        output2 = self.seq2(output1)

        return output2
        # end forward


class Net23(nn.Module):
    # 输入数据：
    #   256 时间步，每个时间步 32 通道脑电信号值
    #   各帧：原始信号
    def __init__(self):
        super(Net23, self).__init__()
        self.lstm = nn.LSTM(32, 16, 3, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(16, 2)

        self.seq1 = nn.Sequential(self.lstm)
        self.seq2 = nn.Sequential(self.fc)
        # end __init__

    def forward(self, x):
        # LSTM 模块
        output1, _ = self.seq1(x)
        output1 = output1[:, -1, :]

        # 全连接模块
        output2 = self.seq2(output1)

        return output2
        # end forward

class Net24(nn.Module):
    # 输入数据：
    #   256 时间步，每个时间步 32 通道脑电信号值
    #   各帧：原始信号
    def __init__(self):
        super(Net24, self).__init__()
        self.lstm = nn.LSTM(32, 16, 3, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(256 * 16, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(p=0.4)

        self.seq1 = nn.Sequential(self.lstm)
        self.seq2 = nn.Sequential(self.fc1, self.dropout, nn.ReLU(), self.fc2,self.dropout, nn.ReLU(), self.fc3)
        # end __init__

    def forward(self, x):
        # LSTM 模块
        output1, _ = self.seq1(x)
        output1 = output1.reshape(output1.shape[0], -1)

        # 全连接模块
        output2 = self.seq2(output1)

        return output2
        # end forward

class Net25(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net25, self).__init__()
        self.fc1 = nn.Linear(224, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 2)

        self.dropout = nn.Dropout(p=0.4)
        self.seq1 = nn.Sequential(self.fc1, self.dropout, nn.ReLU(), self.fc2,
                                  self.dropout, nn.ReLU(), self.fc3,
                                  self.dropout, nn.ReLU(), self.fc4,
                                  self.dropout, nn.ReLU())
        self.seq2 = nn.Sequential(self.fc5, self.dropout, nn.ReLU(), self.fc6)
        # end __init__

    def forward(self, x):
        # 全连接预训练模块
        output1 = self.seq1(x)

        # 全连接微调模块
        output2 = self.seq2(output1)

        return output2
        # end forward


class Net26(nn.Module):
    # 输入数据：
    #   7 帧数据，帧宽 320ms，帧移 160ms
    #   各帧：人工特征
    def __init__(self):
        super(Net26, self).__init__()
        self.lstm = nn.LSTM(224, 16, 3, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(16, 2)

        self.seq1 = nn.Sequential(self.lstm)
        self.seq2 = nn.Sequential(self.fc)
        # end __init__

    def forward(self, x):
        # LSTM 模块
        output1, _ = self.seq1(x)
        output1 = output1[:, -1, :]

        # 全连接模块
        output2 = self.seq2(output1)

        return output2
        # end forward


class Net27(nn.Module):
    # 输入数据：
    #   7 帧数据，帧宽 320ms，帧移 160ms
    #   各帧：人工特征
    def __init__(self):
        super(Net27, self).__init__()
        self.lstm = nn.LSTM(224, 16, 3, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(7 * 16, 16)
        self.fc2 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(p=0.4)

        self.seq1 = nn.Sequential(self.lstm)
        self.seq2 = nn.Sequential(self.fc1, self.dropout, nn.ReLU(), self.fc2)
        # end __init__

    def forward(self, x):
        # LSTM 模块
        output1, _ = self.seq1(x)
        output1 = output1.reshape(output1.shape[0], -1)

        # 全连接模块
        output2 = self.seq2(output1)

        return output2
        # end forward


class Net28(nn.Module):
    # 输入数据：
    #   7 帧数据，帧宽 160ms，帧移 160ms，采样频率 200 Hz
    #   各帧：3 频率通道 * 32 通道 * 采样点
    def __init__(self):
        super(Net28, self).__init__()
        self.pool = nn.AvgPool2d((2, 2), 2)
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 8, (5, 5))
        self.lstm = nn.LSTM(8 * 5 * 5, 16, 3, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(16, 2)

        self.seq1 = nn.Sequential(self.conv1, nn.ReLU(), self.pool, self.conv2,
                                  nn.ReLU(), self.pool)
        self.seq2 = nn.Sequential(self.lstm)
        self.seq3 = nn.Sequential(self.fc)

        # end __init__

    def forward(self, x):
        # CNN 模块
        output1 = torch.empty((x.shape[0], 0, 8 * 5 * 5)).double()
        for step in range(x.shape[1]):
            xStep = self.seq1(x[:, step, :, :, :])
            xStep = xStep.view(-1, 8 * 5 * 5).unsqueeze(dim=1)
            output1 = torch.cat((output1, xStep), 1)

        # LSTM 模块
        output2, _ = self.seq2(output1)
        output2 = output2[:, -1, :]

        # 全连接模块
        output3 = self.seq3(output2)

        return output3
        # end forward


class Net29(nn.Module):
    # 输入数据：
    #   4 帧数据，帧宽 480ms，帧移 160ms，采样频率 200 Hz
    #   各帧：3 频率通道 * 32 通道 * 采样点
    def __init__(self):
        super(Net29, self).__init__()
        self.pool1 = nn.MaxPool2d((2, 4), 2)
        self.pool2 = nn.MaxPool2d((3, 6), 2)
        self.conv1 = nn.Conv2d(3, 6, (5, 15))
        self.conv2 = nn.Conv2d(6, 8, (5, 15))
        self.conv3 = Conv2d(8, 8, (3, 3))
        self.lstm = nn.LSTM(8 * 2 * 4, 8, 3, batch_first=True, dropout=0.4)
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 2)

        self.dropout = nn.Dropout(p=0.4)
        self.seq1 = nn.Sequential(self.conv1, nn.ReLU(), self.pool1,
                                  self.conv2, nn.ReLU(), self.pool1,
                                  self.conv3, nn.ReLU(), self.pool2)
        self.seq2 = nn.Sequential(self.lstm)
        self.seq3 = nn.Sequential(self.fc1, self.dropout, nn.ReLU(), self.fc2)

        # end __init__

    def forward(self, x):
        # CNN 模块
        output1 = torch.empty((x.shape[0], 0, 8 * 2 * 4)).double()
        for step in range(x.shape[1]):
            xStep = self.seq1(x[:, step, :, :, :])
            xStep = xStep.view(-1, 8 * 2 * 4).unsqueeze(dim=1)
            output1 = torch.cat((output1, xStep), 1)

        # LSTM 模块
        output2, _ = self.seq2(output1)
        output2 = output2.reshape(output2.shape[0], -1)

        # 全连接模块
        output3 = self.seq3(output2)

        return output3
        # end forward


# 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def CNN_cross_subject_LR2class_pretraining_Net2():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\左右二分类\\CNN_Net2_pretraining' +
        r'\\learning_curve_CNN_Net2_pretraining_LR2class.txt', 'w')
    fileLog.write('CNN 左右二分类器_预训练（左、右）跨被试\n')
    print('CNN 左右二分类器_预训练（左、右）跨被试')

    # 适用于 Net2、Net29
    FRAME_CNT = 4  # 帧数
    FRAME_WIDTH = 480  # 帧宽
    FRAME_INC = 240  # 帧移
    # 适用于 Net2、Net29

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    classes = ('左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net2()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("CNN_cross_subject_LR2class_pretraining_Net2：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "CNN_cross_subject_LR2class_pretraining_Net2：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[100:]
        y_test = y_test[100:]
        y_test[y_test == 2] = 0
        y_test[y_test == 3] = 1
        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][100:]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j][100:]]
            # end if
        # end for
        y_train[y_train == 2] = 0
        y_train[y_train == 3] = 1
        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        # 信号分帧
        X_train_framed = []
        for X_train_I in X_train:
            X_train_framed.append(
                sp.enFrame3D(X_train_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_train = np.array(X_train_framed)
        X_test_framed = []
        for X_test_I in X_test:
            X_test_framed.append(
                sp.enFrame3D(X_test_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
            pass
        X_test = np.array(X_test_framed)

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,135]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net2()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.0005,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(300):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\左右二分类\\CNN_Net2_pretraining' +
                            '\\%s_%s_Net2.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'CNN_cross_subject_LR2class_pretraining_Net2：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'CNN_cross_subject_LR2class_pretraining_Net2：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})

        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_cross_subject_LR2class_pretraining_Net2：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_cross_subject_LR2class_pretraining_Net2：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\CNN_Net2_pretraining' +
        r'\\learning_curve_CNN_Net2_pretraining_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\左右二分类\\CNN_Net2_pretraining' +
        r'\\confusion_matrix_CNN_Net2_pretraining_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('CNN_Net2_pretraining')
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\CNN_Net2_pretraining' +
        r'\\learning_curve_CNN_Net2_pretraining_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_Net2_pretraining_LR2class')
    # end CNN_cross_subject_LR2class_pretraining_Net2


# 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def CNN_cross_subject_LR2class_pretraining_Net22():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\左右二分类\\CNN_Net22_pretraining' +
        r'\\learning_curve_CNN_Net22_pretraining_LR2class.txt', 'w')
    fileLog.write('CNN 左右二分类器_预训练（左、右）跨被试\n')
    print('CNN 左右二分类器_预训练（左、右）跨被试')

    # 适用于 Net22、Net28
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 160  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net22、Net28

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    classes = ('左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net22()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("CNN_cross_subject_LR2class_pretraining_Net22：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "CNN_cross_subject_LR2class_pretraining_Net22：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[100:]
        y_test = y_test[100:]
        y_test[y_test == 2] = 0
        y_test[y_test == 3] = 1
        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][100:]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j][100:]]
            # end if
        # end for
        y_train[y_train == 2] = 0
        y_train[y_train == 3] = 1
        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        # 信号分帧
        X_train_framed = []
        for X_train_I in X_train:
            X_train_framed.append(
                sp.enFrame3D(X_train_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_train = np.array(X_train_framed)
        X_test_framed = []
        for X_test_I in X_test:
            X_test_framed.append(
                sp.enFrame3D(X_test_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_test = np.array(X_test_framed)

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net22()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.0001,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(500):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\左右二分类\\CNN_Net22_pretraining' +
                            '\\%s_%s_Net22.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'CNN_cross_subject_LR2class_pretraining_Net22：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'CNN_cross_subject_LR2class_pretraining_Net22：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_cross_subject_LR2class_pretraining_Net22：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_cross_subject_LR2class_pretraining_Net22：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\CNN_Net22_pretraining' +
        r'\\learning_curve_CNN_Net22_pretraining_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\左右二分类\\CNN_Net22_pretraining' +
        r'\\confusion_matrix_CNN_Net22_pretraining_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('CNN_Net22_pretraining')
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\CNN_Net22_pretraining' +
        r'\\learning_curve_CNN_Net22_pretraining_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_Net22_pretraining_LR2class')
    # end CNN_cross_subject_LR2class_pretraining_Net22


# 样本集_X(numpy 四维数组(各样本 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def LSTM_cross_subject_LR2class_pretraining_Net23():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\左右二分类\\LSTM_Net23_pretraining' +
        r'\\learning_curve_LSTM_Net23_pretraining_LR2class.txt', 'w')
    fileLog.write('LSTM 左右二分类器_预训练（左、右）跨被试\n')
    print('LSTM 左右二分类器_预训练（左、右）跨被试')


    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    classes = ('左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSet_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()


    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[2],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 对数据样本转置，以匹配 LSTM 输入要求
    X_sampleSetMulti = [
        np.swapaxes(X_sampleSet,1,2) for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net23()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("LSTM_cross_subject_LR2class_pretraining_Net23：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "LSTM_cross_subject_LR2class_pretraining_Net23：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[100:]
        y_test = y_test[100:]
        y_test[y_test == 2] = 0
        y_test[y_test == 3] = 1
        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][100:]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j][100:]]
            # end if
        # end for
        y_train[y_train == 2] = 0
        y_train[y_train == 3] = 1
        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net23()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.005,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(400):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\左右二分类\\LSTM_Net23_pretraining' +
                            '\\%s_%s_Net23.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'LSTM_cross_subject_LR2class_pretraining_Net23：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'LSTM_cross_subject_LR2class_pretraining_Net23：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_LR2class_pretraining_Net23：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_LR2class_pretraining_Net23：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\LSTM_Net23_pretraining' +
        r'\\learning_curve_LSTM_Net23_pretraining_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\左右二分类\\LSTM_Net23_pretraining' +
        r'\\confusion_matrix_LSTM_Net23_pretraining_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('LSTM_Net23_pretraining')
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\LSTM_Net23_pretraining' +
        r'\\learning_curve_LSTM_Net23_pretraining_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net23_pretraining_LR2class')
    # end LSTM_cross_subject_LR2class_pretraining_Net23


# 样本集_X(numpy 四维数组(各样本 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def LSTM_cross_subject_LR2class_pretraining_Net24():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\左右二分类\\LSTM_Net24_pretraining' +
        r'\\learning_curve_LSTM_Net24_pretraining_LR2class.txt', 'w')
    fileLog.write('LSTM 左右二分类器_预训练（左、右）跨被试\n')
    print('LSTM 左右二分类器_预训练（左、右）跨被试')


    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 2)#15
    classes = ('左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSet_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()


    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[2],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 对数据样本转置，以匹配 LSTM 输入要求
    X_sampleSetMulti = [
        np.swapaxes(X_sampleSet,1,2) for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net24()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("LSTM_cross_subject_LR2class_pretraining_Net24：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "LSTM_cross_subject_LR2class_pretraining_Net24：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[100:]
        y_test = y_test[100:]
        y_test[y_test == 2] = 0
        y_test[y_test == 3] = 1

        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][100:,:256,:]] #取前 1280ms 的数据
                y_train = np.r_[y_train, Y_sampleSetMulti[j][100:]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size,:], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net24()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.005,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            #400
            for epoch in range(20):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\左右二分类\\LSTM_Net24_pretraining' +
                            '\\%s_%s_Net24.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'LSTM_cross_subject_LR2class_pretraining_Net24：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'LSTM_cross_subject_LR2class_pretraining_Net24：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_LR2class_pretraining_Net24：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_LR2class_pretraining_Net24：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\LSTM_Net24_pretraining' +
        r'\\learning_curve_LSTM_Net24_pretraining_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\左右二分类\\LSTM_Net24_pretraining' +
        r'\\confusion_matrix_LSTM_Net24_pretraining_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('LSTM_Net24_pretraining')
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\LSTM_Net24_pretraining' +
        r'\\learning_curve_LSTM_Net24_pretraining_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net24_pretraining_LR2class')
    # end LSTM_cross_subject_LR2class_pretraining_Net24


# 样本集_X(numpy 二维数组(各样本 * 各特征))，样本集_Y(numpy 一维数组)
def NN_cross_subject_LR2class_pretraining_Net25():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\左右二分类\\NN_Net25_pretraining' +
        r'\\learning_curve_NN_Net25_pretraining_UD2class.txt', 'w')
    fileLog.write('NN 左右二分类器_预训练（左、右）跨被试\n')
    print('NN 左右二分类器_预训练（左、右）跨被试')
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1

        # 构造样本集
        dataSet = np.c_[X, y]
        print('NN_cross_subject_LR2class_pretraining_Net25：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    net_tmp = Net25()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("NN_cross_subject_LR2class_pretraining_Net25：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "NN_cross_subject_LR2class_pretraining_Net25：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        dataSet_test = dataSetMulti[subjectNo]  # 留下的用于测试的被试样本

        # 提取用于训练的被试集样本
        dataSet_train = np.empty((0, dataSet.shape[1]))
        for j in range(n_subjects):
            if j != subjectNo:
                dataSet_train = np.r_[dataSet_train, dataSetMulti[j]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(dataSet_train.shape[0])
        dataSet_train = dataSet_train[shuffle_index]

        # 训练集特征缩放
        scaler = StandardScaler()
        X_train_scalered = scaler.fit_transform(dataSet_train[:, :-1])
        dataSet_train_scalered = np.c_[X_train_scalered,
                                       dataSet_train[:, -1]]  #特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        # 记录训练集特征缩放参数，测试时会用到
        mean_ = scaler.mean_  #均值
        scale_ = scaler.scale_  #标准差(=np.std(axis=0),注：无ddof=1)

        # 对测试样本集进行特征缩放
        X_test_scalered = (dataSet_test[:, :-1] - mean_) / scale_
        dataSet_test_scalered = np.c_[X_test_scalered,
                                      dataSet_test[:, -1]]  #特征缩放后的测试样本集(X,y)
        dataSet_test_scalered[np.isnan(dataSet_test_scalered)] = 0  # 处理异常特征

        t_sizes = dataSet_train_scalered.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [561,410,560]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(dataSet_train_scalered[:size, :-1],
                                        dataSet_train_scalered[:size,
                                                               -1], transform)
            test_dataset = data_loader(dataSet_test_scalered[:, :-1],
                                       dataSet_test_scalered[:, -1], transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net25()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.002,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            epoch = 0
            while epoch < 20000:
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                # print('test_accu_best = ',test_accu_best)

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.95):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\左右二分类\\NN_Net25_pretraining' +
                            '\\%s_%s_Net25.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
                epoch += 1
                if running_loss / running_loss_initial < 0.2:
                    break
            # end while
            print(
                'NN_cross_subject_LR2class_pretraining_Net25：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'NN_cross_subject_LR2class_pretraining_Net25：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'NN_cross_subject_LR2class_pretraining_Net25：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\NN_Net25_pretraining' +
        r'\\learning_curve_NN_Net25_pretraining_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\左右二分类\\NN_Net25_pretraining' +
        r'\\confusion_matrix_NN_Net25_pretraining_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('NN_Net25_pretraining')
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\NN_Net25_pretraining' +
        r'\\learning_curve_NN_Net25_pretraining_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'NN_Net25_pretraining_LR2class')
    # end NN_cross_subject_LR2class_pretraining_Net25


# 样本集_X(numpy 三维数组(各样本 * 各帧 * 各特征))，样本集_Y(numpy 一维数组)
def LSTM_cross_subject_LR2class_pretraining_Net26():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\左右二分类\\LSTM_Net26_pretraining' +
        r'\\learning_curve_LSTM_Net26_pretraining_LR2class.txt', 'w')
    fileLog.write('LSTM 左右二分类器_预训练（左、右）跨被试\n')
    print('LSTM 左右二分类器_预训练（左、右）跨被试')
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1

        # 构造样本集
        dataSet = [X, y]
        print('LSTM_cross_subject_LR2class_pretraining_Net26：样本数量',
              sum(y == 0), sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    net_tmp = Net26()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("LSTM_cross_subject_LR2class_pretraining_Net26：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "LSTM_cross_subject_LR2class_pretraining_Net26：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        # 留下的用于测试的被试样本
        dataSet_test_X = dataSetMulti[subjectNo][0]
        dataSet_test_Y = dataSetMulti[subjectNo][1]
        # 提取用于训练的被试集样本
        dataSet_train_X = np.empty(
            (0, dataSet[0].shape[1], dataSet[0].shape[2]))
        dataSet_train_Y = np.empty((0, dataSet[1].shape[1]))
        for j in range(n_subjects):
            if j != subjectNo:
                dataSet_train_X = np.r_[dataSet_train_X, dataSetMulti[j][0]]
                dataSet_train_Y = np.r_[dataSet_train_Y, dataSetMulti[j][1]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(dataSet_train_X.shape[0])
        dataSet_train_X = dataSet_train_X[shuffle_index]
        dataSet_train_Y = dataSet_train_Y[shuffle_index]

        dataSet_train_scalered_X = dataSet_train_X
        # 训练集特征缩放
        # scaler = StandardScaler()
        # dataSet_train_scalered_X = scaler.fit_transform(dataSet_train_X[:, :,:])
        dataSet_train_scalered_X[np.isnan(
            dataSet_train_scalered_X)] = 0  # 处理异常特征

        # 记录训练集特征缩放参数，测试时会用到
        # mean_ = scaler.mean_  #均值
        # scale_ = scaler.scale_  #标准差(=np.std(axis=0),注：无ddof=1)

        dataSet_test_scalered = dataSet_test_X
        # 对测试样本集进行特征缩放
        # dataSet_test_scalered = (dataSet_test_X[:,:,:] - mean_) / scale_
        dataSet_test_scalered[np.isnan(dataSet_test_scalered)] = 0  # 处理异常特征

        t_sizes = dataSet_train_scalered_X.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [140,410,560]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(dataSet_train_scalered_X[:size, :],
                                        dataSet_train_Y[:size, :], transform)
            test_dataset = data_loader(dataSet_test_scalered, dataSet_test_Y,
                                       transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=64,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net26()
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.0001)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(800):  # loop over the dataset multiple times
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long().view(-1))
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                # print('test_accu_best = ',test_accu_best)

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels.view(-1)).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels.view(-1)).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    print('\t\ttrain_accu_cur = ', train_accu_cur)
                    if (epoch == 0) or (test_accu_best < test_accu_cur):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\左右二分类\\LSTM_Net26_pretraining' +
                            '\\%s_%s_Net26.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            # end for
            print(
                'LSTM_cross_subject_LR2class_pretraining_Net26：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'LSTM_cross_subject_LR2class_pretraining_Net26：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_LR2class_pretraining_Net26：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\LSTM_Net26_pretraining' +
        r'\\learning_curve_LSTM_Net26_pretraining_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\左右二分类\\LSTM_Net26_pretraining' +
        r'\\confusion_matrix_LSTM_Net26_pretraining_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('LSTM_Net26_pretraining')
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\LSTM_Net26_pretraining' +
        r'\\learning_curve_LSTM_Net26_pretraining_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net26_pretraining_LR2class')
    # end LSTM_cross_subject_LR2class_pretraining_Net26


# 样本集_X(numpy 三维数组(各样本 * 各帧 * 各特征))，样本集_Y(numpy 一维数组)
def LSTM_cross_subject_LR2class_pretraining_Net27():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\左右二分类\\LSTM_Net27_pretraining' +
        r'\\learning_curve_LSTM_Net27_pretraining_LR2class.txt', 'w')
    fileLog.write('LSTM 左右二分类器_预训练（左、右）跨被试\n')
    print('LSTM 左右二分类器_预训练（左、右）跨被试')
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1

        # 构造样本集
        dataSet = [X, y]
        print('LSTM_cross_subject_LR2class_pretraining_Net27：样本数量',
              sum(y == 0), sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    net_tmp = Net27()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("LSTM_cross_subject_LR2class_pretraining_Net27：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "LSTM_cross_subject_LR2class_pretraining_Net27：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        # 留下的用于测试的被试样本
        dataSet_test_X = dataSetMulti[subjectNo][0]
        dataSet_test_Y = dataSetMulti[subjectNo][1]
        # 提取用于训练的被试集样本
        dataSet_train_X = np.empty(
            (0, dataSet[0].shape[1], dataSet[0].shape[2]))
        dataSet_train_Y = np.empty((0, dataSet[1].shape[1]))
        for j in range(n_subjects):
            if j != subjectNo:
                dataSet_train_X = np.r_[dataSet_train_X, dataSetMulti[j][0]]
                dataSet_train_Y = np.r_[dataSet_train_Y, dataSetMulti[j][1]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(dataSet_train_X.shape[0])
        dataSet_train_X = dataSet_train_X[shuffle_index]
        dataSet_train_Y = dataSet_train_Y[shuffle_index]

        dataSet_train_scalered_X = dataSet_train_X
        # 训练集特征缩放
        # scaler = StandardScaler()
        # dataSet_train_scalered_X = scaler.fit_transform(dataSet_train_X[:, :,:])
        dataSet_train_scalered_X[np.isnan(
            dataSet_train_scalered_X)] = 0  # 处理异常特征

        # 记录训练集特征缩放参数，测试时会用到
        # mean_ = scaler.mean_  #均值
        # scale_ = scaler.scale_  #标准差(=np.std(axis=0),注：无ddof=1)

        dataSet_test_scalered = dataSet_test_X
        # 对测试样本集进行特征缩放
        # dataSet_test_scalered = (dataSet_test_X[:,:,:] - mean_) / scale_
        dataSet_test_scalered[np.isnan(dataSet_test_scalered)] = 0  # 处理异常特征

        t_sizes = dataSet_train_scalered_X.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [140,410,560]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(dataSet_train_scalered_X[:size, :],
                                        dataSet_train_Y[:size, :], transform)
            test_dataset = data_loader(dataSet_test_scalered, dataSet_test_Y,
                                       transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=64,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net27()
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.0001)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(800):  # loop over the dataset multiple times
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long().view(-1))
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                # print('test_accu_best = ',test_accu_best)

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels.view(-1)).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels.view(-1)).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    print('\t\ttrain_accu_cur = ', train_accu_cur)
                    if (epoch == 0) or (test_accu_best < test_accu_cur):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\左右二分类\\LSTM_Net27_pretraining' +
                            '\\%s_%s_Net27.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            # end for
            print(
                'LSTM_cross_subject_LR2class_pretraining_Net27：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'LSTM_cross_subject_LR2class_pretraining_Net27：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_LR2class_pretraining_Net27：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\LSTM_Net27_pretraining' +
        r'\\learning_curve_LSTM_Net27_pretraining_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\左右二分类\\LSTM_Net27_pretraining' +
        r'\\confusion_matrix_LSTM_Net27_pretraining_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('LSTM_Net27_pretraining')
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\LSTM_Net27_pretraining' +
        r'\\learning_curve_LSTM_Net27_pretraining_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net27_pretraining_LR2class')
    # end LSTM_cross_subject_LR2class_pretraining_Net27


# 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def CNN_LSTM_cross_subject_LR2class_pretraining_Net28():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\左右二分类\\CNN_LSTM_Net28_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net28_pretraining_LR2class.txt', 'w')
    fileLog.write('CNN_LSTM 左右二分类器_预训练（左、右）跨被试\n')
    print('CNN_LSTM 左右二分类器_预训练（左、右）跨被试')

    # 适用于 Net22、Net28
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 160  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net22、Net28

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net28()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("CNN_LSTM_cross_subject_LR2class_pretraining_Net28：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "CNN_LSTM_cross_subject_LR2class_pretraining_Net28：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[100:]
        y_test = y_test[100:]
        y_test[y_test == 2] = 0
        y_test[y_test == 3] = 1

        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][:100]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j][:100]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        # 信号分帧
        X_train_framed = []
        for X_train_I in X_train:
            X_train_framed.append(
                sp.enFrame3D(X_train_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_train = np.array(X_train_framed)
        X_test_framed = []
        for X_test_I in X_test:
            X_test_framed.append(
                sp.enFrame3D(X_test_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
            pass
        X_test = np.array(X_test_framed)

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net28()
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.0002)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(500):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    print('\t\ttrain_accu_cur = ', train_accu_cur)
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\左右二分类\\CNN_LSTM_Net28_pretraining'
                            + '\\%s_%s_Net28.pkl' %
                            (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'CNN_LSTM_cross_subject_LR2class_pretraining_Net28：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'CNN_LSTM_cross_subject_LR2class_pretraining_Net28：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_LSTM_cross_subject_LR2class_pretraining_Net28：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_LSTM_cross_subject_LR2class_pretraining_Net28：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\CNN_LSTM_Net28_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net28_pretraining_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\CNN_LSTM_Net28_pretraining' +
        r'\\confusion_matrix_CNN_LSTM_Net28_pretraining_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('CNN_LSTM_Net28_pretraining')
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\CNN_LSTM_Net28_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net28_pretraining_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_LSTM_Net28_pretraining_LR2class')
    # end CNN_LSTM_cross_subject_LR2class_pretraining_Net28


# 样本集_X(numpy 四维数组(各样本 * 各频带 * 各数据通道 * 各采样点))，样本集_Y(numpy 一维数组)
def CNN_LSTM_cross_subject_LR2class_pretraining_Net29():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\\左右二分类\\CNN_LSTM_Net29_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net29_pretraining_LR2class.txt', 'w')
    fileLog.write('CNN_LSTM 左右二分类器_预训练（左、右）跨被试\n')
    print('CNN_LSTM 左右二分类器_预训练（左、右）跨被试')

    # 适用于 Net2、Net29
    FRAME_CNT = 4  # 帧数
    FRAME_WIDTH = 480  # 帧宽
    FRAME_INC = 240  # 帧移
    # 适用于 Net2、Net29

    FS_DOWNSAMPLE = 200  # 降采样频率
    TRAIN_SIZES = np.linspace(0.1, 0.5, 15)
    classes = ('左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 留一个被试作为跨被试测试
    net_tmp = Net29()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("CNN_LSTM_cross_subject_LR2class_pretraining_Net29：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "CNN_LSTM_cross_subject_LR2class_pretraining_Net29：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))

        X_test = X_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_X
        y_test = Y_sampleSetMulti[subjectNo]  # 留下的用于测试的被试样本_y

        # 提取二分类数据
        X_test = X_test[100:]
        y_test = y_test[100:]
        y_test[y_test == 2] = 0
        y_test[y_test == 3] = 1

        # 提取用于训练的被试集样本
        X_train = np.empty(
            (0, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        y_train = np.empty((0))
        for j in range(n_subjects):
            if j != subjectNo:
                X_train = np.r_[X_train, X_sampleSetMulti[j][:100]]
                y_train = np.r_[y_train, Y_sampleSetMulti[j][:100]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_index]
        y_train = y_train[shuffle_index]

        # 信号分帧
        X_train_framed = []
        for X_train_I in X_train:
            X_train_framed.append(
                sp.enFrame3D(X_train_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X_train = np.array(X_train_framed)
        X_test_framed = []
        for X_test_I in X_test:
            X_test_framed.append(
                sp.enFrame3D(X_test_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
            pass
        X_test = np.array(X_test_framed)

        t_sizes = X_train.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [513,687]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(X_train[:size, :], y_train[:size],
                                        transform)
            test_dataset = data_loader(X_test, y_test, transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net29()
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.0002)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            for epoch in range(450):  # loop over the dataset multiple times
                print('[%d] loss: %.3f ,%.3f%%' %
                      (epoch + 1, running_loss,
                       100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100*running_loss / running_loss_initial))

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    print('\t\ttrain_accu_cur = ', train_accu_cur)
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.97):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\左右二分类\\CNN_LSTM_Net29_pretraining'
                            + '\\%s_%s_Net29.pkl' %
                            (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
            print(
                'CNN_LSTM_cross_subject_LR2class_pretraining_Net29：%s 训练集容量 %d 训练完成'
                % (NAME_LIST[subjectNo], size))
            fileLog.write(
                'CNN_LSTM_cross_subject_LR2class_pretraining_Net29：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_LSTM_cross_subject_LR2class_pretraining_Net29：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_LSTM_cross_subject_LR2class_pretraining_Net29：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\CNN_LSTM_Net29_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net29_pretraining_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\CNN_LSTM_Net29_pretraining' +
        r'\\confusion_matrix_CNN_LSTM_Net29_pretraining_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('CNN_LSTM_Net29_pretraining')
    fileD = open(
        path + r'\\..\model_all\\左右二分类\\CNN_LSTM_Net29_pretraining' +
        r'\\learning_curve_CNN_LSTM_Net29_pretraining_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_LSTM_Net29_pretraining_LR2class')
    # end CNN_LSTM_cross_subject_LR2class_pretraining_Net29


def CNN_cross_subject_LR2class_fine_Net2():
    fileLog = open(
        os.getcwd() + r'\\..\model\\左右二分类\\CNN_Net2_fine' +
        r'\\learning_curve_CNN_Net2_fine_LR2class.txt', 'w')
    fileLog.write('CNN 左右二分类器_微调（左、右）跨被试\n')
    print('CNN 左右二分类器_微调（左、右）跨被试')

    # 适用于 Net2、Net29
    FRAME_CNT = 4  # 帧数
    FRAME_WIDTH = 480  # 帧宽
    FRAME_INC = 240  # 帧移
    # 适用于 Net2、Net29

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 25)
    classes = ('左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\左右二分类\\CNN_Net2_pretraining' + '\\%s_330_Net2.pkl' % (
            NAME_LIST[subjectNo])
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1

        print('CNN_cross_subject_LR2class_fine_Net2：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 信号分帧
        X_framed = []
        for X_I in X:
            X_framed.append(
                sp.enFrame3D(X_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X = np.array(X_framed)

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.95:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print('CNN_cross_subject_LR2class_fine_Net2：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'CNN_cross_subject_LR2class_fine_Net2：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_cross_subject_LR2class_fine_Net2：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_cross_subject_LR2class_fine_Net2：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\左右二分类\\CNN_Net2_fine' +
        r'\\learning_curve_CNN_Net2_fine_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\左右二分类\\CNN_Net2_fine' +
        r'\\confusion_matrix_CNN_Net2_fine_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('CNN_Net2_fine')
    fileD = open(
        path + r'\\..\model\\左右二分类\\CNN_Net2_fine' +
        r'\\learning_curve_CNN_Net2_fine_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_Net2_fine_LR2class')

    # end CNN_cross_subject_LR2class_fine_Net2


def CNN_cross_subject_LR2class_fine_Net22():
    fileLog = open(
        os.getcwd() + r'\\..\model\\左右二分类\\CNN_Net22_fine' +
        r'\\learning_curve_CNN_Net22_fine_LR2class.txt', 'w')
    fileLog.write('CNN 左右二分类器_微调（左、右）跨被试\n')
    print('CNN 左右二分类器_微调（左、右）跨被试')

    # 适用于 Net22、Net28
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 160  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net22、Net28

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 25)
    classes = ('左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\左右二分类\\CNN_Net22_pretraining' + '\\%s_270_Net22.pkl' % (
            NAME_LIST[subjectNo])
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1
        print('CNN_cross_subject_LR2class_fine_Net22：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 信号分帧
        X_framed = []
        for X_I in X:
            X_framed.append(
                sp.enFrame3D(X_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X = np.array(X_framed)

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.95:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print(
                    'CNN_cross_subject_LR2class_fine_Net22：%s 训练集容量 %d 训练完成' %
                    (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'CNN_cross_subject_LR2class_fine_Net22：%s 训练集容量 %d 训练完成\n'
                    % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_cross_subject_LR2class_fine_Net22：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_cross_subject_LR2class_fine_Net22：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\左右二分类\\CNN_Net22_fine' +
        r'\\learning_curve_CNN_Net22_fine_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\左右二分类\\CNN_Net22_fine' +
        r'\\confusion_matrix_CNN_Net22_fine_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('CNN_Net22_fine')
    fileD = open(
        path + r'\\..\model\\左右二分类\\CNN_Net22_fine' +
        r'\\learning_curve_CNN_Net22_fine_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_Net22_fine_LR2class')

    # end CNN_cross_subject_LR2class_fine_Net22

def LSTM_cross_subject_LR2class_fine_Net23():
    fileLog = open(
        os.getcwd() + r'\\..\model\\左右二分类\\LSTM_Net23_fine' +
        r'\\learning_curve_LSTM_Net23_fine_LR2class.txt', 'w')
    fileLog.write('LSTM 左右二分类器_微调（左、右）跨被试\n')
    print('LSTM 左右二分类器_微调（左、右）跨被试')

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 25)
    classes = ('左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSet_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()


    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[2],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 对数据样本转置，以匹配 LSTM 输入要求
    X_sampleSetMulti = [
        np.swapaxes(X_sampleSet,1,2) for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\左右二分类\\LSTM_Net23_pretraining' + '\\%s_350_Net23.pkl' % (
            NAME_LIST[subjectNo])
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1
        print('LSTM_cross_subject_LR2class_fine_Net23：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=10,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.002)#0.005

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=64,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.95:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print('LSTM_cross_subject_LR2class_fine_Net23：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'LSTM_cross_subject_LR2class_fine_Net23：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_LR2class_fine_Net23：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_LR2class_fine_Net23：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\左右二分类\\LSTM_Net23_fine' +
        r'\\learning_curve_LSTM_Net23_fine_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\左右二分类\\LSTM_Net23_fine' +
        r'\\confusion_matrix_LSTM_Net23_fine_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('LSTM_Net23_fine')
    fileD = open(
        path + r'\\..\model\\左右二分类\\LSTM_Net23_fine' +
        r'\\learning_curve_LSTM_Net23_fine_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net23_fine_LR2class')

    # end LSTM_cross_subject_LR2class_fine_Net23


def LSTM_cross_subject_LR2class_fine_Net24():
    fileLog = open(
        os.getcwd() + r'\\..\model\\左右二分类\\LSTM_Net24_fine' +
        r'\\learning_curve_LSTM_Net24_fine_LR2class.txt', 'w')
    fileLog.write('LSTM 左右二分类器_微调（左、右）跨被试\n')
    print('LSTM 左右二分类器_微调（左、右）跨被试')

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 25)
    classes = ('左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSet_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()


    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[2],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    # 对数据样本转置，以匹配 LSTM 输入要求
    X_sampleSetMulti = [
        np.swapaxes(X_sampleSet,1,2) for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\上下二分类\\LSTM_Net24_pretraining' + '\\%s_330_Net24.pkl' % (
            NAME_LIST[subjectNo])
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        # 提取二分类数据
        X = X[100:,:256,:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1
        print('LSTM_cross_subject_LR2class_fine_Net24：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=10,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.002)#0.005

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=64,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.95:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print('LSTM_cross_subject_LR2class_fine_Net24：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'LSTM_cross_subject_LR2class_fine_Net24：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_LR2class_fine_Net24：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_LR2class_fine_Net24：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\左右二分类\\LSTM_Net24_fine' +
        r'\\learning_curve_LSTM_Net24_fine_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\左右二分类\\LSTM_Net24_fine' +
        r'\\confusion_matrix_LSTM_Net24_fine_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('LSTM_Net24_fine')
    fileD = open(
        path + r'\\..\model\\左右二分类\\LSTM_Net24_fine' +
        r'\\learning_curve_LSTM_Net24_fine_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net24_fine_LR2class')

    # end LSTM_cross_subject_LR2class_fine_Net24


def NN_cross_subject_LR2class_fine_Net25():
    fileLog = open(
        os.getcwd() + r'\\..\model\\左右二分类\\NN_Net25_fine' +
        r'\\learning_curve_NN_Net25_fine_LR2class.txt', 'w')
    fileLog.write('NN 左右二分类器_微调（左、右）跨被试\n')
    print('NN 左右二分类器_微调（左、右）跨被试')

    FINE_SIZES = np.linspace(0, 1.0, 25)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1
        # 构造样本集
        dataSet = np.c_[X, y]
        print('NN_cross_subject_LR2class_fine_Net25：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\左右二分类\\NN_Net25_pretraining' + '\\%s_330_Net25.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        dataSet = dataSetMulti[subjectNo]  # 取出留下的被试样本集

        # 样本集特征缩放
        scaler = StandardScaler()
        X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_train_scalered = np.c_[X_train_scalered,
                                       dataSet[:, -1]]  #特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        print('NN_cross_subject_LR2class_fine_Net25：样本数量',
              sum(dataSet_train_scalered[:, -1] == 0),
              sum(dataSet_train_scalered[:, -1] == 1),
              sum(dataSet_train_scalered[:, -1] == 2),
              sum(dataSet_train_scalered[:, -1] == 3))

        fine_set, test_set = train_test_split(
            dataSet_train_scalered, test_size=0.1)  #划分微调训练集、测试集(默认洗牌)
        X_fine = fine_set[:, :-1]  #微调训练集特征矩阵
        y_fine = fine_set[:, -1]  #微调训练集标签数组
        X_test = test_set[:, :-1]  #测试集特征矩阵
        y_test = test_set[:, -1]  #测试集标签数组

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                epoch = 0
                while epoch < 20000:
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    # print('[%d] loss: %.3f ,%.3f%%' %
                    #       (epoch + 1, running_loss,
                    #        100*running_loss / running_loss_initial))
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))

                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #       (epoch + 1, running_loss,
                        #        100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)

                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.9:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                    epoch += 1
                    if running_loss / running_loss_initial < 0.2:
                        break
                # end while
                print('NN_cross_subject_LR2class_fine_Net25：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'NN_cross_subject_LR2class_fine_Net25：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'NN_cross_subject_LR2class_fine_Net25：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'NN_cross_subject_LR2class_fine_Net25：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\左右二分类\\NN_Net25_fine' +
        r'\\learning_curve_NN_Net25_fine_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\左右二分类\\NN_Net25_fine' +
        r'\\confusion_matrix_NN_Net25_fine_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('NN_Net25_fine')
    fileD = open(
        path + r'\\..\model\\左右二分类\\NN_Net25_fine' +
        r'\\learning_curve_NN_Net25_fine_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'NN_Net25_fine_LR2class')
    # end NN_cross_subject_LR2class_fine_Net25


def LSTM_cross_subject_LR2class_fine_Net26():
    fileLog = open(
        os.getcwd() + r'\\..\model\\左右二分类\\LSTM_Net26_fine' +
        r'\\learning_curve_LSTM_Net26_fine_LR2class.txt', 'w')
    fileLog.write('LSTM 左右二分类器_微调（左、右）跨被试\n')
    print('LSTM 左右二分类器_微调（左、右）跨被试')

    FINE_SIZES = np.linspace(0, 1.0, 25)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1
        # 构造样本集
        dataSet = [X, y]
        print('LSTM_cross_subject_LR2class_fine_Net26：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\左右二分类\\LSTM_Net26_pretraining' + '\\%s_310_Net26.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        # 取出留下的被试样本集
        dataSet_test_X = dataSetMulti[subjectNo][0]
        dataSet_test_Y = dataSetMulti[subjectNo][1]
        dataSet_train_scalered = dataSet_test_X
        # 样本集特征缩放
        # scaler = StandardScaler()
        # X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
        # dataSet_train_scalered = np.c_[X_train_scalered,
        #                                dataSet[:, -1]]  # 特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        print('LSTM_cross_subject_LR2class_fine_Net26：样本数量',
              sum(dataSet_test_Y == 0), sum(dataSet_test_Y == 1),
              sum(dataSet_test_Y == 2), sum(dataSet_test_Y == 3))

        X_test, y_test, X_fine, y_fine = sp.splitSet(
            dataSet_train_scalered, dataSet_test_Y, ratio=0.1,
            shuffle=True)  # 划分微调训练集、测试集(洗牌)

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [180]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.0001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels.view(-1)).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        800):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long().view(-1))
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    # print('[%d] loss: %.3f ,%.3f%%' %
                    #       (epoch + 1, running_loss,
                    #        100*running_loss / running_loss_initial))
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels.view(-1)).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels.view(-1)).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))

                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #       (epoch + 1, running_loss,
                        #        100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)

                        if test_accu_best < test_accu_cur:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                # end for
                print(
                    'LSTM_cross_subject_LR2class_fine_Net26：%s 训练集容量 %d 训练完成' %
                    (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'LSTM_cross_subject_LR2class_fine_Net26：%s 训练集容量 %d 训练完成\n'
                    % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_LR2class_fine_Net26：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_LR2class_fine_Net26：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\左右二分类\\LSTM_Net26_fine' +
        r'\\learning_curve_LSTM_Net26_fine_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\左右二分类\\LSTM_Net26_fine' +
        r'\\confusion_matrix_LSTM_Net26_fine_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('LSTM_Net26_fine')
    fileD = open(
        path + r'\\..\model\\左右二分类\\LSTM_Net26_fine' +
        r'\\learning_curve_LSTM_Net26_fine_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net26_fine_LR2class')
    # end LSTM_cross_subject_LR2class_fine_Net26


def LSTM_cross_subject_LR2class_fine_Net27():
    fileLog = open(
        os.getcwd() + r'\\..\model\\左右二分类\\LSTM_Net27_fine' +
        r'\\learning_curve_LSTM_Net27_fine_LR2class.txt', 'w')
    fileLog.write('LSTM 左右二分类器_微调（左、右）跨被试\n')
    print('LSTM 左右二分类器_微调（左、右）跨被试')

    FINE_SIZES = np.linspace(0, 1.0, 25)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle',
            'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1
        # 构造样本集
        dataSet = [X, y]
        print('LSTM_cross_subject_LR2class_fine_Net27：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\左右二分类\\LSTM_Net27_pretraining' + '\\%s_290_Net27.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        # 取出留下的被试样本集
        dataSet_test_X = dataSetMulti[subjectNo][0]
        dataSet_test_Y = dataSetMulti[subjectNo][1]
        dataSet_train_scalered = dataSet_test_X
        # 样本集特征缩放
        # scaler = StandardScaler()
        # X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
        # dataSet_train_scalered = np.c_[X_train_scalered,
        #                                dataSet[:, -1]]  # 特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        print('LSTM_cross_subject_LR2class_fine_Net27：样本数量',
              sum(dataSet_test_Y == 0), sum(dataSet_test_Y == 1),
              sum(dataSet_test_Y == 2), sum(dataSet_test_Y == 3))

        X_test, y_test, X_fine, y_fine = sp.splitSet(
            dataSet_train_scalered, dataSet_test_Y, ratio=0.1,
            shuffle=True)  # 划分微调训练集、测试集(洗牌)

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [180]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.0001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels.view(-1)).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        800):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long().view(-1))
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    # print('[%d] loss: %.3f ,%.3f%%' %
                    #       (epoch + 1, running_loss,
                    #        100*running_loss / running_loss_initial))
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels.view(-1)).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels.view(-1)).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))

                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #       (epoch + 1, running_loss,
                        #        100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)

                        if test_accu_best < test_accu_cur:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                # end for
                print(
                    'LSTM_cross_subject_LR2class_fine_Net27：%s 训练集容量 %d 训练完成' %
                    (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'LSTM_cross_subject_LR2class_fine_Net27：%s 训练集容量 %d 训练完成\n'
                    % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'LSTM_cross_subject_LR2class_fine_Net27：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'LSTM_cross_subject_LR2class_fine_Net27：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\左右二分类\\LSTM_Net27_fine' +
        r'\\learning_curve_LSTM_Net27_fine_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\左右二分类\\LSTM_Net27_fine' +
        r'\\confusion_matrix_LSTM_Net27_fine_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('LSTM_Net27_fine')
    fileD = open(
        path + r'\\..\model\\左右二分类\\LSTM_Net27_fine' +
        r'\\learning_curve_LSTM_Net27_fine_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'LSTM_Net27_fine_LR2class')
    # end LSTM_cross_subject_LR2class_fine_Net27


def CNN_LSTM_cross_subject_LR2class_fine_Net28():
    fileLog = open(
        os.getcwd() + r'\\..\model\\左右二分类\\CNN_LSTM_Net28_fine' +
        r'\\learning_curve_CNN_LSTM_Net28_fine_LR2class.txt', 'w')
    fileLog.write('CNN_LSTM 左右二分类器_微调（左、右）跨被试\n')
    print('CNN_LSTM 左右二分类器_微调（左、右）跨被试')

    # 适用于 Net22、Net28
    FRAME_CNT = 7  # 帧数
    FRAME_WIDTH = 160  # 帧宽
    FRAME_INC = 160  # 帧移
    # 适用于 Net22、Net28

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 25)
    classes = ('左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\左右二分类\\CNN_LSTM_Net28_pretraining' + '\\%s_330_Net28.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1
        print('CNN_LSTM_cross_subject_LR2class_fine_Net28：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 信号分帧
        X_framed = []
        for X_I in X:
            X_framed.append(
                sp.enFrame3D(X_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X = np.array(X_framed)

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.0001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #     (epoch + 1, running_loss,
                        #     100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.97:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print(
                    'CNN_LSTM_cross_subject_LR2class_fine_Net28：%s 训练集容量 %d 训练完成'
                    % (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'CNN_LSTM_cross_subject_LR2class_fine_Net28：%s 训练集容量 %d 训练完成\n'
                    % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_LSTM_cross_subject_LR2class_fine_Net28：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_LSTM_cross_subject_LR2class_fine_Net28：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\左右二分类\\CNN_LSTM_Net28_fine' +
        r'\\learning_curve_CNN_LSTM_Net28_fine_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\左右二分类\\CNN_LSTM_Net28_fine' +
        r'\\confusion_matrix_CNN_LSTM_Net28_fine_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.myLearning_curve_LIST2DICT_LR2class('CNN_LSTM_Net28_fine')
    fileD = open(
        path + r'\\..\model\\左右二分类\\CNN_LSTM_Net28_fine' +
        r'\\learning_curve_CNN_LSTM_Net28_fine_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_LSTM_Net28_fine_LR2class')
    # end CNN_LSTM_cross_subject_LR2class_fine_Net28


def CNN_LSTM_cross_subject_LR2class_fine_Net29():
    fileLog = open(
        os.getcwd() + r'\\..\model\\左右二分类\\CNN_LSTM_Net29_fine' +
        r'\\learning_curve_CNN_LSTM_Net29_fine_LR2class.txt', 'w')
    fileLog.write('CNN_LSTM\左右二分类器_微调（左、右）跨被试\n')
    print('CNN_LSTM\左右二分类器_微调（左、右）跨被试')

    # 适用于 Net2、Net29
    FRAME_CNT = 4  # 帧数
    FRAME_WIDTH = 480  # 帧宽
    FRAME_INC = 240  # 帧移
    # 适用于 Net2、Net29

    FS_DOWNSAMPLE = 200  # 降采样频率
    FINE_SIZES = np.linspace(0, 1.0, 25)
    classes = ('左', '右')
    # 提取所有被试数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\四分类' +
        '\\single_move_crossSubject_sampleSetMulti_X_y.pickle', 'rb')
    X_sampleSetMulti, Y_sampleSetMulti = pickle.load(fileD)
    fileD.close()

    # 输入检查：输入数据的长度必须满足分帧需求（不能有未满帧）
    assert X_sampleSetMulti[0].shape[3] >= FRAME_INC * (FRAME_CNT -
                                                        1) + FRAME_WIDTH

    # 降采样
    sampleIndex = np.arange(0, X_sampleSetMulti[0].shape[3],
                            1000 / FS_DOWNSAMPLE).astype(int)
    X_sampleSetMulti = [
        X_sampleSet[:, :, :, sampleIndex] for X_sampleSet in X_sampleSetMulti
    ]

    n_subjects = len(NAME_LIST)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\左右二分类\\CNN_LSTM_Net29_pretraining' + '\\%s_330_Net29.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        X = X_sampleSetMulti[subjectNo]  # 取出留下的被试样本_X
        y = Y_sampleSetMulti[subjectNo]  # 取出留下的的被试样本_y

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1
        print('CNN_LSTM_cross_subject_LR2class_fine_Net29：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        # 样本集洗牌
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]

        # 信号分帧
        X_framed = []
        for X_I in X:
            X_framed.append(
                sp.enFrame3D(X_I, FRAME_CNT,
                             int(FRAME_WIDTH / 1000.0 * FS_DOWNSAMPLE),
                             int(FRAME_INC / 1000.0 * FS_DOWNSAMPLE)))
        X = np.array(X_framed)

        # 划分微调训练集、测试集
        X_fine = X[:int(X.shape[0] * 0.9)]
        y_fine = y[:int(y.shape[0] * 0.9)]
        X_test = X[int(X.shape[0] * 0.9):]
        y_test = y[int(y.shape[0] * 0.9):]

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.0001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                for epoch in range(
                        400):  # loop over the dataset multiple times
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))
                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #     (epoch + 1, running_loss,
                        #     100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)
                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.97:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                print(
                    'CNN_LSTM_cross_subject_LR2class_fine_Net29：%s 训练集容量 %d 训练完成'
                    % (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'CNN_LSTM_cross_subject_LR2class_fine_Net29：%s 训练集容量 %d 训练完成\n'
                    % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'CNN_LSTM_cross_subject_LR2class_fine_Net29：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'CNN_LSTM_cross_subject_LR2class_fine_Net29：%s : 测试集准确率  %.3f%%\n'
            % (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\左右二分类\\CNN_LSTM_Net29_fine' +
        r'\\learning_curve_CNN_LSTM_Net29_fine_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\左右二分类\\CNN_LSTM_Net29_fine' +
        r'\\confusion_matrix_CNN_LSTM_Net29_fine_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.myLearning_curve_LIST2DICT_LR2class('CNN_LSTM_Net29_fine')
    fileD = open(
        path + r'\\..\model\\左右二分类\\CNN_LSTM_Net29_fine' +
        r'\\learning_curve_CNN_LSTM_Net29_fine_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'CNN_LSTM_Net29_fine_LR2class')
    # end CNN_LSTM_cross_subject_LR2class_fine_Net29

# 样本集_X(numpy 二维数组(各样本 * 各特征))，样本集_Y(numpy 一维数组)
def NN_cross_subject_4class_pretraining_Net45_multiTask():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\多任务\\四分类\\NN_Net45_pretraining' +
        r'\\learning_curve_NN_Net45_pretraining_4class.txt', 'w')
    fileLog.write('NN 四分类器（上、下、左、右）跨被试\n')
    print('NN 四分类器（上、下、左、右）跨被试')
    TRAIN_SIZES = np.linspace(0.1, 0.5, 3)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\多任务' +
            '\\multi_task_motion_start_motion_end.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = np.c_[X, y]
        print('NN_cross_subject_4class_pretraining_Net45_multiTask：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    net_tmp = Net45()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("NN_cross_subject_4class_pretraining_Net45_multiTask：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "NN_cross_subject_4class_pretraining_Net45_multiTask：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        dataSet_test = dataSetMulti[subjectNo]  # 留下的用于测试的被试样本

        # 提取用于训练的被试集样本
        dataSet_train = np.empty((0, dataSet.shape[1]))
        for j in range(n_subjects):
            if j != subjectNo:
                dataSet_train = np.r_[dataSet_train, dataSetMulti[j]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(dataSet_train.shape[0])
        dataSet_train = dataSet_train[shuffle_index]

        # 训练集特征缩放
        scaler = StandardScaler()
        X_train_scalered = scaler.fit_transform(dataSet_train[:, :-1])
        dataSet_train_scalered = np.c_[X_train_scalered,
                                       dataSet_train[:, -1]]  #特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        # 记录训练集特征缩放参数，测试时会用到
        mean_ = scaler.mean_  #均值
        scale_ = scaler.scale_  #标准差(=np.std(axis=0),注：无ddof=1)

        # 对测试样本集进行特征缩放
        X_test_scalered = (dataSet_test[:, :-1] - mean_) / scale_
        dataSet_test_scalered = np.c_[X_test_scalered,
                                      dataSet_test[:, -1]]  #特征缩放后的测试样本集(X,y)
        dataSet_test_scalered[np.isnan(dataSet_test_scalered)] = 0  # 处理异常特征

        t_sizes = dataSet_train_scalered.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [561,410,560]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(dataSet_train_scalered[:size, :-1],
                                        dataSet_train_scalered[:size,
                                                               -1], transform)
            test_dataset = data_loader(dataSet_test_scalered[:, :-1],
                                       dataSet_test_scalered[:, -1], transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net45()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.002,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            epoch = 0
            while epoch < 20000:
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                # print('test_accu_best = ',test_accu_best)

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.95):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\多任务\\四分类\\NN_Net45_pretraining' +
                            '\\%s_%s_Net45.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
                epoch += 1
                if running_loss / running_loss_initial < 0.2:
                    break
            # end while
            print(
                'NN_cross_subject_4class_pretraining_Net45_multiTask：%s 训练集容量 %d 训练完成' %
                (NAME_LIST[subjectNo], size))
            fileLog.write(
                'NN_cross_subject_4class_pretraining_Net45_multiTask：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'NN_cross_subject_4class_pretraining_Net45_multiTask：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\多任务\\四分类\\NN_Net45_pretraining' +
        r'\\learning_curve_NN_Net45_pretraining_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\多任务\\四分类\\NN_Net45_pretraining' +
        r'\\confusion_matrix_NN_Net45_pretraining_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_4class_multiTask('NN_Net45_pretraining')
    fileD = open(
        path + r'\\..\model_all\\多任务\\四分类\\NN_Net45_pretraining' +
        r'\\learning_curve_NN_Net45_pretraining_4class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'NN_Net45_pretraining_4class')
    # end NN_cross_subject_4class_pretraining_Net45_multiTask


# 样本集_X(numpy 二维数组(各样本 * 各特征))，样本集_Y(numpy 一维数组)
def NN_cross_subject_UD2class_pretraining_Net25_multiTask():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\多任务\\上下二分类\\NN_Net25_pretraining' +
        r'\\learning_curve_NN_Net25_pretraining_UD2class.txt', 'w')
    fileLog.write('NN 上下二分类器（上、下、左、右）跨被试\n')
    print('NN 上下二分类器（上、下、左、右）跨被试')
    TRAIN_SIZES = np.linspace(0.1, 0.5, 3)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\多任务' +
            '\\multi_task_motion_start_motion_end.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[:100]
        y = y[:100]

        # 构造样本集
        dataSet = np.c_[X, y]
        print('NN_cross_subject_UD2class_pretraining_Net25_multiTask：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    net_tmp = Net25()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("NN_cross_subject_UD2class_pretraining_Net25_multiTask：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "NN_cross_subject_UD2class_pretraining_Net25_multiTask：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        dataSet_test = dataSetMulti[subjectNo]  # 留下的用于测试的被试样本

        # 提取用于训练的被试集样本
        dataSet_train = np.empty((0, dataSet.shape[1]))
        for j in range(n_subjects):
            if j != subjectNo:
                dataSet_train = np.r_[dataSet_train, dataSetMulti[j]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(dataSet_train.shape[0])
        dataSet_train = dataSet_train[shuffle_index]

        # 训练集特征缩放
        scaler = StandardScaler()
        X_train_scalered = scaler.fit_transform(dataSet_train[:, :-1])
        dataSet_train_scalered = np.c_[X_train_scalered,
                                       dataSet_train[:, -1]]  #特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        # 记录训练集特征缩放参数，测试时会用到
        mean_ = scaler.mean_  #均值
        scale_ = scaler.scale_  #标准差(=np.std(axis=0),注：无ddof=1)

        # 对测试样本集进行特征缩放
        X_test_scalered = (dataSet_test[:, :-1] - mean_) / scale_
        dataSet_test_scalered = np.c_[X_test_scalered,
                                      dataSet_test[:, -1]]  #特征缩放后的测试样本集(X,y)
        dataSet_test_scalered[np.isnan(dataSet_test_scalered)] = 0  # 处理异常特征

        t_sizes = dataSet_train_scalered.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [561,410,560]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(dataSet_train_scalered[:size, :-1],
                                        dataSet_train_scalered[:size,
                                                               -1], transform)
            test_dataset = data_loader(dataSet_test_scalered[:, :-1],
                                       dataSet_test_scalered[:, -1], transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net25()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.002,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            epoch = 0
            while epoch < 500:
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                # print('test_accu_best = ',test_accu_best)

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.95):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\多任务\\上下二分类\\NN_Net25_pretraining' +
                            '\\%s_%s_Net25.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
                epoch += 1
                if running_loss / running_loss_initial < 0.2:
                    break
            # end while
            print(
                'NN_cross_subject_UD2class_pretraining_Net25_multiTask：%s 训练集容量 %d 训练完成' %
                (NAME_LIST[subjectNo], size))
            fileLog.write(
                'NN_cross_subject_UD2class_pretraining_Net25_multiTask：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'NN_cross_subject_UD2class_pretraining_Net25_multiTask：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\多任务\\上下二分类\\NN_Net25_pretraining' +
        r'\\learning_curve_NN_Net25_pretraining_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\多任务\\上下二分类\\NN_Net25_pretraining' +
        r'\\confusion_matrix_NN_Net25_pretraining_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class_multiTask('NN_Net25_pretraining')
    fileD = open(
        path + r'\\..\model_all\\多任务\\上下二分类\\NN_Net25_pretraining' +
        r'\\learning_curve_NN_Net25_pretraining_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'NN_Net25_pretraining_UD2class')
    # end NN_cross_subject_UD2class_pretraining_Net25_multiTask

# 样本集_X(numpy 二维数组(各样本 * 各特征))，样本集_Y(numpy 一维数组)
def NN_cross_subject_LR2class_pretraining_Net25_multiTask():
    fileLog = open(
        os.getcwd() + r'\\..\model_all\多任务\\左右二分类\\NN_Net25_pretraining' +
        r'\\learning_curve_NN_Net25_pretraining_LR2class.txt', 'w')
    fileLog.write('NN 左右二分类器（上、下、左、右）跨被试\n')
    print('NN 左右二分类器（上、下、左、右）跨被试')
    TRAIN_SIZES = np.linspace(0.1, 0.5, 3)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\多任务' +
            '\\multi_task_motion_start_motion_end.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1

        # 构造样本集
        dataSet = np.c_[X, y]
        print('NN_cross_subject_LR2class_pretraining_Net25_multiTask：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    net_tmp = Net25()
    for x in net_tmp.parameters():
        print(x.data.shape, len(x.data.reshape(-1)))

    print("NN_cross_subject_LR2class_pretraining_Net25_multiTask：参数个数 {}  ".format(
        sum(x.numel() for x in net_tmp.parameters())))
    fileLog.write(
        "NN_cross_subject_LR2class_pretraining_Net25_multiTask：参数个数 {}  \n".format(
            sum(x.numel() for x in net_tmp.parameters())))
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        dataSet_test = dataSetMulti[subjectNo]  # 留下的用于测试的被试样本

        # 提取用于训练的被试集样本
        dataSet_train = np.empty((0, dataSet.shape[1]))
        for j in range(n_subjects):
            if j != subjectNo:
                dataSet_train = np.r_[dataSet_train, dataSetMulti[j]]
            # end if
        # end for

        # 训练样本集洗牌
        shuffle_index = np.random.permutation(dataSet_train.shape[0])
        dataSet_train = dataSet_train[shuffle_index]

        # 训练集特征缩放
        scaler = StandardScaler()
        X_train_scalered = scaler.fit_transform(dataSet_train[:, :-1])
        dataSet_train_scalered = np.c_[X_train_scalered,
                                       dataSet_train[:, -1]]  #特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        # 记录训练集特征缩放参数，测试时会用到
        mean_ = scaler.mean_  #均值
        scale_ = scaler.scale_  #标准差(=np.std(axis=0),注：无ddof=1)

        # 对测试样本集进行特征缩放
        X_test_scalered = (dataSet_test[:, :-1] - mean_) / scale_
        dataSet_test_scalered = np.c_[X_test_scalered,
                                      dataSet_test[:, -1]]  #特征缩放后的测试样本集(X,y)
        dataSet_test_scalered[np.isnan(dataSet_test_scalered)] = 0  # 处理异常特征

        t_sizes = dataSet_train_scalered.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [561,410,560]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            # 训练模型
            transform = None
            train_dataset = data_loader(dataSet_train_scalered[:size, :-1],
                                        dataSet_train_scalered[:size,
                                                               -1], transform)
            test_dataset = data_loader(dataSet_test_scalered[:, :-1],
                                       dataSet_test_scalered[:, -1], transform)

            trainloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=32,
                                                      shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = Net25()
            net = net.double()
            criterion = nn.CrossEntropyLoss()

            weight_p, bias_p = [], []
            for name, param in net.named_parameters():
                if 'bias' in name:
                    bias_p += [param]
                else:
                    weight_p += [param]

            optimizer = optim.SGD([{
                'params': weight_p,
                'weight_decay': 3.117e-3
            }, {
                'params': bias_p,
                'weight_decay': 0
            }],
                                  lr=0.002,
                                  momentum=0.9)

            train_accu_best = 0.0
            test_accu_best = 0.0
            running_loss = 0.1
            running_loss_initial = 0.1
            epoch = 0
            while epoch < 500:
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss = running_loss / size
                if epoch == 0:
                    running_loss_initial = running_loss
                # print('[%d] loss: %.3f ,%.3f%%' %
                #       (epoch + 1, running_loss,
                #        100 * running_loss / running_loss_initial))
                # print('test_accu_best = ',test_accu_best)

                if epoch % 10 == 0:
                    # 计算训练集准确率
                    class_correct_train = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_train[
                                        label.int()] += c.item()
                                else:
                                    class_correct_train[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    train_accu_cur = sum(class_correct_train) / sum(
                        class_total)
                    # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                    #       (100 * sum(class_correct_train) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_train[i] / class_total[i]))

                    # 计算测试集准确率
                    class_correct_test = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    with torch.no_grad():
                        y_true_cm = []
                        y_pred_cm = []
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            c = (predicted == labels).squeeze()
                            y_true_cm = y_true_cm + list(labels)
                            y_pred_cm = y_pred_cm + list(predicted)
                            for i, label in enumerate(labels):
                                if len(c.shape) == 0:
                                    class_correct_test[label.int()] += c.item()
                                else:
                                    class_correct_test[
                                        label.int()] += c[i].item()
                                class_total[label.int()] += 1
                    test_accu_cur = sum(class_correct_test) / sum(class_total)

                    # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                    #       (100 * sum(class_correct_test) / sum(class_total)))
                    # for i in range(4):
                    #     print('\t%5s ：%2d %%' %
                    #           (classes[i],
                    #            100 * class_correct_test[i] / class_total[i]))
                    print('\t\t[%d] loss: %.3f ,%.3f%%' %
                          (epoch + 1, running_loss,
                           100 * running_loss / running_loss_initial))
                    print('\t\ttest_accu_best = ', test_accu_best)
                    print('\t\ttest_accu_cur = ', test_accu_cur)
                    if (epoch == 0) or (
                            test_accu_best < test_accu_cur
                            and running_loss / running_loss_initial < 0.95):
                        train_accu_best = train_accu_cur
                        test_accu_best = test_accu_cur
                        y_true_cm_best = y_true_cm
                        y_pred_cm_best = y_pred_cm
                        torch.save(
                            net,
                            os.getcwd() +
                            r'\\..\\model_all\\多任务\\左右二分类\\NN_Net25_pretraining' +
                            '\\%s_%s_Net25.pkl' % (NAME_LIST[subjectNo], size))

                        print('[%d] loss: %.3f ,%.3f%%' %
                              (epoch + 1, running_loss,
                               100 * running_loss / running_loss_initial))
                        print('train_accu_best = %.3f%%' %
                              (100 * train_accu_best))
                        print('test_accu_best = %.3f%%' %
                              (100 * test_accu_best))
                        fileLog.write('[%d] loss: %.3f ,%.3f%%\n' %
                                      (epoch + 1, running_loss, 100 *
                                       running_loss / running_loss_initial))
                        fileLog.write('train_accu_best = %.3f%%\n' %
                                      (100 * train_accu_best))
                        fileLog.write('test_accu_best = %.3f%%\n' %
                                      (100 * test_accu_best))
                epoch += 1
                if running_loss / running_loss_initial < 0.2:
                    break
            # end while
            print(
                'NN_cross_subject_LR2class_pretraining_Net25_multiTask：%s 训练集容量 %d 训练完成' %
                (NAME_LIST[subjectNo], size))
            fileLog.write(
                'NN_cross_subject_LR2class_pretraining_Net25_multiTask：%s 训练集容量 %d 训练完成\n'
                % (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'NN_cross_subject_LR2class_pretraining_Net25_multiTask：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model_all\\多任务\\左右二分类\\NN_Net25_pretraining' +
        r'\\learning_curve_NN_Net25_pretraining_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model_all\\多任务\\左右二分类\\NN_Net25_pretraining' +
        r'\\confusion_matrix_NN_Net25_pretraining_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class_multiTask('NN_Net25_pretraining')
    fileD = open(
        path + r'\\..\model_all\\多任务\\左右二分类\\NN_Net25_pretraining' +
        r'\\learning_curve_NN_Net25_pretraining_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'NN_Net25_pretraining_LR2class')
    # end NN_cross_subject_LR2class_pretraining_Net25_multiTask

def NN_cross_subject_4class_fine_Net45_multiTask():
    fileLog = open(
        os.getcwd() + r'\\..\model\\多任务\\四分类\\NN_Net45_fine' +
        r'\\learning_curve_NN_Net45_fine_LR2class.txt', 'w')
    fileLog.write('NN 四分类器_微调（左、右）跨被试\n')
    print('NN 四分类器_微调（左、右）跨被试')

    FINE_SIZES = np.linspace(0, 1.0, 3)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\多任务' +
            '\\multi_task_motion_start_motion_end.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = np.c_[X, y]
        print('NN_cross_subject_4class_fine_Net45_multiTask：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\多任务\\四分类\\NN_Net45_pretraining' + '\\%s_700_Net45.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        dataSet = dataSetMulti[subjectNo]  # 取出留下的被试样本集

        # 样本集特征缩放
        scaler = StandardScaler()
        X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_train_scalered = np.c_[X_train_scalered,
                                       dataSet[:, -1]]  #特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        print('NN_cross_subject_4class_fine_Net45_multiTask：样本数量',
              sum(dataSet_train_scalered[:, -1] == 0),
              sum(dataSet_train_scalered[:, -1] == 1),
              sum(dataSet_train_scalered[:, -1] == 2),
              sum(dataSet_train_scalered[:, -1] == 3))

        fine_set, test_set = train_test_split(
            dataSet_train_scalered, test_size=0.1)  #划分微调训练集、测试集(默认洗牌)
        X_fine = fine_set[:, :-1]  #微调训练集特征矩阵
        y_fine = fine_set[:, -1]  #微调训练集标签数组
        X_test = test_set[:, :-1]  #测试集特征矩阵
        y_test = test_set[:, -1]  #测试集标签数组

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                epoch = 0
                while epoch < 20000:
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    # print('[%d] loss: %.3f ,%.3f%%' %
                    #       (epoch + 1, running_loss,
                    #        100*running_loss / running_loss_initial))
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))

                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #       (epoch + 1, running_loss,
                        #        100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)

                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.9:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                    epoch += 1
                    if running_loss / running_loss_initial < 0.2:
                        break
                # end while
                print('NN_cross_subject_4class_fine_Net45_multiTask：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'NN_cross_subject_4class_fine_Net45_multiTask：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'NN_cross_subject_4class_fine_Net45_multiTask：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'NN_cross_subject_4class_fine_Net45_multiTask：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\多任务\\四分类\\NN_Net45_fine' +
        r'\\learning_curve_NN_Net45_fine_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\多任务\\四分类\\NN_Net45_fine' +
        r'\\confusion_matrix_NN_Net45_fine_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_4class_multiTask('NN_Net45_fine')
    fileD = open(
        path + r'\\..\model\\多任务\\四分类\\NN_Net45_fine' +
        r'\\learning_curve_NN_Net45_fine_4class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'NN_Net45_fine_4class')
    # end NN_cross_subject_4class_fine_Net45_multiTask


def NN_cross_subject_UD2class_fine_Net25_multiTask():
    fileLog = open(
        os.getcwd() + r'\\..\model\\多任务\\上下二分类\\NN_Net25_fine' +
        r'\\learning_curve_NN_Net25_fine_LR2class.txt', 'w')
    fileLog.write('NN 上下二分类器_微调（左、右）跨被试\n')
    print('NN 上下二分类器_微调（左、右）跨被试')

    FINE_SIZES = np.linspace(0, 1.0, 3)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\多任务' +
            '\\multi_task_motion_start_motion_end.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[:100]
        y = y[:100]
        # 构造样本集
        dataSet = np.c_[X, y]
        print('NN_cross_subject_UD2class_fine_Net25_multiTask：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\多任务\\上下二分类\\NN_Net25_pretraining' + '\\%s_350_Net25.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        dataSet = dataSetMulti[subjectNo]  # 取出留下的被试样本集

        # 样本集特征缩放
        scaler = StandardScaler()
        X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_train_scalered = np.c_[X_train_scalered,
                                       dataSet[:, -1]]  #特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        print('NN_cross_subject_UD2class_fine_Net25_multiTask：样本数量',
              sum(dataSet_train_scalered[:, -1] == 0),
              sum(dataSet_train_scalered[:, -1] == 1),
              sum(dataSet_train_scalered[:, -1] == 2),
              sum(dataSet_train_scalered[:, -1] == 3))

        fine_set, test_set = train_test_split(
            dataSet_train_scalered, test_size=0.1)  #划分微调训练集、测试集(默认洗牌)
        X_fine = fine_set[:, :-1]  #微调训练集特征矩阵
        y_fine = fine_set[:, -1]  #微调训练集标签数组
        X_test = test_set[:, :-1]  #测试集特征矩阵
        y_test = test_set[:, -1]  #测试集标签数组

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                epoch = 0
                while epoch < 20000:
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    # print('[%d] loss: %.3f ,%.3f%%' %
                    #       (epoch + 1, running_loss,
                    #        100*running_loss / running_loss_initial))
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))

                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #       (epoch + 1, running_loss,
                        #        100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)

                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.9:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                    epoch += 1
                    if running_loss / running_loss_initial < 0.2:
                        break
                # end while
                print('NN_cross_subject_UD2class_fine_Net25_multiTask：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'NN_cross_subject_UD2class_fine_Net25_multiTask：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'NN_cross_subject_UD2class_fine_Net25_multiTask：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'NN_cross_subject_UD2class_fine_Net25_multiTask：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\多任务\\上下二分类\\NN_Net25_fine' +
        r'\\learning_curve_NN_Net25_fine_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\多任务\\上下二分类\\NN_Net25_fine' +
        r'\\confusion_matrix_NN_Net25_fine_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class_multiTask('NN_Net25_fine')
    fileD = open(
        path + r'\\..\model\\多任务\\上下二分类\\NN_Net25_fine' +
        r'\\learning_curve_NN_Net25_fine_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'NN_Net25_fine_UD2class')
    # end NN_cross_subject_UD2class_fine_Net25_multiTask

def NN_cross_subject_LR2class_fine_Net25_multiTask():
    fileLog = open(
        os.getcwd() + r'\\..\model\\多任务\\左右二分类\\NN_Net25_fine' +
        r'\\learning_curve_NN_Net25_fine_LR2class.txt', 'w')
    fileLog.write('NN 左右二分类器_微调（左、右）跨被试\n')
    print('NN 左右二分类器_微调（左、右）跨被试')

    FINE_SIZES = np.linspace(0, 1.0, 3)
    path = os.getcwd()
    dataSetMulti = []

    # 提取所有被试数据
    for name in NAME_LIST:
        # 遍历各被试
        fileD = open(
            path + r'\\..\\data\\' + name + '\\多任务' +
            '\\multi_task_motion_start_motion_end.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X = X[100:]
        y = y[100:]
        y[y == 2] = 0
        y[y == 3] = 1
        # 构造样本集
        dataSet = np.c_[X, y]
        print('NN_cross_subject_LR2class_fine_Net25_multiTask：样本数量', sum(y == 0),
              sum(y == 1), sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for subjectNo in range(n_subjects):
        confusion_matrix_DICT_subjectI = {}
        print('------------------\n%s 的训练情况：' % (NAME_LIST[subjectNo]))
        fileLog.write('------------------\n%s 的训练情况：\n' %
                      (NAME_LIST[subjectNo]))
        path = os.getcwd(
        ) + r'\\..\\model\\多任务\\左右二分类\\NN_Net25_pretraining' + '\\%s_350_Net25.pkl' % (
            NAME_LIST[subjectNo])
        print(path)
        dataSet = dataSetMulti[subjectNo]  # 取出留下的被试样本集

        # 样本集特征缩放
        scaler = StandardScaler()
        X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_train_scalered = np.c_[X_train_scalered,
                                       dataSet[:, -1]]  #特征缩放后的样本集(X,y)
        dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

        print('NN_cross_subject_LR2class_fine_Net25_multiTask：样本数量',
              sum(dataSet_train_scalered[:, -1] == 0),
              sum(dataSet_train_scalered[:, -1] == 1),
              sum(dataSet_train_scalered[:, -1] == 2),
              sum(dataSet_train_scalered[:, -1] == 3))

        fine_set, test_set = train_test_split(
            dataSet_train_scalered, test_size=0.1)  #划分微调训练集、测试集(默认洗牌)
        X_fine = fine_set[:, :-1]  #微调训练集特征矩阵
        y_fine = fine_set[:, -1]  #微调训练集标签数组
        X_test = test_set[:, :-1]  #测试集特征矩阵
        y_test = test_set[:, -1]  #测试集标签数组

        t_sizes = X_fine.shape[0] * FINE_SIZES
        t_sizes = t_sizes.astype(int)
        # t_sizes = [700,1400]  ########################
        scores_train = []
        scores_test = []
        for size in t_sizes:
            confusion_matrix_DICT_subjectI_sizeI = {}
            print('+++++++++ size = %d' % (size))
            fileLog.write('+++++++++ size = %d\n' % (size))
            transform = None
            test_dataset = data_loader(X_test, y_test, transform)
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=4,
                                                     shuffle=False)
            net = torch.load(path)
            net = net.double()
            criterion = nn.CrossEntropyLoss()
            for para in net.seq1.parameters():
                para.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          net.parameters()),
                                   lr=0.001)

            if size == 0:
                # 计算测试集准确率
                class_correct_test = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    y_true_cm_best = []
                    y_pred_cm_best = []
                    for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c = (predicted == labels).squeeze()
                        y_true_cm_best = y_true_cm_best + list(labels)
                        y_pred_cm_best = y_pred_cm_best + list(predicted)
                        for i, label in enumerate(labels):
                            if len(c.shape) == 0:
                                class_correct_test[label.int()] += c.item()
                            else:
                                class_correct_test[label.int()] += c[i].item()
                            class_total[label.int()] += 1
                test_accu_best = sum(class_correct_test) / sum(class_total)
                train_accu_best = test_accu_best
                print('test_accu_best = %.3f%%' % (100 * test_accu_best))
                fileLog.write('test_accu_best = %.3f%%\n' %
                              (100 * test_accu_best))
            else:
                # 微调模型
                train_dataset = data_loader(X_fine[:size, :], y_fine[:size],
                                            transform)
                trainloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=32,
                                                          shuffle=True)

                train_accu_best = 0.0
                test_accu_best = 0.0
                running_loss_initial = 0
                epoch = 0
                while epoch < 20000:
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()  # zero the parameter gradients
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    running_loss = running_loss / size
                    if epoch == 0:
                        running_loss_initial = running_loss
                    # print('[%d] loss: %.3f ,%.3f%%' %
                    #       (epoch + 1, running_loss,
                    #        100*running_loss / running_loss_initial))
                    if epoch % 10 == 0:
                        # 计算训练集准确率
                        class_correct_train = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            for data in trainloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_train[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_train[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        train_accu_cur = sum(class_correct_train) / sum(
                            class_total)
                        # print('CNN_cross_subject_4class：训练集准确率：%d %%' %
                        #       (100 * sum(class_correct_train) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_train[i] / class_total[i]))

                        # 计算测试集准确率
                        class_correct_test = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        with torch.no_grad():
                            y_true_cm = []
                            y_pred_cm = []
                            for data in testloader:
                                images, labels = data
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                c = (predicted == labels).squeeze()
                                y_true_cm = y_true_cm + list(labels)
                                y_pred_cm = y_pred_cm + list(predicted)
                                for i, label in enumerate(labels):
                                    if len(c.shape) == 0:
                                        class_correct_test[
                                            label.int()] += c.item()
                                    else:
                                        class_correct_test[
                                            label.int()] += c[i].item()
                                    class_total[label.int()] += 1
                        test_accu_cur = sum(class_correct_test) / sum(
                            class_total)

                        # print('CNN_cross_subject_4class：测试集准确率：%d %%' %
                        #       (100 * sum(class_correct_test) / sum(class_total)))
                        # for i in range(4):
                        #     print('\t%5s ：%2d %%' %
                        #           (classes[i],
                        #            100 * class_correct_test[i] / class_total[i]))

                        # print('\t\t[%d] loss: %.3f ,%.3f%%' %
                        #       (epoch + 1, running_loss,
                        #        100 * running_loss / running_loss_initial))
                        # print('\t\ttest_accu_best = ', test_accu_best)
                        # print('\t\ttest_accu_cur = ', test_accu_cur)
                        # print('\t\ttrain_accu_cur = ', train_accu_cur)

                        if test_accu_best < test_accu_cur and running_loss / running_loss_initial < 0.9:
                            train_accu_best = train_accu_cur
                            test_accu_best = test_accu_cur
                            y_true_cm_best = y_true_cm
                            y_pred_cm_best = y_pred_cm
                            print('[%d] loss: %.3f ,%.3f%%' %
                                  (epoch + 1, running_loss,
                                   100 * running_loss / running_loss_initial))
                            print('train_accu_best = %.3f%%' %
                                  (100 * train_accu_best))
                            print('test_accu_best = %.3f%%' %
                                  (100 * test_accu_best))
                            fileLog.write(
                                '[%d] loss: %.3f ,%.3f%%\n' %
                                (epoch + 1, running_loss,
                                 100 * running_loss / running_loss_initial))
                            fileLog.write('train_accu_best = %.3f%%\n' %
                                          (100 * train_accu_best))
                            fileLog.write('test_accu_best = %.3f%%\n' %
                                          (100 * test_accu_best))
                    epoch += 1
                    if running_loss / running_loss_initial < 0.2:
                        break
                # end while
                print('NN_cross_subject_LR2class_fine_Net25_multiTask：%s 训练集容量 %d 训练完成' %
                      (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'NN_cross_subject_LR2class_fine_Net25_multiTask：%s 训练集容量 %d 训练完成\n' %
                    (NAME_LIST[subjectNo], size))

            scores_train.append(train_accu_best)
            scores_test.append(test_accu_best)
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_true": np.array([v.item() for v in y_true_cm_best])})
            confusion_matrix_DICT_subjectI_sizeI.update(
                {"y_pred": np.array([v.item() for v in y_pred_cm_best])})
            confusion_matrix_DICT_subjectI.update(
                {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
        # end for
        confusion_matrix_DICT.update(
            {'%s' % NAME_LIST[subjectNo]: confusion_matrix_DICT_subjectI})

        subject_learning_curve_plot = {
            'name': NAME_LIST[subjectNo],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print(
            'NN_cross_subject_LR2class_fine_Net25_multiTask：%s : 测试集准确率  ' %
            NAME_LIST[subjectNo], scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write(
            'NN_cross_subject_LR2class_fine_Net25_multiTask：%s : 测试集准确率  %.3f%%\n' %
            (NAME_LIST[subjectNo], scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\多任务\\左右二分类\\NN_Net25_fine' +
        r'\\learning_curve_NN_Net25_fine_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)
    fileD.close()

    fileD = open(
        path + r'\\..\model\\多任务\\左右二分类\\NN_Net25_fine' +
        r'\\confusion_matrix_NN_Net25_fine_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class_multiTask('NN_Net25_fine')
    fileD = open(
        path + r'\\..\model\\多任务\\左右二分类\\NN_Net25_fine' +
        r'\\learning_curve_NN_Net25_fine_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'NN_Net25_fine_LR2class')
    # end NN_cross_subject_LR2class_fine_Net25_multiTask



# 模型信息
# print_Net_parameters()

# 四分类
# Plot.myLearning_curve_LIST2DICT_4class('LDA_no_cross_subject')

# CNN_cross_subject_4class_pretraining_Net4()
# CNN_cross_subject_4class_pretraining_Net42()
# LSTM_cross_subject_4class_pretraining_Net44()
# NN_cross_subject_4class_pretraining_Net45()
# LSTM_cross_subject_4class_pretraining_Net46()
# LSTM_cross_subject_4class_pretraining_Net47()
# CNN_LSTM_cross_subject_4class_pretraining_Net48()
# CNN_LSTM_cross_subject_4class_pretraining_Net49()

# CNN_cross_subject_4class_fine_Net4()
# CNN_cross_subject_4class_fine_Net42()
# NN_cross_subject_4class_fine_Net45()
# LSTM_cross_subject_4class_fine_Net46()
# LSTM_cross_subject_4class_fine_Net47()
# CNN_LSTM_cross_subject_4class_fine_Net48()
# CNN_LSTM_cross_subject_4class_fine_Net49()

# 上下二分类
# CNN_cross_subject_UD2class_pretraining_Net2()
# CNN_cross_subject_UD2class_pretraining_Net22()
# LSTM_cross_subject_UD2class_pretraining_Net23()
# LSTM_cross_subject_UD2class_pretraining_Net24()
# NN_cross_subject_UD2class_pretraining_Net25()
# LSTM_cross_subject_UD2class_pretraining_Net26()
# LSTM_cross_subject_UD2class_pretraining_Net27()
# CNN_LSTM_cross_subject_UD2class_pretraining_Net28()
# CNN_LSTM_cross_subject_UD2class_pretraining_Net29()

# CNN_cross_subject_UD2class_fine_Net2()
# CNN_cross_subject_UD2class_fine_Net22()
# LSTM_cross_subject_UD2class_fine_Net23()
# LSTM_cross_subject_UD2class_fine_Net24()
# NN_cross_subject_UD2class_fine_Net25()
# LSTM_cross_subject_UD2class_fine_Net26()
# LSTM_cross_subject_UD2class_fine_Net27()
# CNN_LSTM_cross_subject_UD2class_fine_Net28()
# CNN_LSTM_cross_subject_UD2class_fine_Net29()

# 左右二分类
# CNN_cross_subject_LR2class_pretraining_Net2()
# CNN_cross_subject_LR2class_pretraining_Net22()
# LSTM_cross_subject_LR2class_pretraining_Net23()
# NN_cross_subject_LR2class_pretraining_Net25()
# LSTM_cross_subject_LR2class_pretraining_Net26()
# LSTM_cross_subject_LR2class_pretraining_Net27()
# CNN_LSTM_cross_subject_LR2class_pretraining_Net28()
# CNN_LSTM_cross_subject_LR2class_pretraining_Net29()

# CNN_cross_subject_LR2class_fine_Net2()
# CNN_cross_subject_LR2class_fine_Net22()
# LSTM_cross_subject_LR2class_fine_Net23()
# LSTM_cross_subject_LR2class_fine_Net24()
# NN_cross_subject_LR2class_fine_Net25()
# LSTM_cross_subject_LR2class_fine_Net26()
# LSTM_cross_subject_LR2class_fine_Net27()
# CNN_LSTM_cross_subject_LR2class_fine_Net28()
# CNN_LSTM_cross_subject_LR2class_fine_Net29()

# 多任务
# NN_cross_subject_4class_pretraining_Net45_multiTask()
# NN_cross_subject_LR2class_pretraining_Net25_multiTask()
# NN_cross_subject_UD2class_pretraining_Net25_multiTask()

NN_cross_subject_4class_fine_Net45_multiTask()
# NN_cross_subject_UD2class_fine_Net25_multiTask()
# NN_cross_subject_LR2class_fine_Net25_multiTask()