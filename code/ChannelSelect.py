# 通道选择

from ctypes.wintypes import SIZE
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
from sklearn.svm import SVC

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

class Net2504(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2504, self).__init__()
        self.fc1 = nn.Linear(28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
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


class Net2506(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2506, self).__init__()
        self.fc1 = nn.Linear(42, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
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


class Net2508(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2508, self).__init__()
        self.fc1 = nn.Linear(56, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
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



class Net2510(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2510, self).__init__()
        self.fc1 = nn.Linear(70, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
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



class Net2512(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2512, self).__init__()
        self.fc1 = nn.Linear(84, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
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


class Net2514(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2514, self).__init__()
        self.fc1 = nn.Linear(98, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
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


class Net2516(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2516, self).__init__()
        self.fc1 = nn.Linear(112, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
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


class Net2518(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2518, self).__init__()
        self.fc1 = nn.Linear(126, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
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


class Net2520(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2520, self).__init__()
        self.fc1 = nn.Linear(140, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
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

class Net2522(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2522, self).__init__()
        self.fc1 = nn.Linear(154, 1024)
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


class Net2524(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2524, self).__init__()
        self.fc1 = nn.Linear(168, 1024)
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

class Net2526(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2526, self).__init__()
        self.fc1 = nn.Linear(182, 1024)
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

class Net2528(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2528, self).__init__()
        self.fc1 = nn.Linear(196, 1024)
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

class Net2530(nn.Module):
    # 输入数据：
    #   人工特征
    def __init__(self):
        super(Net2530, self).__init__()
        self.fc1 = nn.Linear(210, 1024)
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


# 通道相关性排序_上下二分类-步骤一/二
def NN_cross_subject_UD2class_Train():
    print('NN 上下二分类器_通道选择（上、下）跨被试')

    path = os.getcwd()
    dataSetMulti = np.empty((0,225))

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
        dataSetMulti=np.r_[dataSetMulti,dataSet]
    # end for

    scaler = StandardScaler()
    X_train_scalered = scaler.fit_transform(dataSetMulti[:, :-1])
    dataSet_train_scalered = np.c_[X_train_scalered,
                                    dataSetMulti[:, -1]]  #特征缩放后的样本集(X,y)
    dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

    train_set, test_set = train_test_split(
        dataSet_train_scalered, test_size=0.1)  #划分训练集、测试集(默认洗牌)
    X_train = train_set[:, :-1]  #微调训练集特征矩阵
    y_train= train_set[:, -1]  #微调训练集标签数组
    X_test = test_set[:, :-1]  #测试集特征矩阵
    y_test = test_set[:, -1]  #测试集标签数组

    # 训练模型
    transform = None
    train_dataset = data_loader(X_train,
                                y_train, transform)
    test_dataset = data_loader(X_test,
                                y_test, transform)

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
        running_loss = running_loss / X_train.shape[0]
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
                torch.save(
                    net,
                    os.getcwd() +
                    r'\\..\\model\\通道选择\\' +
                    'ChannelSelect_UD2class.pkl')

                print('[%d] loss: %.3f ,%.3f%%' %
                        (epoch + 1, running_loss,
                        100 * running_loss / running_loss_initial))
                print('train_accu_best = %.3f%%' %
                        (100 * train_accu_best))
                print('test_accu_best = %.3f%%' %
                        (100 * test_accu_best))
        epoch += 1
        if running_loss / running_loss_initial < 0.2:
            break
    # end while
    # end NN_cross_subject_UD2class_Train


# 通道相关性排序_左右二分类-步骤一/二
def NN_cross_subject_LR2class_Train():
    print('NN 左右二分类器_通道选择（左、右）跨被试')

    path = os.getcwd()
    dataSetMulti = np.empty((0,225))

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
        dataSetMulti=np.r_[dataSetMulti,dataSet]
    # end for

    scaler = StandardScaler()
    X_train_scalered = scaler.fit_transform(dataSetMulti[:, :-1])
    dataSet_train_scalered = np.c_[X_train_scalered,
                                    dataSetMulti[:, -1]]  #特征缩放后的样本集(X,y)
    dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

    train_set, test_set = train_test_split(
        dataSet_train_scalered, test_size=0.1)  #划分训练集、测试集(默认洗牌)
    X_train = train_set[:, :-1]  #微调训练集特征矩阵
    y_train= train_set[:, -1]  #微调训练集标签数组
    X_test = test_set[:, :-1]  #测试集特征矩阵
    y_test = test_set[:, -1]  #测试集标签数组

    # 训练模型
    transform = None
    train_dataset = data_loader(X_train,
                                y_train, transform)
    test_dataset = data_loader(X_test,
                                y_test, transform)

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
        running_loss = running_loss / X_train.shape[0]
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
                torch.save(
                    net,
                    os.getcwd() +
                    r'\\..\\model\\通道选择\\' +
                    'ChannelSelect_LR2class.pkl')

                print('[%d] loss: %.3f ,%.3f%%' %
                        (epoch + 1, running_loss,
                        100 * running_loss / running_loss_initial))
                print('train_accu_best = %.3f%%' %
                        (100 * train_accu_best))
                print('test_accu_best = %.3f%%' %
                        (100 * test_accu_best))
        epoch += 1
        if running_loss / running_loss_initial < 0.2:
            break
    # end while
    # end NN_cross_subject_LR2class_Train


# 通道相关性排序_上下二分类-步骤二/二
def NN_cross_subject_UD2class_ChannelSelect():
    print('NN 上下二分类器_通道选择（上、下）跨被试')
    CHANNELNO=[14, 12, 10, 8, 7, 9, 11, 13, 15, 20, 16, 19, 32, 30, 28, 26, 25, 27, 29, 31, 33, 37, 36, 47, 45, 43, 42, 44, 46, 48, 49, 56]
    CHANNEL_CNT_SELECT=16
    path = os.getcwd()
    dataSetMulti = np.empty((0,225))

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
        dataSetMulti=np.r_[dataSetMulti,dataSet]
    # end for

    scaler = StandardScaler()
    X_train_scalered = scaler.fit_transform(dataSetMulti[:, :-1])
    dataSet_train_scalered = np.c_[X_train_scalered,
                                    dataSetMulti[:, -1]]  #特征缩放后的样本集(X,y)
    dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

    # 训练模型
    transform = None
    test_dataset = data_loader(dataSet_train_scalered[:,:-1],
                                dataSet_train_scalered[:,-1], transform)
    # 使用训练后的模型计算关联矩阵
    path = os.getcwd() +r'\\..\\model\\通道选择\\' +'ChannelSelect_UD2class.pkl'
    net = torch.load(path)
    net = net.double()

    G_sum=np.zeros(dataSet_train_scalered[:,:-1].shape[1])
    countArray=np.zeros(64)
    for p in range(30):
        I_std=dataSet_train_scalered[:,:-1].std(axis=0)
        G=np.zeros(I_std.shape[0])
        for i in range(I_std.shape[0]):
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=64,
                                                shuffle=False)
            O1=np.zeros((0,2))
            for data in testloader:
                images, labels = data
                images[:,i]=images[:,i]+0.1*I_std[i]
                outputs = net(images)
                O1=np.r_[O1,outputs.data]

            testloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=64,
                                                shuffle=False)
            O2=np.zeros((0,2))
            for data in testloader:
                images, labels = data
                images[:,i]=images[:,i]-0.1*I_std[i]
                outputs = net(images)
                O2=np.r_[O2,outputs.data]
            
            G[i]=abs((O1-O2)/(2*0.1*I_std[i])).sum()
        G_sum=G_sum+G
        sortIndex=np.argsort(G)[::-1]//7
        # print('通道排序',[CHANNELNO[index] for index in sortIndex])
        channel_selected=[]
        for index in sortIndex:
            if CHANNELNO[index] not in channel_selected:
                channel_selected.append(CHANNELNO[index])
            if len(channel_selected)==CHANNEL_CNT_SELECT:
                break
        print('前 %d 个通道选取完毕'%CHANNEL_CNT_SELECT,channel_selected)
    
    G_channel=[]
    sum_i=0
    for i in range(len(G_sum)):
        sum_i+=G_sum[i]
        if(i%7==6):
            G_channel.append(sum_i)
            sum_i=0
    sortIndex=np.argsort(G_channel)[::-1]
    print('通道排序_channel',[CHANNELNO[index] for index in sortIndex])
    G_channel=np.array(G_channel)
    print((G_channel-G_channel.min())/(G_channel.max()-G_channel.min()))
    # end NN_cross_subject_UD2class_ChannelSelect



# 通道相关性排序_左右二分类-步骤二/二
def NN_cross_subject_LR2class_ChannelSelect():
    print('NN 左右二分类器_通道选择（左、右）跨被试')
    CHANNELNO=[14, 12, 10, 8, 7, 9, 11, 13, 15, 20, 16, 19, 32, 30, 28, 26, 25, 27, 29, 31, 33, 37, 36, 47, 45, 43, 42, 44, 46, 48, 49, 56]
    CHANNEL_CNT_SELECT=16
    path = os.getcwd()
    dataSetMulti = np.empty((0,225))

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
        dataSetMulti=np.r_[dataSetMulti,dataSet]
    # end for

    scaler = StandardScaler()
    X_train_scalered = scaler.fit_transform(dataSetMulti[:, :-1])
    dataSet_train_scalered = np.c_[X_train_scalered,
                                    dataSetMulti[:, -1]]  #特征缩放后的样本集(X,y)
    dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

    # 训练模型
    transform = None
    test_dataset = data_loader(dataSet_train_scalered[:,:-1],
                                dataSet_train_scalered[:,-1], transform)
    # 使用训练后的模型计算关联矩阵
    path = os.getcwd() +r'\\..\\model\\通道选择\\' +'ChannelSelect_LR2class.pkl'
    net = torch.load(path)
    net = net.double()

    G_sum=np.zeros(dataSet_train_scalered[:,:-1].shape[1])
    countArray=np.zeros(64)
    for p in range(30):
        I_std=dataSet_train_scalered[:,:-1].std(axis=0)
        G=np.zeros(I_std.shape[0])
        for i in range(I_std.shape[0]):
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=64,
                                                shuffle=False)
            O1=np.zeros((0,2))
            for data in testloader:
                images, labels = data
                images[:,i]=images[:,i]+0.1*I_std[i]
                outputs = net(images)
                O1=np.r_[O1,outputs.data]

            testloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=64,
                                                shuffle=False)
            O2=np.zeros((0,2))
            for data in testloader:
                images, labels = data
                images[:,i]=images[:,i]-0.1*I_std[i]
                outputs = net(images)
                O2=np.r_[O2,outputs.data]
            
            G[i]=abs((O1-O2)/(2*0.1*I_std[i])).sum()
        G_sum=G_sum+G
        sortIndex=np.argsort(G)[::-1]//7
        # print('通道排序',[CHANNELNO[index] for index in sortIndex])
        channel_selected=[]
        for index in sortIndex:
            if CHANNELNO[index] not in channel_selected:
                channel_selected.append(CHANNELNO[index])
            if len(channel_selected)==CHANNEL_CNT_SELECT:
                break
        print('前 %d 个通道选取完毕'%CHANNEL_CNT_SELECT,channel_selected)
    
    G_channel=[]
    sum_i=0
    for i in range(len(G_sum)):
        sum_i+=G_sum[i]
        if(i%7==6):
            G_channel.append(sum_i)
            sum_i=0
    sortIndex=np.argsort(G_channel)[::-1]
    print('通道排序_channel',[CHANNELNO[index] for index in sortIndex])
    G_channel=np.array(G_channel)
    print((G_channel-G_channel.min())/(G_channel.max()-G_channel.min()))
    # end NN_cross_subject_LR2class_ChannelSelect


# SVM 分类器非跨被试：上下二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_beginTimeFile_UD2class_ChannelSelect() 得到的数据
def SVM_no_cross_subject_UD2class_ChannelSelect():
    CHANNEL_SIZES = np.linspace(4, 32, 15).astype(int)

    for channel_size in CHANNEL_SIZES:
        print('SVM_no_cross_subject_UD2class_ChannelSelect：timeBeginStart = ',channel_size)
        fileLog = open(
            os.getcwd() + r'\\..\model\\通道选择\\上下二分类\\ML\\SVM' +'\\%d'%channel_size+
            r'\\log_SVM_no_cross_subject_UD2class.txt', 'w')
        fileLog.write('SVM 上下二分类器（上、下）非跨被试\n')
        print('SVM 上下二分类器（上、下）非跨被试')
        NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
        TRAIN_SIZES = np.linspace(0.1, 1, 25)
        path = os.getcwd()
        subject_learning_curve_plot_LIST = []
        confusion_matrix_DICT = {}
        for name in NAME_LIST:
            confusion_matrix_DICT_subjectI = {}
            #提取数据
            fileName = 'single_move_ChannelSelect_UD2class_%d.pickle' % channel_size
            fileD = open(
                path + r'\\..\\data\\' + name + '\\四分类\\' +
                fileName, 'rb')
            X, y = pickle.load(fileD)
            fileD.close()

            # 提取二分类数据
            X=X[:100]
            y=y[:100]    

            # 构造样本集
            dataSet = np.c_[X, y]
            print('SVM_no_cross_subject_UD2class_ChannelSelect：样本数量', sum(y == 0), sum(y == 1),
                sum(y == 2), sum(y == 3))

            # 样本集洗牌
            shuffle_index = np.random.permutation(dataSet.shape[0])
            dataSet = dataSet[shuffle_index]

            # 特征缩放
            scaler = StandardScaler()
            X_scalered = scaler.fit_transform(dataSet[:, :-1])
            dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)
            dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征

            #拆分训练集、测试集
            train_set, test_set = train_test_split(dataSet_scalered,
                                                test_size=0.1)  #划分训练集、测试集
            X_train = train_set[:, :-1]  #训练集特征矩阵
            y_train = train_set[:, -1]  #训练集标签数组
            X_test = test_set[:, :-1]  #测试集特征矩阵
            y_test = test_set[:, -1]  #测试集标签数组
            t_sizes = X_train.shape[0] * TRAIN_SIZES
            t_sizes = t_sizes.astype(int)

            # t_sizes = [513]  ########################
            scores_train = []
            scores_test = []
            for size in t_sizes:
                confusion_matrix_DICT_subjectI_sizeI = {}
                print('+++++++++ size = %d' % (size))
                fileLog.write('+++++++++ size = %d\n' % (size))
                # 训练模型
                clf = SVC()
                clf.fit(X_train[:size, :], y_train[:size])
                x_pred = clf.predict(X_train)
                acc_train = sum(x_pred == y_train) / len(y_train)
                print('SVM_no_cross_subject_UD2class_ChannelSelect：%s : 训练集准确率  %.3f%%\n' %
                    (name, acc_train))
                fileLog.write('SVM_no_cross_subject_UD2class_ChannelSelect：%s : 训练集准确率  %.3f%%\n' %
                            (name, acc_train))
                y_pred = clf.predict(X_test)
                acc_test = sum(y_test == y_pred) / len(y_pred)
                print('SVM_no_cross_subject_UD2class_ChannelSelect：%s : 测试集准确率  %.3f%%\n' %
                    (name, acc_test))
                fileLog.write('SVM_no_cross_subject_UD2class_ChannelSelect：%s : 测试集准确率  %.3f%%\n' %
                            (name, acc_test))
                scores_train.append(acc_train)
                scores_test.append(acc_test)
                confusion_matrix_DICT_subjectI_sizeI.update({"y_true": y_test})
                confusion_matrix_DICT_subjectI_sizeI.update({"y_pred": y_pred})
                confusion_matrix_DICT_subjectI.update(
                    {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
            # end for
            confusion_matrix_DICT.update(
                {'%s' % name: confusion_matrix_DICT_subjectI})

            subject_learning_curve_plot = {
                'name': name,
                'train_sizes': t_sizes,
                'train_scores': np.array(scores_train),
                'test_scores': np.array(scores_test),
                'fit_times': []
            }
            print('SVM_no_cross_subject_UD2class_ChannelSelect：%s : 测试集准确率  ' % name,
                scores_test[-1])
            subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
            fileLog.write('SVM_no_cross_subject_UD2class_ChannelSelect：%s : 测试集准确率  %.3f%%\n' %
                        (name, scores_test[-1]))
        # end for
        fileLog.close()

        learning_curve_DICT = {}
        for learning_curve in subject_learning_curve_plot_LIST:
            learning_curve_DICT.update({learning_curve['name']: learning_curve})
        # end for
        learning_curve_DICT.update({'ignoreList': []})
        path = os.getcwd()
        fileD = open(
            path + r'\\..\model\\通道选择\\上下二分类\\ML\\SVM' +'\\%d'%channel_size+
            r'\\learning_curve_UD2class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)

        fileD = open(
            path + r'\\..\model\\通道选择\\上下二分类\\ML\\SVM' +'\\%d'%channel_size+
            r'\\confusion_matrix_UD2class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'learning_curve_UD2classes_intraSubject_SVM_%d'%channel_size)
    # end for
    # end SVM_no_cross_subject_UD2class_ChannelSelect


# SVM 分类器非跨被试：左右二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_beginTimeFile_LR2class_ChannelSelect() 得到的数据
def SVM_no_cross_subject_LR2class_ChannelSelect():
    CHANNEL_SIZES = np.linspace(4, 32, 15).astype(int)

    for channel_size in CHANNEL_SIZES:
        print('SVM_no_cross_subject_LR2class_ChannelSelect：timeBeginStart = ',channel_size)
        fileLog = open(
            os.getcwd() + r'\\..\model\\通道选择\\左右二分类\\ML\\SVM' +'\\%d'%channel_size+
            r'\\log_SVM_no_cross_subject_LR2class.txt', 'w')
        fileLog.write('SVM 左右二分类器（左、右）非跨被试\n')
        print('SVM 左右二分类器（左、右）非跨被试')
        NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
        TRAIN_SIZES = np.linspace(0.1, 1, 25)
        path = os.getcwd()
        subject_learning_curve_plot_LIST = []
        confusion_matrix_DICT = {}
        for name in NAME_LIST:
            confusion_matrix_DICT_subjectI = {}
            #提取数据
            fileName = 'single_move_ChannelSelect_LR2class_%d.pickle' % channel_size
            fileD = open(
                path + r'\\..\\data\\' + name + '\\四分类\\' +
                fileName, 'rb')
            X, y = pickle.load(fileD)
            fileD.close()

            # 提取二分类数据
            X = X[100:]
            y = y[100:]
            y[y == 2] = 0
            y[y == 3] = 1
            # 构造样本集
            dataSet = np.c_[X, y]
            print('SVM_no_cross_subject_LR2class_ChannelSelect：样本数量', sum(y == 0), sum(y == 1),
                sum(y == 2), sum(y == 3))

            # 样本集洗牌
            shuffle_index = np.random.permutation(dataSet.shape[0])
            dataSet = dataSet[shuffle_index]

            # 特征缩放
            scaler = StandardScaler()
            X_scalered = scaler.fit_transform(dataSet[:, :-1])
            dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)
            dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征

            #拆分训练集、测试集
            train_set, test_set = train_test_split(dataSet_scalered,
                                                test_size=0.1)  #划分训练集、测试集
            X_train = train_set[:, :-1]  #训练集特征矩阵
            y_train = train_set[:, -1]  #训练集标签数组
            X_test = test_set[:, :-1]  #测试集特征矩阵
            y_test = test_set[:, -1]  #测试集标签数组
            t_sizes = X_train.shape[0] * TRAIN_SIZES
            t_sizes = t_sizes.astype(int)

            # t_sizes = [513]  ########################
            scores_train = []
            scores_test = []
            for size in t_sizes:
                confusion_matrix_DICT_subjectI_sizeI = {}
                print('+++++++++ size = %d' % (size))
                fileLog.write('+++++++++ size = %d\n' % (size))
                # 训练模型
                clf = SVC()
                clf.fit(X_train[:size, :], y_train[:size])
                x_pred = clf.predict(X_train)
                acc_train = sum(x_pred == y_train) / len(y_train)
                print('SVM_no_cross_subject_LR2class_ChannelSelect：%s : 训练集准确率  %.3f%%\n' %
                    (name, acc_train))
                fileLog.write('SVM_no_cross_subject_LR2class_ChannelSelect：%s : 训练集准确率  %.3f%%\n' %
                            (name, acc_train))
                y_pred = clf.predict(X_test)
                acc_test = sum(y_test == y_pred) / len(y_pred)
                print('SVM_no_cross_subject_LR2class_ChannelSelect：%s : 测试集准确率  %.3f%%\n' %
                    (name, acc_test))
                fileLog.write('SVM_no_cross_subject_LR2class_ChannelSelect：%s : 测试集准确率  %.3f%%\n' %
                            (name, acc_test))
                scores_train.append(acc_train)
                scores_test.append(acc_test)
                confusion_matrix_DICT_subjectI_sizeI.update({"y_true": y_test})
                confusion_matrix_DICT_subjectI_sizeI.update({"y_pred": y_pred})
                confusion_matrix_DICT_subjectI.update(
                    {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
            # end for
            confusion_matrix_DICT.update(
                {'%s' % name: confusion_matrix_DICT_subjectI})

            subject_learning_curve_plot = {
                'name': name,
                'train_sizes': t_sizes,
                'train_scores': np.array(scores_train),
                'test_scores': np.array(scores_test),
                'fit_times': []
            }
            print('SVM_no_cross_subject_LR2class_ChannelSelect：%s : 测试集准确率  ' % name,
                scores_test[-1])
            subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
            fileLog.write('SVM_no_cross_subject_LR2class_ChannelSelect：%s : 测试集准确率  %.3f%%\n' %
                        (name, scores_test[-1]))
        # end for
        fileLog.close()

        learning_curve_DICT = {}
        for learning_curve in subject_learning_curve_plot_LIST:
            learning_curve_DICT.update({learning_curve['name']: learning_curve})
        # end for
        learning_curve_DICT.update({'ignoreList': []})
        path = os.getcwd()
        fileD = open(
            path + r'\\..\model\\通道选择\\左右二分类\\ML\\SVM' +'\\%d'%channel_size+
            r'\\learning_curve_LR2class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)

        fileD = open(
            path + r'\\..\model\\通道选择\\左右二分类\\ML\\SVM' +'\\%d'%channel_size+
            r'\\confusion_matrix_LR2class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'learning_curve_LR2classes_intraSubject_SVM_%d'%channel_size)
    # end for
    # end SVM_no_cross_subject_LR2class_ChannelSelect


# SVM 分类器非跨被试：上下二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_beginTimeFile_UD2class_ChannelSelect() 得到的数据
def SVM_no_cross_subject_4class_ChannelSelect():
    CHANNEL_SIZES = np.linspace(4, 32, 15).astype(int)

    for channel_size in CHANNEL_SIZES:
        print('SVM_no_cross_subject_4class_ChannelSelect：timeBeginStart = ',channel_size)
        fileLog = open(
            os.getcwd() + r'\\..\model\\通道选择\\四分类\\ML\\SVM' +'\\%d'%channel_size+
            r'\\log_SVM_no_cross_subject_4class.txt', 'w')
        fileLog.write('SVM 四分类器（上、下、左、右）非跨被试\n')
        print('SVM 四分类器（上、下、左、右）非跨被试')
        NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
        TRAIN_SIZES = np.linspace(0.1, 1, 25)
        path = os.getcwd()
        subject_learning_curve_plot_LIST = []
        confusion_matrix_DICT = {}
        for name in NAME_LIST:
            confusion_matrix_DICT_subjectI = {}
            #提取数据
            fileName = 'single_move_ChannelSelect_UD2class_%d.pickle' % channel_size
            fileD = open(
                path + r'\\..\\data\\' + name + '\\四分类\\' +
                fileName, 'rb')
            X, y = pickle.load(fileD)
            fileD.close()
            # 构造样本集
            dataSet = np.c_[X, y]
            print('SVM_no_cross_subject_4class_ChannelSelect：样本数量', sum(y == 0), sum(y == 1),
                sum(y == 2), sum(y == 3))

            # 样本集洗牌
            shuffle_index = np.random.permutation(dataSet.shape[0])
            dataSet = dataSet[shuffle_index]

            # 特征缩放
            scaler = StandardScaler()
            X_scalered = scaler.fit_transform(dataSet[:, :-1])
            dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)
            dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征

            #拆分训练集、测试集
            train_set, test_set = train_test_split(dataSet_scalered,
                                                test_size=0.1)  #划分训练集、测试集
            X_train = train_set[:, :-1]  #训练集特征矩阵
            y_train = train_set[:, -1]  #训练集标签数组
            X_test = test_set[:, :-1]  #测试集特征矩阵
            y_test = test_set[:, -1]  #测试集标签数组
            t_sizes = X_train.shape[0] * TRAIN_SIZES
            t_sizes = t_sizes.astype(int)

            # t_sizes = [513]  ########################
            scores_train = []
            scores_test = []
            for size in t_sizes:
                confusion_matrix_DICT_subjectI_sizeI = {}
                print('+++++++++ size = %d' % (size))
                fileLog.write('+++++++++ size = %d\n' % (size))
                # 训练模型
                clf = SVC()
                clf.fit(X_train[:size, :], y_train[:size])
                x_pred = clf.predict(X_train)
                acc_train = sum(x_pred == y_train) / len(y_train)
                print('SVM_no_cross_subject_4class_ChannelSelect：%s : 训练集准确率  %.3f%%\n' %
                    (name, acc_train))
                fileLog.write('SVM_no_cross_subject_4class_ChannelSelect：%s : 训练集准确率  %.3f%%\n' %
                            (name, acc_train))
                y_pred = clf.predict(X_test)
                acc_test = sum(y_test == y_pred) / len(y_pred)
                print('SVM_no_cross_subject_4class_ChannelSelect：%s : 测试集准确率  %.3f%%\n' %
                    (name, acc_test))
                fileLog.write('SVM_no_cross_subject_4class_ChannelSelect：%s : 测试集准确率  %.3f%%\n' %
                            (name, acc_test))
                scores_train.append(acc_train)
                scores_test.append(acc_test)
                confusion_matrix_DICT_subjectI_sizeI.update({"y_true": y_test})
                confusion_matrix_DICT_subjectI_sizeI.update({"y_pred": y_pred})
                confusion_matrix_DICT_subjectI.update(
                    {'%s' % size: confusion_matrix_DICT_subjectI_sizeI})
            # end for
            confusion_matrix_DICT.update(
                {'%s' % name: confusion_matrix_DICT_subjectI})

            subject_learning_curve_plot = {
                'name': name,
                'train_sizes': t_sizes,
                'train_scores': np.array(scores_train),
                'test_scores': np.array(scores_test),
                'fit_times': []
            }
            print('SVM_no_cross_subject_4class_ChannelSelect：%s : 测试集准确率  ' % name,
                scores_test[-1])
            subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
            fileLog.write('SVM_no_cross_subject_4class_ChannelSelect：%s : 测试集准确率  %.3f%%\n' %
                        (name, scores_test[-1]))
        # end for
        fileLog.close()

        learning_curve_DICT = {}
        for learning_curve in subject_learning_curve_plot_LIST:
            learning_curve_DICT.update({learning_curve['name']: learning_curve})
        # end for
        learning_curve_DICT.update({'ignoreList': []})
        path = os.getcwd()
        fileD = open(
            path + r'\\..\model\\通道选择\\四分类\\ML\\SVM' +'\\%d'%channel_size+
            r'\\learning_curve_4class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)

        fileD = open(
            path + r'\\..\model\\通道选择\\四分类\\ML\\SVM' +'\\%d'%channel_size+
            r'\\confusion_matrix_4class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'learning_curve_4classes_intraSubject_SVM_%d'%channel_size)
    # end for
    # end SVM_no_cross_subject_4class_ChannelSelect


# 样本集_X(numpy 二维数组(各样本 * 各特征))，样本集_Y(numpy 一维数组)
def NN_cross_subject_UD2class_pretraining_Net25_ChannelSelect():
    CHANNEL_SIZES = np.linspace(4, 32, 15).astype(int)
    netDict={4:Net2504(),  6:Net2506(),  8:Net2508(), 10:Net2510(), 12:Net2512(), 14:Net2514(), 16:Net2516(), 18:Net2518(), 20:Net2520(), 22:Net2522(), 24:Net2524(), 26:Net2526(), 28:Net2528(), 30:Net2530(),32:Net25()}
    for channel_size in CHANNEL_SIZES:
        print('NN_cross_subject_UD2class_pretraining_Net25_ChannelSelect：channel_size = ',channel_size)
        fileLog = open(
            os.getcwd() + r'\\..\model_all\\通道选择\\上下二分类\\NN_Net25_pretraining' +'\\%d'%channel_size+
            r'\\learning_curve_NN_Net25_pretraining_UD2class.txt', 'w')
    
   
        fileLog.write('NN 上下二分类器_预训练（上、下）跨被试\n')
        print('NN 上下二分类器_预训练（上、下）跨被试')
        TRAIN_SIZES = np.linspace(0.3, 0.5, 8)
        path = os.getcwd()
        dataSetMulti = []

        # 提取所有被试数据
        for name in NAME_LIST:
            # 遍历各被试
            fileName = 'single_move_ChannelSelect_UD2class_%d.pickle' % channel_size
            fileD = open(
                path + r'\\..\\data\\' + name + '\\四分类\\' +
                fileName, 'rb')
            X, y = pickle.load(fileD)
            fileD.close()

            # 提取二分类数据
            X = X[:100]
            y = y[:100]

            # 构造样本集
            dataSet = np.c_[X, y]
            print('NN_cross_subject_UD2class_pretraining_Net25_ChannelSelect：样本数量', sum(y == 0),
                sum(y == 1), sum(y == 2), sum(y == 3))

            dataSetMulti.append(dataSet)
        # end for

        # 留一个被试作为跨被试测试
        net_tmp = netDict[channel_size]
        for x in net_tmp.parameters():
            print(x.data.shape, len(x.data.reshape(-1)))

        print("NN_cross_subject_UD2class_pretraining_Net25_ChannelSelect：参数个数 {}  ".format(
            sum(x.numel() for x in net_tmp.parameters())))
        fileLog.write(
            "NN_cross_subject_UD2class_pretraining_Net25_ChannelSelect：参数个数 {}  \n".format(
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
                net = netDict[channel_size]
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
                                r'\\..\\model_all\\通道选择\\上下二分类\\NN_Net25_pretraining' +'\\%d'%channel_size+
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
                    'NN_cross_subject_UD2class_pretraining_Net25_ChannelSelect：%s 训练集容量 %d 训练完成'
                    % (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'NN_cross_subject_UD2class_pretraining_Net25_ChannelSelect：%s 训练集容量 %d 训练完成\n'
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
                'NN_cross_subject_UD2class_pretraining_Net25_ChannelSelect：%s : 测试集准确率  ' %
                NAME_LIST[subjectNo], scores_test[-1])
            subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        # end for
        fileLog.close()

        learning_curve_DICT = {}
        for learning_curve in subject_learning_curve_plot_LIST:
            learning_curve_DICT.update({learning_curve['name']: learning_curve})
        # end for
        learning_curve_DICT.update({'ignoreList': []})
        path = os.getcwd()
        fileD = open(
            path + r'\\..\model_all\\通道选择\\上下二分类\\NN_Net25_pretraining' +'\\%d'%channel_size+
            r'\\learning_curve_UD2class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)
        fileD.close()

        fileD = open(
            path + r'\\..\model_all\\通道选择\\上下二分类\\NN_Net25_pretraining' +'\\%d'%channel_size+
            r'\\confusion_matrix_UD2class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'NN_Net25_pretraining_UD2class_%d'%channel_size)
    # end for
    # end NN_cross_subject_UD2class_pretraining_Net25_ChannelSelect


# NN_cross_subject_UD2class_Train()
# NN_cross_subject_UD2class_ChannelSelect()

# NN_cross_subject_LR2class_Train()
# NN_cross_subject_LR2class_ChannelSelect()

# SVM_no_cross_subject_UD2class_ChannelSelect()
# SVM_no_cross_subject_LR2class_ChannelSelect()
SVM_no_cross_subject_4class_ChannelSelect()

# NN_cross_subject_UD2class_pretraining_Net25_ChannelSelect() 