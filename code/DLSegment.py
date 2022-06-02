# 深度学习算法-时间段对比

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



# 样本集_X(numpy 二维数组(各样本 * 各特征))，样本集_Y(numpy 一维数组)
def NN_cross_subject_UD2class_pretraining_Net25_multiSegment():
    SEG_START = np.linspace(0, 1280, 9).astype(
        int)  # 0,160,320,480,640,800,960,1120,1280

    for timeBeginStart in SEG_START:
        print('NN_no_cross_subject_UD2class_multiSegment：timeBeginStart = ',timeBeginStart)
        fileLog = open(
            os.getcwd() + r'\\..\model_all\\上下二分类\\NN_Net25_pretraining' +'\\%d'%timeBeginStart+
            r'\\learning_curve_NN_Net25_pretraining_UD2class.txt', 'w')
    
   
        fileLog.write('NN 上下二分类器_预训练（上、下）跨被试\n')
        print('NN 上下二分类器_预训练（上、下）跨被试')
        TRAIN_SIZES = np.linspace(0.3, 0.5, 8)
        path = os.getcwd()
        dataSetMulti = []

        # 提取所有被试数据
        for name in NAME_LIST:
            # 遍历各被试
            fileName = 'single_move_motion_start-%d.pickle' % timeBeginStart
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
            print('NN_cross_subject_UD2class_pretraining_Net25_multiSegment：样本数量', sum(y == 0),
                sum(y == 1), sum(y == 2), sum(y == 3))

            dataSetMulti.append(dataSet)
        # end for

        # 留一个被试作为跨被试测试
        net_tmp = Net25()
        for x in net_tmp.parameters():
            print(x.data.shape, len(x.data.reshape(-1)))

        print("NN_cross_subject_UD2class_pretraining_Net25_multiSegment：参数个数 {}  ".format(
            sum(x.numel() for x in net_tmp.parameters())))
        fileLog.write(
            "NN_cross_subject_UD2class_pretraining_Net25_multiSegment：参数个数 {}  \n".format(
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
                                r'\\..\\model_all\\上下二分类\\NN_Net25_pretraining' +'\\%d'%timeBeginStart+
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
                    'NN_cross_subject_UD2class_pretraining_Net25_multiSegment：%s 训练集容量 %d 训练完成'
                    % (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'NN_cross_subject_UD2class_pretraining_Net25_multiSegment：%s 训练集容量 %d 训练完成\n'
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
                'NN_cross_subject_UD2class_pretraining_Net25_multiSegment：%s : 测试集准确率  ' %
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
            path + r'\\..\model_all\\上下二分类\\NN_Net25_pretraining' +'\\%d'%timeBeginStart+
            r'\\learning_curve_UD2class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)
        fileD.close()

        fileD = open(
            path + r'\\..\model_all\\上下二分类\\NN_Net25_pretraining' +'\\%d'%timeBeginStart+
            r'\\confusion_matrix_UD2class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'NN_Net25_pretraining_UD2class_%d'%timeBeginStart)
    # end for
    # end NN_cross_subject_UD2class_pretraining_Net25_multiSegment



def NN_cross_subject_UD2class_fine_Net25_multiSegment():
    SEG_START = np.linspace(0, 1280, 9).astype(
        int)  # 0,160,320,480,640,800,960,1120,1280   
    sizeDict={0:350,160:350,320:330,480:310,640:350,800:350,960:330,1120:330,1280:350}
    
    for timeBeginStart in SEG_START:
        print('NN_cross_subject_UD2class_fine_Net25_multiSegment：timeBeginStart = ',timeBeginStart)

        fileLog = open(
            os.getcwd() + r'\\..\model\\上下二分类\\NN_Net25_fine'  +'\\%d'%timeBeginStart+
            r'\\learning_curve_NN_Net25_fine_LR2class.txt', 'w')
        fileLog.write('NN 上下二分类器_微调（上、下）跨被试\n')
        print('NN 上下二分类器_微调（上、下）跨被试')

        FINE_SIZES = np.linspace(0, 1.0, 5)#25
        path = os.getcwd()
        dataSetMulti = []

        # 提取所有被试数据
        for name in NAME_LIST:
            # 遍历各被试
            fileName = 'single_move_motion_start-%d.pickle' % timeBeginStart
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
            print('NN_cross_subject_UD2class_fine_Net25_multiSegment：样本数量', sum(y == 0),
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
            ) + r'\\..\\model\\上下二分类\\NN_Net25_pretraining' +'\\%d'%timeBeginStart+ '\\%s_%d_Net25.pkl' % (
                NAME_LIST[subjectNo],sizeDict[timeBeginStart])
            print(path)
            dataSet = dataSetMulti[subjectNo]  # 取出留下的被试样本集

            # 样本集特征缩放
            scaler = StandardScaler()
            X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
            dataSet_train_scalered = np.c_[X_train_scalered,
                                        dataSet[:, -1]]  #特征缩放后的样本集(X,y)
            dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

            print('NN_cross_subject_UD2class_fine_Net25_multiSegment：样本数量',
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
                    print('NN_cross_subject_UD2class_fine_Net25_multiSegment：%s 训练集容量 %d 训练完成' %
                        (NAME_LIST[subjectNo], size))
                    fileLog.write(
                        'NN_cross_subject_UD2class_fine_Net25_multiSegment：%s 训练集容量 %d 训练完成\n' %
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
                'NN_cross_subject_UD2class_fine_Net25_multiSegment：%s : 测试集准确率  ' %
                NAME_LIST[subjectNo], scores_test[-1])
            subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
            fileLog.write(
                'NN_cross_subject_UD2class_fine_Net25_multiSegment：%s : 测试集准确率  %.3f%%\n' %
                (NAME_LIST[subjectNo], scores_test[-1]))
        # end for
        fileLog.close()

        learning_curve_DICT = {}
        for learning_curve in subject_learning_curve_plot_LIST:
            learning_curve_DICT.update({learning_curve['name']: learning_curve})
        # end for
        learning_curve_DICT.update({'ignoreList': []})
 
        path = os.getcwd()
        fileD = open(
            path + r'\\..\model\\上下二分类\\NN_Net25_fine' +'\\%d'%timeBeginStart+
            r'\\learning_curve_UD2class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)
        fileD.close()

        fileD = open(
            path + r'\\..\model\\上下二分类\\NN_Net25_fine' +'\\%d'%timeBeginStart+
            r'\\confusion_matrix_UD2class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'NN_Net25_fine_UD2class_%d'%timeBeginStart)
    # end for
    # end NN_cross_subject_UD2class_fine_Net25_multiSegment





# 样本集_X(numpy 二维数组(各样本 * 各特征))，样本集_Y(numpy 一维数组)
def NN_cross_subject_LR2class_pretraining_Net25_multiSegment():
    SEG_START = np.linspace(0, 1280, 9).astype(
        int)  # 0,160,320,480,640,800,960,1120,1280

    for timeBeginStart in SEG_START:
        print('NN_no_cross_subject_LR2class_multiSegment：timeBeginStart = ',timeBeginStart)
        fileLog = open(
            os.getcwd() + r'\\..\model_all\\左右二分类\\NN_Net25_pretraining' +'\\%d'%timeBeginStart+
            r'\\learning_curve_NN_Net25_pretraining_LR2class.txt', 'w')
    
   
        fileLog.write('NN 左右二分类器_预训练（左、右）跨被试\n')
        print('NN 左右二分类器_预训练（左、右）跨被试')
        TRAIN_SIZES = np.linspace(0.3, 0.5, 8)
        path = os.getcwd()
        dataSetMulti = []

        # 提取所有被试数据
        for name in NAME_LIST:
            # 遍历各被试
            fileName = 'single_move_motion_start-%d.pickle' % timeBeginStart
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
            print('NN_cross_subject_LR2class_pretraining_Net25_multiSegment：样本数量', sum(y == 0),
                sum(y == 1), sum(y == 2), sum(y == 3))

            dataSetMulti.append(dataSet)
        # end for

        # 留一个被试作为跨被试测试
        net_tmp = Net25()
        for x in net_tmp.parameters():
            print(x.data.shape, len(x.data.reshape(-1)))

        print("NN_cross_subject_LR2class_pretraining_Net25_multiSegment：参数个数 {}  ".format(
            sum(x.numel() for x in net_tmp.parameters())))
        fileLog.write(
            "NN_cross_subject_LR2class_pretraining_Net25_multiSegment：参数个数 {}  \n".format(
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
                                r'\\..\\model_all\\左右二分类\\NN_Net25_pretraining' +'\\%d'%timeBeginStart+
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
                    'NN_cross_subject_LR2class_pretraining_Net25_multiSegment：%s 训练集容量 %d 训练完成'
                    % (NAME_LIST[subjectNo], size))
                fileLog.write(
                    'NN_cross_subject_LR2class_pretraining_Net25_multiSegment：%s 训练集容量 %d 训练完成\n'
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
                'NN_cross_subject_LR2class_pretraining_Net25_multiSegment：%s : 测试集准确率  ' %
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
            path + r'\\..\model_all\\左右二分类\\NN_Net25_pretraining' +'\\%d'%timeBeginStart+
            r'\\learning_curve_LR2class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)
        fileD.close()

        fileD = open(
            path + r'\\..\model_all\\左右二分类\\NN_Net25_pretraining' +'\\%d'%timeBeginStart+
            r'\\confusion_matrix_LR2class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'NN_Net25_pretraining_LR2class_%d'%timeBeginStart)
    # end for
    # end NN_cross_subject_LR2class_pretraining_Net25_multiSegment



def NN_cross_subject_LR2class_fine_Net25_multiSegment():
    SEG_START = np.linspace(0, 1280, 9).astype(
        int)  # 0,160,320,480,640,800,960,1120,1280   
    sizeDict={0:350,160:330,320:350,480:330,640:350,800:350,960:350,1120:350,1280:330}
    SEG_START=[1280]
    for timeBeginStart in SEG_START:
        print('NN_cross_subject_LR2class_fine_Net25_multiSegment：timeBeginStart = ',timeBeginStart)

        fileLog = open(
            os.getcwd() + r'\\..\model\\左右二分类\\NN_Net25_fine'  +'\\%d'%timeBeginStart+
            r'\\learning_curve_NN_Net25_fine_LR2class.txt', 'w')
        fileLog.write('NN 左右二分类器_微调（左、右）跨被试\n')
        print('NN 左右二分类器_微调（左、右）跨被试')

        FINE_SIZES = np.linspace(0, 1.0, 25)#25
        path = os.getcwd()
        dataSetMulti = []

        # 提取所有被试数据
        for name in NAME_LIST:
            # 遍历各被试
            fileName = 'single_move_motion_start-%d.pickle' % timeBeginStart
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
            print('NN_cross_subject_LR2class_fine_Net25_multiSegment：样本数量', sum(y == 0),
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
            ) + r'\\..\\model\\左右二分类\\NN_Net25_pretraining' +'\\%d'%timeBeginStart+ '\\%s_%d_Net25.pkl' % (
                NAME_LIST[subjectNo],sizeDict[timeBeginStart])
            print(path)
            dataSet = dataSetMulti[subjectNo]  # 取出留下的被试样本集

            # 样本集特征缩放
            scaler = StandardScaler()
            X_train_scalered = scaler.fit_transform(dataSet[:, :-1])
            dataSet_train_scalered = np.c_[X_train_scalered,
                                        dataSet[:, -1]]  #特征缩放后的样本集(X,y)
            dataSet_train_scalered[np.isnan(dataSet_train_scalered)] = 0  # 处理异常特征

            print('NN_cross_subject_LR2class_fine_Net25_multiSegment：样本数量',
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
                    print('NN_cross_subject_LR2class_fine_Net25_multiSegment：%s 训练集容量 %d 训练完成' %
                        (NAME_LIST[subjectNo], size))
                    fileLog.write(
                        'NN_cross_subject_LR2class_fine_Net25_multiSegment：%s 训练集容量 %d 训练完成\n' %
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
                'NN_cross_subject_LR2class_fine_Net25_multiSegment：%s : 测试集准确率  ' %
                NAME_LIST[subjectNo], scores_test[-1])
            subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
            fileLog.write(
                'NN_cross_subject_LR2class_fine_Net25_multiSegment：%s : 测试集准确率  %.3f%%\n' %
                (NAME_LIST[subjectNo], scores_test[-1]))
        # end for
        fileLog.close()

        learning_curve_DICT = {}
        for learning_curve in subject_learning_curve_plot_LIST:
            learning_curve_DICT.update({learning_curve['name']: learning_curve})
        # end for
        learning_curve_DICT.update({'ignoreList': []})
 
        path = os.getcwd()
        fileD = open(
            path + r'\\..\model\\左右二分类\\NN_Net25_fine' +'\\%d'%timeBeginStart+
            r'\\learning_curve_LR2class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)
        fileD.close()

        fileD = open(
            path + r'\\..\model\\左右二分类\\NN_Net25_fine' +'\\%d'%timeBeginStart+
            r'\\confusion_matrix_LR2class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'NN_Net25_fine_LR2class_%d'%timeBeginStart)
    # end for
    # end NN_cross_subject_LR2class_fine_Net25_multiSegment



# NN_cross_subject_UD2class_pretraining_Net25_multiSegment()
NN_cross_subject_UD2class_fine_Net25_multiSegment()

# NN_cross_subject_LR2class_pretraining_Net25_multiSegment()
# NN_cross_subject_LR2class_fine_Net25_multiSegment()
