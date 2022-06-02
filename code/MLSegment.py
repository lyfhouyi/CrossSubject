# 传统机器学习算法-时间段对比

import os
import pickle
import sys
from io import StringIO

import numpy as np
from sklearn.base import clone
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import Plot

# SVM 分类器非跨被试：上下二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features_multiSegment() 得到的数据
def SVM_no_cross_subject_UD2class_multiSegment():
    SEG_START=np.linspace(-2080, 1120, 41).astype(int) # 窗移 80ms

    for timeBeginStart in SEG_START:
        print('SVM_no_cross_subject_UD2class_multiSegment：timeBeginStart = ',timeBeginStart)
        fileLog = open(
            os.getcwd() + r'\\..\model\\上下二分类\\ML\\SVM' +'\\%d'%timeBeginStart+
            r'\\log_SVM_no_cross_subject_UD2class.txt', 'w')
        fileLog.write('SVM 上下二分类器（上、下）非跨被试\n')
        print('SVM 上下二分类器（上、下）非跨被试')
        NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'xsc', 'yzg'] # 没有 wxc 的数据
        TRAIN_SIZES = np.linspace(0.1, 1, 25)
        path = os.getcwd()
        subject_learning_curve_plot_LIST = []
        confusion_matrix_DICT = {}
        for name in NAME_LIST:
            confusion_matrix_DICT_subjectI = {}
            #提取数据
            fileName = 'single_move_motion_start%d.pickle' % timeBeginStart
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
            print('SVM_no_cross_subject_UD2class_multiSegment：样本数量', sum(y == 0), sum(y == 1),
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
                print('SVM_no_cross_subject_UD2class_multiSegment：%s : 训练集准确率  %.3f%%\n' %
                    (name, acc_train))
                fileLog.write('SVM_no_cross_subject_UD2class_multiSegment：%s : 训练集准确率  %.3f%%\n' %
                            (name, acc_train))
                y_pred = clf.predict(X_test)
                acc_test = sum(y_test == y_pred) / len(y_pred)
                print('SVM_no_cross_subject_UD2class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
                    (name, acc_test))
                fileLog.write('SVM_no_cross_subject_UD2class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
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
            print('SVM_no_cross_subject_UD2class_multiSegment：%s : 测试集准确率  ' % name,
                scores_test[-1])
            subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
            fileLog.write('SVM_no_cross_subject_UD2class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
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
            path + r'\\..\model\\上下二分类\\ML\\SVM' +'\\%d'%timeBeginStart+
            r'\\learning_curve_UD2class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)

        fileD = open(
            path + r'\\..\model\\上下二分类\\ML\\SVM' +'\\%d'%timeBeginStart+
            r'\\confusion_matrix_UD2class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'learning_curve_UD2classes_intraSubject_SVM_%d'%timeBeginStart)
    # end for
    # end SVM_no_cross_subject_UD2class_multiSegment




# LDA 分类器非跨被试：上下二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features_multiSegment() 得到的数据
def LDA_no_cross_subject_UD2class_multiSegment():
    SEG_START=np.linspace(-2080, 1120, 41).astype(int) # 窗移 80ms

    for timeBeginStart in SEG_START:
        print('LDA_no_cross_subject_UD2class_multiSegment：timeBeginStart = ',timeBeginStart)
        fileLog = open(
            os.getcwd() + r'\\..\model\\上下二分类\\ML\\LDA' +'\\%d'%timeBeginStart+
            r'\\log_LDA_no_cross_subject_UD2class.txt', 'w')
        fileLog.write('LDA 上下二分类器（上、下）非跨被试\n')
        print('LDA 上下二分类器（上、下）非跨被试')
        NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'xsc', 'yzg'] # 没有 wxc 的数据
        TRAIN_SIZES = np.linspace(0.1, 1, 25)
        path = os.getcwd()
        subject_learning_curve_plot_LIST = []
        confusion_matrix_DICT = {}
        for name in NAME_LIST:
            confusion_matrix_DICT_subjectI = {}
            #提取数据
            fileName = 'single_move_motion_start%d.pickle' % timeBeginStart
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
            print('LDA_no_cross_subject_UD2class_multiSegment：样本数量', sum(y == 0), sum(y == 1),
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
                clf = LDA()
                clf.fit(X_train[:size, :], y_train[:size])
                x_pred = clf.predict(X_train)
                acc_train = sum(x_pred == y_train) / len(y_train)
                print('LDA_no_cross_subject_UD2class_multiSegment：%s : 训练集准确率  %.3f%%\n' %
                    (name, acc_train))
                fileLog.write('LDA_no_cross_subject_UD2class_multiSegment：%s : 训练集准确率  %.3f%%\n' %
                            (name, acc_train))
                y_pred = clf.predict(X_test)
                acc_test = sum(y_test == y_pred) / len(y_pred)
                print('LDA_no_cross_subject_UD2class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
                    (name, acc_test))
                fileLog.write('LDA_no_cross_subject_UD2class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
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
            print('LDA_no_cross_subject_UD2class_multiSegment：%s : 测试集准确率  ' % name,
                scores_test[-1])
            subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
            fileLog.write('LDA_no_cross_subject_UD2class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
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
            path + r'\\..\model\\上下二分类\\ML\\LDA' +'\\%d'%timeBeginStart+
            r'\\learning_curve_UD2class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)

        fileD = open(
            path + r'\\..\model\\上下二分类\\ML\\LDA' +'\\%d'%timeBeginStart+
            r'\\confusion_matrix_UD2class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'learning_curve_UD2classes_intraSubject_LDA_%d'%timeBeginStart)
    # end for
    # end LDA_no_cross_subject_UD2class_multiSegment

# SVM 分类器非跨被试：左右二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features_multiSegment() 得到的数据
def SVM_no_cross_subject_LR2class_multiSegment():
    SEG_START=np.linspace(-2080, 1120, 41).astype(int) # 窗移 80ms

    for timeBeginStart in SEG_START:
        print('SVM_no_cross_subject_LR2class_multiSegment：timeBeginStart = ',timeBeginStart)
        fileLog = open(
            os.getcwd() + r'\\..\model\\左右二分类\\ML\\SVM' +'\\%d'%timeBeginStart+
            r'\\log_SVM_no_cross_subject_LR2class.txt', 'w')
        fileLog.write('SVM 左右二分类器（左、右）非跨被试\n')
        print('SVM 左右二分类器（左、右）非跨被试')
        NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'xsc', 'yzg'] # 没有 wxc 的数据
        TRAIN_SIZES = np.linspace(0.1, 1, 25)
        path = os.getcwd()
        subject_learning_curve_plot_LIST = []
        confusion_matrix_DICT = {}
        for name in NAME_LIST:
            confusion_matrix_DICT_subjectI = {}
            #提取数据
            fileName = 'single_move_motion_start%d.pickle' % timeBeginStart
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
            print('SVM_no_cross_subject_LR2class_multiSegment：样本数量', sum(y == 0), sum(y == 1),
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
                print('SVM_no_cross_subject_LR2class_multiSegment：%s : 训练集准确率  %.3f%%\n' %
                    (name, acc_train))
                fileLog.write('SVM_no_cross_subject_LR2class_multiSegment：%s : 训练集准确率  %.3f%%\n' %
                            (name, acc_train))
                y_pred = clf.predict(X_test)
                acc_test = sum(y_test == y_pred) / len(y_pred)
                print('SVM_no_cross_subject_LR2class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
                    (name, acc_test))
                fileLog.write('SVM_no_cross_subject_LR2class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
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
            print('SVM_no_cross_subject_LR2class_multiSegment：%s : 测试集准确率  ' % name,
                scores_test[-1])
            subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
            fileLog.write('SVM_no_cross_subject_LR2class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
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
            path + r'\\..\model\\左右二分类\\ML\\SVM' +'\\%d'%timeBeginStart+
            r'\\learning_curve_LR2class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)

        fileD = open(
            path + r'\\..\model\\左右二分类\\ML\\SVM' +'\\%d'%timeBeginStart+
            r'\\confusion_matrix_LR2class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'learning_curve_LR2classes_intraSubject_SVM_%d'%timeBeginStart)
    # end for
    # end SVM_no_cross_subject_LR2class_multiSegment




# LDA 分类器非跨被试：左右二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features_multiSegment() 得到的数据
def LDA_no_cross_subject_LR2class_multiSegment():
    SEG_START=np.linspace(-2080, 1120, 41).astype(int) # 窗移 80ms

    for timeBeginStart in SEG_START:
        print('LDA_no_cross_subject_LR2class_multiSegment：timeBeginStart = ',timeBeginStart)
        fileLog = open(
            os.getcwd() + r'\\..\model\\左右二分类\\ML\\LDA' +'\\%d'%timeBeginStart+
            r'\\log_LDA_no_cross_subject_LR2class.txt', 'w')
        fileLog.write('LDA 左右二分类器（左、右）非跨被试\n')
        print('LDA 左右二分类器（左、右）非跨被试')
        NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'xsc', 'yzg'] # 没有 wxc 的数据
        TRAIN_SIZES = np.linspace(0.1, 1, 25)
        path = os.getcwd()
        subject_learning_curve_plot_LIST = []
        confusion_matrix_DICT = {}
        for name in NAME_LIST:
            confusion_matrix_DICT_subjectI = {}
            #提取数据
            fileName = 'single_move_motion_start%d.pickle' % timeBeginStart
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
            print('LDA_no_cross_subject_LR2class_multiSegment：样本数量', sum(y == 0), sum(y == 1),
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
                clf = LDA()
                clf.fit(X_train[:size, :], y_train[:size])
                x_pred = clf.predict(X_train)
                acc_train = sum(x_pred == y_train) / len(y_train)
                print('LDA_no_cross_subject_LR2class_multiSegment：%s : 训练集准确率  %.3f%%\n' %
                    (name, acc_train))
                fileLog.write('LDA_no_cross_subject_LR2class_multiSegment：%s : 训练集准确率  %.3f%%\n' %
                            (name, acc_train))
                y_pred = clf.predict(X_test)
                acc_test = sum(y_test == y_pred) / len(y_pred)
                print('LDA_no_cross_subject_LR2class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
                    (name, acc_test))
                fileLog.write('LDA_no_cross_subject_LR2class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
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
            print('LDA_no_cross_subject_LR2class_multiSegment：%s : 测试集准确率  ' % name,
                scores_test[-1])
            subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
            fileLog.write('LDA_no_cross_subject_LR2class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
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
            path + r'\\..\model\\左右二分类\\ML\\LDA' +'\\%d'%timeBeginStart+
            r'\\learning_curve_LR2class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)

        fileD = open(
            path + r'\\..\model\\左右二分类\\ML\\LDA' +'\\%d'%timeBeginStart+
            r'\\confusion_matrix_LR2class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'learning_curve_LR2classes_intraSubject_LDA_%d'%timeBeginStart)
    # end for
    # end LDA_no_cross_subject_LR2class_multiSegment

# SVM 分类器非跨被试：四分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features_multiSegment() 得到的数据
def SVM_no_cross_subject_4class_multiSegment():
    SEG_START=np.linspace(-2080, 1120, 41).astype(int) # 窗移 80ms

    for timeBeginStart in SEG_START:
        print('SVM_no_cross_subject_4class_multiSegment：timeBeginStart = ',timeBeginStart)
        fileLog = open(
            os.getcwd() + r'\\..\model\\四分类\\ML\\SVM' +'\\%d'%timeBeginStart+
            r'\\log_SVM_no_cross_subject_4class.txt', 'w')
        fileLog.write('SVM 四分类器（上、下、左、右）非跨被试\n')
        print('SVM 四分类器（上、下、左、右）非跨被试')
        NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'xsc', 'yzg'] # 没有 wxc 的数据
        TRAIN_SIZES = np.linspace(0.1, 1, 25)
        path = os.getcwd()
        subject_learning_curve_plot_LIST = []
        confusion_matrix_DICT = {}
        for name in NAME_LIST:
            confusion_matrix_DICT_subjectI = {}
            #提取数据
            fileName = 'single_move_motion_start%d.pickle' % timeBeginStart
            fileD = open(
                path + r'\\..\\data\\' + name + '\\四分类\\' +
                fileName, 'rb')
            X, y = pickle.load(fileD)
            fileD.close()

            # 构造样本集
            dataSet = np.c_[X, y]
            print('SVM_no_cross_subject_4class_multiSegment：样本数量', sum(y == 0), sum(y == 1),
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
                print('SVM_no_cross_subject_4class_multiSegment：%s : 训练集准确率  %.3f%%\n' %
                    (name, acc_train))
                fileLog.write('SVM_no_cross_subject_4class_multiSegment：%s : 训练集准确率  %.3f%%\n' %
                            (name, acc_train))
                y_pred = clf.predict(X_test)
                acc_test = sum(y_test == y_pred) / len(y_pred)
                print('SVM_no_cross_subject_4class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
                    (name, acc_test))
                fileLog.write('SVM_no_cross_subject_4class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
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
            print('SVM_no_cross_subject_4class_multiSegment：%s : 测试集准确率  ' % name,
                scores_test[-1])
            subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
            fileLog.write('SVM_no_cross_subject_4class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
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
            path + r'\\..\model\\四分类\\ML\\SVM' +'\\%d'%timeBeginStart+
            r'\\learning_curve_4class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)

        fileD = open(
            path + r'\\..\model\\四分类\\ML\\SVM' +'\\%d'%timeBeginStart+
            r'\\confusion_matrix_4class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'learning_curve_4classes_intraSubject_SVM_%d'%timeBeginStart)
    # end for
    # end SVM_no_cross_subject_4class_multiSegment



# LDA 分类器非跨被试：四分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features_multiSegment() 得到的数据
def LDA_no_cross_subject_4class_multiSegment():
    SEG_START=np.linspace(-2080, 1120, 41).astype(int) # 窗移 80ms

    for timeBeginStart in SEG_START:
        print('LDA_no_cross_subject_4class_multiSegment：timeBeginStart = ',timeBeginStart)
        fileLog = open(
            os.getcwd() + r'\\..\model\\四分类\\ML\\LDA' +'\\%d'%timeBeginStart+
            r'\\log_LDA_no_cross_subject_4class.txt', 'w')
        fileLog.write('LDA 四分类器（上、下、左、右）非跨被试\n')
        print('LDA 四分类器（上、下、左、右）非跨被试')
        NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'xsc', 'yzg'] # 没有 wxc 的数据
        TRAIN_SIZES = np.linspace(0.1, 1, 25)
        path = os.getcwd()
        subject_learning_curve_plot_LIST = []
        confusion_matrix_DICT = {}
        for name in NAME_LIST:
            confusion_matrix_DICT_subjectI = {}
            #提取数据
            fileName = 'single_move_motion_start%d.pickle' % timeBeginStart
            fileD = open(
                path + r'\\..\\data\\' + name + '\\四分类\\' +
                fileName, 'rb')
            X, y = pickle.load(fileD)
            fileD.close()

            # 构造样本集
            dataSet = np.c_[X, y]
            print('LDA_no_cross_subject_4class_multiSegment：样本数量', sum(y == 0), sum(y == 1),
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
                clf = LDA()
                clf.fit(X_train[:size, :], y_train[:size])
                x_pred = clf.predict(X_train)
                acc_train = sum(x_pred == y_train) / len(y_train)
                print('LDA_no_cross_subject_4class_multiSegment：%s : 训练集准确率  %.3f%%\n' %
                    (name, acc_train))
                fileLog.write('LDA_no_cross_subject_4class_multiSegment：%s : 训练集准确率  %.3f%%\n' %
                            (name, acc_train))
                y_pred = clf.predict(X_test)
                acc_test = sum(y_test == y_pred) / len(y_pred)
                print('LDA_no_cross_subject_4class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
                    (name, acc_test))
                fileLog.write('LDA_no_cross_subject_4class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
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
            print('LDA_no_cross_subject_4class_multiSegment：%s : 测试集准确率  ' % name,
                scores_test[-1])
            subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
            fileLog.write('LDA_no_cross_subject_4class_multiSegment：%s : 测试集准确率  %.3f%%\n' %
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
            path + r'\\..\model\\四分类\\ML\\LDA' +'\\%d'%timeBeginStart+
            r'\\learning_curve_4class.pickle', 'wb')
        pickle.dump(learning_curve_DICT, fileD)

        fileD = open(
            path + r'\\..\model\\四分类\\ML\\LDA' +'\\%d'%timeBeginStart+
            r'\\confusion_matrix_4class.pickle', 'wb')
        pickle.dump(confusion_matrix_DICT, fileD)
        fileD.close()
        Plot.draw_learning_curve(learning_curve_DICT,
                                'learning_curve_4classes_intraSubject_LDA_%d'%timeBeginStart)
    # end for
    # end LDA_no_cross_subject_4class_multiSegment

# SVM_no_cross_subject_UD2class_multiSegment()
# LDA_no_cross_subject_UD2class_multiSegment()

# SVM_no_cross_subject_LR2class_multiSegment()
# LDA_no_cross_subject_LR2class_multiSegment()

# SVM_no_cross_subject_4class_multiSegment()
# LDA_no_cross_subject_4class_multiSegment()

def test():
    names=np.linspace(-2080, 1120, 41).astype(int)
    for name in names:
        os.mkdir('D:\\硕士学习\\毕业论文\\CrossSubject\\model\\四分类\\ML\\LDA\\%d\\'%name)
        os.mkdir('D:\\硕士学习\\毕业论文\\CrossSubject\\model\\四分类\\ML\\SVM\\%d\\'%name)


# test()