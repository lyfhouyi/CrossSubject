# 传统机器学习算法

import imp
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
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import Plot

# 计算学习曲线
# 必选参数：估计器，特征样本集(numpy 二维数组(行：各样本，列：特征值))，标签样本集(numpy 二维数组(行：各样本，列：标签值))，训练集容量划分，交叉验证折数
# 返回值：训练集容量(numpy 一维数组)，训练集评分(numpy 二维数组(行：遍历各容量值，列：遍历各交叉验证))，测试集评分(numpy 二维数组(行：遍历各容量值，列：遍历各交叉验证))
def myLearning_curve(estimator, X, y, train_sizes, cv):
    skf = StratifiedKFold(n_splits=cv)
    skf.get_n_splits(X, y)
    train_scores = np.empty((train_sizes.shape[0], 0))
    test_scores = np.empty((train_sizes.shape[0], 0))
    # train_sizes=[]

    for train_index, test_index in skf.split(X, y):
        # 遍历各交叉验证
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        t_sizes = train_index.shape[0] * train_sizes
        t_sizes = t_sizes.astype(int)
        scores_train = []
        scores_test = []
        for size in t_sizes:
            # 遍历各训练集容量
            X_trainI = X_train[:size, :]
            y_trainI = y_train[:size]
            clf = clone(estimator)
            clf.fit(X_trainI, y_trainI)

            # 计算评分
            y_pred_train = clf.predict(X_trainI)
            score_train = sum(y_trainI == y_pred_train) / len(y_trainI)
            scores_train.append(score_train)

            y_pred_test = clf.predict(X_test)
            score_test = sum(y_test == y_pred_test) / len(y_test)
            scores_test.append(score_test)
        # end for

        scores_train = np.array(scores_train)
        scores_test = np.array(scores_test)
        train_scores = np.c_[train_scores, scores_train]
        test_scores = np.c_[test_scores, scores_test]
    # end for
    return t_sizes, train_scores, test_scores


# 找到最优的 SVM 分类器非跨被试：二分类（0-上下；1-左右）
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def Find_SVM_no_cross_subject_2class():
    print('SVM 分类器非跨被试')
    #提取数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\cwm\\single_move_motion_start_motion_end.pickle',
        'rb')
    X, y = pickle.load(fileD)
    fileD.close()

    # 二分类：0-上下；1-左右
    y[y == 1] = 0
    y[y != 0] = 1

    # 构造样本集
    dataSet = np.c_[X, y]

    # 样本集洗牌
    shuffle_index = np.random.permutation(dataSet.shape[0])
    dataSet = dataSet[shuffle_index]

    # 特征缩放
    scaler = StandardScaler()
    X_scalered = scaler.fit_transform(dataSet[:, :-1])
    dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)

    #拆分训练集、测试集
    train_set, test_set = train_test_split(dataSet_scalered,
                                           test_size=0.2)  #划分训练集、测试集
    X_train = train_set[:, :-1]  #训练集特征矩阵
    y_train = train_set[:, -1]  #训练集标签数组
    X_test = test_set[:, :-1]  #测试集特征矩阵
    y_test = test_set[:, -1]  #测试集标签数组

    # 训练模型
    param_grid = [{
        'kernel': ['linear', 'rbf'],
        'C': [
            1e-5, 1e-4, 1e-3, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1,
            10, 100
        ],
        'gamma': [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 5, 10],
        'verbose': [False],
        'tol': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    }]
    svm_clf = SVC()
    grid_search = GridSearchCV(svm_clf,
                               param_grid,
                               cv=5,
                               scoring='accuracy',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    svm_best = grid_search.best_estimator_

    # 检验准确率
    y_pred = svm_best.predict(X_test)
    acc = sum(y_test == y_pred) / len(y_pred)
    print('测试集准确率：', acc)

    # train_sizes, train_scores, test_scores =\
    #                 learning_curve(estimator= svm_best,
    #                             X=X,
    #                             y=y,
    #                             train_sizes=np.linspace(0.5, 1.0, 5), #在0.1和1间线性的取10个值
    #                             cv=5,
    #                             n_jobs=1)

    # 测试结果可视化
    plt.figure()
    plt.scatter(np.arange(len(y_pred)), y_pred, c='r')  #测试集标签值
    plt.scatter(np.arange(len(y_test)), y_test, c='b')  #测试集预测输出
    plt.xticks(range(len(y_test)))
    plt.grid()
    plt.show()

    # 提取模型参数(预测输出：P_=X_test*coef_.T+intercept_ 二值化处理：正-1；负-0)
    path = os.getcwd()
    coef_ = svm_best.coef_  #特征权重
    intercept_ = svm_best.intercept_  #偏置项
    print(intercept_.shape)
    print(coef_.shape)
    print(intercept_)
    print(coef_)
    fileD = open(path + r'\\..\params\attention_SVC.txt', 'w')
    head = [
        '注意力分类器\n', '模型：SVC\n',
        '预测输出：P_ = X_test * coef_.T + intercept_  二值化处理：正-1；负-0\n',
        '标签：正常(高水平)-1，异常(低水平)-0\n', "参数格式：偏置项(intercept_) + '#' + 权重(coef_)\n",
        '###\n'
    ]
    fileD.writelines(head)  #写文件头
    fileD.write('%.64f\n' % intercept_)  #写偏置项
    fileD.write('#\n')  #写分隔符
    for wi in list(coef_[0]):
        fileD.write('%.64f\n' % wi)  #写权重
    # end for
    fileD.close()

    # 保存 SVC 转换器(预测输出：P_=X_test*coef_.T+intercept_ 二值化处理：正-1；负-0)
    path = os.getcwd()
    fileD = open(path + r'\\..\saved_model\attention_SVC.pickle', 'wb')
    pickle.dump(svm_best, fileD)  #保存注意力分类器
    fileD.close()
    # end Find_SVM_no_cross_subject_2class

    print('SVM 分类器非跨被试')
    #提取数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\yzg\\single_move_motion_start_motion_end.pickle',
        'rb')
    X, y = pickle.load(fileD)
    fileD.close()

    # 二分类：0-上下；1-左右
    y[y == 1] = 0
    y[y != 0] = 1

    # 构造样本集
    dataSet = np.c_[X, y]
    print('样本数量', sum(y == 0), sum(y == 1), sum(y == 2), sum(y == 3))

    # # 样本集洗牌
    # shuffle_index = np.random.permutation(dataSet.shape[0])
    # dataSet = dataSet[shuffle_index]

    # 特征缩放
    scaler = StandardScaler()
    X_scalered = scaler.fit_transform(dataSet[:, :-1])
    dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)

    dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征
    #拆分训练集、测试集
    train_set, test_set = train_test_split(dataSet_scalered,
                                           test_size=0.2)  #划分训练集、测试集
    X_train = train_set[:, :-1]  #训练集特征矩阵
    y_train = train_set[:, -1]  #训练集标签数组
    X_test = test_set[:, :-1]  #测试集特征矩阵
    y_test = test_set[:, -1]  #测试集标签数组

    ########################
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    # 训练模型
    clf = SGDClassifier(verbose=True, early_stopping=True)
    clf.fit(X_train, y_train)

    sys.stdout = old_stdout
    x_pred = clf.predict(X_train)
    acc_train = sum(x_pred == y_train) / len(y_train)
    print('绘图模型的训练集准确率：', acc_train)

    y_pred = clf.predict(X_test)
    acc_test = sum(y_test == y_pred) / len(y_pred)
    print('绘图模型的测试集准确率：', acc_test)

    loss_history = mystdout.getvalue()
    loss_list = []
    for line in loss_history.split('\n'):
        if (len(line.split("loss: ")) == 1):
            continue
        loss_list.append(float(line.split("loss: ")[-1]))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel("Time in epochs")
    plt.ylabel("Loss")
    plt.show()
    ########################

    ########################
    clf = SGDClassifier(early_stopping=True)
    train_sizes, train_scores, test_scores =\
                    learning_curve(estimator= clf,
                                X=X,
                                y=y,
                                train_sizes=np.linspace(0.5, 1.0, 5), #在0.1和1间线性的取10个值
                                cv=5,
                                n_jobs=1)
    ####################

    ########################
    clf = SGDClassifier(early_stopping=True)
    scores = cross_val_score(clf,
                             dataSet_scalered[:, :-1],
                             dataSet_scalered[:, -1],
                             cv=10)

    print(scores)
    print(np.average(scores))
    ########################
    # end SVM_no_cross_subject_2class


# SVM 分类器非跨被试：四分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features() 得到的数据
def SVM_no_cross_subject_4class():
    fileLog = open(
        os.getcwd() + r'\\..\model\\四分类\\ML\\SVM' +
        r'\\log_SVM_no_cross_subject_4class.txt.txt', 'w')
    fileLog.write('SVM 四分类器（上、下、左、右）非跨被试\n')
    print('SVM 四分类器（上、下、左、右）非跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    TRAIN_SIZES = np.linspace(0.05, 1, 50)
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for name in NAME_LIST:
        confusion_matrix_DICT_subjectI = {}
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = np.c_[X, y]
        print('SVM_no_cross_subject_4class：样本数量', sum(y == 0), sum(y == 1),
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
            print('SVM_no_cross_subject_4class：%s : 训练集准确率  %.3f%%\n' %
                  (name, acc_train))
            fileLog.write('SVM_no_cross_subject_4class：%s : 训练集准确率  %.3f%%\n' %
                          (name, acc_train))
            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('SVM_no_cross_subject_4class：%s : 测试集准确率  %.3f%%\n' %
                  (name, acc_test))
            fileLog.write('SVM_no_cross_subject_4class：%s : 测试集准确率  %.3f%%\n' %
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
        print('SVM_no_cross_subject_4class：%s : 测试集准确率  ' % name,
              scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write('SVM_no_cross_subject_4class：%s : 测试集准确率  %.3f%%\n' %
                      (name, scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML\\SVM' +
        r'\\learning_curve_SVM_no_cross_subject_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)

    fileD = open(
        path + r'\\..\model\\四分类\\ML\\SVM' +
        r'\\confusion_matrix_SVM_no_cross_subject_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'learning_curve_4classes_intraSubject_SVM')
    # end SVM_no_cross_subject_4class



# SVM 分类器非跨被试：上下二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features() 得到的数据
def SVM_no_cross_subject_UD2class():
    fileLog = open(
        os.getcwd() + r'\\..\model\\上下二分类\\ML\\SVM' +
        r'\\log_SVM_no_cross_subject_UD2class.txt.txt', 'w')
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
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X=X[:100]
        y=y[:100]    

        # 构造样本集
        dataSet = np.c_[X, y]
        print('SVM_no_cross_subject_UD2class：样本数量', sum(y == 0), sum(y == 1),
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
            print('SVM_no_cross_subject_UD2class：%s : 训练集准确率  %.3f%%\n' %
                  (name, acc_train))
            fileLog.write('SVM_no_cross_subject_UD2class：%s : 训练集准确率  %.3f%%\n' %
                          (name, acc_train))
            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('SVM_no_cross_subject_UD2class：%s : 测试集准确率  %.3f%%\n' %
                  (name, acc_test))
            fileLog.write('SVM_no_cross_subject_UD2class：%s : 测试集准确率  %.3f%%\n' %
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
        print('SVM_no_cross_subject_UD2class：%s : 测试集准确率  ' % name,
              scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write('SVM_no_cross_subject_UD2class：%s : 测试集准确率  %.3f%%\n' %
                      (name, scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\上下二分类\\ML\\SVM' +
        r'\\learning_curve_SVM_no_cross_subject_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)

    fileD = open(
        path + r'\\..\model\\上下二分类\\ML\\SVM' +
        r'\\confusion_matrix_SVM_no_cross_subject_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('SVM_no_cross_subject')
    fileD = open(
        path + r'\\..\model\\上下二分类\\ML\\SVM' +
        r'\\learning_curve_SVM_no_cross_subject_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'learning_curve_UD2classes_intraSubject_SVM')
    # end SVM_no_cross_subject_UD2class


# SVM 分类器非跨被试：左右二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features() 得到的数据
def SVM_no_cross_subject_LR2class():
    fileLog = open(
        os.getcwd() + r'\\..\model\\左右二分类\\ML\\SVM' +
        r'\\log_SVM_no_cross_subject_UD2class.txt.txt', 'w')
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
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X=X[100:]
        y=y[100:]    
        y[y==2]=0
        y[y==3]=1    

        # 构造样本集
        dataSet = np.c_[X, y]
        print('SVM_no_cross_subject_LR2class：样本数量', sum(y == 0), sum(y == 1),
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
            print('SVM_no_cross_subject_LR2class：%s : 训练集准确率  %.3f%%\n' %
                  (name, acc_train))
            fileLog.write('SVM_no_cross_subject_LR2class：%s : 训练集准确率  %.3f%%\n' %
                          (name, acc_train))
            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('SVM_no_cross_subject_LR2class：%s : 测试集准确率  %.3f%%\n' %
                  (name, acc_test))
            fileLog.write('SVM_no_cross_subject_LR2class：%s : 测试集准确率  %.3f%%\n' %
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
        print('SVM_no_cross_subject_LR2class：%s : 测试集准确率  ' % name,
              scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write('SVM_no_cross_subject_LR2class：%s : 测试集准确率  %.3f%%\n' %
                      (name, scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\左右二分类\\ML\\SVM' +
        r'\\learning_curve_SVM_no_cross_subject_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)

    fileD = open(
        path + r'\\..\model\\左右二分类\\ML\\SVM' +
        r'\\confusion_matrix_SVM_no_cross_subject_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('SVM_no_cross_subject')
    fileD = open(
        path + r'\\..\model\\左右二分类\\ML\\SVM' +
        r'\\learning_curve_SVM_no_cross_subject_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'learning_curve_LR2classes_intraSubject_SVM')
    # end SVM_no_cross_subject_LR2class


# SVM 分类器非跨被试：混合二分类（0-上下；1-左右）
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def SVM_no_cross_subject_mixture2class():
    print('SVM 混合分类分类器（上下、左右）非跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    for name in NAME_LIST:
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 混合二分类：0-上下；1-左右
        y[y == 1] = 0
        y[y != 0] = 1

        # 构造样本集
        dataSet = np.c_[X, y]
        print('SVM_no_cross_subject_mixture2class：样本数量', sum(y == 0),
              sum(y == 1))

        # 样本集洗牌
        shuffle_index = np.random.permutation(dataSet.shape[0])
        dataSet = dataSet[shuffle_index]

        # 特征缩放
        scaler = StandardScaler()
        X_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)

        dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征

        if False:  # 是否绘制 迭代次数-损失值 曲线
            #拆分训练集、测试集
            train_set, test_set = train_test_split(dataSet_scalered,
                                                   test_size=0.2)  #划分训练集、测试集
            X_train = train_set[:, :-1]  #训练集特征矩阵
            y_train = train_set[:, -1]  #训练集标签数组
            X_test = test_set[:, :-1]  #测试集特征矩阵
            y_test = test_set[:, -1]  #测试集标签数组

            ########################
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            # 训练模型
            clf = SGDClassifier(verbose=True, early_stopping=True)
            clf.fit(X_train, y_train)

            sys.stdout = old_stdout
            x_pred = clf.predict(X_train)
            acc_train = sum(x_pred == y_train) / len(y_train)
            print('绘图模型的训练集准确率：', acc_train)

            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('绘图模型的测试集准确率：', acc_test)

            loss_history = mystdout.getvalue()
            loss_list = []
            for line in loss_history.split('\n'):
                if (len(line.split("loss: ")) == 1):
                    continue
                loss_list.append(float(line.split("loss: ")[-1]))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), loss_list)
            plt.xlabel("Time in epochs")
            plt.ylabel("Loss")
            plt.show()
            ########################
        # end if

        ########################
        # 交叉验证
        clf = SGDClassifier(early_stopping=True)
        scores = cross_val_score(clf,
                                 dataSet_scalered[:, :-1],
                                 dataSet_scalered[:, -1],
                                 cv=10)

        print('SVM_no_cross_subject_mixture2class：%s : 交叉验证准确率  ' % name,
              scores)
        print('SVM_no_cross_subject_mixture2class：%s : 交叉验证准确率平均值  ' % name,
              np.average(scores))
        ########################

        ########################
        # 计算学习曲线
        clf = SVC()
        # train_sizes, train_scores, test_scores=myLearning_curve(clf,dataSet_scalered[:, :-1],dataSet_scalered[:, -1],np.linspace(0.2, 1.0, 10),5)

        train_sizes, train_scores, test_scores,fit_times,score_times =\
                        learning_curve(estimator= clf,
                                    X=dataSet_scalered[:, :-1],
                                    y=dataSet_scalered[:, -1],
                                    train_sizes=np.linspace(0.1, 1.0, 100),
                                    cv=10,
                                    n_jobs=-1,
                                    return_times=True)
        subject_learning_curve_plot = {
            'name': name,
            'train_sizes': train_sizes,
            'train_scores': train_scores.mean(axis=1),
            'test_scores': test_scores.mean(axis=1),
            'fit_times': fit_times.mean(axis=1)
        }
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        ########################
    # end for

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML' +
        r'\\learning_curve_SVM_no_cross_subject_mixture2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'learning_curve_2classes_intraSubject_SVM')
    # end SVM_no_cross_subject_mixture2class


# LDA 分类器非跨被试：混合二分类（0-上下；1-左右）
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def LDA_no_cross_subject_mixture2class():
    print('LDA 混合二分类器非跨被试')
    #提取数据
    path = os.getcwd()
    fileD = open(
        path + r'\\..\\data\\kx\\single_move_motion_start_motion_end.pickle',
        'rb')
    X, y = pickle.load(fileD)
    fileD.close()

    # 混合二分类：0-上下；1-左右
    y[y == 1] = 0
    y[y != 0] = 1

    # 构造样本集
    dataSet = np.c_[X, y]

    # 样本集洗牌
    shuffle_index = np.random.permutation(dataSet.shape[0])
    dataSet = dataSet[shuffle_index]

    # 特征缩放
    scaler = StandardScaler()
    X_scalered = scaler.fit_transform(dataSet[:, :-1])
    dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)

    #拆分训练集、测试集
    train_set, test_set = train_test_split(dataSet_scalered,
                                           test_size=0.2)  #划分训练集、测试集
    X_train = train_set[:, :-1]  #训练集特征矩阵
    y_train = train_set[:, -1]  #训练集标签数组
    X_test = test_set[:, :-1]  #测试集特征矩阵
    y_test = test_set[:, -1]  #测试集标签数组

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # 训练模型
    clf = LDA(verbose=True)
    clf.fit(X_train, y_train).flatten()

    x_pred = clf.predict(X_train)
    acc_train = sum(x_pred == y_train) / len(y_train)
    print('训练集准确率：', acc_train)

    y_pred = clf.predict(X_test)
    acc_test = sum(y_test == y_pred) / len(y_pred)
    print('测试集准确率：', acc_test)

    # train_sizes, train_scores, test_scores =\
    #                 learning_curve(estimator= svm_best,
    #                             X=X,
    #                             y=y,
    #                             train_sizes=np.linspace(0.5, 1.0, 5), #在0.1和1间线性的取10个值
    #                             cv=5,
    #                             n_jobs=1)

    ####################

    sys.stdout = old_stdout
    loss_history = mystdout.getvalue()
    loss_list = []
    for line in loss_history.split('\n'):
        if (len(line.split("loss: ")) == 1):
            continue
        loss_list.append(float(line.split("loss: ")[-1]))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel("Time in epochs")
    plt.ylabel("Loss")
    plt.close()

    ####################

    # 测试结果可视化
    plt.figure()
    plt.scatter(np.arange(len(y_pred)), y_pred, c='r')  #测试集标签值
    plt.scatter(np.arange(len(y_test)), y_test, c='b')  #测试集预测输出
    plt.xticks(range(len(y_test)))
    plt.grid()
    plt.show()

    # 保存 LDA 转换器(预测输出：P_=X_test*coef_.T+intercept_ 二值化处理：正-1；负-0)
    path = os.getcwd()
    fileD = open(path + r'\\..\\data\\cwm\\LDA_no_cross_subject_2class.pickle',
                 'wb')
    pickle.dump(clf, fileD)  #保存注意力分类器
    fileD.close()
    # end LDA_no_cross_subject_mixture2class


# 决策树 分类器非跨被试：混合二分类（0-上下；1-左右）
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def DT_no_cross_subject_mixture2class():
    print('DT 混合分类分类器（上下、左右）非跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    for name in NAME_LIST:
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 混合二分类：0-上下；1-左右
        y[y == 1] = 0
        y[y != 0] = 1

        # 构造样本集
        dataSet = np.c_[X, y]
        print('DT_no_cross_subject_mixture2class：样本数量', sum(y == 0),
              sum(y == 1))

        # 样本集洗牌
        shuffle_index = np.random.permutation(dataSet.shape[0])
        dataSet = dataSet[shuffle_index]

        # 特征缩放
        scaler = StandardScaler()
        X_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)

        dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征

        if False:  # 是否绘制 迭代次数-损失值 曲线
            #拆分训练集、测试集
            train_set, test_set = train_test_split(dataSet_scalered,
                                                   test_size=0.2)  #划分训练集、测试集
            X_train = train_set[:, :-1]  #训练集特征矩阵
            y_train = train_set[:, -1]  #训练集标签数组
            X_test = test_set[:, :-1]  #测试集特征矩阵
            y_test = test_set[:, -1]  #测试集标签数组

            ########################
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            # 训练模型
            clf = SGDClassifier(verbose=True, early_stopping=True)
            clf.fit(X_train, y_train)

            sys.stdout = old_stdout
            x_pred = clf.predict(X_train)
            acc_train = sum(x_pred == y_train) / len(y_train)
            print('绘图模型的训练集准确率：', acc_train)

            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('绘图模型的测试集准确率：', acc_test)

            loss_history = mystdout.getvalue()
            loss_list = []
            for line in loss_history.split('\n'):
                if (len(line.split("loss: ")) == 1):
                    continue
                loss_list.append(float(line.split("loss: ")[-1]))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), loss_list)
            plt.xlabel("Time in epochs")
            plt.ylabel("Loss")
            plt.show()
            ########################
        # end if

        ########################
        # 交叉验证
        clf = tree.DecisionTreeClassifier(criterion='gini')
        scores = cross_val_score(clf,
                                 dataSet_scalered[:, :-1],
                                 dataSet_scalered[:, -1],
                                 cv=10)

        print('DT_no_cross_subject_mixture2class：%s : 交叉验证准确率  ' % name,
              scores)
        print('DT_no_cross_subject_mixture2class：%s : 交叉验证准确率平均值  ' % name,
              np.average(scores))
        ########################

        ########################
        # 计算学习曲线
        clf = tree.DecisionTreeClassifier(criterion='gini')
        # train_sizes, train_scores, test_scores=myLearning_curve(clf,dataSet_scalered[:, :-1],dataSet_scalered[:, -1],np.linspace(0.2, 1.0, 10),5)

        train_sizes, train_scores, test_scores,fit_times,score_times =\
                        learning_curve(estimator= clf,
                                    X=dataSet_scalered[:, :-1],
                                    y=dataSet_scalered[:, -1],
                                    train_sizes=np.linspace(0.1, 1.0, 100),
                                    cv=10,
                                    n_jobs=-1,
                                    return_times=True)
        subject_learning_curve_plot = {
            'name': name,
            'train_sizes': train_sizes,
            'train_scores': train_scores.mean(axis=1),
            'test_scores': test_scores.mean(axis=1),
            'fit_times': fit_times.mean(axis=1)
        }
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        ########################
    # end for

    learning_curve_DICT = {}
    for learning_curve_i in subject_learning_curve_plot_LIST:
        learning_curve_DICT.update({learning_curve_i['name']: learning_curve_i})
    # end for
    learning_curve_DICT.update({'ignoreList': []})


    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML' +
        r'\\learning_curve_DT_no_cross_subject_mixture2class.pickle', 'wb')
    pickle.dump(learning_curve_DICT, fileD)

    Plot.draw_learning_curve(learning_curve_DICT,
                             'learning_curve_mixture2class_intraSubject_DT')
    # end DT_no_cross_subject_mixture2class


# 随机森林 分类器非跨被试：混合二分类（0-上下；1-左右）
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def RF_no_cross_subject_mixture2class():
    print('RF 混合分类分类器（上下、左右）非跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    for name in NAME_LIST:
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 混合二分类：0-上下；1-左右
        y[y == 1] = 0
        y[y != 0] = 1

        # 构造样本集
        dataSet = np.c_[X, y]
        print('RF_no_cross_subject_mixture2class：样本数量', sum(y == 0),
              sum(y == 1))

        # 样本集洗牌
        shuffle_index = np.random.permutation(dataSet.shape[0])
        dataSet = dataSet[shuffle_index]

        # 特征缩放
        scaler = StandardScaler()
        X_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)

        dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征

        if False:  # 是否绘制 迭代次数-损失值 曲线
            #拆分训练集、测试集
            train_set, test_set = train_test_split(dataSet_scalered,
                                                   test_size=0.2)  #划分训练集、测试集
            X_train = train_set[:, :-1]  #训练集特征矩阵
            y_train = train_set[:, -1]  #训练集标签数组
            X_test = test_set[:, :-1]  #测试集特征矩阵
            y_test = test_set[:, -1]  #测试集标签数组

            ########################
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            # 训练模型
            clf = SGDClassifier(verbose=True, early_stopping=True)
            clf.fit(X_train, y_train)

            sys.stdout = old_stdout
            x_pred = clf.predict(X_train)
            acc_train = sum(x_pred == y_train) / len(y_train)
            print('绘图模型的训练集准确率：', acc_train)

            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('绘图模型的测试集准确率：', acc_test)

            loss_history = mystdout.getvalue()
            loss_list = []
            for line in loss_history.split('\n'):
                if (len(line.split("loss: ")) == 1):
                    continue
                loss_list.append(float(line.split("loss: ")[-1]))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), loss_list)
            plt.xlabel("Time in epochs")
            plt.ylabel("Loss")
            plt.show()
            ########################
        # end if

        ########################
        # 交叉验证
        clf = RandomForestClassifier(n_estimators=10, criterion='gini')    
        scores = cross_val_score(clf,
                                 dataSet_scalered[:, :-1],
                                 dataSet_scalered[:, -1],
                                 cv=10)

        print('RF_no_cross_subject_mixture2class：%s : 交叉验证准确率  ' % name,
              scores)
        print('RF_no_cross_subject_mixture2class：%s : 交叉验证准确率平均值  ' % name,
              np.average(scores))
        ########################

        ########################
        # 计算学习曲线
        clf = RandomForestClassifier(n_estimators=10, criterion='gini') 
        # train_sizes, train_scores, test_scores=myLearning_curve(clf,dataSet_scalered[:, :-1],dataSet_scalered[:, -1],np.linspace(0.2, 1.0, 10),5)

        train_sizes, train_scores, test_scores,fit_times,score_times =\
                        learning_curve(estimator= clf,
                                    X=dataSet_scalered[:, :-1],
                                    y=dataSet_scalered[:, -1],
                                    train_sizes=np.linspace(0.1, 1.0, 100),
                                    cv=10,
                                    n_jobs=-1,
                                    return_times=True)
        subject_learning_curve_plot = {
            'name': name,
            'train_sizes': train_sizes,
            'train_scores': train_scores.mean(axis=1),
            'test_scores': test_scores.mean(axis=1),
            'fit_times': fit_times.mean(axis=1)
        }
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        ########################
    # end for

    learning_curve_DICT = {}
    for learning_curve_i in subject_learning_curve_plot_LIST:
        learning_curve_DICT.update({learning_curve_i['name']: learning_curve_i})
    # end for
    learning_curve_DICT.update({'ignoreList': []})


    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML' +
        r'\\learning_curve_RF_no_cross_subject_mixture2class.pickle', 'wb')
    pickle.dump(learning_curve_DICT, fileD)

    Plot.draw_learning_curve(learning_curve_DICT,
                             'learning_curve_mixture2class_intraSubject_RF')
    # end RF_no_cross_subject_mixture2class


# SVM 分类器跨被试：四分类
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def SVM_cross_subject_4class():
    print('SVM 四分类器（上、下、左、右）跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    TRAIN_SIZES = np.linspace(0.1, 1, 100)
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
        print('SVM_cross_subject_4class：样本数量', sum(y == 0), sum(y == 1),
              sum(y == 2), sum(y == 3))

        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    for i in range(n_subjects):
        dataSet_test = dataSetMulti[i]  # 留下的用于测试的被试样本
        # 提取用于训练的被试集样本
        dataSet_train = np.empty((0, dataSet.shape[1]))
        for j in range(n_subjects):
            if j != i:
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

        t_sizes = dataSet_train_scalered.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        scores_train = []
        scores_test = []
        for size in t_sizes:
            # 训练模型
            clf = SGDClassifier(eta0=1e-6,
                                learning_rate='adaptive',
                                early_stopping=True)
            clf.fit(dataSet_train_scalered[:size, :-1],
                    dataSet_train_scalered[:size, -1])
            # 计算训练集评分
            y_train_pred = clf.predict(dataSet_train_scalered[:, :-1])
            acc_train = sum(
                y_train_pred == dataSet_train_scalered[:,
                                                       -1]) / len(y_train_pred)
            # print('训练集准确率：', acc_train)
            scores_train.append(acc_train)
            #### 这部分如果使用需要更改
            if False:  # 是否绘制 迭代次数-损失值 曲线
                #拆分训练集、测试集
                train_set, test_set = train_test_split(
                    dataSet_scalered, test_size=0.2)  #划分训练集、测试集
                X_train = train_set[:, :-1]  #训练集特征矩阵
                y_train = train_set[:, -1]  #训练集标签数组
                X_test = test_set[:, :-1]  #测试集特征矩阵
                y_test = test_set[:, -1]  #测试集标签数组

                ########################
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                # 训练模型
                clf = SGDClassifier(verbose=True,
                                    eta0=1e-2,
                                    learning_rate='adaptive',
                                    early_stopping=True)
                clf.fit(X_train, y_train)

                sys.stdout = old_stdout
                x_pred = clf.predict(X_train)
                acc_train = sum(x_pred == y_train) / len(y_train)
                print('绘图模型的训练集准确率：', acc_train)

                y_pred = clf.predict(X_test)
                acc_test = sum(y_test == y_pred) / len(y_pred)
                print('绘图模型的测试集准确率：', acc_test)

                loss_history = mystdout.getvalue()
                loss_list = []
                for line in loss_history.split('\n'):
                    if (len(line.split("loss: ")) == 1):
                        continue
                    loss_list.append(float(line.split("loss: ")[-1]))
                plt.figure()
                plt.plot(np.arange(len(loss_list)), loss_list)
                plt.xlabel("Time in epochs")
                plt.ylabel("Loss")
                plt.show()
                ########################
            # end if
            #### 这部分如果使用需要更改

            # 对测试样本集进行特征缩放
            X_test_scalered = (dataSet_test[:, :-1] - mean_) / scale_
            dataSet_test_scalered = np.c_[X_test_scalered,
                                          dataSet_test[:,
                                                       -1]]  #特征缩放后的测试样本集(X,y)
            dataSet_test_scalered[np.isnan(
                dataSet_test_scalered)] = 0  # 处理异常特征

            y_test_pred = clf.predict(dataSet_test_scalered[:, :-1])
            acc_test = sum(
                y_test_pred == dataSet_test_scalered[:, -1]) / len(y_test_pred)
            # print('测试集准确率：', acc_test)
            scores_test.append(acc_test)
        # end for
        subject_learning_curve_plot = {
            'name': NAME_LIST[i],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print('SVM_cross_subject_4class：%s : 测试集准确率  ' % NAME_LIST[i],
              scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML' +
        r'\\learning_curve_SVM_cross_subject_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'learning_curve_4classes_crossSubject_SVM')
    # end SVM_cross_subject_4class


# SVM 分类器跨被试：混合二分类（0-上下；1-左右）
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def SVM_cross_subject_mixture2class():
    print('SVM 混合二分类分类器（上下、左右）跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    TRAIN_SIZES = np.linspace(0.1, 1, 100)
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

        # 混合二分类：0-上下；1-左右
        y[y == 1] = 0
        y[y != 0] = 1

        # 构造样本集
        dataSet = np.c_[X, y]
        print('SVM_cross_subject_mixture2class：样本数量', sum(y == 0), sum(y == 1))
        dataSetMulti.append(dataSet)
    # end for

    # 留一个被试作为跨被试测试
    n_subjects = len(dataSetMulti)
    subject_learning_curve_plot_LIST = []
    for i in range(n_subjects):
        dataSet_test = dataSetMulti[i]  # 留下的用于测试的被试样本
        # 提取用于训练的被试集样本
        dataSet_train = np.empty((0, dataSet.shape[1]))
        for j in range(n_subjects):
            if j != i:
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

        t_sizes = dataSet_train_scalered.shape[0] * TRAIN_SIZES
        t_sizes = t_sizes.astype(int)
        scores_train = []
        scores_test = []
        for size in t_sizes:
            # 训练模型
            clf = SGDClassifier(eta0=1e-3,
                                learning_rate='adaptive',
                                early_stopping=True)
            clf.fit(dataSet_train_scalered[:size, :-1],
                    dataSet_train_scalered[:size, -1])

            # 计算训练集评分
            y_train_pred = clf.predict(dataSet_train_scalered[:, :-1])
            acc_train = sum(
                y_train_pred == dataSet_train_scalered[:,
                                                       -1]) / len(y_train_pred)
            # print('训练集准确率：', acc_train)
            scores_train.append(acc_train)
            #### 这部分如果使用需要更改
            if False:  # 是否绘制 迭代次数-损失值 曲线
                #拆分训练集、测试集
                train_set, test_set = train_test_split(
                    dataSet_scalered, test_size=0.2)  #划分训练集、测试集
                X_train = train_set[:, :-1]  #训练集特征矩阵
                y_train = train_set[:, -1]  #训练集标签数组
                X_test = test_set[:, :-1]  #测试集特征矩阵
                y_test = test_set[:, -1]  #测试集标签数组

                ########################
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                # 训练模型
                clf = SGDClassifier(verbose=True,
                                    eta0=1e-2,
                                    learning_rate='adaptive',
                                    early_stopping=True)
                clf.fit(X_train, y_train)

                sys.stdout = old_stdout
                x_pred = clf.predict(X_train)
                acc_train = sum(x_pred == y_train) / len(y_train)
                print('绘图模型的训练集准确率：', acc_train)

                y_pred = clf.predict(X_test)
                acc_test = sum(y_test == y_pred) / len(y_pred)
                print('绘图模型的测试集准确率：', acc_test)

                loss_history = mystdout.getvalue()
                loss_list = []
                for line in loss_history.split('\n'):
                    if (len(line.split("loss: ")) == 1):
                        continue
                    loss_list.append(float(line.split("loss: ")[-1]))
                plt.figure()
                plt.plot(np.arange(len(loss_list)), loss_list)
                plt.xlabel("Time in epochs")
                plt.ylabel("Loss")
                plt.show()
                ########################
            # end if
            #### 这部分如果使用需要更改

            # 对测试样本集进行特征缩放
            X_test_scalered = (dataSet_test[:, :-1] - mean_) / scale_
            dataSet_test_scalered = np.c_[X_test_scalered,
                                          dataSet_test[:,
                                                       -1]]  #特征缩放后的测试样本集(X,y)
            dataSet_test_scalered[np.isnan(
                dataSet_test_scalered)] = 0  # 处理异常特征

            y_test_pred = clf.predict(dataSet_test_scalered[:, :-1])
            acc_test = sum(
                y_test_pred == dataSet_test_scalered[:, -1]) / len(y_test_pred)
            # print('测试集准确率：', acc_test)
            scores_test.append(acc_test)
        # end for
        subject_learning_curve_plot = {
            'name': NAME_LIST[i],
            'train_sizes': t_sizes,
            'train_scores': np.array(scores_train),
            'test_scores': np.array(scores_test),
            'fit_times': []
        }
        print('SVM_cross_subject_4class：%s : 测试集准确率  ' % NAME_LIST[i],
              scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
    # end for

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML' +
        r'\\learning_curve_SVM_cross_subject_mixture2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'learning_curve_2classes_crossSubject_SVM')
    # end SVM_cross_subject_mixture2class


# LDA 分类器非跨被试：四分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features() 得到的数据
def LDA_no_cross_subject_4class():
    fileLog = open(
        os.getcwd() + r'\\..\model\\四分类\\ML\\LDA' +
        r'\\log_LDA_no_cross_subject_4class.txt.txt', 'w')
    fileLog.write('LDA 四分类器（上、下、左、右）非跨被试\n')
    print('LDA 四分类器（上、下、左、右）非跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    TRAIN_SIZES = np.linspace(0.05, 1, 50)
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for name in NAME_LIST:
        confusion_matrix_DICT_subjectI = {}
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = np.c_[X, y]
        print('LDA_no_cross_subject_4class：样本数量', sum(y == 0), sum(y == 1),
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
            print('LDA_no_cross_subject_4class：%s : 训练集准确率  %.3f%%\n' %
                  (name, acc_train))
            fileLog.write('LDA_no_cross_subject_4class：%s : 训练集准确率  %.3f%%\n' %
                          (name, acc_train))
            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('LDA_no_cross_subject_4class：%s : 测试集准确率  %.3f%%\n' %
                  (name, acc_test))
            fileLog.write('LDA_no_cross_subject_4class：%s : 测试集准确率  %.3f%%\n' %
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
        print('LDA_no_cross_subject_4class：%s : 测试集准确率  ' % name,
              scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write('LDA_no_cross_subject_4class：%s : 测试集准确率  %.3f%%\n' %
                      (name, scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML\\LDA' +
        r'\\learning_curve_LDA_no_cross_subject_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)

    fileD = open(
        path + r'\\..\model\\四分类\\ML\\LDA' +
        r'\\confusion_matrix_LDA_no_cross_subject_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()

    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'learning_curve_4classes_intraSubject_LDA')
    # end LDA_no_cross_subject_4class


# LDA 分类器非跨被试：上下二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features() 得到的数据
def LDA_no_cross_subject_UD2class():
    fileLog = open(
        os.getcwd() + r'\\..\model\\上下二分类\\ML\\LDA' +
        r'\\log_LDA_no_cross_subject_UD2class.txt.txt', 'w')
    fileLog.write('LDA 上下二分类器（上、下）非跨被试\n')
    print('LDA 上下二分类器（上、下）非跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    TRAIN_SIZES = np.linspace(0.1, 1, 25)
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for name in NAME_LIST:
        confusion_matrix_DICT_subjectI = {}
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X=X[:100]
        y=y[:100]    

        # 构造样本集
        dataSet = np.c_[X, y]
        print('LDA_no_cross_subject_UD2class：样本数量', sum(y == 0), sum(y == 1),
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
            print('LDA_no_cross_subject_UD2class：%s : 训练集准确率  %.3f%%\n' %
                  (name, acc_train))
            fileLog.write('LDA_no_cross_subject_UD2class：%s : 训练集准确率  %.3f%%\n' %
                          (name, acc_train))
            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('LDA_no_cross_subject_UD2class：%s : 测试集准确率  %.3f%%\n' %
                  (name, acc_test))
            fileLog.write('LDA_no_cross_subject_UD2class：%s : 测试集准确率  %.3f%%\n' %
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
        print('LDA_no_cross_subject_UD2class：%s : 测试集准确率  ' % name,
              scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write('LDA_no_cross_subject_UD2class：%s : 测试集准确率  %.3f%%\n' %
                      (name, scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\上下二分类\\ML\\LDA' +
        r'\\learning_curve_LDA_no_cross_subject_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)

    fileD = open(
        path + r'\\..\model\\上下二分类\\ML\\LDA' +
        r'\\confusion_matrix_LDA_no_cross_subject_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class('LDA_no_cross_subject')
    fileD = open(
        path + r'\\..\model\\上下二分类\\ML\\LDA' +
        r'\\learning_curve_LDA_no_cross_subject_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'learning_curve_UD2classes_intraSubject_LDA')
    # end LDA_no_cross_subject_UD2class


# LDA 分类器非跨被试：左右二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features() 得到的数据
def LDA_no_cross_subject_LR2class():
    fileLog = open(
        os.getcwd() + r'\\..\model\\左右二分类\\ML\\LDA' +
        r'\\log_LDA_no_cross_subject_UD2class.txt.txt', 'w')
    fileLog.write('LDA 左右二分类器（左、右）非跨被试\n')
    print('LDA 左右二分类器（左、右）非跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    TRAIN_SIZES = np.linspace(0.1, 1, 25)
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for name in NAME_LIST:
        confusion_matrix_DICT_subjectI = {}
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X=X[100:]
        y=y[100:]    
        y[y==2]=0
        y[y==3]=1    

        # 构造样本集
        dataSet = np.c_[X, y]
        print('LDA_no_cross_subject_LR2class：样本数量', sum(y == 0), sum(y == 1),
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
            print('LDA_no_cross_subject_LR2class：%s : 训练集准确率  %.3f%%\n' %
                  (name, acc_train))
            fileLog.write('LDA_no_cross_subject_LR2class：%s : 训练集准确率  %.3f%%\n' %
                          (name, acc_train))
            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('LDA_no_cross_subject_LR2class：%s : 测试集准确率  %.3f%%\n' %
                  (name, acc_test))
            fileLog.write('LDA_no_cross_subject_LR2class：%s : 测试集准确率  %.3f%%\n' %
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
        print('LDA_no_cross_subject_LR2class：%s : 测试集准确率  ' % name,
              scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write('LDA_no_cross_subject_LR2class：%s : 测试集准确率  %.3f%%\n' %
                      (name, scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\左右二分类\\ML\\LDA' +
        r'\\learning_curve_LDA_no_cross_subject_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)

    fileD = open(
        path + r'\\..\model\\左右二分类\\ML\\LDA' +
        r'\\confusion_matrix_LDA_no_cross_subject_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class('LDA_no_cross_subject')
    fileD = open(
        path + r'\\..\model\\左右二分类\\ML\\LDA' +
        r'\\learning_curve_LDA_no_cross_subject_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'learning_curve_LR2classes_intraSubject_LDA')
    # end LDA_no_cross_subject_LR2class


# 决策树 分类器非跨被试：四分类
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def DT_no_cross_subject_4class():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    for name in NAME_LIST:
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = np.c_[X, y]
        print('DT_no_cross_subject_4class：样本数量', sum(y == 0),
              sum(y == 1))

        # 样本集洗牌
        shuffle_index = np.random.permutation(dataSet.shape[0])
        dataSet = dataSet[shuffle_index]

        # 特征缩放
        scaler = StandardScaler()
        X_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)

        dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征

        if False:  # 是否绘制 迭代次数-损失值 曲线
            #拆分训练集、测试集
            train_set, test_set = train_test_split(dataSet_scalered,
                                                   test_size=0.2)  #划分训练集、测试集
            X_train = train_set[:, :-1]  #训练集特征矩阵
            y_train = train_set[:, -1]  #训练集标签数组
            X_test = test_set[:, :-1]  #测试集特征矩阵
            y_test = test_set[:, -1]  #测试集标签数组

            ########################
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            # 训练模型
            clf = SGDClassifier(verbose=True, early_stopping=True)
            clf.fit(X_train, y_train)

            sys.stdout = old_stdout
            x_pred = clf.predict(X_train)
            acc_train = sum(x_pred == y_train) / len(y_train)
            print('绘图模型的训练集准确率：', acc_train)

            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('绘图模型的测试集准确率：', acc_test)

            loss_history = mystdout.getvalue()
            loss_list = []
            for line in loss_history.split('\n'):
                if (len(line.split("loss: ")) == 1):
                    continue
                loss_list.append(float(line.split("loss: ")[-1]))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), loss_list)
            plt.xlabel("Time in epochs")
            plt.ylabel("Loss")
            plt.show()
            ########################
        # end if

        ########################
        # 交叉验证
        clf = tree.DecisionTreeClassifier(criterion='gini')
        scores = cross_val_score(clf,
                                 dataSet_scalered[:, :-1],
                                 dataSet_scalered[:, -1],
                                 cv=10)

        print('DT_no_cross_subject_4class：%s : 交叉验证准确率  ' % name,
              scores)
        print('DT_no_cross_subject_4class：%s : 交叉验证准确率平均值  ' % name,
              np.average(scores))
        ########################

        ########################
        # 计算学习曲线
        clf = tree.DecisionTreeClassifier(criterion='gini')
        # train_sizes, train_scores, test_scores=myLearning_curve(clf,dataSet_scalered[:, :-1],dataSet_scalered[:, -1],np.linspace(0.2, 1.0, 10),5)

        train_sizes, train_scores, test_scores,fit_times,score_times =\
                        learning_curve(estimator= clf,
                                    X=dataSet_scalered[:, :-1],
                                    y=dataSet_scalered[:, -1],
                                    train_sizes=np.linspace(0.1, 1.0, 100),
                                    cv=10,
                                    n_jobs=-1,
                                    return_times=True)
        subject_learning_curve_plot = {
            'name': name,
            'train_sizes': train_sizes,
            'train_scores': train_scores.mean(axis=1),
            'test_scores': test_scores.mean(axis=1),
            'fit_times': fit_times.mean(axis=1)
        }
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        ########################
    # end for

    learning_curve_DICT = {}
    for learning_curve_i in subject_learning_curve_plot_LIST:
        learning_curve_DICT.update({learning_curve_i['name']: learning_curve_i})
    # end for
    learning_curve_DICT.update({'ignoreList': []})


    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML' +
        r'\\learning_curve_DT_no_cross_subject_4class.pickle', 'wb')
    pickle.dump(learning_curve_DICT, fileD)

    Plot.draw_learning_curve(learning_curve_DICT,
                             'learning_curve_4class_intraSubject_DT')
    # end DT_no_cross_subject_4class

# 决策树 分类器非跨被试：上下二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def DT_no_cross_subject_UD2class():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    for name in NAME_LIST:
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X=X[:100]
        y=y[:100]    
        
        # 构造样本集
        dataSet = np.c_[X, y]
        print('DT_no_cross_subject_UD2class：样本数量', sum(y == 0),
              sum(y == 1))

        # 样本集洗牌
        shuffle_index = np.random.permutation(dataSet.shape[0])
        dataSet = dataSet[shuffle_index]

        # 特征缩放
        scaler = StandardScaler()
        X_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)

        dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征

        if False:  # 是否绘制 迭代次数-损失值 曲线
            #拆分训练集、测试集
            train_set, test_set = train_test_split(dataSet_scalered,
                                                   test_size=0.2)  #划分训练集、测试集
            X_train = train_set[:, :-1]  #训练集特征矩阵
            y_train = train_set[:, -1]  #训练集标签数组
            X_test = test_set[:, :-1]  #测试集特征矩阵
            y_test = test_set[:, -1]  #测试集标签数组

            ########################
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            # 训练模型
            clf = SGDClassifier(verbose=True, early_stopping=True)
            clf.fit(X_train, y_train)

            sys.stdout = old_stdout
            x_pred = clf.predict(X_train)
            acc_train = sum(x_pred == y_train) / len(y_train)
            print('绘图模型的训练集准确率：', acc_train)

            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('绘图模型的测试集准确率：', acc_test)

            loss_history = mystdout.getvalue()
            loss_list = []
            for line in loss_history.split('\n'):
                if (len(line.split("loss: ")) == 1):
                    continue
                loss_list.append(float(line.split("loss: ")[-1]))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), loss_list)
            plt.xlabel("Time in epochs")
            plt.ylabel("Loss")
            plt.show()
            ########################
        # end if

        ########################
        # 交叉验证
        clf = tree.DecisionTreeClassifier(criterion='gini')
        scores = cross_val_score(clf,
                                 dataSet_scalered[:, :-1],
                                 dataSet_scalered[:, -1],
                                 cv=10)

        print('DT_no_cross_subject_UD2class：%s : 交叉验证准确率  ' % name,
              scores)
        print('DT_no_cross_subject_UD2class：%s : 交叉验证准确率平均值  ' % name,
              np.average(scores))
        ########################

        ########################
        # 计算学习曲线
        clf = tree.DecisionTreeClassifier(criterion='gini')
        # train_sizes, train_scores, test_scores=myLearning_curve(clf,dataSet_scalered[:, :-1],dataSet_scalered[:, -1],np.linspace(0.2, 1.0, 10),5)

        train_sizes, train_scores, test_scores,fit_times,score_times =\
                        learning_curve(estimator= clf,
                                    X=dataSet_scalered[:, :-1],
                                    y=dataSet_scalered[:, -1],
                                    train_sizes=np.linspace(0.1, 1.0, 100),
                                    cv=10,
                                    n_jobs=-1,
                                    return_times=True)
        subject_learning_curve_plot = {
            'name': name,
            'train_sizes': train_sizes,
            'train_scores': train_scores.mean(axis=1),
            'test_scores': test_scores.mean(axis=1),
            'fit_times': fit_times.mean(axis=1)
        }
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        ########################
    # end for

    learning_curve_DICT = {}
    for learning_curve_i in subject_learning_curve_plot_LIST:
        learning_curve_DICT.update({learning_curve_i['name']: learning_curve_i})
    # end for
    learning_curve_DICT.update({'ignoreList': []})


    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML' +
        r'\\learning_curve_DT_no_cross_subject_UD2class.pickle', 'wb')
    pickle.dump(learning_curve_DICT, fileD)

    Plot.draw_learning_curve(learning_curve_DICT,
                             'learning_curve_UD2class_intraSubject_DT')
    # end DT_no_cross_subject_UD2class

# 决策树 分类器非跨被试：左右二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def DT_no_cross_subject_LR2class():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    for name in NAME_LIST:
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X=X[100:]
        y=y[100:]    
        y[y==2]=0
        y[y==3]=1    
        
        # 构造样本集
        dataSet = np.c_[X, y]
        print('DT_no_cross_subject_LR2class：样本数量', sum(y == 0),
              sum(y == 1))

        # 样本集洗牌
        shuffle_index = np.random.permutation(dataSet.shape[0])
        dataSet = dataSet[shuffle_index]

        # 特征缩放
        scaler = StandardScaler()
        X_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)

        dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征

        if False:  # 是否绘制 迭代次数-损失值 曲线
            #拆分训练集、测试集
            train_set, test_set = train_test_split(dataSet_scalered,
                                                   test_size=0.2)  #划分训练集、测试集
            X_train = train_set[:, :-1]  #训练集特征矩阵
            y_train = train_set[:, -1]  #训练集标签数组
            X_test = test_set[:, :-1]  #测试集特征矩阵
            y_test = test_set[:, -1]  #测试集标签数组

            ########################
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            # 训练模型
            clf = SGDClassifier(verbose=True, early_stopping=True)
            clf.fit(X_train, y_train)

            sys.stdout = old_stdout
            x_pred = clf.predict(X_train)
            acc_train = sum(x_pred == y_train) / len(y_train)
            print('绘图模型的训练集准确率：', acc_train)

            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('绘图模型的测试集准确率：', acc_test)

            loss_history = mystdout.getvalue()
            loss_list = []
            for line in loss_history.split('\n'):
                if (len(line.split("loss: ")) == 1):
                    continue
                loss_list.append(float(line.split("loss: ")[-1]))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), loss_list)
            plt.xlabel("Time in epochs")
            plt.ylabel("Loss")
            plt.show()
            ########################
        # end if

        ########################
        # 交叉验证
        clf = tree.DecisionTreeClassifier(criterion='gini')
        scores = cross_val_score(clf,
                                 dataSet_scalered[:, :-1],
                                 dataSet_scalered[:, -1],
                                 cv=10)

        print('DT_no_cross_subject_LR2class：%s : 交叉验证准确率  ' % name,
              scores)
        print('DT_no_cross_subject_LR2class：%s : 交叉验证准确率平均值  ' % name,
              np.average(scores))
        ########################

        ########################
        # 计算学习曲线
        clf = tree.DecisionTreeClassifier(criterion='gini')
        # train_sizes, train_scores, test_scores=myLearning_curve(clf,dataSet_scalered[:, :-1],dataSet_scalered[:, -1],np.linspace(0.2, 1.0, 10),5)

        train_sizes, train_scores, test_scores,fit_times,score_times =\
                        learning_curve(estimator= clf,
                                    X=dataSet_scalered[:, :-1],
                                    y=dataSet_scalered[:, -1],
                                    train_sizes=np.linspace(0.1, 1.0, 100),
                                    cv=10,
                                    n_jobs=-1,
                                    return_times=True)
        subject_learning_curve_plot = {
            'name': name,
            'train_sizes': train_sizes,
            'train_scores': train_scores.mean(axis=1),
            'test_scores': test_scores.mean(axis=1),
            'fit_times': fit_times.mean(axis=1)
        }
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        ########################
    # end for

    learning_curve_DICT = {}
    for learning_curve_i in subject_learning_curve_plot_LIST:
        learning_curve_DICT.update({learning_curve_i['name']: learning_curve_i})
    # end for
    learning_curve_DICT.update({'ignoreList': []})


    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML' +
        r'\\learning_curve_DT_no_cross_subject_LR2class.pickle', 'wb')
    pickle.dump(learning_curve_DICT, fileD)

    Plot.draw_learning_curve(learning_curve_DICT,
                             'learning_curve_LR2class_intraSubject_DT')
    # end DT_no_cross_subject_LR2class


# 随机森林 分类器非跨被试：四分类
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def RF_no_cross_subject_4class():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    for name in NAME_LIST:
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = np.c_[X, y]
        print('RF_no_cross_subject_4class：样本数量', sum(y == 0),
              sum(y == 1))

        # 样本集洗牌
        shuffle_index = np.random.permutation(dataSet.shape[0])
        dataSet = dataSet[shuffle_index]

        # 特征缩放
        scaler = StandardScaler()
        X_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)

        dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征

        if False:  # 是否绘制 迭代次数-损失值 曲线
            #拆分训练集、测试集
            train_set, test_set = train_test_split(dataSet_scalered,
                                                   test_size=0.2)  #划分训练集、测试集
            X_train = train_set[:, :-1]  #训练集特征矩阵
            y_train = train_set[:, -1]  #训练集标签数组
            X_test = test_set[:, :-1]  #测试集特征矩阵
            y_test = test_set[:, -1]  #测试集标签数组

            ########################
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            # 训练模型
            clf = SGDClassifier(verbose=True, early_stopping=True)
            clf.fit(X_train, y_train)

            sys.stdout = old_stdout
            x_pred = clf.predict(X_train)
            acc_train = sum(x_pred == y_train) / len(y_train)
            print('绘图模型的训练集准确率：', acc_train)

            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('绘图模型的测试集准确率：', acc_test)

            loss_history = mystdout.getvalue()
            loss_list = []
            for line in loss_history.split('\n'):
                if (len(line.split("loss: ")) == 1):
                    continue
                loss_list.append(float(line.split("loss: ")[-1]))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), loss_list)
            plt.xlabel("Time in epochs")
            plt.ylabel("Loss")
            plt.show()
            ########################
        # end if

        ########################
        # 交叉验证
        clf = RandomForestClassifier(n_estimators=10, criterion='gini') 
        scores = cross_val_score(clf,
                                 dataSet_scalered[:, :-1],
                                 dataSet_scalered[:, -1],
                                 cv=10)

        print('RF_no_cross_subject_4class：%s : 交叉验证准确率  ' % name,
              scores)
        print('RF_no_cross_subject_4class：%s : 交叉验证准确率平均值  ' % name,
              np.average(scores))
        ########################

        ########################
        # 计算学习曲线
        clf = RandomForestClassifier(n_estimators=10, criterion='gini') 
        # train_sizes, train_scores, test_scores=myLearning_curve(clf,dataSet_scalered[:, :-1],dataSet_scalered[:, -1],np.linspace(0.2, 1.0, 10),5)

        train_sizes, train_scores, test_scores,fit_times,score_times =\
                        learning_curve(estimator= clf,
                                    X=dataSet_scalered[:, :-1],
                                    y=dataSet_scalered[:, -1],
                                    train_sizes=np.linspace(0.1, 1.0, 100),
                                    cv=10,
                                    n_jobs=-1,
                                    return_times=True)
        subject_learning_curve_plot = {
            'name': name,
            'train_sizes': train_sizes,
            'train_scores': train_scores.mean(axis=1),
            'test_scores': test_scores.mean(axis=1),
            'fit_times': fit_times.mean(axis=1)
        }
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        ########################
    # end for

    learning_curve_DICT = {}
    for learning_curve_i in subject_learning_curve_plot_LIST:
        learning_curve_DICT.update({learning_curve_i['name']: learning_curve_i})
    # end for
    learning_curve_DICT.update({'ignoreList': []})


    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML' +
        r'\\learning_curve_RF_no_cross_subject_4class.pickle', 'wb')
    pickle.dump(learning_curve_DICT, fileD)

    Plot.draw_learning_curve(learning_curve_DICT,
                             'learning_curve_4class_intraSubject_RF')
    # end RF_no_cross_subject_4class

# 随机森林 分类器非跨被试：上下二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def RF_no_cross_subject_UD2class():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    for name in NAME_LIST:
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X=X[:100]
        y=y[:100]    
        
        # 构造样本集
        dataSet = np.c_[X, y]
        print('RF_no_cross_subject_UD2class：样本数量', sum(y == 0),
              sum(y == 1))

        # 样本集洗牌
        shuffle_index = np.random.permutation(dataSet.shape[0])
        dataSet = dataSet[shuffle_index]

        # 特征缩放
        scaler = StandardScaler()
        X_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)

        dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征

        if False:  # 是否绘制 迭代次数-损失值 曲线
            #拆分训练集、测试集
            train_set, test_set = train_test_split(dataSet_scalered,
                                                   test_size=0.2)  #划分训练集、测试集
            X_train = train_set[:, :-1]  #训练集特征矩阵
            y_train = train_set[:, -1]  #训练集标签数组
            X_test = test_set[:, :-1]  #测试集特征矩阵
            y_test = test_set[:, -1]  #测试集标签数组

            ########################
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            # 训练模型
            clf = SGDClassifier(verbose=True, early_stopping=True)
            clf.fit(X_train, y_train)

            sys.stdout = old_stdout
            x_pred = clf.predict(X_train)
            acc_train = sum(x_pred == y_train) / len(y_train)
            print('绘图模型的训练集准确率：', acc_train)

            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('绘图模型的测试集准确率：', acc_test)

            loss_history = mystdout.getvalue()
            loss_list = []
            for line in loss_history.split('\n'):
                if (len(line.split("loss: ")) == 1):
                    continue
                loss_list.append(float(line.split("loss: ")[-1]))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), loss_list)
            plt.xlabel("Time in epochs")
            plt.ylabel("Loss")
            plt.show()
            ########################
        # end if

        ########################
        # 交叉验证
        clf = RandomForestClassifier(n_estimators=10, criterion='gini') 
        scores = cross_val_score(clf,
                                 dataSet_scalered[:, :-1],
                                 dataSet_scalered[:, -1],
                                 cv=10)

        print('RF_no_cross_subject_UD2class：%s : 交叉验证准确率  ' % name,
              scores)
        print('RF_no_cross_subject_UD2class：%s : 交叉验证准确率平均值  ' % name,
              np.average(scores))
        ########################

        ########################
        # 计算学习曲线
        clf = RandomForestClassifier(n_estimators=10, criterion='gini') 
        # train_sizes, train_scores, test_scores=myLearning_curve(clf,dataSet_scalered[:, :-1],dataSet_scalered[:, -1],np.linspace(0.2, 1.0, 10),5)

        train_sizes, train_scores, test_scores,fit_times,score_times =\
                        learning_curve(estimator= clf,
                                    X=dataSet_scalered[:, :-1],
                                    y=dataSet_scalered[:, -1],
                                    train_sizes=np.linspace(0.1, 1.0, 100),
                                    cv=10,
                                    n_jobs=-1,
                                    return_times=True)
        subject_learning_curve_plot = {
            'name': name,
            'train_sizes': train_sizes,
            'train_scores': train_scores.mean(axis=1),
            'test_scores': test_scores.mean(axis=1),
            'fit_times': fit_times.mean(axis=1)
        }
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        ########################
    # end for

    learning_curve_DICT = {}
    for learning_curve_i in subject_learning_curve_plot_LIST:
        learning_curve_DICT.update({learning_curve_i['name']: learning_curve_i})
    # end for
    learning_curve_DICT.update({'ignoreList': []})


    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML' +
        r'\\learning_curve_RF_no_cross_subject_UD2class.pickle', 'wb')
    pickle.dump(learning_curve_DICT, fileD)

    Plot.draw_learning_curve(learning_curve_DICT,
                             'learning_curve_UD2class_intraSubject_RF')
    # end RF_no_cross_subject_UD2class

# 随机森林 分类器非跨被试：左右二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_single_move_artificial_features() 得到的数据
def RF_no_cross_subject_LR2class():
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    for name in NAME_LIST:
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\四分类' +
            '\\single_move_motion_start_motion_end_beginTimeFile.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X=X[100:]
        y=y[100:]    
        y[y==2]=0
        y[y==3]=1    
        
        # 构造样本集
        dataSet = np.c_[X, y]
        print('RF_no_cross_subject_LR2class：样本数量', sum(y == 0),
              sum(y == 1))

        # 样本集洗牌
        shuffle_index = np.random.permutation(dataSet.shape[0])
        dataSet = dataSet[shuffle_index]

        # 特征缩放
        scaler = StandardScaler()
        X_scalered = scaler.fit_transform(dataSet[:, :-1])
        dataSet_scalered = np.c_[X_scalered, dataSet[:, -1]]  #特征缩放后的样本集(X,y)

        dataSet_scalered[np.isnan(dataSet_scalered)] = 0  # 处理异常特征

        if False:  # 是否绘制 迭代次数-损失值 曲线
            #拆分训练集、测试集
            train_set, test_set = train_test_split(dataSet_scalered,
                                                   test_size=0.2)  #划分训练集、测试集
            X_train = train_set[:, :-1]  #训练集特征矩阵
            y_train = train_set[:, -1]  #训练集标签数组
            X_test = test_set[:, :-1]  #测试集特征矩阵
            y_test = test_set[:, -1]  #测试集标签数组

            ########################
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            # 训练模型
            clf = SGDClassifier(verbose=True, early_stopping=True)
            clf.fit(X_train, y_train)

            sys.stdout = old_stdout
            x_pred = clf.predict(X_train)
            acc_train = sum(x_pred == y_train) / len(y_train)
            print('绘图模型的训练集准确率：', acc_train)

            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('绘图模型的测试集准确率：', acc_test)

            loss_history = mystdout.getvalue()
            loss_list = []
            for line in loss_history.split('\n'):
                if (len(line.split("loss: ")) == 1):
                    continue
                loss_list.append(float(line.split("loss: ")[-1]))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), loss_list)
            plt.xlabel("Time in epochs")
            plt.ylabel("Loss")
            plt.show()
            ########################
        # end if

        ########################
        # 交叉验证
        clf = RandomForestClassifier(n_estimators=10, criterion='gini') 
        scores = cross_val_score(clf,
                                 dataSet_scalered[:, :-1],
                                 dataSet_scalered[:, -1],
                                 cv=10)

        print('RF_no_cross_subject_LR2class：%s : 交叉验证准确率  ' % name,
              scores)
        print('RF_no_cross_subject_LR2class：%s : 交叉验证准确率平均值  ' % name,
              np.average(scores))
        ########################

        ########################
        # 计算学习曲线
        clf = RandomForestClassifier(n_estimators=10, criterion='gini') 
        # train_sizes, train_scores, test_scores=myLearning_curve(clf,dataSet_scalered[:, :-1],dataSet_scalered[:, -1],np.linspace(0.2, 1.0, 10),5)

        train_sizes, train_scores, test_scores,fit_times,score_times =\
                        learning_curve(estimator= clf,
                                    X=dataSet_scalered[:, :-1],
                                    y=dataSet_scalered[:, -1],
                                    train_sizes=np.linspace(0.1, 1.0, 100),
                                    cv=10,
                                    n_jobs=-1,
                                    return_times=True)
        subject_learning_curve_plot = {
            'name': name,
            'train_sizes': train_sizes,
            'train_scores': train_scores.mean(axis=1),
            'test_scores': test_scores.mean(axis=1),
            'fit_times': fit_times.mean(axis=1)
        }
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        ########################
    # end for

    learning_curve_DICT = {}
    for learning_curve_i in subject_learning_curve_plot_LIST:
        learning_curve_DICT.update({learning_curve_i['name']: learning_curve_i})
    # end for
    learning_curve_DICT.update({'ignoreList': []})


    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\四分类\\ML' +
        r'\\learning_curve_RF_no_cross_subject_LR2class.pickle', 'wb')
    pickle.dump(learning_curve_DICT, fileD)

    Plot.draw_learning_curve(learning_curve_DICT,
                             'learning_curve_LR2class_intraSubject_RF')
    # end RF_no_cross_subject_LR2class

# SVM 分类器非跨被试：四分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features() 得到的数据
def SVM_no_cross_subject_4class_multiTask():
    fileLog = open(
        os.getcwd() + r'\\..\model\\多任务\\四分类\\SVM' +
        r'\\log_SVM_no_cross_subject_UD2class.txt.txt', 'w')
    fileLog.write('SVM 四分类器（左、右）非跨被试\n')
    print('SVM 四分类器（左、右）非跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    TRAIN_SIZES = np.linspace(0.1, 1, 25)
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for name in NAME_LIST:
        confusion_matrix_DICT_subjectI = {}
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\多任务' +
            '\\multi_task_motion_start_motion_end.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 构造样本集
        dataSet = np.c_[X, y]
        print('SVM_no_cross_subject_4class_multiTask：样本数量', sum(y == 0), sum(y == 1),
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
            print('SVM_no_cross_subject_4class_multiTask：%s : 训练集准确率  %.3f%%\n' %
                  (name, acc_train))
            fileLog.write('SVM_no_cross_subject_4class_multiTask：%s : 训练集准确率  %.3f%%\n' %
                          (name, acc_train))
            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('SVM_no_cross_subject_4class_multiTask：%s : 测试集准确率  %.3f%%\n' %
                  (name, acc_test))
            fileLog.write('SVM_no_cross_subject_4class_multiTask：%s : 测试集准确率  %.3f%%\n' %
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
        print('SVM_no_cross_subject_4class_multiTask：%s : 测试集准确率  ' % name,
              scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write('SVM_no_cross_subject_4class_multiTask：%s : 测试集准确率  %.3f%%\n' %
                      (name, scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\多任务\\四分类\\SVM' +
        r'\\learning_curve_SVM_no_cross_subject_4class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)

    fileD = open(
        path + r'\\..\model\\多任务\\四分类\\SVM' +
        r'\\confusion_matrix_SVM_no_cross_subject_4class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_4class_multiTask('SVM_no_cross_subject')
    fileD = open(
        path + r'\\..\model\\多任务\\四分类\\SVM' +
        r'\\learning_curve_SVM_no_cross_subject_4class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'learning_curve_4classes_intraSubject_SVM')
    # end SVM_no_cross_subject_4class_multiTask

# SVM 分类器非跨被试：上下二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features() 得到的数据
def SVM_no_cross_subject_UD2class_multiTask():
    fileLog = open(
        os.getcwd() + r'\\..\model\\多任务\\上下二分类\\SVM' +
        r'\\log_SVM_no_cross_subject_UD2class.txt.txt', 'w')
    fileLog.write('SVM 上下二分类器（左、右）非跨被试\n')
    print('SVM 上下二分类器（左、右）非跨被试')
    NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']
    TRAIN_SIZES = np.linspace(0.1, 1, 25)
    path = os.getcwd()
    subject_learning_curve_plot_LIST = []
    confusion_matrix_DICT = {}
    for name in NAME_LIST:
        confusion_matrix_DICT_subjectI = {}
        #提取数据
        fileD = open(
            path + r'\\..\\data\\' + name + '\\多任务' +
            '\\multi_task_motion_start_motion_end.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X=X[:100]
        y=y[:100]      

        # 构造样本集
        dataSet = np.c_[X, y]
        print('SVM_no_cross_subject_UD2class_multiTask：样本数量', sum(y == 0), sum(y == 1),
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
            print('SVM_no_cross_subject_UD2class_multiTask：%s : 训练集准确率  %.3f%%\n' %
                  (name, acc_train))
            fileLog.write('SVM_no_cross_subject_UD2class_multiTask：%s : 训练集准确率  %.3f%%\n' %
                          (name, acc_train))
            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('SVM_no_cross_subject_UD2class_multiTask：%s : 测试集准确率  %.3f%%\n' %
                  (name, acc_test))
            fileLog.write('SVM_no_cross_subject_UD2class_multiTask：%s : 测试集准确率  %.3f%%\n' %
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
        print('SVM_no_cross_subject_UD2class_multiTask：%s : 测试集准确率  ' % name,
              scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write('SVM_no_cross_subject_UD2class_multiTask：%s : 测试集准确率  %.3f%%\n' %
                      (name, scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\多任务\\上下二分类\\SVM' +
        r'\\learning_curve_SVM_no_cross_subject_UD2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)

    fileD = open(
        path + r'\\..\model\\多任务\\上下二分类\\SVM' +
        r'\\confusion_matrix_SVM_no_cross_subject_UD2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_UD2class_multiTask('SVM_no_cross_subject')
    fileD = open(
        path + r'\\..\model\\多任务\\上下二分类\\SVM' +
        r'\\learning_curve_SVM_no_cross_subject_UD2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'learning_curve_UD2classes_intraSubject_SVM')
    # end SVM_no_cross_subject_UD2class_multiTask

# SVM 分类器非跨被试：左右二分类
# 数据格式：
#   人工提取特征，使用 get_EEG_features() 得到的数据
def SVM_no_cross_subject_LR2class_multiTask():
    fileLog = open(
        os.getcwd() + r'\\..\model\\多任务\\左右二分类\\SVM' +
        r'\\log_SVM_no_cross_subject_UD2class.txt.txt', 'w')
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
        fileD = open(
            path + r'\\..\\data\\' + name + '\\多任务' +
            '\\multi_task_motion_start_motion_end.pickle', 'rb')
        X, y = pickle.load(fileD)
        fileD.close()

        # 提取二分类数据
        X=X[100:]
        y=y[100:]    
        y[y==2]=0
        y[y==3]=1    

        # 构造样本集
        dataSet = np.c_[X, y]
        print('SVM_no_cross_subject_LR2class_multiTask：样本数量', sum(y == 0), sum(y == 1),
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
            print('SVM_no_cross_subject_LR2class_multiTask：%s : 训练集准确率  %.3f%%\n' %
                  (name, acc_train))
            fileLog.write('SVM_no_cross_subject_LR2class_multiTask：%s : 训练集准确率  %.3f%%\n' %
                          (name, acc_train))
            y_pred = clf.predict(X_test)
            acc_test = sum(y_test == y_pred) / len(y_pred)
            print('SVM_no_cross_subject_LR2class_multiTask：%s : 测试集准确率  %.3f%%\n' %
                  (name, acc_test))
            fileLog.write('SVM_no_cross_subject_LR2class_multiTask：%s : 测试集准确率  %.3f%%\n' %
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
        print('SVM_no_cross_subject_LR2class_multiTask：%s : 测试集准确率  ' % name,
              scores_test[-1])
        subject_learning_curve_plot_LIST.append(subject_learning_curve_plot)
        fileLog.write('SVM_no_cross_subject_LR2class_multiTask：%s : 测试集准确率  %.3f%%\n' %
                      (name, scores_test[-1]))
    # end for
    fileLog.close()

    path = os.getcwd()
    fileD = open(
        path + r'\\..\model\\多任务\\左右二分类\\SVM' +
        r'\\learning_curve_SVM_no_cross_subject_LR2class.pickle', 'wb')
    pickle.dump(subject_learning_curve_plot_LIST, fileD)

    fileD = open(
        path + r'\\..\model\\多任务\\左右二分类\\SVM' +
        r'\\confusion_matrix_SVM_no_cross_subject_LR2class.pickle', 'wb')
    pickle.dump(confusion_matrix_DICT, fileD)
    fileD.close()
    Plot.myLearning_curve_LIST2DICT_LR2class_multiTask('SVM_no_cross_subject')
    fileD = open(
        path + r'\\..\model\\多任务\\左右二分类\\SVM' +
        r'\\learning_curve_SVM_no_cross_subject_LR2class.pickle', 'rb')
    subject_learning_curve_plot_LIST = pickle.load(fileD)
    fileD.close()
    Plot.draw_learning_curve(subject_learning_curve_plot_LIST,
                             'learning_curve_LR2classes_intraSubject_SVM')
    # end SVM_no_cross_subject_LR2class_multiTask

# Find_SVM_no_cross_subject_2class()
# SVM_no_cross_subject_mixture2class()
# SVM_cross_subject_4class()
# SVM_cross_subject_mixture2class()

# DT_no_cross_subject_mixture2class()
# DT_no_cross_subject_4class()
# DT_no_cross_subject_UD2class()
# DT_no_cross_subject_LR2class()

# RF_no_cross_subject_mixture2class()
# RF_no_cross_subject_4class()
# RF_no_cross_subject_UD2class()
# RF_no_cross_subject_LR2class()

# 下面是已经训练好的，无需再运行
# SVM_no_cross_subject_4class()
# LDA_no_cross_subject_4class()

# SVM_no_cross_subject_UD2class()
# SVM_no_cross_subject_LR2class()

# LDA_no_cross_subject_UD2class()
# LDA_no_cross_subject_LR2class()

# SVM_no_cross_subject_4class_multiTask()
# SVM_no_cross_subject_UD2class_multiTask()
# SVM_no_cross_subject_LR2class_multiTask()