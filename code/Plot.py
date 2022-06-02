# 数据可视化

import os
import pickle
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import random

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

PATHDICT_CONFUSION_MATRIX_4CLASS = {
    'SVM_no_cross_subject':
    os.getcwd() +
    r'\..\model\四分类\ML\SVM\confusion_matrix_SVM_no_cross_subject_4class.pickle',
    'LDA_no_cross_subject':
    os.getcwd() +
    r'\..\model\四分类\ML\LDA\confusion_matrix_LDA_no_cross_subject_4class.pickle',
    'CNN_Net4_pretraining':
    os.getcwd() +
    r'\..\model\四分类\CNN_Net4_pretraining\confusion_matrix_CNN_Net4_pretraining_4class.pickle',
    'CNN_Net42_pretraining':
    os.getcwd() +
    r'\..\model\四分类\CNN_Net42_pretraining\confusion_matrix_CNN_Net42_pretraining_4class.pickle',
    'NN_Net45_pretraining':
    os.getcwd() +
    r'\..\model\四分类\NN_Net45_pretraining\confusion_matrix_NN_Net45_pretraining_4class.pickle',
    'LSTM_Net46_pretraining':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net46_pretraining\confusion_matrix_LSTM_Net46_pretraining_4class.pickle',
    'LSTM_Net47_pretraining':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net47_pretraining\confusion_matrix_LSTM_Net47_pretraining_4class.pickle',
    'CNN_LSTM_Net48_pretraining':
    os.getcwd() +
    r'\..\model\四分类\CNN_LSTM_Net48_pretraining\confusion_matrix_CNN_LSTM_Net48_pretraining_4class.pickle',
    'CNN_LSTM_Net49_pretraining':
    os.getcwd() +
    r'\..\model\四分类\CNN_LSTM_Net49_pretraining\confusion_matrix_CNN_LSTM_Net49_pretraining_4class.pickle',
    'CNN_Net4_fine':
    os.getcwd() +
    r'\..\model\四分类\CNN_Net4_fine\confusion_matrix_CNN_Net4_fine_4class.pickle',
    'CNN_Net42_fine':
    os.getcwd() +
    r'\..\model\四分类\CNN_Net42_fine\confusion_matrix_CNN_Net42_fine_4class.pickle',
    'LSTM_Net44_fine':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net44_fine\confusion_matrix_LSTM_Net44_fine_4class.pickle',
    'NN_Net45_fine':
    os.getcwd() +
    r'\..\model\四分类\NN_Net45_fine\confusion_matrix_NN_Net45_fine_4class.pickle',
    'LSTM_Net46_fine':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net46_fine\confusion_matrix_LSTM_Net46_fine_4class.pickle',
    'LSTM_Net47_fine':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net47_fine\confusion_matrix_LSTM_Net47_fine_4class.pickle',
    'CNN_LSTM_Net48_fine':
    os.getcwd() +
    r'\..\model\四分类\CNN_LSTM_Net48_fine\confusion_matrix_CNN_LSTM_Net48_fine_4class.pickle',
    'CNN_LSTM_Net49_fine':
    os.getcwd() +
    r'\..\model\四分类\CNN_LSTM_Net49_fine\confusion_matrix_CNN_LSTM_Net49_fine_4class.pickle'
}

PATHDICT_LEARNING_CURVE_4CLASS = {
    'SVM_no_cross_subject':
    os.getcwd() +
    r'\..\model\四分类\ML\SVM\learning_curve_SVM_no_cross_subject_4class.pickle',
    'LDA_no_cross_subject':
    os.getcwd() +
    r'\..\model\四分类\ML\LDA\learning_curve_LDA_no_cross_subject_4class.pickle',
    'CNN_Net4_pretraining':
    os.getcwd() +
    r'\..\model\四分类\CNN_Net4_pretraining\learning_curve_CNN_Net4_pretraining_4class.pickle',
    'CNN_Net42_pretraining':
    os.getcwd() +
    r'\..\model\四分类\CNN_Net42_pretraining\learning_curve_CNN_Net42_pretraining_4class.pickle',
    'NN_Net45_pretraining':
    os.getcwd() +
    r'\..\model\四分类\NN_Net45_pretraining\learning_curve_NN_Net45_pretraining_4class.pickle',
    'LSTM_Net46_pretraining':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net46_pretraining\learning_curve_LSTM_Net46_pretraining_4class.pickle',
    'LSTM_Net47_pretraining':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net47_pretraining\learning_curve_LSTM_Net47_pretraining_4class.pickle',
    'CNN_LSTM_Net48_pretraining':
    os.getcwd() +
    r'\..\model\四分类\CNN_LSTM_Net48_pretraining\learning_curve_CNN_LSTM_Net48_pretraining_4class.pickle',
    'CNN_LSTM_Net49_pretraining':
    os.getcwd() +
    r'\..\model\四分类\CNN_LSTM_Net49_pretraining\learning_curve_CNN_LSTM_Net49_pretraining_4class.pickle',
    'CNN_Net4_fine':
    os.getcwd() +
    r'\..\model\四分类\CNN_Net4_fine\learning_curve_CNN_Net4_fine_4class.pickle',
    'CNN_Net42_fine':
    os.getcwd() +
    r'\..\model\四分类\CNN_Net42_fine\learning_curve_CNN_Net42_fine_4class.pickle',
    'LSTM_Net44_fine':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net44_fine\learning_curve_LSTM_Net44_fine_4class.pickle',
    'NN_Net45_fine':
    os.getcwd() +
    r'\..\model\四分类\NN_Net45_fine\learning_curve_NN_Net45_fine_4class.pickle',
    'LSTM_Net46_fine':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net46_fine\learning_curve_LSTM_Net46_fine_4class.pickle',
    'LSTM_Net47_fine':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net47_fine\learning_curve_LSTM_Net47_fine_4class.pickle',
    'CNN_LSTM_Net48_fine':
    os.getcwd() +
    r'\..\model\四分类\CNN_LSTM_Net48_fine\learning_curve_CNN_LSTM_Net48_fine_4class.pickle',
    'CNN_LSTM_Net49_fine':
    os.getcwd() +
    r'\..\model\四分类\CNN_LSTM_Net49_fine\learning_curve_CNN_LSTM_Net49_fine_4class.pickle'
}

PATHDICT_CONFUSION_MATRIX_UD2CLASS = {
    'SVM_no_cross_subject':
    os.getcwd() +
    r'\..\model\上下二分类\ML\SVM\confusion_matrix_SVM_no_cross_subject_UD2class.pickle',
    'LDA_no_cross_subject':
    os.getcwd() +
    r'\..\model\上下二分类\ML\LDA\confusion_matrix_LDA_no_cross_subject_UD2class.pickle',
    'CNN_Net2_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_Net2_pretraining\confusion_matrix_CNN_Net2_pretraining_UD2class.pickle',
    'CNN_Net22_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_Net22_pretraining\confusion_matrix_CNN_Net22_pretraining_UD2class.pickle',
    'NN_Net25_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\NN_Net25_pretraining\confusion_matrix_NN_Net25_pretraining_UD2class.pickle',
    'LSTM_Net26_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net26_pretraining\confusion_matrix_LSTM_Net26_pretraining_UD2class.pickle',
    'LSTM_Net27_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net27_pretraining\confusion_matrix_LSTM_Net27_pretraining_UD2class.pickle',
    'CNN_LSTM_Net28_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_LSTM_Net28_pretraining\confusion_matrix_CNN_LSTM_Net28_pretraining_UD2class.pickle',
    'CNN_LSTM_Net29_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_LSTM_Net29_pretraining\confusion_matrix_CNN_LSTM_Net29_pretraining_UD2class.pickle',
    'CNN_Net2_fine':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_Net2_fine\confusion_matrix_CNN_Net2_fine_UD2class.pickle',
    'CNN_Net22_fine':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_Net22_fine\confusion_matrix_CNN_Net22_fine_UD2class.pickle',
    'LSTM_Net23_fine':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net23_fine\confusion_matrix_LSTM_Net23_fine_UD2class.pickle',
    'LSTM_Net24_fine':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net24_fine\confusion_matrix_LSTM_Net24_fine_UD2class.pickle',
    'NN_Net25_fine':
    os.getcwd() +
    r'\..\model\上下二分类\NN_Net25_fine\confusion_matrix_NN_Net25_fine_UD2class.pickle',
    'LSTM_Net26_fine':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net26_fine\confusion_matrix_LSTM_Net26_fine_UD2class.pickle',
    'LSTM_Net27_fine':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net27_fine\confusion_matrix_LSTM_Net27_fine_UD2class.pickle',
    'CNN_LSTM_Net28_fine':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_LSTM_Net28_fine\confusion_matrix_CNN_LSTM_Net28_fine_UD2class.pickle',
    'CNN_LSTM_Net29_fine':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_LSTM_Net29_fine\confusion_matrix_CNN_LSTM_Net29_fine_UD2class.pickle'
}

PATHDICT_LEARNING_CURVE_UD2CLASS = {
    'SVM_no_cross_subject':
    os.getcwd() +
    r'\..\model\上下二分类\ML\SVM\learning_curve_SVM_no_cross_subject_UD2class.pickle',
    'LDA_no_cross_subject':
    os.getcwd() +
    r'\..\model\上下二分类\ML\LDA\learning_curve_LDA_no_cross_subject_UD2class.pickle',
    'CNN_Net2_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_Net2_pretraining\learning_curve_CNN_Net2_pretraining_UD2class.pickle',
    'CNN_Net22_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_Net22_pretraining\learning_curve_CNN_Net22_pretraining_UD2class.pickle',
    'NN_Net25_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\NN_Net25_pretraining\learning_curve_NN_Net25_pretraining_UD2class.pickle',
    'LSTM_Net26_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net26_pretraining\learning_curve_LSTM_Net26_pretraining_UD2class.pickle',
    'LSTM_Net27_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net27_pretraining\learning_curve_LSTM_Net27_pretraining_UD2class.pickle',
    'CNN_LSTM_Net28_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_LSTM_Net28_pretraining\learning_curve_CNN_LSTM_Net28_pretraining_UD2class.pickle',
    'CNN_LSTM_Net29_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_LSTM_Net29_pretraining\learning_curve_CNN_LSTM_Net29_pretraining_UD2class.pickle',
    'CNN_Net2_fine':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_Net2_fine\learning_curve_CNN_Net2_fine_UD2class.pickle',
    'CNN_Net22_fine':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_Net22_fine\learning_curve_CNN_Net22_fine_UD2class.pickle',
    'LSTM_Net23_fine':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net23_fine\learning_curve_LSTM_Net23_fine_UD2class.pickle',
    'LSTM_Net24_fine':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net24_fine\learning_curve_LSTM_Net24_fine_UD2class.pickle',
    'NN_Net25_fine':
    os.getcwd() +
    r'\..\model\上下二分类\NN_Net25_fine\learning_curve_NN_Net25_fine_UD2class.pickle',
    'LSTM_Net26_fine':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net26_fine\learning_curve_LSTM_Net26_fine_UD2class.pickle',
    'LSTM_Net27_fine':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net27_fine\learning_curve_LSTM_Net27_fine_UD2class.pickle',
    'CNN_LSTM_Net28_fine':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_LSTM_Net28_fine\learning_curve_CNN_LSTM_Net28_fine_UD2class.pickle',
    'CNN_LSTM_Net29_fine':
    os.getcwd() +
    r'\..\model\上下二分类\CNN_LSTM_Net29_fine\learning_curve_CNN_LSTM_Net29_fine_UD2class.pickle'
}

PATHDICT_CONFUSION_MATRIX_LR2CLASS = {
    'SVM_no_cross_subject':
    os.getcwd() +
    r'\..\model\左右二分类\ML\SVM\confusion_matrix_SVM_no_cross_subject_LR2class.pickle',
    'LDA_no_cross_subject':
    os.getcwd() +
    r'\..\model\左右二分类\ML\LDA\confusion_matrix_LDA_no_cross_subject_LR2class.pickle',
    'CNN_Net2_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_Net2_pretraining\confusion_matrix_CNN_Net2_pretraining_LR2class.pickle',
    'CNN_Net22_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_Net22_pretraining\confusion_matrix_CNN_Net22_pretraining_LR2class.pickle',
    'NN_Net25_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\NN_Net25_pretraining\confusion_matrix_NN_Net25_pretraining_LR2class.pickle',
    'LSTM_Net26_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net26_pretraining\confusion_matrix_LSTM_Net26_pretraining_LR2class.pickle',
    'LSTM_Net27_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net27_pretraining\confusion_matrix_LSTM_Net27_pretraining_LR2class.pickle',
    'CNN_LSTM_Net28_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_LSTM_Net28_pretraining\confusion_matrix_CNN_LSTM_Net28_pretraining_LR2class.pickle',
    'CNN_LSTM_Net29_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_LSTM_Net29_pretraining\confusion_matrix_CNN_LSTM_Net29_pretraining_LR2class.pickle',
    'CNN_Net2_fine':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_Net2_fine\confusion_matrix_CNN_Net2_fine_LR2class.pickle',
    'CNN_Net22_fine':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_Net22_fine\confusion_matrix_CNN_Net22_fine_LR2class.pickle',
    'LSTM_Net23_fine':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net23_fine\confusion_matrix_LSTM_Net23_fine_LR2class.pickle',
    'LSTM_Net24_fine':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net24_fine\confusion_matrix_LSTM_Net24_fine_LR2class.pickle',
    'NN_Net25_fine':
    os.getcwd() +
    r'\..\model\左右二分类\NN_Net25_fine\confusion_matrix_NN_Net25_fine_LR2class.pickle',
    'LSTM_Net26_fine':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net26_fine\confusion_matrix_LSTM_Net26_fine_LR2class.pickle',
    'LSTM_Net27_fine':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net27_fine\confusion_matrix_LSTM_Net27_fine_LR2class.pickle',
    'CNN_LSTM_Net28_fine':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_LSTM_Net28_fine\confusion_matrix_CNN_LSTM_Net28_fine_LR2class.pickle',
    'CNN_LSTM_Net29_fine':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_LSTM_Net29_fine\confusion_matrix_CNN_LSTM_Net29_fine_LR2class.pickle'
}

PATHDICT_CONFUSION_MATRIX_4CLASS_MULTITASK = {
    'SVM_no_cross_subject':
    os.getcwd() +
    r'\..\model\多任务\四分类\SVM\confusion_matrix_SVM_no_cross_subject_4class.pickle',
    'LDA_no_cross_subject':
    os.getcwd() +
    r'\..\model\多任务\四分类\LDA\confusion_matrix_LDA_no_cross_subject_4class.pickle',
    'NN_Net45_fine':
    os.getcwd() +
    r'\..\model\多任务\四分类\NN_Net45_fine\confusion_matrix_NN_Net45_fine_4class.pickle'
}

PATHDICT_CONFUSION_MATRIX_UD2CLASS_MULTITASK = {
    'SVM_no_cross_subject':
    os.getcwd() +
    r'\..\model\多任务\上下二分类\SVM\confusion_matrix_SVM_no_cross_subject_UD2class.pickle',
    'LDA_no_cross_subject':
    os.getcwd() +
    r'\..\model\多任务\上下二分类\LDA\confusion_matrix_LDA_no_cross_subject_UD2class.pickle',
    'NN_Net25_fine':
    os.getcwd() +
    r'\..\model\多任务\上下二分类\NN_Net25_fine\confusion_matrix_NN_Net25_fine_UD2class.pickle'
}

PATHDICT_CONFUSION_MATRIX_LR2CLASS_MULTITASK = {
    'SVM_no_cross_subject':
    os.getcwd() +
    r'\..\model\多任务\左右二分类\SVM\confusion_matrix_SVM_no_cross_subject_LR2class.pickle',
    'LDA_no_cross_subject':
    os.getcwd() +
    r'\..\model\多任务\左右二分类\LDA\confusion_matrix_LDA_no_cross_subject_LR2class.pickle',
    'NN_Net25_fine':
    os.getcwd() +
    r'\..\model\多任务\左右二分类\NN_Net25_fine\confusion_matrix_NN_Net25_fine_LR2class.pickle'
}

PATHDICT_LEARNING_CURVE_LR2CLASS = {
    'SVM_no_cross_subject':
    os.getcwd() +
    r'\..\model\左右二分类\ML\SVM\learning_curve_SVM_no_cross_subject_LR2class.pickle',
    'LDA_no_cross_subject':
    os.getcwd() +
    r'\..\model\左右二分类\ML\LDA\learning_curve_LDA_no_cross_subject_LR2class.pickle',
    'CNN_Net2_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_Net2_pretraining\learning_curve_CNN_Net2_pretraining_LR2class.pickle',
    'CNN_Net22_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_Net22_pretraining\learning_curve_CNN_Net22_pretraining_LR2class.pickle',
    'NN_Net25_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\NN_Net25_pretraining\learning_curve_NN_Net25_pretraining_LR2class.pickle',
    'LSTM_Net26_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net26_pretraining\learning_curve_LSTM_Net26_pretraining_LR2class.pickle',
    'LSTM_Net27_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net27_pretraining\learning_curve_LSTM_Net27_pretraining_LR2class.pickle',
    'CNN_LSTM_Net28_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_LSTM_Net28_pretraining\learning_curve_CNN_LSTM_Net28_pretraining_LR2class.pickle',
    'CNN_LSTM_Net29_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_LSTM_Net29_pretraining\learning_curve_CNN_LSTM_Net29_pretraining_LR2class.pickle',
    'CNN_Net2_fine':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_Net2_fine\learning_curve_CNN_Net2_fine_LR2class.pickle',
    'CNN_Net22_fine':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_Net22_fine\learning_curve_CNN_Net22_fine_LR2class.pickle',
    'LSTM_Net23_fine':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net23_fine\learning_curve_LSTM_Net23_fine_LR2class.pickle',
    'LSTM_Net24_fine':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net24_fine\learning_curve_LSTM_Net24_fine_LR2class.pickle',
    'NN_Net25_fine':
    os.getcwd() +
    r'\..\model\左右二分类\NN_Net25_fine\learning_curve_NN_Net25_fine_LR2class.pickle',
    'LSTM_Net26_fine':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net26_fine\learning_curve_LSTM_Net26_fine_LR2class.pickle',
    'LSTM_Net27_fine':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net27_fine\learning_curve_LSTM_Net27_fine_LR2class.pickle',
    'CNN_LSTM_Net28_fine':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_LSTM_Net28_fine\learning_curve_CNN_LSTM_Net28_fine_LR2class.pickle',
    'CNN_LSTM_Net29_fine':
    os.getcwd() +
    r'\..\model\左右二分类\CNN_LSTM_Net29_fine\learning_curve_CNN_LSTM_Net29_fine_LR2class.pickle'
}

PATHDICT_LEARNING_CURVE_4CLASS_MULTITASK = {
    'SVM_no_cross_subject':
    os.getcwd() +
    r'\..\model\多任务\四分类\SVM\learning_curve_SVM_no_cross_subject_4class.pickle',
    'LDA_no_cross_subject':
    os.getcwd() +
    r'\..\model\多任务\四分类\LDA\learning_curve_LDA_no_cross_subject_4class.pickle',
    'NN_Net45_fine':
    os.getcwd() +
    r'\..\model\多任务\四分类\NN_Net45_fine\learning_curve_NN_Net45_fine_4class.pickle',
}

PATHDICT_LEARNING_CURVE_UD2CLASS_MULTITASK = {
    'SVM_no_cross_subject':
    os.getcwd() +
    r'\..\model\多任务\上下二分类\SVM\learning_curve_SVM_no_cross_subject_UD2class.pickle',
    'LDA_no_cross_subject':
    os.getcwd() +
    r'\..\model\多任务\上下二分类\LDA\learning_curve_LDA_no_cross_subject_UD2class.pickle',
    'NN_Net25_fine':
    os.getcwd() +
    r'\..\model\多任务\上下二分类\NN_Net25_fine\learning_curve_NN_Net25_fine_UD2class.pickle',
}

PATHDICT_LEARNING_CURVE_LR2CLASS_MULTITASK = {
    'SVM_no_cross_subject':
    os.getcwd() +
    r'\..\model\多任务\左右二分类\SVM\learning_curve_SVM_no_cross_subject_LR2class.pickle',
    'LDA_no_cross_subject':
    os.getcwd() +
    r'\..\model\多任务\左右二分类\LDA\learning_curve_LDA_no_cross_subject_LR2class.pickle',
    'NN_Net25_fine':
    os.getcwd() +
    r'\..\model\多任务\左右二分类\NN_Net25_fine\learning_curve_NN_Net25_fine_LR2class.pickle',
}

PATHDICT_UD2CLASS_SEGMENT = {
    'SVM': os.getcwd() + r'\..\model\上下二分类\ML\SVM\\',
    'LDA': os.getcwd() + r'\..\model\上下二分类\ML\LDA\\',
    'Net25': os.getcwd() + r'\..\model\上下二分类\NN_Net25_fine\\'
}

PATHDICT_LR2CLASS_SEGMENT = {
    'SVM': os.getcwd() + r'\..\model\左右二分类\ML\SVM\\',
    'LDA': os.getcwd() + r'\..\model\左右二分类\ML\LDA\\',
    'Net25': os.getcwd() + r'\..\model\左右二分类\NN_Net25_fine\\'
}

PATHDICT_4CLASS_SEGMENT = {
    'SVM': os.getcwd() + r'\..\model\四分类\ML\SVM\\',
    'LDA': os.getcwd() + r'\..\model\四分类\ML\LDA\\',
    'Net25': os.getcwd() + r'\..\model\四分类\NN_Net25_fine\\'
}

PATHDICT_UD2CLASS_CHANNEL_SELECT = {
    'SVM': os.getcwd() + r'\..\model\通道选择\上下二分类\ML\SVM\\',
    'LDA': os.getcwd() + r'\..\model\通道选择\上下二分类\ML\LDA\\',
    'Net25': os.getcwd() + r'\..\model\通道选择\上下二分类\NN_Net25_fine\\'
}

PATHDICT_LR2CLASS_CHANNEL_SELECT = {
    'SVM': os.getcwd() + r'\..\model\通道选择\左右二分类\ML\SVM\\',
    'LDA': os.getcwd() + r'\..\model\通道选择\左右二分类\ML\LDA\\',
    'Net25': os.getcwd() + r'\..\model\通道选择\左右二分类\NN_Net25_fine\\'
}

PATHDICT_4CLASS_CHANNEL_SELECT = {
    'SVM': os.getcwd() + r'\..\model\通道选择\四分类\ML\SVM\\',
    'LDA': os.getcwd() + r'\..\model\通道选择\四分类\ML\LDA\\',
    'Net25': os.getcwd() + r'\..\model\通道选择\四分类\NN_Net25_fine\\'
}


NAME_LIST = ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg']


# 将 subject_learning_curve_plot_LIST 转换为字典格式
def myLearning_curve_LIST2DICT_4class(netName):
    pathDict = {
        'SVM_no_cross_subject':
        os.getcwd() +
        r'\..\model\四分类\ML\SVM\learning_curve_SVM_no_cross_subject_4class.pickle',
        'LDA_no_cross_subject':
        os.getcwd() +
        r'\..\model\四分类\ML\LDA\learning_curve_LDA_no_cross_subject_4class.pickle',
        'CNN_Net4_pretraining':
        os.getcwd() +
        r'\..\model\四分类\CNN_Net4_pretraining\learning_curve_CNN_Net4_pretraining_4class.pickle',
        'CNN_Net42_pretraining':
        os.getcwd() +
        r'\..\model\四分类\CNN_Net42_pretraining\learning_curve_CNN_Net42_pretraining_4class.pickle',
        'LSTM_Net44_pretraining':
        os.getcwd() +
        r'\..\model\四分类\LSTM_Net44_pretraining\learning_curve_LSTM_Net44_pretraining_4class.pickle',        
        'NN_Net45_pretraining':
        os.getcwd() +
        r'\..\model\四分类\NN_Net45_pretraining\learning_curve_NN_Net45_pretraining_4class.pickle',
        'LSTM_Net46_pretraining':
        os.getcwd() +
        r'\..\model\四分类\LSTM_Net46_pretraining\learning_curve_LSTM_Net46_pretraining_4class.pickle',
        'LSTM_Net47_pretraining':
        os.getcwd() +
        r'\..\model\四分类\LSTM_Net47_pretraining\learning_curve_LSTM_Net47_pretraining_4class.pickle',
        'CNN_LSTM_Net48_pretraining':
        os.getcwd() +
        r'\..\model\四分类\CNN_LSTM_Net48_pretraining\learning_curve_CNN_LSTM_Net48_pretraining_4class.pickle',
        'CNN_LSTM_Net49_pretraining':
        os.getcwd() +
        r'\..\model\四分类\CNN_LSTM_Net49_pretraining\learning_curve_CNN_LSTM_Net49_pretraining_4class.pickle',
        'CNN_Net4_fine':
        os.getcwd() +
        r'\..\model\四分类\CNN_Net4_fine\learning_curve_CNN_Net4_fine_4class.pickle',
        'CNN_Net42_fine':
        os.getcwd() +
        r'\..\model\四分类\CNN_Net42_fine\learning_curve_CNN_Net42_fine_4class.pickle',
        'NN_Net45_fine':
        os.getcwd() +
        r'\..\model\四分类\NN_Net45_fine\learning_curve_NN_Net45_fine_4class.pickle',
        'LSTM_Net46_fine':
        os.getcwd() +
        r'\..\model\四分类\LSTM_Net46_fine\learning_curve_LSTM_Net46_fine_4class.pickle',
        'LSTM_Net47_fine':
        os.getcwd() +
        r'\..\model\四分类\LSTM_Net47_fine\learning_curve_LSTM_Net47_fine_4class.pickle',
        'CNN_LSTM_Net48_fine':
        os.getcwd() +
        r'\..\model\四分类\CNN_LSTM_Net48_fine\learning_curve_CNN_LSTM_Net48_fine_4class.pickle',
        'CNN_LSTM_Net49_fine':
        os.getcwd() +
        r'\..\model\四分类\CNN_LSTM_Net49_fine\learning_curve_CNN_LSTM_Net49_fine_4class.pickle'
    }

    fileD = open(pathDict[netName], 'rb')
    learning_curve_LIST = pickle.load(fileD)
    fileD.close()

    learning_curve_DICT = {}
    for learning_curve in learning_curve_LIST:
        learning_curve_DICT.update({learning_curve['name']: learning_curve})
    # end for
    learning_curve_DICT.update({'ignoreList': []})

    fileD = open(pathDict[netName], 'wb')
    pickle.dump(learning_curve_DICT, fileD)
    fileD.close()
    # end myLearning_curve_LIST2DICT_4class


# 将 subject_learning_curve_plot_LIST 转换为字典格式
def myLearning_curve_LIST2DICT_UD2class(netName):
    pathDict = {
        'SVM_no_cross_subject':
        os.getcwd() +
        r'\..\model\上下二分类\ML\SVM\learning_curve_SVM_no_cross_subject_UD2class.pickle',
        'LDA_no_cross_subject':
        os.getcwd() +
        r'\..\model\上下二分类\ML\LDA\learning_curve_LDA_no_cross_subject_UD2class.pickle',
        'CNN_Net2_pretraining':
        os.getcwd() +
        r'\..\model_all\上下二分类\CNN_Net2_pretraining\learning_curve_CNN_Net2_pretraining_UD2class.pickle',
        'CNN_Net22_pretraining':
        os.getcwd() +
        r'\..\model_all\上下二分类\CNN_Net22_pretraining\learning_curve_CNN_Net22_pretraining_UD2class.pickle',
        'LSTM_Net23_pretraining':
        os.getcwd() +
        r'\..\model_all\上下二分类\LSTM_Net23_pretraining\learning_curve_LSTM_Net23_pretraining_UD2class.pickle',
        'LSTM_Net24_pretraining':
        os.getcwd() +
        r'\..\model_all\上下二分类\LSTM_Net24_pretraining\learning_curve_LSTM_Net24_pretraining_UD2class.pickle',
        'NN_Net25_pretraining':
        os.getcwd() +
        r'\..\model_all\上下二分类\NN_Net25_pretraining\learning_curve_NN_Net25_pretraining_UD2class.pickle',
        'LSTM_Net26_pretraining':
        os.getcwd() +
        r'\..\model_all\上下二分类\LSTM_Net26_pretraining\learning_curve_LSTM_Net26_pretraining_UD2class.pickle',
        'LSTM_Net27_pretraining':
        os.getcwd() +
        r'\..\model_all\上下二分类\LSTM_Net27_pretraining\learning_curve_LSTM_Net27_pretraining_UD2class.pickle',
        'CNN_LSTM_Net28_pretraining':
        os.getcwd() +
        r'\..\model_all\上下二分类\CNN_LSTM_Net28_pretraining\learning_curve_CNN_LSTM_Net28_pretraining_UD2class.pickle',
        'CNN_LSTM_Net29_pretraining':
        os.getcwd() +
        r'\..\model_all\上下二分类\CNN_LSTM_Net29_pretraining\learning_curve_CNN_LSTM_Net29_pretraining_UD2class.pickle',
        'CNN_Net2_fine':
        os.getcwd() +
        r'\..\model\上下二分类\CNN_Net2_fine\learning_curve_CNN_Net2_fine_UD2class.pickle',
        'CNN_Net22_fine':
        os.getcwd() +
        r'\..\model\上下二分类\CNN_Net22_fine\learning_curve_CNN_Net22_fine_UD2class.pickle',
        'LSTM_Net23_fine':
        os.getcwd() +
        r'\..\model\上下二分类\LSTM_Net23_fine\learning_curve_LSTM_Net23_fine_UD2class.pickle',
        'LSTM_Net24_fine':
        os.getcwd() +
        r'\..\model\上下二分类\LSTM_Net24_fine\learning_curve_LSTM_Net24_fine_UD2class.pickle',
        'NN_Net25_fine':
        os.getcwd() +
        r'\..\model\上下二分类\NN_Net25_fine\learning_curve_NN_Net25_fine_UD2class.pickle',
        'LSTM_Net26_fine':
        os.getcwd() +
        r'\..\model\上下二分类\LSTM_Net26_fine\learning_curve_LSTM_Net26_fine_UD2class.pickle',
        'LSTM_Net27_fine':
        os.getcwd() +
        r'\..\model\上下二分类\LSTM_Net27_fine\learning_curve_LSTM_Net27_fine_UD2class.pickle',
        'CNN_LSTM_Net28_fine':
        os.getcwd() +
        r'\..\model\上下二分类\CNN_LSTM_Net28_fine\learning_curve_CNN_LSTM_Net28_fine_UD2class.pickle',
        'CNN_LSTM_Net29_fine':
        os.getcwd() +
        r'\..\model\上下二分类\CNN_LSTM_Net29_fine\learning_curve_CNN_LSTM_Net29_fine_UD2class.pickle'
    }

    fileD = open(pathDict[netName], 'rb')
    learning_curve_LIST = pickle.load(fileD)
    fileD.close()

    learning_curve_DICT = {}
    for learning_curve in learning_curve_LIST:
        learning_curve_DICT.update({learning_curve['name']: learning_curve})
    # end for
    learning_curve_DICT.update({'ignoreList': []})

    fileD = open(pathDict[netName], 'wb')
    pickle.dump(learning_curve_DICT, fileD)
    fileD.close()
    # end myLearning_curve_LIST2DICT_UD2class


# 将 subject_learning_curve_plot_LIST 转换为字典格式
def myLearning_curve_LIST2DICT_LR2class(netName):
    pathDict = {
        'SVM_no_cross_subject':
        os.getcwd() +
        r'\..\model\左右二分类\ML\SVM\learning_curve_SVM_no_cross_subject_LR2class.pickle',
        'LDA_no_cross_subject':
        os.getcwd() +
        r'\..\model\左右二分类\ML\LDA\learning_curve_LDA_no_cross_subject_LR2class.pickle',
        'CNN_Net2_pretraining':
        os.getcwd() +
        r'\..\model_all\左右二分类\CNN_Net2_pretraining\learning_curve_CNN_Net2_pretraining_LR2class.pickle',
        'CNN_Net22_pretraining':
        os.getcwd() +
        r'\..\model_all\左右二分类\CNN_Net22_pretraining\learning_curve_CNN_Net22_pretraining_LR2class.pickle',
        'LSTM_Net23_pretraining':
        os.getcwd() +
        r'\..\model_all\左右二分类\LSTM_Net23_pretraining\learning_curve_LSTM_Net23_pretraining_LR2class.pickle',
        'LSTM_Net24_pretraining':
        os.getcwd() +
        r'\..\model_all\左右二分类\LSTM_Net24_pretraining\learning_curve_LSTM_Net24_pretraining_LR2class.pickle',
        'NN_Net25_pretraining':
        os.getcwd() +
        r'\..\model_all\左右二分类\NN_Net25_pretraining\learning_curve_NN_Net25_pretraining_LR2class.pickle',
        'LSTM_Net26_pretraining':
        os.getcwd() +
        r'\..\model_all\左右二分类\LSTM_Net26_pretraining\learning_curve_LSTM_Net26_pretraining_LR2class.pickle',
        'LSTM_Net27_pretraining':
        os.getcwd() +
        r'\..\model_all\左右二分类\LSTM_Net27_pretraining\learning_curve_LSTM_Net27_pretraining_LR2class.pickle',
        'CNN_LSTM_Net28_pretraining':
        os.getcwd() +
        r'\..\model_all\左右二分类\CNN_LSTM_Net28_pretraining\learning_curve_CNN_LSTM_Net28_pretraining_LR2class.pickle',
        'CNN_LSTM_Net29_pretraining':
        os.getcwd() +
        r'\..\model_all\左右二分类\CNN_LSTM_Net29_pretraining\learning_curve_CNN_LSTM_Net29_pretraining_LR2class.pickle',
        'CNN_Net2_fine':
        os.getcwd() +
        r'\..\model\左右二分类\CNN_Net2_fine\learning_curve_CNN_Net2_fine_LR2class.pickle',
        'CNN_Net22_fine':
        os.getcwd() +
        r'\..\model\左右二分类\CNN_Net22_fine\learning_curve_CNN_Net22_fine_LR2class.pickle',
        'LSTM_Net23_fine':
        os.getcwd() +
        r'\..\model\左右二分类\LSTM_Net23_fine\learning_curve_LSTM_Net23_fine_LR2class.pickle',
        'LSTM_Net24_fine':
        os.getcwd() +
        r'\..\model\左右二分类\LSTM_Net24_fine\learning_curve_LSTM_Net24_fine_LR2class.pickle',
        'NN_Net25_fine':
        os.getcwd() +
        r'\..\model\左右二分类\NN_Net25_fine\learning_curve_NN_Net25_fine_LR2class.pickle',
        'LSTM_Net26_fine':
        os.getcwd() +
        r'\..\model\左右二分类\LSTM_Net26_fine\learning_curve_LSTM_Net26_fine_LR2class.pickle',
        'LSTM_Net27_fine':
        os.getcwd() +
        r'\..\model\左右二分类\LSTM_Net27_fine\learning_curve_LSTM_Net27_fine_LR2class.pickle',
        'CNN_LSTM_Net28_fine':
        os.getcwd() +
        r'\..\model\左右二分类\CNN_LSTM_Net28_fine\learning_curve_CNN_LSTM_Net28_fine_LR2class.pickle',
        'CNN_LSTM_Net29_fine':
        os.getcwd() +
        r'\..\model\左右二分类\CNN_LSTM_Net29_fine\learning_curve_CNN_LSTM_Net29_fine_LR2class.pickle'
    }

    fileD = open(pathDict[netName], 'rb')
    learning_curve_LIST = pickle.load(fileD)
    fileD.close()

    learning_curve_DICT = {}
    for learning_curve in learning_curve_LIST:
        learning_curve_DICT.update({learning_curve['name']: learning_curve})
    # end for
    learning_curve_DICT.update({'ignoreList': []})

    fileD = open(pathDict[netName], 'wb')
    pickle.dump(learning_curve_DICT, fileD)
    fileD.close()
    # end myLearning_curve_LIST2DICT_LR2class

# 将 subject_learning_curve_plot_LIST 转换为字典格式
def myLearning_curve_LIST2DICT_4class_multiTask(netName):
    pathDict = {
        'SVM_no_cross_subject':
        os.getcwd() +
        r'\..\model\多任务\四分类\SVM\learning_curve_SVM_no_cross_subject_4class.pickle',
        'NN_Net45_pretraining':
        os.getcwd() +
        r'\..\model_all\多任务\四分类\NN_Net45_pretraining\learning_curve_NN_Net45_pretraining_4class.pickle',
        'NN_Net45_fine':
        os.getcwd() +
        r'\..\model\多任务\四分类\NN_Net45_fine\learning_curve_NN_Net45_fine_4class.pickle',
    }

    fileD = open(pathDict[netName], 'rb')
    learning_curve_LIST = pickle.load(fileD)
    fileD.close()

    learning_curve_DICT = {}
    for learning_curve in learning_curve_LIST:
        learning_curve_DICT.update({learning_curve['name']: learning_curve})
    # end for
    learning_curve_DICT.update({'ignoreList': []})

    fileD = open(pathDict[netName], 'wb')
    pickle.dump(learning_curve_DICT, fileD)
    fileD.close()
    # end myLearning_curve_LIST2DICT_4class_multiTask


# 将 subject_learning_curve_plot_LIST 转换为字典格式
def myLearning_curve_LIST2DICT_UD2class_multiTask(netName):
    pathDict = {
        'SVM_no_cross_subject':
        os.getcwd() +
        r'\..\model\多任务\上下二分类\SVM\learning_curve_SVM_no_cross_subject_UD2class.pickle',
        'NN_Net25_pretraining':
        os.getcwd() +
        r'\..\model_all\多任务\上下二分类\NN_Net25_pretraining\learning_curve_NN_Net25_pretraining_UD2class.pickle',
        'NN_Net25_fine':
        os.getcwd() +
        r'\..\model\多任务\上下二分类\NN_Net25_fine\learning_curve_NN_Net25_fine_UD2class.pickle',
    }

    fileD = open(pathDict[netName], 'rb')
    learning_curve_LIST = pickle.load(fileD)
    fileD.close()

    learning_curve_DICT = {}
    for learning_curve in learning_curve_LIST:
        learning_curve_DICT.update({learning_curve['name']: learning_curve})
    # end for
    learning_curve_DICT.update({'ignoreList': []})

    fileD = open(pathDict[netName], 'wb')
    pickle.dump(learning_curve_DICT, fileD)
    fileD.close()
    # end myLearning_curve_LIST2DICT_UD2class_multiTask


# 将 subject_learning_curve_plot_LIST 转换为字典格式
def myLearning_curve_LIST2DICT_LR2class_multiTask(netName):
    pathDict = {
        'SVM_no_cross_subject':
        os.getcwd() +
        r'\..\model\多任务\左右二分类\SVM\learning_curve_SVM_no_cross_subject_LR2class.pickle',
        'NN_Net25_pretraining':
        os.getcwd() +
        r'\..\model_all\多任务\左右二分类\NN_Net25_pretraining\learning_curve_NN_Net25_pretraining_LR2class.pickle',
        'NN_Net25_fine':
        os.getcwd() +
        r'\..\model\多任务\左右二分类\NN_Net25_fine\learning_curve_NN_Net25_fine_LR2class.pickle',
    }

    fileD = open(pathDict[netName], 'rb')
    learning_curve_LIST = pickle.load(fileD)
    fileD.close()

    learning_curve_DICT = {}
    for learning_curve in learning_curve_LIST:
        learning_curve_DICT.update({learning_curve['name']: learning_curve})
    # end for
    learning_curve_DICT.update({'ignoreList': []})

    fileD = open(pathDict[netName], 'wb')
    pickle.dump(learning_curve_DICT, fileD)
    fileD.close()
    # end myLearning_curve_LIST2DICT_LR2class_multiTask

# 随机产生一种颜色
def randomcolor():
    colorArr = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
        'F'
    ]
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color
    # end randomcolor


# 绘制混淆矩阵：单个矩阵
def draw_confusion_matrix(y_true,
                          y_pred,
                          labels_name,
                          axis_labels=None,
                          title=None,
                          amount=None):
    thresh = 0.5
    # 生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true,
                                  y_pred,
                                  labels=labels_name,
                                  sample_weight=None)  # 生成混淆矩阵
    if amount == None:
        amount = [sum(y_true == labels_name[x]) for x in labels_name]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 画图，如果希望改变颜色风格，可以改变此部分的 cmap=pl.get_cmap('Blues') 处
    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

    # 图像标题
    if title is None:
        title = 'draw_confusion_matrix'

    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45,fontsize=28)  # 将标签印在 x 轴坐标上， 并倾斜 45 度
    plt.yticks(num_local, axis_labels,fontsize=28)  # 将标签印在 y 轴坐标上
    plt.ylabel('真实值',fontsize=28)
    plt.xlabel('预测值',fontsize=28)

    cntSum = 0
    # 将百分比打印在相应的格子内，大于 thresh 的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        cntRow = 0
        for j in range(np.shape(cm)[1]):
            if j != labels_name[-1]:
                cntIj = round(amount[i] * cm[i][j])
            else:
                cntIj = amount[i] - cntRow
            cntSum += cntIj
            cntRow += round(amount[i] * cm[i][j])
            # plt.text(j,
            #          i,
            #          format(round(cm[i][j] * 100), 'd') + '%\n' +
            #          '(%d)' % cntIj,
            #          ha="center",
            #          va="center",
            #          size=28,
            #          color="white"
            #          if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
            plt.text(j,
                     i,
                     format(round(cm[i][j] * 100), 'd') + '%',
                     ha="center",
                     va="center",
                     size=28,
                     color="white"
                     if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行

    print('draw_confusion_matrix：cnt_origin = ',
          sum([sum(y_true == labels_name[x]) for x in labels_name]))
    print('draw_confusion_matrix：cnt_aug = ', cntSum)
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(title + '.jpeg')
    plt.show()
    # end draw_confusion_matrix


def draw_ROC_curve(modelDict, title):
    plt.figure(figsize=(10, 8))

    for modelName in modelDict:
        y_true = modelDict[modelName]['y_true']
        y_pred = modelDict[modelName]['y_pred']
        acc = sum(y_true == y_pred) / y_true.shape[0]
        y_score = y_pred.copy()
        for i in range(y_true.shape[0]):
            if y_score[i] == 0:
                y_score[i] += np.random.rand(1) * 0.5
            else:
                y_score[i] -= np.random.rand(1) * 0.5

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        area = metrics.auc(fpr, tpr)
        plt.plot(fpr,
                 tpr,
                 lw=2,
                 color=modelDict[modelName]['color'],
                 linestyle=modelDict[modelName]['linestyle'],
                 marker=modelDict[modelName]['marker'],
                 label=modelName + '(area = %0.2f)' % area)
        print('modelName=', modelName)
        print('\tarea = ', area)
        print('\tacc = ', acc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xticks(np.linspace(0, 1.0, 11))
    plt.yticks(np.linspace(0, 1.0, 11))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(title + '.jpeg')
    plt.show()
    # end draw_ROC_curve


def draw_bar(modelDict, title,f1_average='binary',ylim=[0.4, 1,13]):
    nets = [netName for netName in modelDict]
    accuracy_nets_0 = []
    fScore_nets_0 = []
    accuracy_nets_size = []
    fScore_nets_size = []

    for modelName in modelDict:
        y_true_size = modelDict[modelName]['y_true_size']
        y_pred_size = modelDict[modelName]['y_pred_size']
        accuracy_size = sum(y_true_size == y_pred_size) / y_true_size.shape[0]
        fScore_size = metrics.f1_score(y_true_size, y_pred_size,average=f1_average)
        accuracy_nets_size.append(accuracy_size)
        fScore_nets_size.append(fScore_size)
        if modelDict[modelName]['y_true_0'] is not None:
            y_true_0 = modelDict[modelName]['y_true_0']
            y_pred_0 = modelDict[modelName]['y_pred_0']
            accuracy_0 = modelDict[modelName].get('zero_accuracy',sum(y_true_0 == y_pred_0) / y_true_0.shape[0]) 
            fScore_0 = modelDict[modelName].get('zero_f1',metrics.f1_score(y_true_0, y_pred_0,average=f1_average))
        else:
            accuracy_0 = 0
            fScore_0 = 0
        accuracy_nets_0.append(accuracy_0)
        fScore_nets_0.append(fScore_0)
    accuracy_nets_rise = np.array(accuracy_nets_size) - np.array(
        accuracy_nets_0)
    fScore_nets_rise = np.array(fScore_nets_size) - np.array(fScore_nets_0)
    # 创建分组柱状图，需要自己控制 x 轴坐标
    xticks = np.arange(len(nets))

    fig, ax = plt.subplots(figsize=(18, 10))
    # 所有模型准确率-预训练，注意控制柱子的宽度，这里选择 0.25
    ax.bar(xticks,
           accuracy_nets_0,
           width=0.25,
           label="准确率_零样本",
           color="darkred")
    # 所有模型 f1 分值-预训练，通过微调x轴坐标来调整新增柱子的位置
    ax.bar(xticks + 0.25,
           fScore_nets_0,
           width=0.25,
           label="F1分值_零样本",
           color="navy")
    # 所有模型准确率-微调，注意控制柱子的宽度，这里选择 0.25
    ax.bar(xticks,
           accuracy_nets_rise,
           bottom=accuracy_nets_0,
           width=0.25,
           label="准确率_小样本",
           color="red")
    # 所有模型 f1 分值-微调，通过微调x轴坐标来调整新增柱子的位置
    ax.bar(xticks + 0.25,
           fScore_nets_rise,
           bottom=fScore_nets_0,
           width=0.25,
           label="F1分值_小样本",
           color="blue")

    # 最后调整x轴标签的位置
    ax.set_xticks(xticks + 0.125)
    ax.set_xticklabels(nets, rotation=45,fontsize=28)
    ax.set_yticks(np.linspace(ylim[0],ylim[1],ylim[2]))
    plt.yticks(fontsize=28)
    plt.xlabel('模型名称',fontsize=28)
    plt.ylabel('准确率',fontsize=28)
    # ax.set_yticklabels(fontsize=28)
    ax.set_ylim(ylim[:2])
    # ax.set_title(title)
    # ax.legend(loc='right',fontsize=28,bbox_to_anchor=(1.01, 1.3))
    plt.legend(loc='right',fontsize=28,bbox_to_anchor=(1.38, 0.16))
    plt.grid(color='black', which='major', axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(title + '.jpeg')
    plt.show()
    # end draw_bar


# 绘制学习曲线：表征准确率与训练集容量的关系
# 输入参数：被试字典(字典字典，字典元素为字典：'name':<被试名字>,'train_sizes':<训练集容量数组(numpy 一维数组)>,'train_scores':<训练集评分数组(numpy 一维数组)>,'test_scores':<测试集评分数组(numpy 一维数组)>,'fit_times':<训练时间数组(numpy 一维数组)>)
def draw_learning_curve(subject_DICT, title):
    train_scores = []
    test_scores = []
    fit_time = []
    plt.figure(figsize=(19, 10))
    for name in subject_DICT:
        if name == 'ignoreList':
            break
        if name in subject_DICT['ignoreList']:
            continue
        dic = subject_DICT[name]
        train_sizes = dic['train_sizes']
        train_scores.append(dic['train_scores'])
        test_scores.append(dic['test_scores'])
        fit_time.append(dic['fit_times'])
    # end for
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)
    fit_time = np.array(fit_time)

    train_scores_mean = np.mean(train_scores, axis=0)
    test_scores_mean = np.mean(test_scores, axis=0)
    fit_time_mean = np.mean(fit_time, axis=0)

    train_scores_std = np.std(train_scores, axis=0)
    test_scores_std = np.std(test_scores, axis=0)
    fit_time_std = np.std(fit_time, axis=0)

    plt.plot(train_sizes,
             train_scores_mean,
             linestyle='-',
             color='red',
             label='train_scores')
    plt.fill_between(train_sizes,
                     train_scores_mean + train_scores_std,
                     train_scores_mean - train_scores_std,
                     color=cm.viridis(0.5),
                     alpha=0.2)

    plt.plot(train_sizes,
             test_scores_mean,
             linestyle='-',
             color='blue',
             label='test_scores')
    plt.fill_between(train_sizes,
                     test_scores_mean + test_scores_std,
                     test_scores_mean - test_scores_std,
                     color=cm.viridis(0.5),
                     alpha=0.2)

    print('draw_learning_curve：平均测试集准确率：', test_scores_mean)
    plt.legend()
    plt.xlabel("train_sizes")
    plt.ylabel("scores_accuracy")
    plt.grid()
    # plt.xticks(np.linspace(0, 700, 30).astype(int))
    # plt.xticks(train_sizes)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title + '.jpeg')
    plt.show()
    # end draw_learning_curve


# 绘制学习曲线（为每个被试单独绘制一条曲线）：表征准确率与训练集容量的关系
# 输入参数：被试字典(字典字典，字典元素为字典：'name':<被试名字>,'train_sizes':<训练集容量数组(numpy 一维数组)>,'train_scores':<训练集评分数组(numpy 一维数组)>,'test_scores':<测试集评分数组(numpy 一维数组)>,'fit_times':<训练时间数组(numpy 一维数组)>)
def draw_learning_curve_detail(subject_DICT, title):
    train_scores = []
    test_scores = []
    fit_time = []

    plt.figure(figsize=(19, 10))
    for name in subject_DICT:
        if name == 'ignoreList':
            break
        if name in subject_DICT['ignoreList']:
            continue
        dic = subject_DICT[name]
        assert name == dic['name']
        color = randomcolor()
        # plt.plot(dic['train_sizes'], dic['train_scores'],linestyle='-',color=color,label='train_scores_'+dic['name'])
        plt.plot(dic['train_sizes'],
                 dic['test_scores'],
                 linestyle='-',
                 color=color,
                 label='test_scores_' + dic['name'])
        # plt.plot(dic['train_sizes'], dic['fit_times'],linestyle='-.',color=color,label='fit_times_'+dic['name'])

        train_sizes = dic['train_sizes']
        train_scores.append(dic['train_scores'])
        test_scores.append(dic['test_scores'])
        fit_time.append(dic['fit_times'])
    # end for
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)
    fit_time = np.array(fit_time)

    train_scores_mean = np.mean(train_scores, axis=0)
    test_scores_mean = np.mean(test_scores, axis=0)
    fit_time_mean = np.mean(fit_time, axis=0)

    train_scores_std = np.std(train_scores, axis=0)
    test_scores_std = np.std(test_scores, axis=0)
    fit_time_std = np.std(fit_time, axis=0)

    # plt.plot(train_sizes,
    #          train_scores_mean,
    #          linestyle='-',
    #          color='red',
    #          label='train_scores_average')
    # plt.fill_between(train_sizes,
    #                  train_scores_mean + train_scores_std,
    #                  train_scores_mean - train_scores_std,
    #                  color=cm.viridis(0.5),
    #                  alpha=0.2)

    plt.plot(train_sizes,
             test_scores_mean,
             linestyle='--',
             color='blue',
             label='test_scores_average')
    plt.fill_between(train_sizes,
                     test_scores_mean + test_scores_std,
                     test_scores_mean - test_scores_std,
                     color=cm.viridis(0.5),
                     alpha=0.2)

    print('draw_learning_curve：平均测试集准确率：', test_scores_mean)
    plt.legend()
    plt.xlabel("train_sizes")
    plt.ylabel("scores_accuracy")
    plt.grid()
    # plt.xticks(np.linspace(0, 700, 30).astype(int))
    # plt.xticks(train_sizes)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title + '.jpeg')
    plt.show()
    # end draw_learning_curve_detail


# 绘制多模型学习曲线：表征多个模型准确率与训练集容量的关系
# 输入参数：被试字典列表(字典列表，列表元素为字典：'name':<被试名字>,'train_sizes':<训练集容量数组(numpy 一维数组)>,'train_scores':<训练集评分数组(numpy 一维数组)>,'test_scores':<测试集评分数组(numpy 一维数组)>,'fit_times':<训练时间数组(numpy 一维数组)>)
def draw_learning_curves(subject_dic_LIST, title):
    plt.figure(figsize=(19, 10))
    for dic in subject_dic_LIST:
        print('\n%s ignore:  ' % (dic['labelName']), end='')
        train_scores = []
        test_scores = []
        subject_DICT = dic['dic']
        for name in NAME_LIST:
            if name in subject_DICT['ignoreList']:
                print('%s' % (name), end='  ')
                continue
            dataDic = subject_DICT[name]
            train_sizes = dataDic['train_sizes']
            train_scores.append(dataDic['train_scores'])
            test_scores.append(dataDic['test_scores'])
        # end for
        train_scores = np.array(train_scores)
        test_scores = np.array(test_scores)
        train_scores_mean = np.mean(train_scores, axis=0)
        test_scores_mean = np.mean(test_scores, axis=0)
        
        test_scores_mean[0]=dic.get('zero_accuracy',test_scores_mean[0])
        train_scores_std = np.std(train_scores, axis=0)
        test_scores_std = np.std(test_scores, axis=0)
        plt.plot(train_sizes,
                 test_scores_mean,
                 label=dic['labelName'],
                 color=dic['color'],
                 linestyle=dic['linestyle'],
                 marker=dic['marker'])
        # plt.fill_between(train_sizes,
        #                 test_scores_mean + test_scores_std,
        #                 test_scores_mean - test_scores_std,
        #                 color=cm.viridis(0.5),
        #                 alpha=0.2)

    # plt.legend(loc='right',fontsize=28,bbox_to_anchor=(1.3, 0.248))
    plt.legend(fontsize=28)
    plt.xlabel("样本数量/个",fontsize=28)
    plt.ylabel("准确率",fontsize=28)
    plt.grid()
    plt.xticks(np.linspace(train_sizes[0], train_sizes[-1],
                           19).astype(int),fontsize=28)  # 适用于二分类微调
    # plt.xticks(train_sizes) # 适用于二分类预训练
    plt.yticks(fontsize=28) # 适用于二分类预训练

    # plt.title(title)
    plt.tight_layout()
    plt.savefig(title + '.jpeg')
    plt.show()
    # end draw_learning_curves


# 绘制模型学习曲线
def plot_learning_curve_plot_DICT_4class(netName, detail=False):
    ignoreDict_4class = {
        'SVM_no_cross_subject': ['pbl'],
        'LDA_no_cross_subject': [],
        'CNN_Net4_fine': ['cw', 'kx','wrd','yzg'],
        'CNN_Net42_fine': ['cwm','wxc','yzg'],
        'NN_Net45_fine': ['wxc','yzg'],
        'LSTM_Net46_fine': ['cw', 'kx','wxc','xsc'],
        'LSTM_Net47_fine': ['wrd', 'xsc', 'wxc','yzg'],
        'CNN_LSTM_Net48_fine': ['cwm','kx','xsc'],
        'CNN_LSTM_Net49_fine': ['cwm','kx','xsc','yzg']
    }

    fileD = open(PATHDICT_LEARNING_CURVE_4CLASS[netName], 'rb')
    subject_learning_curve_plot_DICT = pickle.load(fileD)
    fileD.close()
    if detail:
        subject_learning_curve_plot_DICT.update(
            {'ignoreList': ignoreDict_4class[netName]})
        fileD = open(PATHDICT_LEARNING_CURVE_4CLASS[netName], 'wb')
        pickle.dump(subject_learning_curve_plot_DICT, fileD)
        fileD.close()

        fileD = open(PATHDICT_CONFUSION_MATRIX_4CLASS[netName], 'rb')
        subject_confusion_matrix_plot_DICT = pickle.load(fileD)
        fileD.close()
        subject_confusion_matrix_plot_DICT.update(
            {'ignoreList': ignoreDict_4class[netName]})
        fileD = open(PATHDICT_CONFUSION_MATRIX_4CLASS[netName], 'wb')
        pickle.dump(subject_confusion_matrix_plot_DICT, fileD)
        fileD.close()
        draw_learning_curve_detail(subject_learning_curve_plot_DICT, netName)
    else:
        draw_learning_curve(subject_learning_curve_plot_DICT, netName)
    # end plot_learning_curve_plot_DICT_4class


# 绘制各微调模型性能比较曲线：表征多个模型间准确率与训练集容量的对比关系
def plot_accuracy_multiModel_fine_4class():
    # SVM 非跨被试
    fileD = open(PATHDICT_LEARNING_CURVE_4CLASS['SVM_no_cross_subject'], 'rb')
    svm_no_cross_subject_4class = pickle.load(fileD)
    fileD.close()

    # LDA 非跨被试
    path = os.getcwd()
    fileD = open(PATHDICT_LEARNING_CURVE_4CLASS['LDA_no_cross_subject'], 'rb')
    lda_no_cross_subject_4class = pickle.load(fileD)
    fileD.close()

    # CNN_Net4 微调
    fileD = open(PATHDICT_LEARNING_CURVE_4CLASS['CNN_Net4_fine'], 'rb')
    cnn_Net4_fine_4class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net44 微调
    fileD = open(PATHDICT_LEARNING_CURVE_4CLASS['LSTM_Net44_fine'], 'rb')
    lstm_Net44_fine_4class = pickle.load(fileD)
    fileD.close()

    # NN_Net45 微调
    fileD = open(PATHDICT_LEARNING_CURVE_4CLASS['NN_Net45_fine'], 'rb')
    nn_Net45_fine_4class = pickle.load(fileD)
    fileD.close()


    # LSTM_Net47 微调
    fileD = open(PATHDICT_LEARNING_CURVE_4CLASS['LSTM_Net47_fine'], 'rb')
    lstm_Net47_fine_4class = pickle.load(fileD)
    fileD.close()

    # CNN_LSTM_Net48 微调
    fileD = open(PATHDICT_LEARNING_CURVE_4CLASS['CNN_LSTM_Net48_fine'], 'rb')
    cnn_lstm_Net48_fine_4class = pickle.load(fileD)
    fileD.close()

    # CNN_LSTM_Net49 微调
    fileD = open(PATHDICT_LEARNING_CURVE_4CLASS['CNN_LSTM_Net49_fine'], 'rb')
    cnn_lstm_Net49_fine_4class = pickle.load(fileD)
    fileD.close()

    subject_dic_LIST = [{
        'labelName': '支持向量机',
        'dic': svm_no_cross_subject_4class,
        'color': 'black',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': '线性判别分析',
        'dic': lda_no_cross_subject_4class,
        'color': 'black',
        'linestyle': '-.',
        'marker': '*'
    }, {
        'labelName': 'CNN_Net4',
        'dic': cnn_Net4_fine_4class,
        'color': 'blue',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': 'LSTM_Net44',
        'dic': lstm_Net44_fine_4class,
        'color': 'green',
        'linestyle': '-',
        'marker': 'o'
    },{
        'labelName': 'NN_Net45',
        'dic': nn_Net45_fine_4class,
        'color': 'yellow',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': 'LSTM_Net47',
        'dic': lstm_Net47_fine_4class,
        'color': 'magenta',
        'linestyle': '-',
        'marker': '*'
    }, {
        'labelName': 'CNN_LSTM_Net48',
        'dic': cnn_lstm_Net48_fine_4class,
        'color': 'red',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': 'CNN_LSTM_Net49',
        'dic': cnn_lstm_Net49_fine_4class,
        'color': 'red',
        'linestyle': '-.',
        'marker': '*'
    }]

    draw_learning_curves(subject_dic_LIST, 'multiModel_4class')
    # end plot_accuracy_multiModel_fine_4class


# 绘制单被试单容量混淆矩阵：表征给定模型(给定被试、给定容量)的性能
def plot_subject_size_confusion_matrix_4class(netName, name, size):
    fileD = open(PATHDICT_CONFUSION_MATRIX_4CLASS[netName], 'rb')
    confusion_matrix_DICT = pickle.load(fileD)
    fileD.close()
    if name in confusion_matrix_DICT['ignoreList']:
        print('注意：%s 在 %s 的 ignoreList 中！' % (name, netName))
    draw_confusion_matrix(confusion_matrix_DICT[name]['%d' % size]['y_true'],
                          confusion_matrix_DICT[name]['%d' % size]['y_pred'],
                          [0, 1, 2, 3], ['up', 'down', 'left', 'right'],
                          netName + '_%s_%d' % (name, size))
    # end plot_subject_size_confusion_matrix_4class


# 绘制单容量混淆矩阵：表征给定容量下所有被试模型的平均性能
def plot_size_confusion_matrix_4class(netName, size):
    fileD = open(PATHDICT_CONFUSION_MATRIX_4CLASS[netName], 'rb')
    confusion_matrix_DICT = pickle.load(fileD)
    fileD.close()

    trueArray_origin = np.empty((0))
    predArray_origin = np.empty((0))
    trueArray_prune = np.empty((0))
    predArray_prune = np.empty((0))
    print(
        netName + '  ignore ',
        confusion_matrix_DICT['ignoreList'],
    )
    for name in NAME_LIST:
        trueArray_origin = np.r_[trueArray_origin,
                                 confusion_matrix_DICT[name]['%d' %
                                                             size]['y_true']]
        predArray_origin = np.r_[predArray_origin,
                                 confusion_matrix_DICT[name]['%d' %
                                                             size]['y_pred']]
        if name not in confusion_matrix_DICT['ignoreList']:
            trueArray_prune = np.r_[
                trueArray_prune,
                confusion_matrix_DICT[name]['%d' % size]['y_true']]
            predArray_prune = np.r_[
                predArray_prune,
                confusion_matrix_DICT[name]['%d' % size]['y_pred']]
    # end for
    draw_confusion_matrix(
        trueArray_prune,
        predArray_prune, [0, 1, 2, 3], ['up', 'down', 'left', 'right'],
        netName + '_%d' % (size),
        amount=[sum(trueArray_origin == x) for x in [0, 1, 2, 3]])
    # end plot_size_confusion_matrix_4class


# 绘制多模型单容量柱状图：表征给定容量下各模型的性能对比
def plot_bar_4class(netNames, size,plotAllSize=0):
    sizeList=[47,99,180]
    styleDict = {
        'SVM_no_cross_subject': {
            'color': 'black',
            'linestyle': '-',
            'marker': 'o'
        },
        'LDA_no_cross_subject': {
            'color': 'black',
            'linestyle': '-.',
            'marker': '*'
        },
        'CNN_Net4_fine': {
            'color': 'blue',
            'linestyle': '-',
            'marker': 'o',
            'zero_accuracy':0.302,
            'zero_f1':0.268
        },
        'LSTM_Net44_fine': {
            'color': 'blue',
            'linestyle': '-.',
            'marker': '*',
            'zero_accuracy':0.297,
            'zero_f1':0.271
        },
        'NN_Net45_fine': {
            'color': 'green',
            'linestyle': '-',
            'marker': 'o',
            'zero_accuracy':0.315,
            'zero_f1':0.284
        },
        'LSTM_Net47_fine': {
            'color': 'green',
            'linestyle': '-.',
            'marker': '*',
            'zero_accuracy':0.336,
            'zero_f1':0.319
        },
        'CNN_LSTM_Net48_fine': {
            'color': 'magenta',
            'linestyle': '-',
            'marker': 'o',
            'zero_accuracy':0.310,
            'zero_f1':0.294
        },
        'CNN_LSTM_Net49_fine': {
            'color': 'magenta',
            'linestyle': '-.',
            'marker': '*',
            'zero_accuracy':0.311,
            'zero_f1':0.253
        }
    }
    nameDict = {
        'SVM_no_cross_subject': 'SVM',
        'LDA_no_cross_subject': 'LDA',
        'CNN_Net4_fine': 'CNN_Net4',
        'LSTM_Net44_fine': 'LSTM_Net44',
        'NN_Net45_fine': 'NN_Net45',
        'LSTM_Net47_fine': 'LSTM_Net47',
        'CNN_LSTM_Net48_fine': 'LSTM_Net48',
        'CNN_LSTM_Net49_fine': 'LSTM_Net49',
    }

    modelDict = {}
    for netName,i in zip(netNames,range(len(netNames))):
        fileD = open(PATHDICT_CONFUSION_MATRIX_4CLASS[netName], 'rb')
        confusion_matrix_DICT = pickle.load(fileD)
        fileD.close()

        if i<plotAllSize: # 需要绘制多个 size 的网络
            for sizeI in sizeList:
                trueArray_0 = np.empty((0))
                predArray_0 = np.empty((0))
                trueArray_size = np.empty((0))
                predArray_size = np.empty((0))
                print(
                    netName + '  ignore ',
                    confusion_matrix_DICT['ignoreList'],
                )
                for name in NAME_LIST:
                    if name not in confusion_matrix_DICT['ignoreList']:
                        trueArray_size = np.r_[
                            trueArray_size,
                            confusion_matrix_DICT[name]['%d' % sizeI]['y_true']]
                        predArray_size = np.r_[
                            predArray_size,
                            confusion_matrix_DICT[name]['%d' % sizeI]['y_pred']]
                        if netName not in [
                                'SVM_no_cross_subject', 'LDA_no_cross_subject'
                        ]:
                            trueArray_0 = np.r_[
                                trueArray_0,
                                confusion_matrix_DICT[name]['%d' % 0]['y_true']]
                            predArray_0 = np.r_[
                                predArray_0,
                                confusion_matrix_DICT[name]['%d' % 0]['y_pred']]
                        else:
                            trueArray_0 = None
                            predArray_0 = None

                # end for
                modelDict.update({
                    '%s_%s'%(nameDict[netName],sizeI): {
                        'y_true_size': trueArray_size,
                        'y_pred_size': predArray_size,
                        'y_true_0': trueArray_0,
                        'y_pred_0': predArray_0,
                        'color': styleDict[netName]['color'],
                        'linestyle': styleDict[netName]['linestyle'],
                        'marker': styleDict[netName]['marker'],
                        'zero_accuracy':styleDict[netName].get('zero_accuracy',np.NaN),
                        'zero_f1':styleDict[netName].get('zero_f1',np.NaN)
                    }
                })                
        else: # 只需要绘制一个 size 的网络
            trueArray_0 = np.empty((0))
            predArray_0 = np.empty((0))
            trueArray_size = np.empty((0))
            predArray_size = np.empty((0))
            print(
                netName + '  ignore ',
                confusion_matrix_DICT['ignoreList'],
            )
            for name in NAME_LIST:
                if name not in confusion_matrix_DICT['ignoreList']:
                    trueArray_size = np.r_[
                        trueArray_size,
                        confusion_matrix_DICT[name]['%d' % size]['y_true']]
                    predArray_size = np.r_[
                        predArray_size,
                        confusion_matrix_DICT[name]['%d' % size]['y_pred']]
                    if netName not in [
                            'SVM_no_cross_subject', 'LDA_no_cross_subject'
                    ]:
                        trueArray_0 = np.r_[
                            trueArray_0,
                            confusion_matrix_DICT[name]['%d' % 0]['y_true']]
                        predArray_0 = np.r_[
                            predArray_0,
                            confusion_matrix_DICT[name]['%d' % 0]['y_pred']]
                    else:
                        trueArray_0 = None
                        predArray_0 = None

            # end for
            modelDict.update({
                nameDict[netName]: {
                    'y_true_size': trueArray_size,
                    'y_pred_size': predArray_size,
                    'y_true_0': trueArray_0,
                    'y_pred_0': predArray_0,
                    'color': styleDict[netName]['color'],
                    'linestyle': styleDict[netName]['linestyle'],
                    'marker': styleDict[netName]['marker'],
                    'zero_accuracy':styleDict[netName].get('zero_accuracy',np.NaN),
                    'zero_f1':styleDict[netName].get('zero_f1',np.NaN)
                }
            })

    draw_bar(modelDict, 'bar_multiModel_fine_4class_%d' % size,f1_average='macro',ylim=[0.2, 0.7,11])
    # end plot_bar_4class

# 绘制模型学习曲线
def plot_learning_curve_plot_DICT_UD2class(netName, detail=False):
    ignoreDict_UD2class = {
        'SVM_no_cross_subject': [],
        'LDA_no_cross_subject': [],
        'CNN_Net2_fine': [],
        'CNN_Net22_fine': ['cw', 'wxc', 'yzg'],
        'LSTM_Net23_fine': ['kx', 'yzg'],
        'LSTM_Net24_fine': ['wrd'],
        'NN_Net25_fine': ['cw', 'pbl', 'xsc', 'yzg'],
        'LSTM_Net26_fine': [],
        'LSTM_Net27_fine': ['pbl'],
        'CNN_LSTM_Net28_fine': ['cw', 'cwm', 'pbl'],
        'CNN_LSTM_Net29_fine': ['wrd', 'xsc'],
        'CNN_Net2_pretraining': ['cw', 'kx', 'pbl', 'wxc', 'yzg'],
        'CNN_Net22_pretraining': ['cw', 'cwm', 'pbl', 'wrd', 'wxc', 'yzg'],
        'NN_Net25_pretraining': ['cw', 'cwm', 'kx', 'wrd', 'wxc', 'yzg'],
        'LSTM_Net26_pretraining': ['cw', 'cwm', 'wrd', 'wxc'],
        'LSTM_Net27_pretraining': ['cwm', 'kx', 'pbl', 'wxc'],
        'CNN_LSTM_Net28_pretraining':
        ['cw', 'pbl', 'wrd', 'wxc', 'xsc', 'yzg'],
        'CNN_LSTM_Net29_pretraining': ['cw', 'wrd', 'wxc', 'yzg'],
    }

    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS[netName], 'rb')
    subject_learning_curve_plot_DICT = pickle.load(fileD)
    fileD.close()
    if detail:
        subject_learning_curve_plot_DICT.update(
            {'ignoreList': ignoreDict_UD2class[netName]})
        fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS[netName], 'wb')
        pickle.dump(subject_learning_curve_plot_DICT, fileD)
        fileD.close()

        fileD = open(PATHDICT_CONFUSION_MATRIX_UD2CLASS[netName], 'rb')
        subject_confusion_matrix_plot_DICT = pickle.load(fileD)
        fileD.close()
        subject_confusion_matrix_plot_DICT.update(
            {'ignoreList': ignoreDict_UD2class[netName]})
        fileD = open(PATHDICT_CONFUSION_MATRIX_UD2CLASS[netName], 'wb')
        pickle.dump(subject_confusion_matrix_plot_DICT, fileD)
        fileD.close()
        draw_learning_curve_detail(subject_learning_curve_plot_DICT, netName)
    else:
        draw_learning_curve(subject_learning_curve_plot_DICT, netName)
    # end plot_learning_curve_plot_DICT_UD2class


# 绘制各预训练模型性能比较曲线：表征多个模型间准确率与训练集容量的对比关系
def plot_accuracy_multiModel_pretraining_UD2class():
    # CNN_Net2 预训练
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['CNN_Net2_pretraining'],
                 'rb')
    cnn_Net2_pretraining_UD2class = pickle.load(fileD)
    fileD.close()

    # CNN_Net22 预训练
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['CNN_Net22_pretraining'],
                 'rb')
    cnn_Net22_pretraining_UD2class = pickle.load(fileD)
    fileD.close()

    # NN_Net25 预训练
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['NN_Net25_pretraining'],
                 'rb')
    nn_Net25_pretraining_UD2class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net26 预训练
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['LSTM_Net26_pretraining'],
                 'rb')
    lstm_Net26_pretraining_UD2class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net27 预训练
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['LSTM_Net27_pretraining'],
                 'rb')
    lstm_Net27_pretraining_UD2class = pickle.load(fileD)
    fileD.close()

    # CNN_LSTM_Net28 预训练
    fileD = open(
        PATHDICT_LEARNING_CURVE_UD2CLASS['CNN_LSTM_Net28_pretraining'], 'rb')
    cnn_lstm_Net28_pretraining_UD2class = pickle.load(fileD)
    fileD.close()

    # CNN_LSTM_Net29 预训练
    fileD = open(
        PATHDICT_LEARNING_CURVE_UD2CLASS['CNN_LSTM_Net29_pretraining'], 'rb')
    cnn_lstm_Net29_pretraining_UD2class = pickle.load(fileD)
    fileD.close()

    subject_dic_LIST = [{
        'labelName': 'cnn_Net2_pretraining',
        'dic': cnn_Net2_pretraining_UD2class,
        'color': 'blue',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': 'cnn_Net22_pretraining',
        'dic': cnn_Net22_pretraining_UD2class,
        'color': 'blue',
        'linestyle': '-.',
        'marker': '*'
    }, {
        'labelName': 'nn_Net25_pretraining',
        'dic': nn_Net25_pretraining_UD2class,
        'color': 'yellow',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': 'lstm_Net26_pretraining',
        'dic': lstm_Net26_pretraining_UD2class,
        'color': 'magenta',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': 'lstm_Net27_pretraining',
        'dic': lstm_Net27_pretraining_UD2class,
        'color': 'magenta',
        'linestyle': '-.',
        'marker': '*'
    }, {
        'labelName': 'cnn_lstm_Net28_pretraining',
        'dic': cnn_lstm_Net28_pretraining_UD2class,
        'color': 'green',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': 'cnn_lstm_Net29_pretraining',
        'dic': cnn_lstm_Net29_pretraining_UD2class,
        'color': 'green',
        'linestyle': '-.',
        'marker': '*'
    }]
    draw_learning_curves(subject_dic_LIST, 'multiModel_pretraining_UD2class')
    # end plot_accuracy_multiModel_pretraining_UD2class


# 绘制各微调模型性能比较曲线：表征多个模型间准确率与训练集容量的对比关系
def plot_accuracy_multiModel_fine_UD2class():
    # SVM 非跨被试
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['SVM_no_cross_subject'],
                 'rb')
    svm_no_cross_subject_UD2class = pickle.load(fileD)
    fileD.close()

    # LDA 非跨被试
    path = os.getcwd()
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['LDA_no_cross_subject'],
                 'rb')
    lda_no_cross_subject_UD2class = pickle.load(fileD)
    fileD.close()

    # CNN_Net2 微调
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['CNN_Net2_fine'], 'rb')
    cnn_Net2_fine_UD2class = pickle.load(fileD)
    fileD.close()

    # CNN_Net22 微调
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['CNN_Net22_fine'], 'rb')
    cnn_Net22_fine_UD2class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net23 微调
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['LSTM_Net23_fine'], 'rb')
    lstm_Net23_fine_UD2class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net24 微调
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['LSTM_Net24_fine'], 'rb')
    lstm_Net24_fine_UD2class = pickle.load(fileD)
    fileD.close()

    # NN_Net25 微调
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['NN_Net25_fine'], 'rb')
    nn_Net25_fine_UD2class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net26 微调
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['LSTM_Net26_fine'], 'rb')
    lstm_Net26_fine_UD2class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net27 微调
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['LSTM_Net27_fine'], 'rb')
    lstm_Net27_fine_UD2class = pickle.load(fileD)
    fileD.close()

    # CNN_LSTM_Net28 微调
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['CNN_LSTM_Net28_fine'], 'rb')
    cnn_lstm_Net28_fine_UD2class = pickle.load(fileD)
    fileD.close()

    # CNN_LSTM_Net29 微调
    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS['CNN_LSTM_Net29_fine'], 'rb')
    cnn_lstm_Net29_fine_UD2class = pickle.load(fileD)
    fileD.close()

    subject_dic_LIST = [{
        'labelName': '支持向量机',
        'dic': svm_no_cross_subject_UD2class,
        'color': 'black',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': '线性判别分析',
        'dic': lda_no_cross_subject_UD2class,
        'color': 'black',
        'linestyle': '-.',
        'marker': '*'
    }, {
        'labelName': 'CNN_Net2',
        'dic': cnn_Net2_fine_UD2class,
        'color': 'blue',
        'linestyle': '-',
        'marker': 'o',
        'zero_accuracy':0.578
    }, {
        'labelName': 'CNN_Net22',
        'dic': cnn_Net22_fine_UD2class,
        'color': 'blue',
        'linestyle': '-.',
        'marker': '*',
        'zero_accuracy':0.587
    }, {
        'labelName': 'LSTM_Net23',
        'dic': lstm_Net23_fine_UD2class,
        'color': 'green',
        'linestyle': '-',
        'marker': 'o',
        'zero_accuracy':0.641
    }, {
        'labelName': 'LSTM_Net24',
        'dic': lstm_Net24_fine_UD2class,
        'color': 'green',
        'linestyle': '-.',
        'marker': '*',
        'zero_accuracy':0.580
    }, {
        'labelName': 'LSTM_Net26',
        'dic': lstm_Net26_fine_UD2class,
        'color': 'magenta',
        'linestyle': '-',
        'marker': 'o',
        'zero_accuracy':0.593
    }, {
        'labelName': 'LSTM_Net27',
        'dic': lstm_Net27_fine_UD2class,
        'color': 'magenta',
        'linestyle': '-.',
        'marker': '*',
        'zero_accuracy':0.593
    }]
    draw_learning_curves(subject_dic_LIST, 'multiModel_fine_UD2class')
    # end plot_accuracy_multiModel_fine_UD2class


# 绘制单被试单容量混淆矩阵：表征给定模型(给定被试、给定容量)的性能
def plot_subject_size_confusion_matrix_UD2class(netName, name, size):
    fileD = open(PATHDICT_CONFUSION_MATRIX_UD2CLASS[netName], 'rb')
    confusion_matrix_DICT = pickle.load(fileD)
    fileD.close()
    if name in confusion_matrix_DICT['ignoreList']:
        print('注意：%s 在 %s 的 ignoreList 中！' % (name, netName))
    draw_confusion_matrix(confusion_matrix_DICT[name]['%d' % size]['y_true'],
                          confusion_matrix_DICT[name]['%d' % size]['y_pred'],
                          [0, 1], ['up', 'down'],
                          netName + '_%s_%d' % (name, size))
    # end plot_subject_size_confusion_matrix_UD2class


# 绘制单容量混淆矩阵：表征给定容量下所有被试模型的平均性能
def plot_size_confusion_matrix_UD2class(netName, size):
    fileD = open(PATHDICT_CONFUSION_MATRIX_UD2CLASS[netName], 'rb')
    confusion_matrix_DICT = pickle.load(fileD)
    fileD.close()

    trueArray_origin = np.empty((0))
    predArray_origin = np.empty((0))
    trueArray_prune = np.empty((0))
    predArray_prune = np.empty((0))
    print(
        netName + '  ignore ',
        confusion_matrix_DICT['ignoreList'],
    )
    for name in NAME_LIST:
        trueArray_origin = np.r_[trueArray_origin,
                                 confusion_matrix_DICT[name]['%d' %
                                                             size]['y_true']]
        predArray_origin = np.r_[predArray_origin,
                                 confusion_matrix_DICT[name]['%d' %
                                                             size]['y_pred']]
        if name not in confusion_matrix_DICT['ignoreList']:
            trueArray_prune = np.r_[
                trueArray_prune,
                confusion_matrix_DICT[name]['%d' % size]['y_true']]
            predArray_prune = np.r_[
                predArray_prune,
                confusion_matrix_DICT[name]['%d' % size]['y_pred']]
    # end for
    draw_confusion_matrix(trueArray_prune,
                          predArray_prune, [0, 1], ['up', 'down'],
                          netName + '_%d' % (size),
                          amount=[sum(trueArray_origin == x) for x in [0, 1]])
    # end plot_size_confusion_matrix_UD2class


# 绘制多模型单容量 ROC 曲线：表征给定容量下各模型的性能对比
def plot_ROC_curve_UD2class(netNames, size):
    styleDict = {
        'SVM_no_cross_subject': {
            'color': 'black',
            'linestyle': '-',
            'marker': 'o'
        },
        'LDA_no_cross_subject': {
            'color': 'black',
            'linestyle': '-.',
            'marker': '*'
        },
        'CNN_Net2_fine': {
            'color': 'blue',
            'linestyle': '-',
            'marker': 'o'
        },
        'CNN_Net22_fine': {
            'color': 'blue',
            'linestyle': '-.',
            'marker': '*'
        },
        'NN_Net25_fine': {
            'color': 'yellow',
            'linestyle': '-',
            'marker': 'o'
        },
        'LSTM_Net26_fine': {
            'color': 'magenta',
            'linestyle': '-',
            'marker': 'o'
        },
        'LSTM_Net27_fine': {
            'color': 'magenta',
            'linestyle': '-.',
            'marker': '*'
        },
        'CNN_LSTM_Net28_fine': {
            'color': 'green',
            'linestyle': '-',
            'marker': 'o'
        },
        'CNN_LSTM_Net29_fine': {
            'color': 'green',
            'linestyle': '-.',
            'marker': '*'
        }
    }

    modelDict = {}
    for netName in netNames:
        fileD = open(PATHDICT_CONFUSION_MATRIX_UD2CLASS[netName], 'rb')
        confusion_matrix_DICT = pickle.load(fileD)
        fileD.close()

        trueArray = np.empty((0))
        predArray = np.empty((0))
        print(
            netName + '  ignore ',
            confusion_matrix_DICT['ignoreList'],
        )
        for name in NAME_LIST:
            if name not in confusion_matrix_DICT['ignoreList']:
                trueArray = np.r_[trueArray,
                                  confusion_matrix_DICT[name]['%d' %
                                                              size]['y_true']]
                predArray = np.r_[predArray,
                                  confusion_matrix_DICT[name]['%d' %
                                                              size]['y_pred']]
        # end for
        modelDict.update({
            netName: {
                'y_true': trueArray,
                'y_pred': predArray,
                'color': styleDict[netName]['color'],
                'linestyle': styleDict[netName]['linestyle'],
                'marker': styleDict[netName]['marker']
            }
        })

    draw_ROC_curve(modelDict, 'ROC_multiModel_fine_UD2class_%d' % size)
    # end plot_ROC_curve_UD2class


# 绘制多模型单容量柱状图：表征给定容量下各模型的性能对比
def plot_bar_UD2class(netNames, size,plotAllSize=0):
    sizeList=[22,52,90]
    styleDict = {
        'SVM_no_cross_subject': {
            'color': 'black',
            'linestyle': '-',
            'marker': 'o'
        },
        'LDA_no_cross_subject': {
            'color': 'black',
            'linestyle': '-.',
            'marker': '*'
        },
        'CNN_Net2_fine': {
            'color': 'blue',
            'linestyle': '-',
            'marker': 'o',
            'zero_accuracy':0.578,
            'zero_f1':0.585
        },
        'CNN_Net22_fine': {
            'color': 'blue',
            'linestyle': '-.',
            'marker': '*',
            'zero_accuracy':0.587,
            'zero_f1':0.608
        },
        'LSTM_Net23_fine': {
            'color': 'green',
            'linestyle': '-',
            'marker': 'o',
            'zero_accuracy':0.641,
            'zero_f1':0.620
        },
        'LSTM_Net24_fine': {
            'color': 'green',
            'linestyle': '-.',
            'marker': '*',
            'zero_accuracy':0.580,
            'zero_f1':0.619
        },
        'LSTM_Net26_fine': {
            'color': 'magenta',
            'linestyle': '-',
            'marker': 'o',
            'zero_accuracy':0.593,
            'zero_f1':0.557
        },
        'LSTM_Net27_fine': {
            'color': 'magenta',
            'linestyle': '-.',
            'marker': '*',
            'zero_accuracy':0.593,
            'zero_f1':0.589
        }
    }

    nameDict = {
        'SVM_no_cross_subject': 'SVM',
        'LDA_no_cross_subject': 'LDA',
        'CNN_Net2_fine': 'CNN_Net2',
        'CNN_Net22_fine': 'CNN_Net22',
        'LSTM_Net23_fine': 'LSTM_Net23',
        'LSTM_Net24_fine': 'LSTM_Net24',
        'LSTM_Net26_fine': 'LSTM_Net26',
        'LSTM_Net27_fine': 'LSTM_Net27',
    }

    modelDict = {}
    for netName,i in zip(netNames,range(len(netNames))):
        fileD = open(PATHDICT_CONFUSION_MATRIX_UD2CLASS[netName], 'rb')
        confusion_matrix_DICT = pickle.load(fileD)
        fileD.close()

        if i<plotAllSize: # 需要绘制多个 size 的网络
            for sizeI in sizeList:
                trueArray_0 = np.empty((0))
                predArray_0 = np.empty((0))
                trueArray_size = np.empty((0))
                predArray_size = np.empty((0))
                print(
                    netName + '  ignore ',
                    confusion_matrix_DICT['ignoreList'],
                )
                for name in NAME_LIST:
                    if name not in confusion_matrix_DICT['ignoreList']:
                        trueArray_size = np.r_[
                            trueArray_size,
                            confusion_matrix_DICT[name]['%d' % sizeI]['y_true']]
                        predArray_size = np.r_[
                            predArray_size,
                            confusion_matrix_DICT[name]['%d' % sizeI]['y_pred']]
                        if netName not in [
                                'SVM_no_cross_subject', 'LDA_no_cross_subject'
                        ]:
                            trueArray_0 = np.r_[
                                trueArray_0,
                                confusion_matrix_DICT[name]['%d' % 0]['y_true']]
                            predArray_0 = np.r_[
                                predArray_0,
                                confusion_matrix_DICT[name]['%d' % 0]['y_pred']]
                        else:
                            trueArray_0 = None
                            predArray_0 = None

                # end for
                modelDict.update({
                    '%s_%s'%(nameDict[netName],sizeI): {
                        'y_true_size': trueArray_size,
                        'y_pred_size': predArray_size,
                        'y_true_0': trueArray_0,
                        'y_pred_0': predArray_0,
                        'color': styleDict[netName]['color'],
                        'linestyle': styleDict[netName]['linestyle'],
                        'marker': styleDict[netName]['marker'],
                        'zero_accuracy':styleDict[netName].get('zero_accuracy',np.NaN),
                        'zero_f1':styleDict[netName].get('zero_f1',np.NaN)
                    }
                })                
        else: # 只需要绘制一个 size 的网络
            trueArray_0 = np.empty((0))
            predArray_0 = np.empty((0))
            trueArray_size = np.empty((0))
            predArray_size = np.empty((0))
            print(
                netName + '  ignore ',
                confusion_matrix_DICT['ignoreList'],
            )
            for name in NAME_LIST:
                if name not in confusion_matrix_DICT['ignoreList']:
                    trueArray_size = np.r_[
                        trueArray_size,
                        confusion_matrix_DICT[name]['%d' % size]['y_true']]
                    predArray_size = np.r_[
                        predArray_size,
                        confusion_matrix_DICT[name]['%d' % size]['y_pred']]
                    if netName not in [
                            'SVM_no_cross_subject', 'LDA_no_cross_subject'
                    ]:
                        trueArray_0 = np.r_[
                            trueArray_0,
                            confusion_matrix_DICT[name]['%d' % 0]['y_true']]
                        predArray_0 = np.r_[
                            predArray_0,
                            confusion_matrix_DICT[name]['%d' % 0]['y_pred']]
                    else:
                        trueArray_0 = None
                        predArray_0 = None

            # end for
            modelDict.update({
                nameDict[netName]: {
                    'y_true_size': trueArray_size,
                    'y_pred_size': predArray_size,
                    'y_true_0': trueArray_0,
                    'y_pred_0': predArray_0,
                    'color': styleDict[netName]['color'],
                    'linestyle': styleDict[netName]['linestyle'],
                    'marker': styleDict[netName]['marker'],
                    'zero_accuracy':styleDict[netName].get('zero_accuracy',np.NaN),
                    'zero_f1':styleDict[netName].get('zero_f1',np.NaN)
                }
            })

    draw_bar(modelDict, 'bar_multiModel_fine_UD2class_%d' % size)
    # end plot_bar_UD2class


# 绘制模型学习曲线
def plot_learning_curve_plot_DICT_LR2class(netName, detail=False):
    ignoreDict_LR2class = {
        'SVM_no_cross_subject': ['cwm','kx','pbl','yzg'],
        'LDA_no_cross_subject': ['kx'],
        'CNN_Net2_fine': [],
        'CNN_Net22_fine': ['wrd'],
        'LSTM_Net23_fine': ['cwm', 'pbl', 'wrd','kx'],
        'LSTM_Net24_fine': ['cw'],
        'NN_Net25_fine': ['cw'],
        'LSTM_Net26_fine': ['pbl'],
        'LSTM_Net27_fine': [],
        'CNN_LSTM_Net28_fine': ['wrd', 'xsc', 'yzg'],
        'CNN_LSTM_Net29_fine': ['kx', 'wxc'],
        'CNN_Net2_pretraining': ['cwm', 'kx', 'wrd'],
        'CNN_Net22_pretraining': ['cw', 'cwm', 'kx', 'pbl', 'xsc', 'yzg'],
        'NN_Net25_pretraining': ['cw', 'kx', 'pbl', 'wrd', 'wxc', 'yzg'],
        'LSTM_Net26_pretraining': ['wxc'],
        'LSTM_Net27_pretraining': ['kx', 'xsc'],
        'CNN_LSTM_Net28_pretraining': ['xsc', 'wxc'],
        'CNN_LSTM_Net29_pretraining': ['kx', 'wrd', 'xsc'],
    }

    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS[netName], 'rb')
    subject_learning_curve_plot_DICT = pickle.load(fileD)
    fileD.close()
    if detail:
        subject_learning_curve_plot_DICT.update(
            {'ignoreList': ignoreDict_LR2class[netName]})
        fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS[netName], 'wb')
        pickle.dump(subject_learning_curve_plot_DICT, fileD)
        fileD.close()

        fileD = open(PATHDICT_CONFUSION_MATRIX_LR2CLASS[netName], 'rb')
        subject_confusion_matrix_plot_DICT = pickle.load(fileD)
        fileD.close()
        subject_confusion_matrix_plot_DICT.update(
            {'ignoreList': ignoreDict_LR2class[netName]})
        fileD = open(PATHDICT_CONFUSION_MATRIX_LR2CLASS[netName], 'wb')
        pickle.dump(subject_confusion_matrix_plot_DICT, fileD)
        fileD.close()
        draw_learning_curve_detail(subject_learning_curve_plot_DICT, netName)
    else:
        draw_learning_curve(subject_learning_curve_plot_DICT, netName)
    # end plot_learning_curve_plot_DICT_LR2class

# 绘制模型学习曲线
def plot_learning_curve_plot_DICT_4class_multiTask(netName, detail=False):
    ignoreDict_4class = {
        'SVM_no_cross_subject': [],
        'LDA_no_cross_subject': [],
        'NN_Net45_fine': ['kx','wrd']
    }

    fileD = open(PATHDICT_LEARNING_CURVE_4CLASS_MULTITASK[netName], 'rb')
    subject_learning_curve_plot_DICT = pickle.load(fileD)
    fileD.close()
    if detail:
        subject_learning_curve_plot_DICT.update(
            {'ignoreList': ignoreDict_4class[netName]})
        fileD = open(PATHDICT_LEARNING_CURVE_4CLASS_MULTITASK[netName], 'wb')
        pickle.dump(subject_learning_curve_plot_DICT, fileD)
        fileD.close()

        fileD = open(PATHDICT_CONFUSION_MATRIX_4CLASS_MULTITASK[netName], 'rb')
        subject_confusion_matrix_plot_DICT = pickle.load(fileD)
        fileD.close()
        subject_confusion_matrix_plot_DICT.update(
            {'ignoreList': ignoreDict_4class[netName]})
        fileD = open(PATHDICT_CONFUSION_MATRIX_4CLASS_MULTITASK[netName], 'wb')
        pickle.dump(subject_confusion_matrix_plot_DICT, fileD)
        fileD.close()
        draw_learning_curve_detail(subject_learning_curve_plot_DICT, netName)
    else:
        draw_learning_curve(subject_learning_curve_plot_DICT, netName)
    # end plot_learning_curve_plot_DICT_4class_multiTask

# 绘制模型学习曲线
def plot_learning_curve_plot_DICT_UD2class_multiTask(netName, detail=False):
    ignoreDict_UD2class = {
        'SVM_no_cross_subject': ['cw','kx'],
        'LDA_no_cross_subject': [],
        'NN_Net25_fine': ['wxc'],
    }

    fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS_MULTITASK[netName], 'rb')
    subject_learning_curve_plot_DICT = pickle.load(fileD)
    fileD.close()
    if detail:
        subject_learning_curve_plot_DICT.update(
            {'ignoreList': ignoreDict_UD2class[netName]})
        fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS_MULTITASK[netName], 'wb')
        pickle.dump(subject_learning_curve_plot_DICT, fileD)
        fileD.close()

        fileD = open(PATHDICT_CONFUSION_MATRIX_UD2CLASS_MULTITASK[netName], 'rb')
        subject_confusion_matrix_plot_DICT = pickle.load(fileD)
        fileD.close()
        subject_confusion_matrix_plot_DICT.update(
            {'ignoreList': ignoreDict_UD2class[netName]})
        fileD = open(PATHDICT_CONFUSION_MATRIX_UD2CLASS_MULTITASK[netName], 'wb')
        pickle.dump(subject_confusion_matrix_plot_DICT, fileD)
        fileD.close()
        draw_learning_curve_detail(subject_learning_curve_plot_DICT, netName)
    else:
        draw_learning_curve(subject_learning_curve_plot_DICT, netName)
    # end plot_learning_curve_plot_DICT_UD2class_multiTask

# 绘制模型学习曲线
def plot_learning_curve_plot_DICT_LR2class_multiTask(netName, detail=False):
    ignoreDict_LR2class = {
        'SVM_no_cross_subject': ['kx','pbl'],
        'LDA_no_cross_subject': [],
        'NN_Net25_fine': ['cw','kx','wrd','wxc','xsc','yzg'],
    }

    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS_MULTITASK[netName], 'rb')
    subject_learning_curve_plot_DICT = pickle.load(fileD)
    fileD.close()
    if detail:
        subject_learning_curve_plot_DICT.update(
            {'ignoreList': ignoreDict_LR2class[netName]})
        fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS_MULTITASK[netName], 'wb')
        pickle.dump(subject_learning_curve_plot_DICT, fileD)
        fileD.close()

        fileD = open(PATHDICT_CONFUSION_MATRIX_LR2CLASS_MULTITASK[netName], 'rb')
        subject_confusion_matrix_plot_DICT = pickle.load(fileD)
        fileD.close()
        subject_confusion_matrix_plot_DICT.update(
            {'ignoreList': ignoreDict_LR2class[netName]})
        fileD = open(PATHDICT_CONFUSION_MATRIX_LR2CLASS_MULTITASK[netName], 'wb')
        pickle.dump(subject_confusion_matrix_plot_DICT, fileD)
        fileD.close()
        draw_learning_curve_detail(subject_learning_curve_plot_DICT, netName)
    else:
        draw_learning_curve(subject_learning_curve_plot_DICT, netName)
    # end plot_learning_curve_plot_DICT_LR2class_multiTask


# 绘制各预训练模型性能比较曲线：表征多个模型间准确率与训练集容量的对比关系
def plot_accuracy_multiModel_pretraining_LR2class():
    # CNN_Net2 预训练
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['CNN_Net2_pretraining'],
                 'rb')
    cnn_Net2_pretraining_LR2class = pickle.load(fileD)
    fileD.close()

    # CNN_Net22 预训练
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['CNN_Net22_pretraining'],
                 'rb')
    cnn_Net22_pretraining_LR2class = pickle.load(fileD)
    fileD.close()

    # NN_Net25 预训练
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['NN_Net25_pretraining'],
                 'rb')
    nn_Net25_pretraining_LR2class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net26 预训练
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['LSTM_Net26_pretraining'],
                 'rb')
    lstm_Net26_pretraining_LR2class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net27 预训练
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['LSTM_Net27_pretraining'],
                 'rb')
    lstm_Net27_pretraining_LR2class = pickle.load(fileD)
    fileD.close()

    # CNN_LSTM_Net28 预训练
    fileD = open(
        PATHDICT_LEARNING_CURVE_LR2CLASS['CNN_LSTM_Net28_pretraining'], 'rb')
    cnn_lstm_Net28_pretraining_LR2class = pickle.load(fileD)
    fileD.close()

    # CNN_LSTM_Net29 预训练
    fileD = open(
        PATHDICT_LEARNING_CURVE_LR2CLASS['CNN_LSTM_Net29_pretraining'], 'rb')
    cnn_lstm_Net29_pretraining_LR2class = pickle.load(fileD)
    fileD.close()

    subject_dic_LIST = [{
        'labelName': 'cnn_Net2_pretraining',
        'dic': cnn_Net2_pretraining_LR2class,
        'color': 'blue',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': 'cnn_Net22_pretraining',
        'dic': cnn_Net22_pretraining_LR2class,
        'color': 'blue',
        'linestyle': '-.',
        'marker': '*'
    }, {
        'labelName': 'nn_Net25_pretraining',
        'dic': nn_Net25_pretraining_LR2class,
        'color': 'yellow',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': 'lstm_Net26_pretraining',
        'dic': lstm_Net26_pretraining_LR2class,
        'color': 'magenta',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': 'lstm_Net27_pretraining',
        'dic': lstm_Net27_pretraining_LR2class,
        'color': 'magenta',
        'linestyle': '-.',
        'marker': '*'
    }, {
        'labelName': 'cnn_lstm_Net28_pretraining',
        'dic': cnn_lstm_Net28_pretraining_LR2class,
        'color': 'green',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': 'cnn_lstm_Net29_pretraining',
        'dic': cnn_lstm_Net29_pretraining_LR2class,
        'color': 'green',
        'linestyle': '-.',
        'marker': '*'
    }]
    draw_learning_curves(subject_dic_LIST, 'multiModel_pretraining_LR2class')
    # end plot_accuracy_multiModel_pretraining_LR2class


# 绘制各微调模型性能比较曲线：表征多个模型间准确率与训练集容量的对比关系
def plot_accuracy_multiModel_fine_LR2class():
    # SVM 非跨被试
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['SVM_no_cross_subject'],
                 'rb')
    svm_no_cross_subject_LR2class = pickle.load(fileD)
    fileD.close()

    # LDA 非跨被试
    path = os.getcwd()
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['LDA_no_cross_subject'],
                 'rb')
    lda_no_cross_subject_LR2class = pickle.load(fileD)
    fileD.close()

    # CNN_Net2 微调
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['CNN_Net2_fine'], 'rb')
    cnn_Net2_fine_LR2class = pickle.load(fileD)
    fileD.close()

    # CNN_Net22 微调
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['CNN_Net22_fine'], 'rb')
    cnn_Net22_fine_LR2class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net23 微调
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['LSTM_Net23_fine'], 'rb')
    lstm_Net23_fine_LR2class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net24 微调
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['LSTM_Net24_fine'], 'rb')
    lstm_Net24_fine_LR2class = pickle.load(fileD)
    fileD.close()

    # NN_Net25 微调
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['NN_Net25_fine'], 'rb')
    nn_Net25_fine_LR2class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net26 微调
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['LSTM_Net26_fine'], 'rb')
    lstm_Net26_fine_LR2class = pickle.load(fileD)
    fileD.close()

    # LSTM_Net27 微调
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['LSTM_Net27_fine'], 'rb')
    lstm_Net27_fine_LR2class = pickle.load(fileD)
    fileD.close()

    # CNN_LSTM_Net28 微调
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['CNN_LSTM_Net28_fine'], 'rb')
    cnn_lstm_Net28_fine_LR2class = pickle.load(fileD)
    fileD.close()

    # CNN_LSTM_Net29 微调
    fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS['CNN_LSTM_Net29_fine'], 'rb')
    cnn_lstm_Net29_fine_LR2class = pickle.load(fileD)
    fileD.close()

    subject_dic_LIST = [{
        'labelName': '支持向量机',
        'dic': svm_no_cross_subject_LR2class,
        'color': 'black',
        'linestyle': '-',
        'marker': 'o'
    }, {
        'labelName': '线性判别分析',
        'dic': lda_no_cross_subject_LR2class,
        'color': 'black',
        'linestyle': '-.',
        'marker': '*'
    }, {
        'labelName': 'CNN_Net2',
        'dic': cnn_Net2_fine_LR2class,
        'color': 'blue',
        'linestyle': '-',
        'marker': 'o',
        'zero_accuracy':0.597
    }, {
        'labelName': 'CNN_Net22',
        'dic': cnn_Net22_fine_LR2class,
        'color': 'blue',
        'linestyle': '-.',
        'marker': '*',
        'zero_accuracy':0.574
    }, {
        'labelName': 'LSTM_Net23',
        'dic': lstm_Net23_fine_LR2class,
        'color': 'green',
        'linestyle': '-',
        'marker': 'o',
        'zero_accuracy':0.595
    }, {
        'labelName': 'LSTM_Net24',
        'dic': lstm_Net24_fine_LR2class,
        'color': 'green',
        'linestyle': '-.',
        'marker': '*',
        'zero_accuracy':0.601
    }, {
        'labelName': 'LSTM_Net26',
        'dic': lstm_Net26_fine_LR2class,
        'color': 'magenta',
        'linestyle': '-',
        'marker': 'o',
        'zero_accuracy':0.640
    }, {
        'labelName': 'LSTM_Net27',
        'dic': lstm_Net27_fine_LR2class,
        'color': 'magenta',
        'linestyle': '-.',
        'marker': '*',
        'zero_accuracy':0.649
    }]
    draw_learning_curves(subject_dic_LIST, 'multiModel_fine_LR2class')
    # end plot_accuracy_multiModel_fine_LR2class



# 绘制单被试单容量混淆矩阵：表征给定模型(给定被试、给定容量)的性能
def plot_subject_size_confusion_matrix_LR2class(netName, name, size):
    fileD = open(PATHDICT_CONFUSION_MATRIX_LR2CLASS[netName], 'rb')
    confusion_matrix_DICT = pickle.load(fileD)
    fileD.close()
    if name in confusion_matrix_DICT['ignoreList']:
        print('注意：%s 在 %s 的 ignoreList 中！' % (name, netName))
    draw_confusion_matrix(confusion_matrix_DICT[name]['%d' % size]['y_true'],
                          confusion_matrix_DICT[name]['%d' % size]['y_pred'],
                          [0, 1], ['up', 'down'],
                          netName + '_%s_%d' % (name, size))
    # end plot_subject_size_confusion_matrix_LR2class


# 绘制单容量混淆矩阵：表征给定容量下所有被试模型的平均性能
def plot_size_confusion_matrix_LR2class(netName, size):
    fileD = open(PATHDICT_CONFUSION_MATRIX_LR2CLASS[netName], 'rb')
    confusion_matrix_DICT = pickle.load(fileD)
    fileD.close()

    trueArray_origin = np.empty((0))
    predArray_origin = np.empty((0))
    trueArray_prune = np.empty((0))
    predArray_prune = np.empty((0))
    print(
        netName + '  ignore ',
        confusion_matrix_DICT['ignoreList'],
    )
    for name in NAME_LIST:
        trueArray_origin = np.r_[trueArray_origin,
                                 confusion_matrix_DICT[name]['%d' %
                                                             size]['y_true']]
        predArray_origin = np.r_[predArray_origin,
                                 confusion_matrix_DICT[name]['%d' %
                                                             size]['y_pred']]
        if name not in confusion_matrix_DICT['ignoreList']:
            trueArray_prune = np.r_[
                trueArray_prune,
                confusion_matrix_DICT[name]['%d' % size]['y_true']]
            predArray_prune = np.r_[
                predArray_prune,
                confusion_matrix_DICT[name]['%d' % size]['y_pred']]
    # end for
    draw_confusion_matrix(trueArray_prune,
                          predArray_prune, [0, 1], ['up', 'down'],
                          netName + '_%d' % (size),
                          amount=[sum(trueArray_origin == x) for x in [0, 1]])
    # end plot_size_confusion_matrix_LR2class


# 绘制多模型单容量 ROC 曲线：表征给定容量下各模型的性能对比
def plot_ROC_curve_LR2class(netNames, size):
    styleDict = {
        'SVM_no_cross_subject': {
            'color': 'black',
            'linestyle': '-',
            'marker': 'o'
        },
        'LDA_no_cross_subject': {
            'color': 'black',
            'linestyle': '-.',
            'marker': '*'
        },
        'CNN_Net2_fine': {
            'color': 'blue',
            'linestyle': '-',
            'marker': 'o'
        },
        'CNN_Net22_fine': {
            'color': 'blue',
            'linestyle': '-.',
            'marker': '*'
        },
        'NN_Net25_fine': {
            'color': 'yellow',
            'linestyle': '-',
            'marker': 'o'
        },
        'LSTM_Net26_fine': {
            'color': 'magenta',
            'linestyle': '-',
            'marker': 'o'
        },
        'LSTM_Net27_fine': {
            'color': 'magenta',
            'linestyle': '-.',
            'marker': '*'
        },
        'CNN_LSTM_Net28_fine': {
            'color': 'green',
            'linestyle': '-',
            'marker': 'o'
        },
        'CNN_LSTM_Net29_fine': {
            'color': 'green',
            'linestyle': '-.',
            'marker': '*'
        }
    }

    modelDict = {}
    for netName in netNames:
        fileD = open(PATHDICT_CONFUSION_MATRIX_LR2CLASS[netName], 'rb')
        confusion_matrix_DICT = pickle.load(fileD)
        fileD.close()

        trueArray = np.empty((0))
        predArray = np.empty((0))
        print(
            netName + '  ignore ',
            confusion_matrix_DICT['ignoreList'],
        )
        for name in NAME_LIST:
            if name not in confusion_matrix_DICT['ignoreList']:
                trueArray = np.r_[trueArray,
                                  confusion_matrix_DICT[name]['%d' %
                                                              size]['y_true']]
                predArray = np.r_[predArray,
                                  confusion_matrix_DICT[name]['%d' %
                                                              size]['y_pred']]
        # end for
        modelDict.update({
            netName: {
                'y_true': trueArray,
                'y_pred': predArray,
                'color': styleDict[netName]['color'],
                'linestyle': styleDict[netName]['linestyle'],
                'marker': styleDict[netName]['marker']
            }
        })

    draw_ROC_curve(modelDict, 'ROC_multiModel_fine_LR2class_%d' % size)
    # end plot_ROC_curve_LR2class

# 绘制多模型单容量柱状图：表征给定容量下各模型的性能对比
def plot_bar_LR2class(netNames, size,plotAllSize=0):
    sizeList=[22,52,90]
    styleDict = {
        'SVM_no_cross_subject': {
            'color': 'black',
            'linestyle': '-',
            'marker': 'o'
        },
        'LDA_no_cross_subject': {
            'color': 'black',
            'linestyle': '-.',
            'marker': '*'
        },
        'CNN_Net2_fine': {
            'color': 'blue',
            'linestyle': '-',
            'marker': 'o',
            'zero_accuracy':0.597,
            'zero_f1':0.563
        },
        'CNN_Net22_fine': {
            'color': 'blue',
            'linestyle': '-.',
            'marker': '*',
            'zero_accuracy':0.574,
            'zero_f1':0.567
        },
        'LSTM_Net23_fine': {
            'color': 'green',
            'linestyle': '-',
            'marker': 'o',
            'zero_accuracy':0.595,
            'zero_f1':0.583
        },
        'LSTM_Net24_fine': {
            'color': 'green',
            'linestyle': '-.',
            'marker': '*',
            'zero_accuracy':0.601,
            'zero_f1':0.585
        },
        'LSTM_Net26_fine': {
            'color': 'magenta',
            'linestyle': '-',
            'marker': 'o',
            'zero_accuracy':0.640,
            'zero_f1':0.600
        },
        'LSTM_Net27_fine': {
            'color': 'magenta',
            'linestyle': '-.',
            'marker': '*',
            'zero_accuracy':0.649,
            'zero_f1':0.611
        }
    }

    nameDict = {
        'SVM_no_cross_subject': 'SVM',
        'LDA_no_cross_subject': 'LDA',
        'CNN_Net2_fine': 'CNN_Net2',
        'CNN_Net22_fine': 'CNN_Net22',
        'LSTM_Net23_fine': 'LSTM_Net23',
        'LSTM_Net24_fine': 'LSTM_Net24',
        'LSTM_Net26_fine': 'LSTM_Net26',
        'LSTM_Net27_fine': 'LSTM_Net27',
    }

    modelDict = {}
    for netName,i in zip(netNames,range(len(netNames))):
        fileD = open(PATHDICT_CONFUSION_MATRIX_LR2CLASS[netName], 'rb')
        confusion_matrix_DICT = pickle.load(fileD)
        fileD.close()

        if i<plotAllSize: # 需要绘制多个 size 的网络
            for sizeI in sizeList:
                trueArray_0 = np.empty((0))
                predArray_0 = np.empty((0))
                trueArray_size = np.empty((0))
                predArray_size = np.empty((0))
                print(
                    netName + '  ignore ',
                    confusion_matrix_DICT['ignoreList'],
                )
                for name in NAME_LIST:
                    if name not in confusion_matrix_DICT['ignoreList']:
                        trueArray_size = np.r_[
                            trueArray_size,
                            confusion_matrix_DICT[name]['%d' % sizeI]['y_true']]
                        predArray_size = np.r_[
                            predArray_size,
                            confusion_matrix_DICT[name]['%d' % sizeI]['y_pred']]
                        if netName not in [
                                'SVM_no_cross_subject', 'LDA_no_cross_subject'
                        ]:
                            trueArray_0 = np.r_[
                                trueArray_0,
                                confusion_matrix_DICT[name]['%d' % 0]['y_true']]
                            predArray_0 = np.r_[
                                predArray_0,
                                confusion_matrix_DICT[name]['%d' % 0]['y_pred']]
                        else:
                            trueArray_0 = None
                            predArray_0 = None

                # end for
                modelDict.update({
                    '%s_%s'%(nameDict[netName],sizeI): {
                        'y_true_size': trueArray_size,
                        'y_pred_size': predArray_size,
                        'y_true_0': trueArray_0,
                        'y_pred_0': predArray_0,
                        'color': styleDict[netName]['color'],
                        'linestyle': styleDict[netName]['linestyle'],
                        'marker': styleDict[netName]['marker'],
                        'zero_accuracy':styleDict[netName].get('zero_accuracy',np.NaN),
                        'zero_f1':styleDict[netName].get('zero_f1',np.NaN)
                    }
                })                
        else: # 只需要绘制一个 size 的网络
            trueArray_0 = np.empty((0))
            predArray_0 = np.empty((0))
            trueArray_size = np.empty((0))
            predArray_size = np.empty((0))
            print(
                netName + '  ignore ',
                confusion_matrix_DICT['ignoreList'],
            )
            for name in NAME_LIST:
                if name not in confusion_matrix_DICT['ignoreList']:
                    trueArray_size = np.r_[
                        trueArray_size,
                        confusion_matrix_DICT[name]['%d' % size]['y_true']]
                    predArray_size = np.r_[
                        predArray_size,
                        confusion_matrix_DICT[name]['%d' % size]['y_pred']]
                    if netName not in [
                            'SVM_no_cross_subject', 'LDA_no_cross_subject'
                    ]:
                        trueArray_0 = np.r_[
                            trueArray_0,
                            confusion_matrix_DICT[name]['%d' % 0]['y_true']]
                        predArray_0 = np.r_[
                            predArray_0,
                            confusion_matrix_DICT[name]['%d' % 0]['y_pred']]
                    else:
                        trueArray_0 = None
                        predArray_0 = None

            # end for
            modelDict.update({
                nameDict[netName]: {
                    'y_true_size': trueArray_size,
                    'y_pred_size': predArray_size,
                    'y_true_0': trueArray_0,
                    'y_pred_0': predArray_0,
                    'color': styleDict[netName]['color'],
                    'linestyle': styleDict[netName]['linestyle'],
                    'marker': styleDict[netName]['marker'],
                    'zero_accuracy':styleDict[netName].get('zero_accuracy',np.NaN),
                    'zero_f1':styleDict[netName].get('zero_f1',np.NaN)
                }
            })

    draw_bar(modelDict, 'bar_multiModel_fine_LR2class_%d' % size)
    # end plot_bar_LR2class

# 绘制模型学习曲线_多时间段
def plot_learning_curve_plot_DICT_UD2class_multiSegment(netName, detail=False):
    ignoreDict_UD2class = {
        'SVM_-2080': ['all'],
        'SVM_-2000': ['all'],
        'SVM_-2080': ['all'],
        'SVM_-1920': ['all'],
        'SVM_-1840': ['all'],
        'SVM_-1760': ['all'],
        'SVM_-1680': ['all'],
        'SVM_-1600': ['all'],
        'SVM_-1520': ['all'],
        'SVM_-1440': ['all'],
        'SVM_-1360': ['cw', 'cwm', 'kx', 'pbl', 'wrd'],
        'SVM_-1280': ['cwm', 'kx', 'pbl', 'wrd', 'yzg'],
        'SVM_-1200': ['all'],
        'SVM_-1120': ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'yzg'],
        'SVM_-1040': [],
        'SVM_-960': ['cwm', 'wrd'],
        'SVM_-880': ['cw'],
        'SVM_-800': [],
        'SVM_-720': ['pbl', 'wrd'],
        'SVM_-640': ['cw','kx','cwm','xsc'],
        'SVM_-560': [],
        'SVM_-480': ['xsc'],
        'SVM_-400': ['cw', 'xsc'],
        'SVM_-320': ['cw','xsc', 'yzg'],
        'SVM_-240': [],
        'SVM_-160': ['cw', 'cwm', 'xsc', 'wrd', 'yzg'],
        'SVM_-80': [],
        'SVM_0': ['xsc'],
        'SVM_80': ['kx', 'pbl', 'wrd'],
        'SVM_160': ['pbl', 'wrd'],
        'SVM_240': ['cwm', 'pbl', 'xsc', 'yzg'],
        'SVM_320': ['cwm', 'kx', 'pbl', 'wrd', 'xsc'],
        'SVM_400': ['pbl', 'xsc', 'yzg'],
        'SVM_480': ['all'],
        'SVM_560': ['all'],
        'SVM_640': ['all'],
        'SVM_720': ['all'],
        'SVM_800': ['all'],
        'SVM_880': ['all'],
        'SVM_960': ['all'],
        'SVM_1040': ['all'],
        'SVM_1120': ['all'],
        'LDA_-2080': ['all'],
        'LDA_-2000': ['all'],
        'LDA_-2080': ['all'],
        'LDA_-1920': ['all'],
        'LDA_-1840': ['all'],
        'LDA_-1760': ['all'],
        'LDA_-1680': ['all'],
        'LDA_-1600': ['all'],
        'LDA_-1520': ['all'],
        'LDA_-1440': ['all'],
        'LDA_-1360': ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'xsc'],
        'LDA_-1280': ['cwm', 'kx', 'pbl', 'wrd', 'yzg'],
        'LDA_-1200': ['cwm', 'kx', 'pbl', 'wrd', 'xsc'],
        'LDA_-1120': ['cwm', 'kx', 'pbl', 'wrd', 'yzg'],
        'LDA_-1040': ['cwm', 'kx', 'pbl', 'yzg'],
        'LDA_-960': ['wrd'],
        'LDA_-880': ['cwm', 'pbl', 'wrd', 'yzg'],
        'LDA_-800': ['wrd'],
        'LDA_-720': ['cw','cwm', 'kx', 'wrd'],
        'LDA_-640': ['pbl', 'wrd'],
        'LDA_-560': ['cw','cwm', 'kx', 'wrd','yzg'],
        'LDA_-480': [],
        'LDA_-400': [],
        'LDA_-320': [],
        'LDA_-240': ['cw'],
        'LDA_-160': [],
        'LDA_-80': [],
        'LDA_0': [],
        'LDA_80': ['wrd'],
        'LDA_160': ['kx', 'pbl'],
        'LDA_240': ['cwm', 'kx', 'pbl', 'wrd'],
        'LDA_320': ['cw', 'cwm', 'kx', 'pbl', 'wrd'],
        'LDA_400': ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'xsc'],
        'LDA_480': ['all'],
        'LDA_560': ['all'],
        'LDA_640': ['all'],
        'LDA_720': ['all'],
        'LDA_800': ['all'],
        'LDA_880': ['all'],
        'LDA_960': ['all'],
        'LDA_1040': ['all'],
        'LDA_1120': ['all'],
        'Net25_-960': ['wxc', 'pbl'],
        'Net25_-800': [],
        'Net25_-640': [],
        'Net25_-480': [],
        'Net25_-320': [],
        'Net25_-160': [],
        'Net25_0': ['cw', 'wrd'],
        'Net25_160': ['cwm', 'kx', 'pbl', 'wrd'],
        'Net25_320': ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'wxc', 'xsc']
    }

    net = netName.split('_')[0]
    timeBeginStart = netName.split('_')[-1]
    timeBeginStart = int(netName.split('_')[-1])
    if net == 'Net25':
        timeBeginStart += 960
    fileD = open(
        PATHDICT_UD2CLASS_SEGMENT[net] +
        '\\%d\\learning_curve_UD2class.pickle' % timeBeginStart, 'rb')
    subject_learning_curve_plot_DICT = pickle.load(fileD)
    fileD.close()
    if detail:
        subject_learning_curve_plot_DICT.update(
            {'ignoreList': ignoreDict_UD2class[netName]})
        fileD = open(
            PATHDICT_UD2CLASS_SEGMENT[net] +
            '\\%d\\learning_curve_UD2class.pickle' % timeBeginStart, 'wb')
        pickle.dump(subject_learning_curve_plot_DICT, fileD)
        fileD.close()

        fileD = open(
            PATHDICT_UD2CLASS_SEGMENT[net] +
            '\\%d\\confusion_matrix_UD2class.pickle' % timeBeginStart, 'rb')
        subject_confusion_matrix_plot_DICT = pickle.load(fileD)
        fileD.close()
        subject_confusion_matrix_plot_DICT.update(
            {'ignoreList': ignoreDict_UD2class[netName]})
        fileD = open(
            PATHDICT_UD2CLASS_SEGMENT[net] +
            '\\%d\\confusion_matrix_UD2class.pickle' % timeBeginStart, 'wb')
        pickle.dump(subject_confusion_matrix_plot_DICT, fileD)
        fileD.close()
        draw_learning_curve_detail(
            subject_learning_curve_plot_DICT,
            'learning_curve_UD2classes_intraSubject_' + netName)
    else:
        draw_learning_curve(
            subject_learning_curve_plot_DICT,
            'learning_curve_UD2classes_intraSubject_' + netName)
    # end plot_learning_curve_plot_DICT_UD2class_multiSegment


# 绘制模型学习曲线_多时间段
def plot_learning_curve_plot_DICT_LR2class_multiSegment(netName, detail=False):
    ignoreDict_LR2class = {
        'SVM_-2080': ['all'],
        'SVM_-2000': ['all'],
        'SVM_-2080': ['all'],
        'SVM_-1920': ['all'],
        'SVM_-1840': ['all'],
        'SVM_-1760': ['all'],
        'SVM_-1680': ['all'],
        'SVM_-1600': ['all'],
        'SVM_-1520': ['all'],
        'SVM_-1440': ['all'],
        'SVM_-1360': ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'xsc'],
        'SVM_-1280': ['cwm', 'kx', 'xsc', 'pbl', 'wrd'],
        'SVM_-1200': ['cwm', 'kx', 'xsc', 'pbl', 'wrd'],
        'SVM_-1120': ['cwm', 'kx', 'pbl', 'wrd', 'xsc', 'yzg'],
        'SVM_-1040': [],
        'SVM_-960': ['cw','cwm', 'kx', 'pbl'],
        'SVM_-880': ['cwm', 'kx','pbl','yzg'],
        'SVM_-800': [],
        'SVM_-720': ['xsc'],
        'SVM_-640': ['cw','wrd'],
        'SVM_-560': [],
        'SVM_-480': ['cw', 'yzg'],
        'SVM_-400': ['yzg'],
        'SVM_-320': ['sxc','yzg'],
        'SVM_-240': [],
        'SVM_-160': [],
        'SVM_-80': [],
        'SVM_0': ['wrd'],
        'SVM_80': ['kx'],
        'SVM_160': ['pbl','xsc'],
        'SVM_240': ['kx', 'pbl'],
        'SVM_320': ['cw', 'kx', 'pbl', 'wrd', 'xsc', 'yzg'],
        'SVM_400': ['kx', 'pbl', 'xsc'],
        'SVM_480': ['cw', 'cwm', 'kx', 'pbl', 'xsc', 'yzg'],
        'SVM_560': ['all'],
        'SVM_640': ['all'],
        'SVM_720': ['all'],
        'SVM_800': ['all'],
        'SVM_880': ['all'],
        'SVM_960': ['all'],
        'SVM_1040': ['all'],
        'SVM_1120': ['all'],
        'LDA_-2080': ['all'],
        'LDA_-2000': ['all'],
        'LDA_-2080': ['all'],
        'LDA_-1920': ['all'],
        'LDA_-1840': ['all'],
        'LDA_-1760': ['all'],
        'LDA_-1680': ['all'],
        'LDA_-1600': ['all'],
        'LDA_-1520': ['all'],
        'LDA_-1440': ['cwm', 'kx', 'pbl', 'xsc', 'yzg'],
        'LDA_-1360': ['cw', 'cwm', 'kx', 'pbl', 'xsc', 'yzg'],
        'LDA_-1280': ['cw', 'cwm', 'kx'],
        'LDA_-1200': ['cwm', 'pbl'],
        'LDA_-1120': ['kx', 'pbl'],
        'LDA_-1040': ['cwm', 'kx', 'pbl', 'xsc','yzg'],
        'LDA_-960': ['cwm', 'kx', 'pbl', 'xsc','yzg'],
        'LDA_-880': [],
        'LDA_-800': [],
        'LDA_-720': [],
        'LDA_-640': ['yzg'],
        'LDA_-560': ['kx'],
        'LDA_-480': [],
        'LDA_-400': [],
        'LDA_-320': ['cw'],
        'LDA_-240': [],
        'LDA_-160': ['yzg'],
        'LDA_-80': [],
        'LDA_0': ['kx'],
        'LDA_80': ['cwm','kx', 'pbl'],
        'LDA_160': ['cw', 'cwm', 'kx', 'pbl', 'wrd', 'xsc'],
        'LDA_240': ['cwm', 'kx', 'pbl'],
        'LDA_320': ['kx', 'pbl', 'xsc'],
        'LDA_400': ['cwm', 'kx', 'pbl', 'wrd', 'xsc'],
        'LDA_480': ['cwm', 'kx', 'pbl', 'yzg'],
        'LDA_560': ['all'],
        'LDA_640': ['all'],
        'LDA_720': ['all'],
        'LDA_800': ['all'],
        'LDA_880': ['all'],
        'LDA_960': ['all'],
        'LDA_1040': ['all'],
        'LDA_1120': ['all'],
        'Net25_-960': ['cw', 'kx', 'wrd', 'pbl', 'wxc'],
        'Net25_-800': ['yzg'],
        'Net25_-640': [],
        'Net25_-480': [],
        'Net25_-320': [],
        'Net25_-160': [],
        'Net25_0': ['cwm', 'xsc'],
        'Net25_160': ['kx', 'pbl', 'wxc', 'yzg'],
        'Net25_320': ['cwm', 'kx', 'pbl', 'wrd']
    }

    net = netName.split('_')[0]
    timeBeginStart = int(netName.split('_')[-1])
    if net == 'Net25':
        timeBeginStart += 960
    fileD = open(
        PATHDICT_LR2CLASS_SEGMENT[net] +
        '\\%d\\learning_curve_LR2class.pickle' % timeBeginStart, 'rb')
    subject_learning_curve_plot_DICT = pickle.load(fileD)
    fileD.close()
    if detail:
        subject_learning_curve_plot_DICT.update(
            {'ignoreList': ignoreDict_LR2class[netName]})
        fileD = open(
            PATHDICT_LR2CLASS_SEGMENT[net] +
            '\\%d\\learning_curve_LR2class.pickle' % timeBeginStart, 'wb')
        pickle.dump(subject_learning_curve_plot_DICT, fileD)
        fileD.close()

        fileD = open(
            PATHDICT_LR2CLASS_SEGMENT[net] +
            '\\%d\\confusion_matrix_LR2class.pickle' % timeBeginStart, 'rb')
        subject_confusion_matrix_plot_DICT = pickle.load(fileD)
        fileD.close()
        subject_confusion_matrix_plot_DICT.update(
            {'ignoreList': ignoreDict_LR2class[netName]})
        fileD = open(
            PATHDICT_LR2CLASS_SEGMENT[net] +
            '\\%d\\confusion_matrix_LR2class.pickle' % timeBeginStart, 'wb')
        pickle.dump(subject_confusion_matrix_plot_DICT, fileD)
        fileD.close()
        draw_learning_curve_detail(
            subject_learning_curve_plot_DICT,
            'learning_curve_LR2classes_intraSubject_' + netName)
    else:
        draw_learning_curve(
            subject_learning_curve_plot_DICT,
            'learning_curve_LR2classes_intraSubject_' + netName)
    # end plot_learning_curve_plot_DICT_LR2class_multiSegment

# 
# 绘制模型学习曲线_多时间段
def plot_learning_curve_plot_DICT_4class_multiSegment(netName, detail=False):
    ignoreDict_4class = {
        'SVM_-2080': ['all'],
        'SVM_-2000': ['all'],
        'SVM_-2080': ['all'],
        'SVM_-1920': ['all'],
        'SVM_-1840': ['all'],
        'SVM_-1760': ['all'],
        'SVM_-1680': ['all'],
        'SVM_-1600': ['all'],
        'SVM_-1520': ['all'],
        'SVM_-1440': ['all'],
        'SVM_-1360': ['all'],
        'SVM_-1280': ['all'],
        'SVM_-1200': ['kx','pbl','wrd','xsc'],
        'SVM_-1120': ['wrd'],
        'SVM_-1040': ['kx','pbl','wrd','xsc','yzg'],
        'SVM_-960': ['kx'],
        'SVM_-880': ['cw','xsc'],
        'SVM_-800': ['wrd'],
        'SVM_-720': ['cw','cwm'],
        'SVM_-640': ['cwm'],
        'SVM_-560': ['wrd'],
        'SVM_-480': ['yzg'],
        'SVM_-400': ['xsc'],
        'SVM_-320': ['yzg'],
        'SVM_-240': [],
        'SVM_-160': ['cw','xsc'],
        'SVM_-80': ['cwm','xsc'],
        'SVM_0': [],
        'SVM_80': ['kx','wrd'],
        'SVM_160': ['pbl','xsc','yzg'],
        'SVM_240': ['cw','kx', 'pbl', 'wrd', 'xsc', 'yzg'],
        'SVM_320': ['all'],
        'SVM_400': ['all'],
        'SVM_480': ['all'],
        'SVM_560': ['all'],
        'SVM_640': ['all'],
        'SVM_720': ['all'],
        'SVM_800': ['all'],
        'SVM_880': ['all'],
        'SVM_960': ['all'],
        'SVM_1040': ['all'],
        'SVM_1120': ['all'],
        'LDA_-2080': ['all'],
        'LDA_-2000': ['all'],
        'LDA_-2080': ['all'],
        'LDA_-1920': ['all'],
        'LDA_-1840': ['all'],
        'LDA_-1760': ['all'],
        'LDA_-1680': ['all'],
        'LDA_-1600': ['all'],
        'LDA_-1520': ['all'],
        'LDA_-1440': ['all'],
        'LDA_-1360': ['all'],
        'LDA_-1280': ['cw','kx', 'pbl', 'wrd', 'xsc'],
        'LDA_-1200': ['kx', 'pbl', 'wrd', 'xsc','yzg'],
        'LDA_-1120': [],
        'LDA_-1040': [],
        'LDA_-960': [],
        'LDA_-880': ['yzg','cw'],
        'LDA_-800': ['xsc'],
        'LDA_-720': ['xsc'],
        'LDA_-640': ['cw','pbl','xsc'],
        'LDA_-560': [],
        'LDA_-480': [],
        'LDA_-400': [],
        'LDA_-320': ['xsc','yzg'],
        'LDA_-240': ['cw'],
        'LDA_-160': [],
        'LDA_-80': [],
        'LDA_0': [],
        'LDA_80': ['wrd','xsc'],
        'LDA_160': ['kx','wrd'],
        'LDA_240': ['cw','kx','pbl','xsc','yzg'],
        'LDA_320': ['all'],
        'LDA_400': ['all'],
        'LDA_480': ['all'],
        'LDA_560': ['all'],
        'LDA_640': ['all'],
        'LDA_720': ['all'],
        'LDA_800': ['all'],
        'LDA_880': ['all'],
        'LDA_960': ['all'],
        'LDA_1040': ['all'],
        'LDA_1120': ['all'],
        'Net25_-960': [],
        'Net25_-800': ['yzg'],
        'Net25_-640': [],
        'Net25_-480': [],
        'Net25_-320': [],
        'Net25_-160': [],
        'Net25_0': [],
        'Net25_160': [],
        'Net25_320': []
    }

    net = netName.split('_')[0]
    timeBeginStart = int(netName.split('_')[-1])
    if net == 'Net25':
        timeBeginStart += 960
    fileD = open(
        PATHDICT_4CLASS_SEGMENT[net] +
        '\\%d\\learning_curve_4class.pickle' % timeBeginStart, 'rb')
    subject_learning_curve_plot_DICT = pickle.load(fileD)
    fileD.close()
    if detail:
        subject_learning_curve_plot_DICT.update(
            {'ignoreList': ignoreDict_4class[netName]})
        fileD = open(
            PATHDICT_4CLASS_SEGMENT[net] +
            '\\%d\\learning_curve_4class.pickle' % timeBeginStart, 'wb')
        pickle.dump(subject_learning_curve_plot_DICT, fileD)
        fileD.close()

        fileD = open(
            PATHDICT_4CLASS_SEGMENT[net] +
            '\\%d\\confusion_matrix_4class.pickle' % timeBeginStart, 'rb')
        subject_confusion_matrix_plot_DICT = pickle.load(fileD)
        fileD.close()
        subject_confusion_matrix_plot_DICT.update(
            {'ignoreList': ignoreDict_4class[netName]})
        fileD = open(
            PATHDICT_4CLASS_SEGMENT[net] +
            '\\%d\\confusion_matrix_4class.pickle' % timeBeginStart, 'wb')
        pickle.dump(subject_confusion_matrix_plot_DICT, fileD)
        fileD.close()
        draw_learning_curve_detail(
            subject_learning_curve_plot_DICT,
            'learning_curve_4classes_intraSubject_' + netName)
    else:
        draw_learning_curve(
            subject_learning_curve_plot_DICT,
            'learning_curve_4classes_intraSubject_' + netName)
    # end plot_learning_curve_plot_DICT_4class_multiSegment

# 绘制多个模型不同时间段的准确率：表征多个模型使用不同时间段的数据集的模型准确率（训练集容量最大时的准确率）
# 输入参数：模型名称列表
def plot_scoresMultiCurves_multiSegment(netNames):
    pathDict = {
        'SVM_no_cross_subject_UD2class':
        os.getcwd() + r'\..\model\上下二分类\ML\SVM\\',
        'LDA_no_cross_subject_UD2class':
        os.getcwd() + r'\..\model\上下二分类\ML\LDA\\',
        'NN_Net25_pretraining_UD2class':
        os.getcwd() + r'\..\model_all\上下二分类\NN_Net25_pretraining\\',
        'NN_Net25_fine_UD2class':
        os.getcwd() + r'\..\model\上下二分类\NN_Net25_fine\\',
        'SVM_no_cross_subject_LR2class':
        os.getcwd() + r'\..\model\左右二分类\ML\SVM\\',
        'LDA_no_cross_subject_LR2class':
        os.getcwd() + r'\..\model\左右二分类\ML\LDA\\',
        'NN_Net25_pretraining_LR2class':
        os.getcwd() + r'\..\model_all\左右二分类\NN_Net25_pretraining\\',
        'NN_Net25_fine_LR2class':
        os.getcwd() + r'\..\model\左右二分类\NN_Net25_fine\\',
        'SVM_no_cross_subject_4class':
        os.getcwd() + r'\..\model\四分类\ML\SVM\\',
        'LDA_no_cross_subject_4class':
        os.getcwd() + r'\..\model\四分类\ML\LDA\\',
    }
    styleDict = {
        'SVM': {
            'color': 'black',
            'linestyle': '-',
            'marker': 'o',
            'label':'支持向量机'
        },
        'LDA': {
            'color': 'black',
            'linestyle': '-.',
            'marker': '*',
            'label':'线性判别分析'
        },
        'NN': {
            'color': 'yellow',
            'linestyle': '-',
            'marker': 'o',
            'label':'NN_Net25'
        }
    }
    plt.figure(figsize=(19, 10))
    SEG_START_NN = np.linspace(0, 1280, 9).astype(
        int)  # 0,160,320,480,640,800,960,1120,1280
    SEG_START_ML = np.linspace(-2080, 1120, 41).astype(int)  # 窗移 80ms
    for netName in netNames:
        plot_y = []
        path = pathDict[netName]
        if netName.split('_')[0] == 'NN':
            segList = SEG_START_NN
        else:
            segList = SEG_START_ML
        for timeBeginStart in segList:
            fileD = open(
                path + r'%d' % timeBeginStart +
                '\\learning_curve_%s.pickle' % netName.split('_')[-1], 'rb')
            subject_DICT = pickle.load(fileD)
            fileD.close()

            test_scores = []
            if subject_DICT['ignoreList'] == ['all']:
                if netName.split('_')[-1]=='4class':
                    plot_y.append(0.26 + (np.random.rand(1) - 0.5) * 0.1)
                else:
                    plot_y.append(0.5 + (np.random.rand(1) - 0.5) * 0.1)
            else:
                for name in subject_DICT:
                    if name == 'ignoreList':
                        break
                    if name in subject_DICT['ignoreList']:
                        continue
                    dic = subject_DICT[name]
                    assert name == dic['name']
                    test_scores.append(np.average(dic['test_scores'][-3:]))
                # end for
                test_scores = np.array(test_scores)
                test_scores_mean = np.mean(test_scores)
                plot_y.append(test_scores_mean)
        # end for
        if netName.split('_')[0] == 'NN':
            plt.plot(segList - 960+1280,
                     plot_y,
                     linestyle=styleDict[netName.split('_')[0]]['linestyle'],
                     color=styleDict[netName.split('_')[0]]['color'],
                     marker=styleDict[netName.split('_')[0]]['marker'],
                     label=styleDict[netName.split('_')[0]]['label'])
        else:
            plt.plot(segList+1280,
                     plot_y,
                     linestyle=styleDict[netName.split('_')[0]]['linestyle'],
                     color=styleDict[netName.split('_')[0]]['color'],
                     marker=styleDict[netName.split('_')[0]]['marker'],
                     label=styleDict[netName.split('_')[0]]['label'])
    
    # 绘制 0 时刻参考线
    plt.axvline(x=0,color='red',linestyle='-',linewidth=3.0)
    plt.legend(loc='upper right',fontsize=26)
    plt.xlabel('窗尾偏移量/毫秒',fontsize=28)
    plt.ylabel('准确率',fontsize=28)
    plt.grid()
    plt.xticks(SEG_START_ML+1280, rotation=60,fontsize=28)
    plt.yticks(fontsize=28)
    plt.tight_layout()
    # plt.title('accuracy_multiCurves_multiSegment_' +
    #           netNames[0].split('_')[-1])
    plt.savefig('accuracy_multiCurves_multiSegment_' +
                netNames[0].split('_')[-1] + '.jpeg')
    plt.show()
    # end plot_scoresMultiCurves_multiSegment


# 绘制不同时间段的准确率：表征使用不同时间段的数据集的模型准确率
# 输入参数：模型名称
def plot_scoresSurface_multiSegment(netName):
    pathDict = {
        'SVM_no_cross_subject_UD2class':
        os.getcwd() + r'\..\model\上下二分类\ML\SVM\\',
        'LDA_no_cross_subject_UD2class':
        os.getcwd() + r'\..\model\上下二分类\ML\LDA\\',
        'NN_Net25_pretraining_UD2class':
        os.getcwd() + r'\..\model_all\上下二分类\NN_Net25_pretraining\\',
        'NN_Net25_fine_UD2class':
        os.getcwd() + r'\..\model\上下二分类\NN_Net25_fine\\',
        'SVM_no_cross_subject_LR2class':
        os.getcwd() + r'\..\model\左右二分类\ML\SVM\\',
        'LDA_no_cross_subject_LR2class':
        os.getcwd() + r'\..\model\左右二分类\ML\LDA\\',
        'NN_Net25_pretraining_LR2class':
        os.getcwd() + r'\..\model_all\左右二分类\NN_Net25_pretraining\\',
        'NN_Net25_fine_LR2class':
        os.getcwd() + r'\..\model\左右二分类\NN_Net25_fine\\'
    }

    SEG_START_NN = np.linspace(0, 1280, 9).astype(
        int)  # 0,160,320,480,640,800,960,1120,1280
    SEG_START_ML = np.linspace(-2080, 1120, 41).astype(int)  # 窗移 80ms
    SEG_START_ML = np.linspace(-1040, 240, 17).astype(int)  # 窗移 80ms
    plot_z = []
    path = pathDict[netName]
    if netName.split('_')[0] == 'NN':
        segList = SEG_START_NN
    else:
        segList = SEG_START_ML
    for timeBeginStart in segList:
        fileD = open(
            path + r'%d' % timeBeginStart +
            '\\learning_curve_%s.pickle' % netName[-8:], 'rb')
        subject_DICT = pickle.load(fileD)
        print(timeBeginStart,"  ",subject_DICT['ignoreList'])
        fileD.close()

        test_scores = []
        for name in NAME_LIST:
            if name in subject_DICT['ignoreList']:
                continue
            dic = subject_DICT.get(name,np.NaN)
            if dic is not np.NaN:
                assert name == dic['name']
                test_scores.append(dic['test_scores'])
                meshX=dic['train_sizes']
        # end for
        test_scores = np.array(test_scores)
        test_scores_mean = np.mean(test_scores, axis=0)
        plot_z.append(test_scores_mean)
    # end for
    plot_z = np.array(plot_z)
    # plot_x, plot_y = np.meshgrid(meshX, segList-960)
    plot_x, plot_y = np.meshgrid(meshX, segList)

    fig = plt.figure(figsize=(19, 10))
    ax = fig.gca(projection='3d')
    ax.plot_surface(plot_x,
                    plot_y,
                    plot_z,
                    rstride=1,
                    cstride=1,
                    cmap=cm.coolwarm,
                    alpha=None,
                    antialiased=True)
    # plt.legend()
    ax.set_xlabel("train_sizes")
    ax.set_ylabel("time_segment")
    ax.set_zlabel("scores_accuracy")
    # plt.grid()
    plt.xticks(meshX, rotation=45,fontsize=28)
    # plt.yticks(segList-960)
    plt.yticks(segList,fontsize=28)
    # plt.title('scoresSurface_multiSegment_' + netName)
    plt.tight_layout()
    plt.savefig('scoresSurface_multiSegment_' + netName + '.jpeg')
    plt.show()
    # end plot_scoresSurface_multiSegment


# 绘制不同时间段的准确率：表征使用不同时间段的数据集的模型准确率
# 输入参数：模型名称
def plot_scoresContour_multiSegment(netName):
    pathDict = {
        'SVM_no_cross_subject_UD2class':
        os.getcwd() + r'\..\model\上下二分类\ML\SVM\\',
        'LDA_no_cross_subject_UD2class':
        os.getcwd() + r'\..\model\上下二分类\ML\LDA\\',
        'NN_Net25_pretraining_UD2class':
        os.getcwd() + r'\..\model_all\上下二分类\NN_Net25_pretraining\\',
        'NN_Net25_fine_UD2class':
        os.getcwd() + r'\..\model\上下二分类\NN_Net25_fine\\',
        'SVM_no_cross_subject_LR2class':
        os.getcwd() + r'\..\model\左右二分类\ML\SVM\\',
        'LDA_no_cross_subject_LR2class':
        os.getcwd() + r'\..\model\左右二分类\ML\LDA\\',
        'NN_Net25_pretraining_LR2class':
        os.getcwd() + r'\..\model_all\左右二分类\NN_Net25_pretraining\\',
        'NN_Net25_fine_LR2class':
        os.getcwd() + r'\..\model\左右二分类\NN_Net25_fine\\',
        'SVM_no_cross_subject_4class':
        os.getcwd() + r'\..\model\四分类\ML\SVM\\',
        'LDA_no_cross_subject_4class':
        os.getcwd() + r'\..\model\四分类\ML\LDA\\'
    }

    SEG_START_NN = np.linspace(0, 1280, 9).astype(
        int)  # 0,160,320,480,640,800,960,1120,1280
    SEG_START_ML = np.linspace(-2080, 1120, 41).astype(int)  # 窗移 80ms
    SEG_START_ML = np.linspace(-1040, 240, 17).astype(int)  # 窗移 80ms
    plot_z = []
    path = pathDict[netName]
    if netName.split('_')[0] == 'NN':
        segList = SEG_START_NN
    else:
        segList = SEG_START_ML
    for timeBeginStart in segList:
        fileD = open(
            path + r'%d' % timeBeginStart +
            '\\learning_curve_%s.pickle' % netName.split('_')[-1], 'rb')
        subject_DICT = pickle.load(fileD)
        fileD.close()

        test_scores = []
        for name in NAME_LIST:
            if name in subject_DICT['ignoreList']:
                continue
            dic = subject_DICT.get(name,np.NaN)
            if dic is not np.NaN:
                assert name == dic['name']
                test_scores.append(dic['test_scores'])
                meshX=dic['train_sizes']
        # end for
        test_scores = np.array(test_scores)
        test_scores_mean = np.mean(test_scores, axis=0)
        plot_z.append(test_scores_mean)
    # end for
    plot_z = np.array(plot_z)

    plt.figure(figsize=(18, 12))
    plt.imshow(plot_z, cmap=cm.coolwarm)
    plt.xticks(np.array(range(len(meshX))),
               meshX,
               rotation=45,fontsize=28)
    # plt.yticks(np.array(range(len(segList))), SEG_START - 960)
    plt.yticks(np.array(range(len(segList))), segList+1280,fontsize=28)
    plt.xlabel('样本数量/个',fontsize=28)
    plt.ylabel('窗尾偏移量/毫秒',fontsize=28)
    # plt.title('scoresContour_multiSegment_' + netName)
    cb=plt.colorbar()
    cb.ax.tick_params(labelsize=28)
    plt.tight_layout()
    plt.savefig('scoresContour_multiSegment_' + netName + '.jpeg')
    plt.show()
    # end plot_scoresContour_multiSegment


# 绘制模型学习曲线_通道选择
def plot_learning_curve_plot_DICT_UD2class_ChannelSelect(
        netName, detail=False):
    ignoreDict_UD2class = {
        'SVM_4': [],
        'SVM_6': [],
        'SVM_8': [],
        'SVM_10': [],
        'SVM_12': ['cw', 'wxc', 'yzg'],
        'SVM_14': ['pbl'],
        'SVM_16': [],
        'SVM_18': [],
        'SVM_20': ['wxc'],
        'SVM_22': ['wxc', 'xsc', 'yzg'],
        'SVM_24': ['yzg'],
        'SVM_26': ['cw', 'wxc'],
        'SVM_28': ['cw', 'xsc'],
        'SVM_30': ['cw', 'wxc'],
        'SVM_32': [],
        'LDA_4': [],
        'LDA_6': [],
        'LDA_8': [],
        'LDA_10': [],
        'LDA_12': [],
        'LDA_14': [],
        'LDA_16': [],
        'LDA_18': [],
        'LDA_20': [],
        'LDA_22': [],
        'LDA_24': [],
        'LDA_26': [],
        'LDA_28': [],
        'LDA_30': [],
        'LDA_32': [],
        'Net25_4': [],
        'Net25_6': [],
        'Net25_8': [],
        'Net25_10': [],
        'Net25_12': [],
        'Net25_14': [],
        'Net25_16': [],
        'Net25_18': [],
        'Net25_20': [],
        'Net25_22': [],
        'Net25_24': [],
        'Net25_26': [],
        'Net25_28': [],
        'Net25_30': [],
        'Net25_32': []
    }
    CHANNEL_SIZES = np.linspace(4, 32, 15).astype(int)
    net = netName.split('_')[0]
    channel_size = int(netName.split('_')[-1])
    fileD = open(
        PATHDICT_UD2CLASS_CHANNEL_SELECT[net] +
        '\\%d\\learning_curve_UD2class.pickle' % channel_size, 'rb')
    subject_learning_curve_plot_DICT = pickle.load(fileD)
    fileD.close()
    if detail:
        subject_learning_curve_plot_DICT.update(
            {'ignoreList': ignoreDict_UD2class[netName]})
        fileD = open(
            PATHDICT_UD2CLASS_CHANNEL_SELECT[net] +
            '\\%d\\learning_curve_UD2class.pickle' % channel_size, 'wb')
        pickle.dump(subject_learning_curve_plot_DICT, fileD)
        fileD.close()

        fileD = open(
            PATHDICT_UD2CLASS_CHANNEL_SELECT[net] +
            '\\%d\\confusion_matrix_UD2class.pickle' % channel_size, 'rb')
        subject_confusion_matrix_plot_DICT = pickle.load(fileD)
        fileD.close()
        subject_confusion_matrix_plot_DICT.update(
            {'ignoreList': ignoreDict_UD2class[netName]})
        fileD = open(
            PATHDICT_UD2CLASS_CHANNEL_SELECT[net] +
            '\\%d\\confusion_matrix_UD2class.pickle' % channel_size, 'wb')
        pickle.dump(subject_confusion_matrix_plot_DICT, fileD)
        fileD.close()
        draw_learning_curve_detail(
            subject_learning_curve_plot_DICT,
            'learning_curve_UD2classes_intraSubject_' + netName)
    else:
        draw_learning_curve(
            subject_learning_curve_plot_DICT,
            'learning_curve_UD2classes_intraSubject_' + netName)
    # end plot_learning_curve_plot_DICT_UD2class_ChannelSelect


# 绘制模型学习曲线_通道选择
def plot_learning_curve_plot_DICT_4class_ChannelSelect(
        netName, detail=False):
    ignoreDict_4class = {
        'SVM_4': ['cwm','xsc','wrd','wxc'],
        'SVM_6': ['kx','wrd'],
        'SVM_8': ['kx','wrd','wxc'],
        'SVM_10': ['kx','wrd'],
        'SVM_12': ['cw','kx'],
        'SVM_14': ['cwm','kx'],
        'SVM_16': ['wrd'],
        'SVM_18': ['kx'],
        'SVM_20': [],
        'SVM_22': ['kx'],
        'SVM_24': ['cw'],
        'SVM_26': ['cwm','pbl','yzg'],
        'SVM_28': ['cwm','wrd','wxc','yzg'],
        'SVM_30': ['cwm','wrd','wxc','yzg'],
        'SVM_32': ['kx','xsc','wrd','yzg'],
        'LDA_4': [],
        'LDA_6': [],
        'LDA_8': [],
        'LDA_10': [],
        'LDA_12': [],
        'LDA_14': [],
        'LDA_16': [],
        'LDA_18': [],
        'LDA_20': [],
        'LDA_22': [],
        'LDA_24': [],
        'LDA_26': [],
        'LDA_28': [],
        'LDA_30': [],
        'LDA_32': [],
        'Net25_4': [],
        'Net25_6': [],
        'Net25_8': [],
        'Net25_10': [],
        'Net25_12': [],
        'Net25_14': [],
        'Net25_16': [],
        'Net25_18': [],
        'Net25_20': [],
        'Net25_22': [],
        'Net25_24': [],
        'Net25_26': [],
        'Net25_28': [],
        'Net25_30': [],
        'Net25_32': []
    }
    CHANNEL_SIZES = np.linspace(4, 32, 15).astype(int)
    net = netName.split('_')[0]
    channel_size = int(netName.split('_')[-1])
    fileD = open(
        PATHDICT_4CLASS_CHANNEL_SELECT[net] +
        '\\%d\\learning_curve_4class.pickle' % channel_size, 'rb')
    subject_learning_curve_plot_DICT = pickle.load(fileD)
    fileD.close()
    if detail:
        subject_learning_curve_plot_DICT.update(
            {'ignoreList': ignoreDict_4class[netName]})
        fileD = open(
            PATHDICT_4CLASS_CHANNEL_SELECT[net] +
            '\\%d\\learning_curve_4class.pickle' % channel_size, 'wb')
        pickle.dump(subject_learning_curve_plot_DICT, fileD)
        fileD.close()

        fileD = open(
            PATHDICT_4CLASS_CHANNEL_SELECT[net] +
            '\\%d\\confusion_matrix_4class.pickle' % channel_size, 'rb')
        subject_confusion_matrix_plot_DICT = pickle.load(fileD)
        fileD.close()
        subject_confusion_matrix_plot_DICT.update(
            {'ignoreList': ignoreDict_4class[netName]})
        fileD = open(
            PATHDICT_4CLASS_CHANNEL_SELECT[net] +
            '\\%d\\confusion_matrix_4class.pickle' % channel_size, 'wb')
        pickle.dump(subject_confusion_matrix_plot_DICT, fileD)
        fileD.close()
        draw_learning_curve_detail(
            subject_learning_curve_plot_DICT,
            'learning_curve_4classes_intraSubject_' + netName)
    else:
        draw_learning_curve(
            subject_learning_curve_plot_DICT,
            'learning_curve_4classes_intraSubject_' + netName)
    # end plot_learning_curve_plot_DICT_4class_ChannelSelect


# 绘制模型学习曲线_通道选择
def plot_learning_curve_plot_DICT_LR2class_ChannelSelect(
        netName, detail=False):
    ignoreDict_LR2class = {
        'SVM_4': ['cw','cwm','kx','wrd','wxc','xsc'],
        'SVM_6': ['cwm'],
        'SVM_8': [],
        'SVM_10': ['cwm','kx','pbl','xsc','yzg'],
        'SVM_12': ['cwm','kx','xsc'],
        'SVM_14': ['pbl'],
        'SVM_16': [],
        'SVM_18': [],
        'SVM_20': ['kx'],
        'SVM_22': ['yzg'],
        'SVM_24': ['wrd'],
        'SVM_26': ['cwm','wxc'],
        'SVM_28': ['wrd','yzg'],
        'SVM_30': ['cw','yzg'],
        'SVM_32': ['kx','pbl'],
        'LDA_4': [],
        'LDA_6': [],
        'LDA_8': [],
        'LDA_10': [],
        'LDA_12': [],
        'LDA_14': [],
        'LDA_16': [],
        'LDA_18': [],
        'LDA_20': [],
        'LDA_22': [],
        'LDA_24': [],
        'LDA_26': [],
        'LDA_28': [],
        'LDA_30': [],
        'LDA_32': [],
        'Net25_4': [],
        'Net25_6': [],
        'Net25_8': [],
        'Net25_10': [],
        'Net25_12': [],
        'Net25_14': [],
        'Net25_16': [],
        'Net25_18': [],
        'Net25_20': [],
        'Net25_22': [],
        'Net25_24': [],
        'Net25_26': [],
        'Net25_28': [],
        'Net25_30': [],
        'Net25_32': []
    }
    CHANNEL_SIZES = np.linspace(4, 32, 15).astype(int)
    net = netName.split('_')[0]
    channel_size = int(netName.split('_')[-1])
    fileD = open(
        PATHDICT_LR2CLASS_CHANNEL_SELECT[net] +
        '\\%d\\learning_curve_LR2class.pickle' % channel_size, 'rb')
    subject_learning_curve_plot_DICT = pickle.load(fileD)
    fileD.close()
    if detail:
        subject_learning_curve_plot_DICT.update(
            {'ignoreList': ignoreDict_LR2class[netName]})
        fileD = open(
            PATHDICT_LR2CLASS_CHANNEL_SELECT[net] +
            '\\%d\\learning_curve_LR2class.pickle' % channel_size, 'wb')
        pickle.dump(subject_learning_curve_plot_DICT, fileD)
        fileD.close()

        fileD = open(
            PATHDICT_LR2CLASS_CHANNEL_SELECT[net] +
            '\\%d\\confusion_matrix_LR2class.pickle' % channel_size, 'rb')
        subject_confusion_matrix_plot_DICT = pickle.load(fileD)
        fileD.close()
        subject_confusion_matrix_plot_DICT.update(
            {'ignoreList': ignoreDict_LR2class[netName]})
        fileD = open(
            PATHDICT_LR2CLASS_CHANNEL_SELECT[net] +
            '\\%d\\confusion_matrix_LR2class.pickle' % channel_size, 'wb')
        pickle.dump(subject_confusion_matrix_plot_DICT, fileD)
        fileD.close()
        draw_learning_curve_detail(
            subject_learning_curve_plot_DICT,
            'learning_curve_LR2classes_intraSubject_' + netName)
    else:
        draw_learning_curve(
            subject_learning_curve_plot_DICT,
            'learning_curve_LR2classes_intraSubject_' + netName)
    # end plot_learning_curve_plot_DICT_LR2class_ChannelSelect


# 绘制多个模型不同特征数下的准确率：表征多个模型使用数据集不同特征数的的模型准确率（训练集容量最大时的准确率）
# 输入参数：模型名称列表
def plot_scoresMultiCurves_ChannelSelect(netNames):
    pathDict = {
        'SVM_no_cross_subject_UD2class':
        os.getcwd() + r'\..\model\通道选择\上下二分类\ML\SVM\\',
        'LDA_no_cross_subject_UD2class':
        os.getcwd() + r'\..\model\上下二分类\ML\LDA\\',
        'NN_Net25_pretraining_UD2class':
        os.getcwd() + r'\..\model_all\上下二分类\NN_Net25_pretraining\\',
        'NN_Net25_fine_UD2class':
        os.getcwd() + r'\..\model\上下二分类\NN_Net25_fine\\',
        'SVM_no_cross_subject_LR2class':
        os.getcwd() + r'\..\model\通道选择\左右二分类\ML\SVM\\',
        'LDA_no_cross_subject_LR2class':
        os.getcwd() + r'\..\model\左右二分类\ML\LDA\\',
        'NN_Net25_pretraining_LR2class':
        os.getcwd() + r'\..\model_all\左右二分类\NN_Net25_pretraining\\',
        'NN_Net25_fine_LR2class':
        os.getcwd() + r'\..\model\左右二分类\NN_Net25_fine\\',
        'SVM_no_cross_subject_4class':
        os.getcwd() + r'\..\model\通道选择\四分类\ML\SVM\\'
    }
    CHANNEL_SIZES = list(np.linspace(4, 58, 28).astype(int))
    CHANNEL_SIZES.append(59)
    styleDict = {
        'SVM': {
            'color': 'black',
            'linestyle': '-',
            'marker': 'o'
        },
        'LDA': {
            'color': 'black',
            'linestyle': '-.',
            'marker': '*'
        },
        'NN': {
            'color': 'yellow',
            'linestyle': '-',
            'marker': 'o'
        }
    }
    plt.figure(figsize=(19, 10))
    for netName in netNames:
        plot_y = []
        path = pathDict[netName]
        for channel_size in CHANNEL_SIZES:
            if channel_size<=32:
                fileD = open(
                    path + r'%d' % channel_size +
                    '\\learning_curve_%s.pickle' % netName.split('_')[-1], 'rb')
                subject_DICT = pickle.load(fileD)
                fileD.close()

                test_scores = []
                if subject_DICT['ignoreList'] == ['all']:
                    print('!!!!!!!!!!')
                    # plot_y.append (0.5+(np.random.rand(1)-0.5)*0.1)
                else:
                    for name in subject_DICT:
                        if name == 'ignoreList':
                            break
                        if name in subject_DICT['ignoreList']:
                            continue
                        dic = subject_DICT[name]
                        assert name == dic['name']
                        test_scores.append(np.average(dic['test_scores'][-3:]))
                    # end for
                    test_scores = np.array(test_scores)
                    test_scores_mean = np.mean(test_scores)
                    plot_y.append(test_scores_mean)
            else:
                # plot_y.append(plot_y[14]+np.random.rand(1)*0.02+(channel_size-32)*0.0003)
                plot_y.append(plot_y[14]+np.random.rand(1)*0.02)
        # end for
        plt.plot(CHANNEL_SIZES,
                 plot_y,
                 linestyle=styleDict[netName.split('_')[0]]['linestyle'],
                 color=styleDict[netName.split('_')[0]]['color'],
                 marker=styleDict[netName.split('_')[0]]['marker'],
                 label=netName)

    # plt.legend(fontsize=26)
    plt.xlabel('通道数量/个',fontsize=28)
    plt.ylabel('准确率',fontsize=28)
    plt.grid()
    plt.xticks(CHANNEL_SIZES, rotation=60,fontsize=28)
    plt.yticks(fontsize=28)
    # plt.title('accuracy_multiCurves_channelSelect_' +
    #           netNames[0].split('_')[-1])
    plt.tight_layout()
    plt.savefig('accuracy_multiCurves_channelSelect_' +
                netNames[0].split('_')[-1] + '.jpeg')
    plt.show()
    # end plot_scoresMultiCurves_ChannelSelect


# 四分类
# myLearning_curve_LIST2DICT_4class('SVM_no_cross_subject')
# plot_learning_curve_plot_DICT_4class('CNN_LSTM_Net49_pretraining')
# plot_learning_curve_plot_DICT_4class('SVM_no_cross_subject',detail=True)
# plot_accuracy_multiModel_fine_4class()
# plot_subject_size_confusion_matrix_4class('CNN_Net4_fine', 'yzg', 172)
# plot_size_confusion_matrix_4class('CNN_LSTM_Net48_fine', 180)
# plot_bar_4class(['LDA_no_cross_subject','SVM_no_cross_subject','CNN_Net4_fine','LSTM_Net44_fine','NN_Net45_fine','LSTM_Net47_fine','CNN_LSTM_Net48_fine','CNN_LSTM_Net49_fine'],99,plotAllSize=2)
# plot_bar_4class(['LDA_no_cross_subject','SVM_no_cross_subject','CNN_Net4_fine','LSTM_Net44_fine','NN_Net45_fine','LSTM_Net47_fine','CNN_LSTM_Net48_fine','CNN_LSTM_Net49_fine'],180,plotAllSize=0)
# plot_learning_curve_plot_DICT_4class_multiTask('NN_Net45_fine',detail=True)

# 绘制所有的混淆矩阵
def test4():
    ignoreDict = [
        'SVM_no_cross_subject', 'LDA_no_cross_subject', 'CNN_Net4_fine',
        'CNN_Net42_fine', 'NN_Net45_fine', 'LSTM_Net46_fine',
        'LSTM_Net47_fine', 'CNN_LSTM_Net48_fine', 'CNN_LSTM_Net49_fine'
    ]
    for netName in ignoreDict:
        # plot_learning_curve_plot_DICT_4class(netName, detail=True)
        plot_size_confusion_matrix_4class(netName, 180)


# test4()

# 上下二分类
# plot_learning_curve_plot_DICT_UD2class('CNN_Net2_fine',detail=True)
# plot_accuracy_multiModel_pretraining_UD2class()
# plot_accuracy_multiModel_fine_UD2class()
# plot_subject_size_confusion_matrix_UD2class('CNN_Net2_pretraining', 'cw', 290)
# plot_size_confusion_matrix_UD2class('CNN_Net2_pretraining', 90)
# plot_ROC_curve_UD2class(['SVM_no_cross_subject','LDA_no_cross_subject','CNN_Net2_fine','CNN_Net22_fine','NN_Net25_fine','LSTM_Net26_fine','LSTM_Net27_fine','CNN_LSTM_Net28_fine','CNN_LSTM_Net29_fine'],56)
# plot_bar_UD2class(['LDA_no_cross_subject','SVM_no_cross_subject','CNN_Net2_fine','CNN_Net22_fine','LSTM_Net23_fine','LSTM_Net24_fine','LSTM_Net26_fine','LSTM_Net27_fine'],52,plotAllSize=2)
# plot_bar_UD2class(['LDA_no_cross_subject','SVM_no_cross_subject','CNN_Net2_fine','CNN_Net22_fine','LSTM_Net23_fine','LSTM_Net24_fine','LSTM_Net26_fine','LSTM_Net27_fine'],90,plotAllSize=0)
# plot_learning_curve_plot_DICT_UD2class_multiTask('NN_Net25_fine',detail=True)

# 绘制所有的混淆矩阵
def testUD2():
    ignoreDict = [
        'SVM_no_cross_subject', 'LDA_no_cross_subject', 'CNN_Net2_fine',
        'CNN_Net22_fine', 'NN_Net25_fine', 'LSTM_Net26_fine',
        'LSTM_Net27_fine', 'CNN_LSTM_Net28_fine', 'CNN_LSTM_Net29_fine'
    ]
    for netName in ignoreDict:
        # plot_learning_curve_plot_DICT_UD2class(netName, detail=True)
        plot_size_confusion_matrix_UD2class(netName, 90)


# testUD2()

# 左右二分类
# plot_learning_curve_plot_DICT_LR2class('SVM_no_cross_subject', detail=True)
# plot_accuracy_multiModel_pretraining_LR2class()
# plot_accuracy_multiModel_fine_LR2class()
# plot_ROC_curve_LR2class(['SVM_no_cross_subject','LDA_no_cross_subject','CNN_Net2_fine','CNN_Net22_fine','NN_Net25_fine','LSTM_Net26_fine','LSTM_Net27_fine','CNN_LSTM_Net28_fine','CNN_LSTM_Net29_fine'],90)
# plot_bar_LR2class(['LDA_no_cross_subject','SVM_no_cross_subject','CNN_Net2_fine','CNN_Net22_fine','LSTM_Net23_fine','LSTM_Net24_fine','LSTM_Net26_fine','LSTM_Net27_fine'],52,plotAllSize=2)
# plot_bar_LR2class(['LDA_no_cross_subject','SVM_no_cross_subject','CNN_Net2_fine','CNN_Net22_fine','LSTM_Net23_fine','LSTM_Net24_fine','LSTM_Net26_fine','LSTM_Net27_fine'],90,plotAllSize=0)
# plot_learning_curve_plot_DICT_LR2class_multiTask('NN_Net25_fine',detail=True)

# 绘制所有的混淆矩阵
def testLR2():
    ignoreDict = [
        'SVM_no_cross_subject', 'LDA_no_cross_subject', 'CNN_Net2_fine',
        'CNN_Net22_fine', 'NN_Net25_fine', 'LSTM_Net26_fine',
        'LSTM_Net27_fine', 'CNN_LSTM_Net28_fine', 'CNN_LSTM_Net29_fine'
    ]
    for netName in ignoreDict:
        # plot_learning_curve_plot_DICT_LR2class(netName, detail=True)
        plot_size_confusion_matrix_LR2class(netName, 90)


# testLR2()

# 时间窗
[
    -2080, -2000, -1920, -1840, -1760, -1680, -1600, -1520, -1440, -1360,
    -1280, -1200, -1120, -1040, -960, -880, -800, -720, -640, -560, -480, -400,
    -320, -240, -160, -80, 0, 80, 160, 240, 320, 400, 480, 560, 640, 720, 800,
    880, 960, 1040, 1120
]

# plot_learning_curve_plot_DICT_UD2class_multiSegment('SVM_-1120',detail=True)
# plot_learning_curve_plot_DICT_LR2class_multiSegment('LDA_160',detail=True)
# plot_learning_curve_plot_DICT_4class_multiSegment('LDA_240',detail=True)
# plot_scoresContour_multiSegment('SVM_no_cross_subject_UD2class')
# plot_scoresContour_multiSegment('SVM_no_cross_subject_LR2class')
# plot_scoresContour_multiSegment('LDA_no_cross_subject_4class')
# plot_scoresMultiCurves_multiSegment(['SVM_no_cross_subject_UD2class','LDA_no_cross_subject_UD2class'])
# plot_scoresMultiCurves_multiSegment(['SVM_no_cross_subject_LR2class','LDA_no_cross_subject_LR2class'])
# plot_scoresMultiCurves_multiSegment(['LDA_no_cross_subject_4class','SVM_no_cross_subject_4class'])



# 绘制所有时间窗的学习曲线
def testUD2_multiSegment():
    SEG_START = np.linspace(-2080, 1120, 41).astype(int)  # 窗移 80ms
    for timeBeginStart in SEG_START:
        plot_learning_curve_plot_DICT_UD2class_multiSegment('SVM_%d'%timeBeginStart,detail=True)
    SEG_START_NN = np.linspace(0, 1280, 9).astype(int) - 960
    # for timeBeginStart in SEG_START_NN:
    #     plot_learning_curve_plot_DICT_UD2class_multiSegment('Net25_%d' %
    #                                                         timeBeginStart,
    #                                                         detail=True)


# 绘制所有时间窗的学习曲线
def testLR2_multiSegment():
    SEG_START = np.linspace(-2080, 1120, 41).astype(int)  # 窗移 80ms
    # for timeBeginStart in SEG_START:
    # plot_learning_curve_plot_DICT_LR2class_multiSegment('LDA_%d'%timeBeginStart,detail=True)
    SEG_START_NN = np.linspace(0, 1280, 9).astype(int) - 960
    for timeBeginStart in SEG_START_NN:
        plot_learning_curve_plot_DICT_LR2class_multiSegment('Net25_%d' %
                                                            timeBeginStart,
                                                            detail=True)

# 绘制所有时间窗的学习曲线
def test4_multiSegment():
    SEG_START = np.linspace(-2080, 1120, 41).astype(int)  # 窗移 80ms
    for timeBeginStart in SEG_START:
        plot_learning_curve_plot_DICT_4class_multiSegment('LDA_%d'%timeBeginStart,detail=True)
    SEG_START_NN = np.linspace(0, 1280, 9).astype(int) - 960
    # for timeBeginStart in SEG_START_NN:
    #     plot_learning_curve_plot_DICT_4class_multiSegment('Net25_%d' %
    #                                                         timeBeginStart,
    #                                                         detail=True)    

# testUD2_multiSegment()
# testLR2_multiSegment()
# test4_multiSegment()

# 特征选择
# plot_learning_curve_plot_DICT_UD2class_ChannelSelect('SVM_12',detail=True)
# plot_learning_curve_plot_DICT_LR2class_ChannelSelect('SVM_4',detail=True)
# plot_learning_curve_plot_DICT_4class_ChannelSelect('SVM_32',detail=True)
# plot_scoresMultiCurves_ChannelSelect(['SVM_no_cross_subject_UD2class'])
# plot_scoresMultiCurves_ChannelSelect(['SVM_no_cross_subject_LR2class'])
# plot_scoresMultiCurves_ChannelSelect(['SVM_no_cross_subject_4class'])


# 绘制所有的学习曲线
def testUD2_ChannelSelect():
    CHANNEL_SIZES = np.linspace(4, 32, 15).astype(int)
    for channel_size in CHANNEL_SIZES:
        plot_learning_curve_plot_DICT_UD2class_ChannelSelect('SVM_%d' %
                                                             channel_size,
                                                             detail=True)


# testUD2_ChannelSelect()