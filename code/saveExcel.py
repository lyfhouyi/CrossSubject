# 模型结果写入 Excel

import os
import pickle

import numpy as np
import xlwt
from sklearn import metrics

NAMEDICT={'cw':'S1','cwm':'S2', 'kx':'S3', 'pbl':'S4', 'wrd':'S5', 'wxc':'S6', 'xsc':'S7', 'yzg':'S8'}


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
    'LSTM_Net44_pretraining':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net44_pretraining\confusion_matrix_LSTM_Net44_pretraining_4class.pickle',    
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
    'LSTM_Net23_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net23_pretraining\confusion_matrix_LSTM_Net23_pretraining_UD2class.pickle',
    'LSTM_Net24_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net24_pretraining\confusion_matrix_LSTM_Net24_pretraining_UD2class.pickle',      
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
    'LSTM_Net23_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net23_pretraining\confusion_matrix_LSTM_Net23_pretraining_LR2class.pickle',
    'LSTM_Net24_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net24_pretraining\confusion_matrix_LSTM_Net24_pretraining_LR2class.pickle',  
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
    'LSTM_Net44_pretraining':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net44_pretraining\learning_curve_LSTM_Net44_pretraining_4class.pickle',    
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
    'LSTM_Net44_fine':
    os.getcwd() +
    r'\..\model\四分类\LSTM_Net44_fine\learning_curve_LSTM_Net44_fine_4class.pickle',
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
    'LSTM_Net23_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net23_pretraining\learning_curve_LSTM_Net23_pretraining_UD2class.pickle',
    'LSTM_Net24_pretraining':
    os.getcwd() +
    r'\..\model\上下二分类\LSTM_Net24_pretraining\learning_curve_LSTM_Net24_pretraining_UD2class.pickle',
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
    'LSTM_Net23_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net23_pretraining\learning_curve_LSTM_Net23_pretraining_LR2class.pickle',
    'LSTM_Net24_pretraining':
    os.getcwd() +
    r'\..\model\左右二分类\LSTM_Net24_pretraining\learning_curve_LSTM_Net24_pretraining_LR2class.pickle',  
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
    r'\..\model\多任务\四分类\NN_Net45_fine\learning_curve_NN_Net45_fine_4class.pickle'
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
    r'\..\model\多任务\上下二分类\NN_Net25_fine\learning_curve_NN_Net25_fine_UD2class.pickle'
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
    r'\..\model\多任务\左右二分类\NN_Net25_fine\learning_curve_NN_Net25_fine_LR2class.pickle'
}

# 将模型预测准确率写入 Excel：上下二分类预训练
def pretraining_medium_UD2class():
    sizeDict={'CNN_Net2_pretraining':290,'CNN_Net22_pretraining':330,'LSTM_Net23_pretraining':330,'LSTM_Net24_pretraining':330,'LSTM_Net26_pretraining':290,'LSTM_Net27_pretraining':330}
    netList=['CNN_Net2_pretraining','CNN_Net22_pretraining','LSTM_Net23_pretraining','LSTM_Net24_pretraining','LSTM_Net26_pretraining','LSTM_Net27_pretraining','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('中等规模350左右')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            resultDict=subject_learning_curve_plot_LIST[subjectName]

            # 取 sizeDict 指示的那个，sizeDict 指示的与微调用到的相同。
            if netName in ['LDA_no_cross_subject','SVM_no_cross_subject']:
                ind=len(resultDict['train_sizes'])-1
            else:
                ind=list(resultDict['train_sizes']).index(sizeDict[netName]) # 取 sizeDict 指示的那个，sizeDict 指示的与微调用到的相同。
            sub_net_value=resultDict['test_scores'][ind]
            ws.write(i+1,j+1,sub_net_value)
            netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\pretraining_UD2class_中等规模.xls')

    for netName in netList:
        fileD = open(PATHDICT_CONFUSION_MATRIX_UD2CLASS[netName], 'rb')
        confusion_matrix_DICT = pickle.load(fileD)
        fileD.close()
        accuracy_list=[]
        f1Score_list=[]
        for subjectName in NAMEDICT:
            accuracy_list.append(sum(confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_true']==confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_pred'])/len(confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_true']))
            f1Score_list.append(metrics.f1_score(confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_true'],confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_pred']))    
        print(netName)
        print('accuracy_avg = %.3f'%(sum(accuracy_list)/len(accuracy_list)))
        print('f1_score_avg = %.3f'%(sum(f1Score_list)/len(f1Score_list)))
    # end pretraining_medium_UD2class


# 将模型预测准确率写入 Excel：左右二分类预训练
def pretraining_medium_LR2class():
    sizeDict={'CNN_Net2_pretraining':330,'CNN_Net22_pretraining':270,'LSTM_Net23_pretraining':350,'LSTM_Net24_pretraining':330,'LSTM_Net26_pretraining':310,'LSTM_Net27_pretraining':290}
    netList=['CNN_Net2_pretraining','CNN_Net22_pretraining','LSTM_Net23_pretraining','LSTM_Net24_pretraining','LSTM_Net26_pretraining','LSTM_Net27_pretraining','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('中等规模350左右')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            resultDict=subject_learning_curve_plot_LIST[subjectName]

            # 取 sizeDict 指示的那个，sizeDict 指示的与微调用到的相同。
            if netName in ['LDA_no_cross_subject','SVM_no_cross_subject']:
                ind=len(resultDict['train_sizes'])-1
            else:
                ind=list(resultDict['train_sizes']).index(sizeDict[netName]) # 取 sizeDict 指示的那个，sizeDict 指示的与微调用到的相同。
            sub_net_value=resultDict['test_scores'][ind]
            ws.write(i+1,j+1,sub_net_value)
            netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\pretraining_LR2class_中等规模.xls')

    for netName in netList:
        fileD = open(PATHDICT_CONFUSION_MATRIX_LR2CLASS[netName], 'rb')
        confusion_matrix_DICT = pickle.load(fileD)
        fileD.close()
        accuracy_list=[]
        f1Score_list=[]
        for subjectName in NAMEDICT:
            accuracy_list.append(sum(confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_true']==confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_pred'])/len(confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_true']))
            f1Score_list.append(metrics.f1_score(confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_true'],confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_pred']))    
        print(netName)
        print('accuracy_avg = %.3f'%(sum(accuracy_list)/len(accuracy_list)))
        print('f1_score_avg = %.3f'%(sum(f1Score_list)/len(f1Score_list)))
    # end pretraining_medium_LR2class


# 将模型预测准确率写入 Excel：四分类预训练
def pretraining_medium_4class():
    sizeDict={'CNN_Net4_pretraining':622,'LSTM_Net44_pretraining':661,'NN_Net45_pretraining':700,'LSTM_Net47_pretraining':680,'CNN_LSTM_Net48_pretraining':603,'CNN_LSTM_Net49_pretraining':680}
    netList=['CNN_Net4_pretraining','LSTM_Net44_pretraining','NN_Net45_pretraining','LSTM_Net47_pretraining','CNN_LSTM_Net48_pretraining','CNN_LSTM_Net49_pretraining','LDA_no_cross_subject','SVM_no_cross_subject']
    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('中等规模700左右')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_4CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            resultDict=subject_learning_curve_plot_LIST[subjectName]

            # 取 sizeDict 指示的那个，sizeDict 指示的与微调用到的相同。
            if netName in ['LDA_no_cross_subject','SVM_no_cross_subject']:
                ind=len(resultDict['train_sizes'])-1
            else:
                ind=list(resultDict['train_sizes']).index(sizeDict[netName]) # 取 sizeDict 指示的那个，sizeDict 指示的与微调用到的相同。
            sub_net_value=resultDict['test_scores'][ind]
            ws.write(i+1,j+1,sub_net_value)
            netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\pretraining_4class_中等规模.xls')
    
    for netName in netList:
        fileD = open(PATHDICT_CONFUSION_MATRIX_4CLASS[netName], 'rb')
        confusion_matrix_DICT = pickle.load(fileD)
        fileD.close()
        accuracy_list=[]
        f1Score_list=[]
        for subjectName in NAMEDICT:
            accuracy_list.append(sum(confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_true']==confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_pred'])/len(confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_true']))
            f1Score_list.append(metrics.f1_score(confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_true'],confusion_matrix_DICT[subjectName]['%d' % sizeDict[netName]]['y_pred'],average='macro'))    
        print(netName)
        print('accuracy_avg = %.3f'%(sum(accuracy_list)/len(accuracy_list)))
        print('f1_score_avg = %.3f'%(sum(f1Score_list)/len(f1Score_list)))    
    # end pretraining_medium_4class

# 将模型预测准确率写入 Excel：上下二分类预训练
def pretraining_large_UD2class():
    sizeDict={'CNN_Net2_pretraining':290,'CNN_Net22_pretraining':330,'LSTM_Net23_pretraining':330,'LSTM_Net24_pretraining':330,'LSTM_Net26_pretraining':290,'LSTM_Net27_pretraining':330}
    netList=['CNN_Net2_pretraining','CNN_Net22_pretraining','LSTM_Net23_pretraining','LSTM_Net24_pretraining','LSTM_Net26_pretraining','LSTM_Net27_pretraining','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('大规模700左右')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            resultDict=subject_learning_curve_plot_LIST[subjectName]

            # 取 sizeDict 指示的那个，sizeDict 指示的与微调用到的相同。
            if netName in ['LDA_no_cross_subject','SVM_no_cross_subject']:
                ind=len(resultDict['train_sizes'])-1
            else:
                ind=list(resultDict['test_scores']).index(resultDict['test_scores'].max()) # 取各训练集容量中，准确率最高的那个。
            sub_net_value=resultDict['test_scores'][ind]
            ws.write(i+1,j+1,sub_net_value)
            netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\pretraining_UD2class_大规模.xls')
    # end pretraining_large_UD2class



# 将模型预测准确率写入 Excel：上下二分类预训练
def pretraining_large_LR2class():
    sizeDict={'CNN_Net2_pretraining':330,'CNN_Net22_pretraining':270,'LSTM_Net23_pretraining':350,'LSTM_Net24_pretraining':330,'LSTM_Net26_pretraining':310,'LSTM_Net27_pretraining':290}
    netList=['CNN_Net2_pretraining','CNN_Net22_pretraining','LSTM_Net23_pretraining','LSTM_Net24_pretraining','LSTM_Net26_pretraining','LSTM_Net27_pretraining','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('大规模700左右')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            resultDict=subject_learning_curve_plot_LIST[subjectName]

            # 取 sizeDict 指示的那个，sizeDict 指示的与微调用到的相同。
            if netName in ['LDA_no_cross_subject','SVM_no_cross_subject']:
                ind=len(resultDict['train_sizes'])-1
            else:
                ind=list(resultDict['test_scores']).index(resultDict['test_scores'].max()) # 取各训练集容量中，准确率最高的那个。
            sub_net_value=resultDict['test_scores'][ind]
            ws.write(i+1,j+1,sub_net_value)
            netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\pretraining_LR2class_大规模.xls')
    # end pretraining_large_LR2class

# 将模型预测准确率写入 Excel：四分类预训练
def pretraining_large_4class():
    sizeDict={'CNN_Net4_pretraining':622,'LSTM_Net44_pretraining':661,'NN_Net45_pretraining':700,'LSTM_Net47_pretraining':680,'CNN_LSTM_Net48_pretraining':603,'CNN_LSTM_Net49_pretraining':680}
    netList=['CNN_Net4_pretraining','LSTM_Net44_pretraining','NN_Net45_pretraining','LSTM_Net47_pretraining','CNN_LSTM_Net48_pretraining','CNN_LSTM_Net49_pretraining','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('大规模1400左右')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_4CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            resultDict=subject_learning_curve_plot_LIST[subjectName]

            # 取 sizeDict 指示的那个，sizeDict 指示的与微调用到的相同。
            if netName in ['LDA_no_cross_subject','SVM_no_cross_subject']:
                ind=len(resultDict['train_sizes'])-1
            else:
                ind=list(resultDict['test_scores']).index(resultDict['test_scores'].max()) # 取各训练集容量中，准确率最高的那个。
            sub_net_value=resultDict['test_scores'][ind]
            ws.write(i+1,j+1,sub_net_value)
            netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\pretraining_4class_大规模.xls')
    # end pretraining_large_4class


# 将模型预测准确率写入 Excel：上下二分类微调
def fine_small_UD2class():
    netList=['CNN_Net2_fine','CNN_Net22_fine','LSTM_Net23_fine','LSTM_Net24_fine','LSTM_Net26_fine','LSTM_Net27_fine','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('小规模22')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            if subjectName in subject_learning_curve_plot_LIST['ignoreList']:
                pass
            else:
                resultDict=subject_learning_curve_plot_LIST[subjectName]
                ind=list(resultDict['train_sizes']).index(22) # 微调样本集容量为 22 的那个
                sub_net_value=resultDict['test_scores'][ind]
                ws.write(i+1,j+1,sub_net_value)
                netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\fine_UD2class_小规模.xls')
    # end fine_small_UD2class

# 将模型预测准确率写入 Excel：上下二分类微调
def fine_medium_UD2class():
    netList=['CNN_Net2_fine','CNN_Net22_fine','LSTM_Net23_fine','LSTM_Net24_fine','LSTM_Net26_fine','LSTM_Net27_fine','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('中等规模52')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            if subjectName in subject_learning_curve_plot_LIST['ignoreList']:
                pass
            else:
                resultDict=subject_learning_curve_plot_LIST[subjectName]
                ind=list(resultDict['train_sizes']).index(52) # 微调样本集容量为 52 的那个
                sub_net_value=resultDict['test_scores'][ind]
                ws.write(i+1,j+1,sub_net_value)
                netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\fine_UD2class_中等规模.xls')
    # end fine_medium_UD2class

# 将模型预测准确率写入 Excel：上下二分类微调
def fine_large_UD2class():
    netList=['CNN_Net2_fine','CNN_Net22_fine','LSTM_Net23_fine','LSTM_Net24_fine','LSTM_Net26_fine','LSTM_Net27_fine','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('大规模90')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            if subjectName in subject_learning_curve_plot_LIST['ignoreList']:
                pass
            else:
                resultDict=subject_learning_curve_plot_LIST[subjectName]
                ind=list(resultDict['train_sizes']).index(90) # 微调样本集容量为 90 的那个
                sub_net_value=resultDict['test_scores'][ind]
                ws.write(i+1,j+1,sub_net_value)
                netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\fine_UD2class_大规模.xls')
    # end fine_large_UD2class

# 将模型预测准确率写入 Excel：左右二分类微调
def fine_small_LR2class():
    netList=['CNN_Net2_fine','CNN_Net22_fine','LSTM_Net23_fine','LSTM_Net24_fine','LSTM_Net26_fine','LSTM_Net27_fine','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('小规模22')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            if subjectName in subject_learning_curve_plot_LIST['ignoreList']:
                pass
            else:
                resultDict=subject_learning_curve_plot_LIST[subjectName]
                ind=list(resultDict['train_sizes']).index(22) # 微调样本集容量为 22 的那个
                sub_net_value=resultDict['test_scores'][ind]
                ws.write(i+1,j+1,sub_net_value)
                netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\fine_LR2class_小规模.xls')
    # end fine_small_LR2class

# 将模型预测准确率写入 Excel：左右二分类微调
def fine_medium_LR2class():
    netList=['CNN_Net2_fine','CNN_Net22_fine','LSTM_Net23_fine','LSTM_Net24_fine','LSTM_Net26_fine','LSTM_Net27_fine','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('中等规模52')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            if subjectName in subject_learning_curve_plot_LIST['ignoreList']:
                pass
            else:
                resultDict=subject_learning_curve_plot_LIST[subjectName]
                ind=list(resultDict['train_sizes']).index(52) # 微调样本集容量为 52 的那个
                sub_net_value=resultDict['test_scores'][ind]
                ws.write(i+1,j+1,sub_net_value)
                netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\fine_LR2class_中等规模.xls')
    # end fine_medium_LR2class

# 将模型预测准确率写入 Excel：左右二分类微调
def fine_large_LR2class():
    netList=['CNN_Net2_fine','CNN_Net22_fine','LSTM_Net23_fine','LSTM_Net24_fine','LSTM_Net26_fine','LSTM_Net27_fine','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('大规模90')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            if subjectName in subject_learning_curve_plot_LIST['ignoreList']:
                pass
            else:
                resultDict=subject_learning_curve_plot_LIST[subjectName]
                ind=list(resultDict['train_sizes']).index(90) # 微调样本集容量为 90 的那个
                sub_net_value=resultDict['test_scores'][ind]
                ws.write(i+1,j+1,sub_net_value)
                netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\fine_LR2class_大规模.xls')
    # end fine_large_LR2class


# 将模型预测准确率写入 Excel：四分类微调
def fine_small_4class():
    netList=['CNN_Net4_fine','LSTM_Net44_fine','NN_Net45_fine','LSTM_Net47_fine','CNN_LSTM_Net48_fine','CNN_LSTM_Net49_fine','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('小规模47')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_4CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            if subjectName in subject_learning_curve_plot_LIST['ignoreList']:
                pass
            else:
                resultDict=subject_learning_curve_plot_LIST[subjectName]
                ind=list(resultDict['train_sizes']).index(47) # 微调样本集容量为 22 的那个
                sub_net_value=resultDict['test_scores'][ind]
                ws.write(i+1,j+1,sub_net_value)
                netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\fine_4class_小规模.xls')
    # end fine_small_4class


# 将模型预测准确率写入 Excel：四分类微调
def fine_medium_4class():
    netList=['CNN_Net4_fine','LSTM_Net44_fine','NN_Net45_fine','LSTM_Net47_fine','CNN_LSTM_Net48_fine','CNN_LSTM_Net49_fine','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('中等规模99')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_4CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            if subjectName in subject_learning_curve_plot_LIST['ignoreList']:
                pass
            else:
                resultDict=subject_learning_curve_plot_LIST[subjectName]
                ind=list(resultDict['train_sizes']).index(99) # 微调样本集容量为 52 的那个
                sub_net_value=resultDict['test_scores'][ind]
                ws.write(i+1,j+1,sub_net_value)
                netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\fine_4class_中等规模.xls')
    # end fine_medium_4class


# 将模型预测准确率写入 Excel：左右二分类微调
def fine_large_4class():
    netList=['CNN_Net4_fine','LSTM_Net44_fine','NN_Net45_fine','LSTM_Net47_fine','CNN_LSTM_Net48_fine','CNN_LSTM_Net49_fine','LDA_no_cross_subject','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('大规模180')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_4CLASS[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            if subjectName in subject_learning_curve_plot_LIST['ignoreList']:
                pass
            else:
                resultDict=subject_learning_curve_plot_LIST[subjectName]
                ind=list(resultDict['train_sizes']).index(180) # 微调样本集容量为 90 的那个
                sub_net_value=resultDict['test_scores'][ind]
                ws.write(i+1,j+1,sub_net_value)
                netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\fine_4class_大规模.xls')
    # end fine_large_4class


# 将模型预测准确率写入 Excel：上下二分类微调
def fine_large_UD2class_multiTask():
    netList=['NN_Net25_fine','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('大规模90')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_UD2CLASS_MULTITASK[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            if subjectName in subject_learning_curve_plot_LIST['ignoreList']:
                pass
            else:
                resultDict=subject_learning_curve_plot_LIST[subjectName]
                ind=list(resultDict['train_sizes']).index(90) # 微调样本集容量为 90 的那个
                sub_net_value=resultDict['test_scores'][ind]
                ws.write(i+1,j+1,sub_net_value)
                netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\多任务\\fine_UD2class_大规模.xls')
    # end fine_large_UD2class_multiTask

# 将模型预测准确率写入 Excel：左右二分类微调
def fine_large_LR2class_multiTask():
    netList=['NN_Net25_fine','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('大规模90')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_LR2CLASS_MULTITASK[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            if subjectName in subject_learning_curve_plot_LIST['ignoreList']:
                pass
            else:
                resultDict=subject_learning_curve_plot_LIST[subjectName]
                ind=list(resultDict['train_sizes']).index(90) # 微调样本集容量为 90 的那个
                sub_net_value=resultDict['test_scores'][ind]
                ws.write(i+1,j+1,sub_net_value)
                netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\多任务\\fine_LR2class_大规模.xls')
    # end fine_large_LR2class_multiTask

# 将模型预测准确率写入 Excel：左右二分类微调
def fine_large_4class_multiTask():
    netList=['NN_Net45_fine','SVM_no_cross_subject']

    fileExcel=xlwt.Workbook(encoding='ascii')
    ws=fileExcel.add_sheet('大规模180')
   
    # 写首行
    for subjectName,i in zip(NAMEDICT,range(len(NAMEDICT))):
        ws.write(0,i+1,label=NAMEDICT[subjectName])
    ws.write(0,len(NAMEDICT)+1,label='Average ± SD')

    # 写各模型准确率
    for netName,i in zip(netList,range(len(netList))):
        fileD = open(PATHDICT_LEARNING_CURVE_4CLASS_MULTITASK[netName], 'rb')
        subject_learning_curve_plot_LIST = pickle.load(fileD)
        ws.write(i+1,0,label=netName)
        netValues=[]
        for subjectName,j in zip(NAMEDICT,range(len(NAMEDICT))):
            if subjectName in subject_learning_curve_plot_LIST['ignoreList']:
                pass
            else:
                resultDict=subject_learning_curve_plot_LIST[subjectName]
                ind=list(resultDict['train_sizes']).index(180) # 微调样本集容量为 90 的那个
                sub_net_value=resultDict['test_scores'][ind]
                ws.write(i+1,j+1,sub_net_value)
                netValues.append(sub_net_value)
        netValues=np.array(netValues)
        avg=netValues.mean()
        std=netValues.std() # 对应 Excel 中的 STDEV.P()
        ws.write(i+1,len(NAMEDICT)+1,label='%.3f ± %.3f'%(avg,std))
        print(netName)
    fileD.close()
    fileExcel.save(os.getcwd() + r'\\..\result\\多任务\\fine_4class_大规模.xls')
    # end fine_large_4class_multiTask

# pretraining_medium_UD2class()
# pretraining_large_UD2class()
# pretraining_medium_LR2class()
# pretraining_large_LR2class()
# pretraining_medium_4class()
# pretraining_large_4class()

# fine_small_UD2class()
# fine_medium_UD2class()
# fine_large_UD2class()
# fine_small_LR2class()
# fine_medium_LR2class()
# fine_large_LR2class()
# fine_small_4class()
# fine_medium_4class()
# fine_large_4class()


# fine_large_UD2class_multiTask()
# fine_large_LR2class_multiTask()
# fine_large_4class_multiTask()
