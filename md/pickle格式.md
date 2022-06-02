# 微调数据

深度学习模型：[ 0,  3,  7, 11, 15, 18, 22, 26, 30, 33, 37, 41, 45, 48, 52, 56, 60, 63, 67, 71, 75, 78, 82, 86, 90]

机器学习模型：[ 9, 12, 15, 19, 22, 25, 29, 32, 36, 39, 42, 46, 49, 52, 56, 59, 62, 66, 69, 73, 76, 79, 83, 86, 90]

共同：15，22，52，56，86，90

# 数据

## single_move_motion_start_motion_end

CrossSubject\data\cwm\单任务运动\single_move_motion_start_motion_end.pickle

get_EEG_single_move_artificial_features()

包含单任务运动 1 个被试 4 个方向的数据，已完成了预处理、人工特征提取。

featureArray, labelArray 分别为 样本集特征矩阵(numpy 二维数组(行：各样本；列：特征值))，样本集标签矩阵(numpy 一维数组)。

运动开始时刻的判断是编程自动判断的。

## single_move_eeg_with_label

CrossSubject\data\cwm\单任务运动\single_move_eeg_with_label.pickle

get_EEG_single_move()

包含单任务运动 1个被试 4 个方向的全部数据，已完成了预处理。

dataArraySet, labelArray 分别为样本集数组(numpy 三维数组(各样本 \* 各通道 \* 采样点))，样本集标签数组(numpy 二维数组(各样本 \* 标签信息(标签值，运动开始时刻，运动结束时刻)))。

运动开始时刻的判断是编程自动判断的。

## single_move_motion_start_motion_end_beginTimeFile

CrossSubject\data\cwm\单任务运动\single_move_motion_start_motion_end_beginTimeFile.pickle

get_EEG_features()

包含单任务运动 1 个被试 4 个方向的数据，已完成了预处理、人工特征提取。

featureArray, labelArray 分别为 样本集特征矩阵(numpy 二维数组(行：各样本；列：特征值))，样本集标签矩阵(numpy 二维数组(行：各样本；列：标签))。

运动开始时刻的判断是逐实验手动判断的。

## single_move_motion_start_motion_end_multiFrame_beginTimeFile

CrossSubject\data\cwm\单任务运动\single_move_motion_start_motion_end_multiFrame_beginTimeFile.pickle

get_EEG_features_multiFrame()

根据输入参数进行分帧。

包含单任务运动 1 个被试 4 个方向的数据，已完成了分帧、预处理、人工特征提取。

featureArray, labelArray 分别为 样本集特征矩阵(numpy 三维数组(各样本 * 各帧 * 特征值))，样本集标签矩阵(numpy 二维数组(行：各样本；列：标签))。

运动开始时刻的判断是逐实验手动判断的。

## single_move_eeg_with_label_beginTimeFile

CrossSubject\data\cwm\四分类\single_move_eeg_with_label_beginTimeFile.pickle

get_EEG_single_move_beginTimeFile()

包含单任务运动 1个被试 4 个方向的全部数据，已完成了预处理。

dataArraySet, labelArray 分别为样本集数组(numpy 三维数组(各样本 \* 各通道 \* 采样点))，样本集标签数组(numpy 二维数组(各样本 \* 标签信息(标签值，运动开始时刻)))。

运动开始时刻的判断是逐实验手动判断的。

## single_move_crossSubject_sampleSet_X_y

CrossSubject\data\四分类\single_move_crossSubject_sampleSetMulti_X_y.pickle

getSampleSet_beginTimeFile()

包含单任务运动 8 个被试 4 个方向的全部数据，已完成了预处理。

X_sampleSetMulti, Y_sampleSetMulti 分别为 样本集\_X 列表（共 8 个元素 ），样本集\_y 列表（共 8 个元素），各元素内容如下：

样本集\_X(numpy 四维数组(各样本 \* 各数据通道 \* 各采样点))。

样本集_Y(numpy 一维数组)。

运动开始时刻的判断是逐实验手动判断的。

## single_move_crossSubject_sampleSetMulti_X_y

CrossSubject\data\四分类\single_move_crossSubject_sampleSetMulti_X_y.pickle

getSampleSetMulti_beginTimeFile()

包含单任务运动 8 个被试 4 个方向的全部数据，已完成了预处理。

X_sampleSetMulti, Y_sampleSetMulti 分别为 样本集\_X 列表（共 8 个元素 ），样本集\_y 列表（共 8 个元素），各元素内容如下：

样本集\_X(numpy 四维数组(各样本 \* 各频带 \* 各数据通道 \* 各采样点))。

样本集_Y(numpy 一维数组)。

运动开始时刻的判断是逐实验手动判断的。

## single_move_motion_start-0

CrossSubject\data\cwm\四分类\single_move_motion_start-0.pickle

get_EEG_features_multiSegment()

包含单任务运动 1 个被试 4 个方向的数据，已完成了预处理、人工特征提取。

featureArray, labelArray 分别为 样本集特征矩阵(numpy 二维数组(行：各样本；列：特征值))，样本集标签矩阵(numpy 二维数组(行：各样本；列：标签))。

数据时间段：[actBeginTime-%d,actBeginTime-%d+1280]

运动开始时刻的判断是逐实验手动判断的。

# 模型结果

## learning_curve_CNN_Net42_pretraining_4class

跨被试四分类

CrossSubject\model\单任务运动\CNN_Net42_pretraining\learning_curve_CNN_Net42_pretraining_4class.pickle

CNN_cross_subject_4class_pretraining()

Net42 预训练所得模型的测试结果，包括各被试的不同预训练训练集大小下的训练集准确率及测试集准确率。

## learning_curve_CNN_Net42_fine_4class

跨被试四分类

CrossSubject\model\单任务运动\CNN_Net42_fine\learning_curve_CNN_Net42_fine_4class.pickle

CNN_cross_subject_4class_fine()

Net42 微调所得模型的测试结果，包括各被试的不同微调训练集大小下的训练集准确率及测试集准确率。

注：预训练模型使用相应被试的由 `CNN_cross_subject_4class_pretraining()` 得到的某预训练训练集大小下的预训练模型。

## learning_curve_SVM_cross_subject_4class

跨被试四分类

CrossSubject\model\单任务运动\ML\learning_curve_SVM_cross_subject_4class.pickle

SVM_cross_subject_4class()

使用 SVM 模型的跨被试四分类测试结果，包括各被试的训练集准确率及测试集准确率。

## learning_curve_SVM_cross_subject_mixture2class

跨被试混合二分类

CrossSubject\model\单任务运动\ML\learning_curve_SVM_cross_subject_mixture2class.pickle

SVM_cross_subject_mixture2class()

使用 SVM 模型的跨被试混合二分类测试结果，包括各被试的训练集准确率及测试集准确率。

## learning_curve_SVM_no_cross_subject_4class

CrossSubject\model\单任务运动\ML\learning_curve_SVM_no_cross_subject_4class.pickle

非跨被试四分类

SVM_no_cross_subject_4class()

使用 SVM 模型的非跨被试四分类测试结果，包括各被试的训练集准确率及测试集准确率。

## learning_curve_SVM_no_cross_subject_mixture2class

CrossSubject\model\单任务运动\ML\learning_curve_SVM_no_cross_subject_mixture2class.pickle

非跨被试混合二分类

SVM_no_cross_subject_mixture2class()

使用 SVM 模型的非跨被试混合二分类测试结果，包括各被试的训练集准确率及测试集准确率。

## learning_curve_LR2class

CrossSubject\model\左右二分类\NN_Net25_fine\0\learning_curve_LR2class.pickle

跨被试左右二分类

NN_cross_subject_LR2class_fine_Net25_multiSegment()

使用Net25模型的跨被试左右二分类各时间窗微调结果，包括各被试的训练集准确率及测试集准确率。

