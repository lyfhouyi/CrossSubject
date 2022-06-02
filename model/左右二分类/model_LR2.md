---
typora-copy-images-to: ./
---



# 模型训练进度

| 模型  | 预训练 | 微调 |
| ----- | ------ | ---- |
| Net2  | √      | √    |
| Net22 | √      | √    |
| Net25 | √      | √    |
| Net26 | √      | √    |
| Net27 | √      | √    |
| Net28 | √      | √    |
| Net29 | √      | √    |



# LDA

## 数据源

起始于运动开始时刻前 640 ms，结束于开始时刻后 640 ms。

## 输入

人工提取的特征。

## 结果

`\model\左右二分类\ML\LDA\learning_curve_LDA_no_cross_subject_LR2class.pickle`



# SVM

## 数据源

起始于运动开始时刻前 640 ms，结束于开始时刻后 640 ms。

## 输入

人工提取的特征。

## 结果

`\model\左右二分类\ML\SVM\learning_curve_SVM_no_cross_subject_LR2class.pickle`



# Net2

## 数据源

起始于运动开始时刻前 640 ms。

## 输入

* 4 帧数据，帧宽 480 ms，帧移 160 ms，采样频率 200 Hz。

* 各帧：3 频率通道 \* 32 通道 \* 采样点。

## 模型

![Net2](D:\硕士学习\毕业论文\CrossSubject\model\左右二分类\Net2.JPG)

* 预训练
  * 各帧分别送入 CNN 层，将各帧 CNN 层的输出级联送入全连接层。
  * 各帧使用相同的 CNN 层参数。
* 微调
  * 只微调全连接层。

## 结果

* 预训练

  `\model_all\左右二分类\CNN_Net2_pretraining`

* 微调

  `\model\左右二分类\CNN_Net2_fine`



# Net22

## 数据源

起始于运动开始时刻前 640 ms。

## 输入

![Net22](D:\硕士学习\毕业论文\CrossSubject\model\左右二分类\Net22.JPG)

- 7 帧数据，帧宽 160 ms，帧移 160 ms，采样频率 200 Hz。
- 各帧：3 频率通道 \* 32 通道 \* 采样点。

## 模型

- 预训练
  - 各帧分别送入 CNN 层，将各帧 CNN 层的输出级联送入全连接层。
  - 各帧使用相同的 CNN 层参数。

- 微调
  - 只微调全连接层。

## 结果

- 预训练

  `\model_all\左右二分类\CNN_Net22_pretraining`

- 微调

  `\model\左右二分类\CNN_Net22_fine`



# Net25

## 数据源

起始于运动开始时刻前 640 ms，结束于开始时刻后 640 ms。

## 输入

人工提取的特征。

## 模型

![Net25](D:\硕士学习\毕业论文\CrossSubject\model\左右二分类\Net25.JPG)

- 预训练
  - 输入数据直接送入 NN 网络。
- 微调
  - 只微调最后两层（第五、第六层）。

## 结果

- 预训练

  `\model_all\左右二分类\NN_Net25_pretraining`

- 微调

  `\model\左右二分类\NN_Net25_fine`



# Net26

## 数据源

起始于运动开始时刻前 640 ms。

## 输入

- 7 帧数据，帧宽 320 ms，帧移 160 ms。
- 各帧：人工提取的特征。

## 模型

![Net26](D:\硕士学习\毕业论文\CrossSubject\model\左右二分类\Net26.JPG)

- 预训练
  - 每帧作为一个时间步。
  - 各时间步依次送入 LSTM 层，将最后一时间步 LSTM 层的输出送入全连接层。
- 微调
  - 只微调全连接层。

## 结果

- 预训练

  `\model_all\左右二分类\LSTM_Net26_pretraining`

- 微调

  `\model\左右二分类\LSTM_Net26_fine`



# Net27

## 数据源

起始于运动开始时刻前 640 ms。

## 输入

- 7 帧数据，帧宽 320 ms，帧移 160 ms。

- 各帧：人工提取的特征。

## 模型

![Net27](D:\硕士学习\毕业论文\CrossSubject\model\左右二分类\Net27.JPG)

- 预训练
  - 每帧作为一个时间步。
  - 各时间步依次送入 LSTM 层，将各时间步 LSTM 层的输出级联送入全连接层。
- 微调
  - 只微调全连接层。

## 结果

- 预训练

  `\model_all\左右二分类\LSTM_Net27_pretraining`

- 微调

  `\model\左右二分类\LSTM_Net27_fine`



# Net28

## 数据源

起始于运动开始时刻前 640 ms。

## 输入

- 7 帧数据，帧宽 160 ms，帧移 160 ms，采样频率 200 Hz。
- 各帧：3 频率通道 \* 32 通道 \* 采样点。

## 模型

![Net28](D:\硕士学习\毕业论文\CrossSubject\model\左右二分类\Net28.JPG)

- 预训练
  - 各帧分别送入 CNN 层，将各帧 CNN 层的输出送入 LSTM 层。
  - 各帧使用相同的 CNN 层参数。
  - 每帧 CNN 层的输出作为一个时间步。
  - 各时间步依次送入 LSTM 层，将最后一时间步 LSTM 层的输出送入全连接层。
- 微调
  - 只微调全连接层。

## 结果

- 预训练

  `\model_all\左右二分类\CNN_LSTM_Net28_pretraining`

- 微调

  `\model\左右二分类\CNN_LSTM_Net28_fine`



# Net29

## 数据源

起始于运动开始时刻前 640 ms。

## 输入

- 4 帧数据，帧宽 480 ms，帧移 160 ms，采样频率 200 Hz。
- 各帧：3 频率通道 \* 32 通道 \* 采样点。

## 模型

![Net29](D:\硕士学习\毕业论文\CrossSubject\model\左右二分类\Net29.JPG)

- 预训练
  - 各帧分别送入 CNN 层，将各帧 CNN 层的输出送入 LSTM 层。
  - 各帧使用相同的 CNN 层参数。
  - 每帧 CNN 层的输出作为一个时间步。
  - 各时间步依次送入 LSTM 层，将各时间步 LSTM 层的输出级联送入全连接层。
- 微调
  - 只微调全连接层。

## 结果

- 预训练

  `\model_all\左右二分类\CNN_LSTM_Net29_pretraining`

- 微调

  `\model\左右二分类\CNN_LSTM_Net29_fine`


