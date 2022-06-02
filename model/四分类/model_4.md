---
typora-copy-images-to: ./
---



# 模型训练进度

| 模型  | 预训练 | 微调 |
| ----- | ------ | ---- |
| Net4  | √      | √    |
| Net42 | √      | √    |
| Net45 | √      | √    |
| Net46 | √      | √    |
| Net47 | √      | √    |
| Net48 | √      | √    |
| Net49 | √      | √    |



# LDA

## 数据源

起始于运动开始时刻前 640 ms，结束于开始时刻后 640 ms。

## 输入

人工提取的特征。

## 结果

`\model\四分类\ML\LDA\learning_curve_LDA_no_cross_subject_4class.pickle`



# SVM

## 数据源

起始于运动开始时刻前 640 ms，结束于开始时刻后 640 ms。

## 输入

人工提取的特征。

## 结果

`\model\四分类\ML\SVM\learning_curve_SVM_no_cross_subject_4class.pickle`



# Net4

## 数据源

起始于运动开始时刻前 640 ms。

## 输入

* 4 帧数据，帧宽 480 ms，帧移 160 ms，采样频率 200 Hz。

* 各帧：3 频率通道 \* 32 通道 \* 采样点。

## 模型

![Net4](D:\硕士学习\毕业论文\CrossSubject\model\四分类\Net4.JPG)

* 预训练
  * 各帧分别送入 CNN 层，将各帧 CNN 层的输出级联送入全连接层。
  * 各帧使用相同的 CNN 层参数。
* 微调
  * 只微调全连接层。

## 结果

* 预训练

  `\model_all\四分类\CNN_Net4_pretraining`

* 微调

  `\model\四分类\CNN_Net4_fine`



# Net42

## 数据源

起始于运动开始时刻前 640 ms。

## 输入

![Net42](D:\硕士学习\毕业论文\CrossSubject\model\四分类\Net42.JPG)

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

  `\model_all\四分类\CNN_Net42_pretraining`

- 微调

  `\model\四分类\CNN_Net42_fine`



# Net45

## 数据源

起始于运动开始时刻前 640 ms，结束于开始时刻后 640 ms。

## 输入

人工提取的特征。

## 模型

![Net45](D:\硕士学习\毕业论文\CrossSubject\model\四分类\Net45.JPG)

- 预训练
  - 输入数据直接送入 NN 网络。
- 微调
  - 只微调最后两层（第五、第六层）。

## 结果

- 预训练

  `\model_all\四分类\NN_Net45_pretraining`

- 微调

  `\model\四分类\NN_Net45_fine`



# Net46

## 数据源

起始于运动开始时刻前 640 ms。

## 输入

- 7 帧数据，帧宽 320 ms，帧移 160 ms。
- 各帧：人工提取的特征。

## 模型

![Net46](D:\硕士学习\毕业论文\CrossSubject\model\四分类\Net46.JPG)

- 预训练
  - 每帧作为一个时间步。
  - 各时间步依次送入 LSTM 层，将最后一时间步 LSTM 层的输出送入全连接层。
- 微调
  - 只微调全连接层。

## 结果

- 预训练

  `\model_all\四分类\LSTM_Net46_pretraining`

- 微调

  `\model\四分类\LSTM_Net46_fine`



# Net47

## 数据源

起始于运动开始时刻前 640 ms。

## 输入

- 7 帧数据，帧宽 320 ms，帧移 160 ms。

- 各帧：人工提取的特征。

## 模型

![Net47](D:\硕士学习\毕业论文\CrossSubject\model\四分类\Net47.JPG)

- 预训练
  - 每帧作为一个时间步。
  - 各时间步依次送入 LSTM 层，将各时间步 LSTM 层的输出级联送入全连接层。
- 微调
  - 只微调全连接层。

## 结果

- 预训练

  `\model_all\四分类\LSTM_Net47_pretraining`

- 微调

  `\model\四分类\LSTM_Net47_fine`



# Net48

## 数据源

起始于运动开始时刻前 640 ms。

## 输入

- 7 帧数据，帧宽 160 ms，帧移 160 ms，采样频率 200 Hz。
- 各帧：3 频率通道 \* 32 通道 \* 采样点。

## 模型

![Net48](D:\硕士学习\毕业论文\CrossSubject\model\四分类\Net48.JPG)

- 预训练
  - 各帧分别送入 CNN 层，将各帧 CNN 层的输出送入 LSTM 层。
  - 各帧使用相同的 CNN 层参数。
  - 每帧 CNN 层的输出作为一个时间步。
  - 各时间步依次送入 LSTM 层，将最后一时间步 LSTM 层的输出送入全连接层。
- 微调
  - 只微调全连接层。

## 结果

- 预训练

  `\model_all\四分类\CNN_LSTM_Net48_pretraining`

- 微调

  `\model\四分类\CNN_LSTM_Net48_fine`



# Net49

## 数据源

起始于运动开始时刻前 640 ms。

## 输入

- 4 帧数据，帧宽 480 ms，帧移 160 ms，采样频率 200 Hz。
- 各帧：3 频率通道 \* 32 通道 \* 采样点。

## 模型

![Net49](D:\硕士学习\毕业论文\CrossSubject\model\四分类\Net49.JPG)

- 预训练
  - 各帧分别送入 CNN 层，将各帧 CNN 层的输出送入 LSTM 层。
  - 各帧使用相同的 CNN 层参数。
  - 每帧 CNN 层的输出作为一个时间步。
  - 各时间步依次送入 LSTM 层，将各时间步 LSTM 层的输出级联送入全连接层。
- 微调
  - 只微调全连接层。

## 结果

- 预训练

  `\model_all\四分类\CNN_LSTM_Net49_pretraining`

- 微调

  `\model\四分类\CNN_LSTM_Net49_fine`


