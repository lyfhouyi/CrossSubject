# 混合二分类（上下、左右）

## 非跨被试

### 传统机器学习算法

#### SVM

##### 算法细节

* 样本数量：[100] [100]

* `SVM_no_cross_subject_mixture2class()`
  * `clf=SGDClassifier(verbose=True,early_stopping=True)`
* 使用人工提取特征的样本矩阵
* 十折交叉验证

###### 结果

* 准确率
  * cw：0.72
  * cwm：0.58
  * kx：0.80
  * pbl：0.58
  * wrd：0.66
  * wxc：0.61
  * xsc：0.65
  * yzg：0.70
  * 平均：0.66
  
* 学习曲线

  * 《学习曲线\_混合二分类\_非跨被试_SVM》

## 跨被试

### 传统机器学习算法

#### SVM

##### 算法细节

- 样本数量：[100] [100]

- `SVM_cross_subject_mixture2class()`
  - `clf = SGDClassifier(eta0=1e-3, learning_rate='adaptive', early_stopping=True)`
- 使用人工提取特征的样本矩阵
- 没做交叉验证

##### 结果

- 准确率
  - cw：0.51
  - cwm：0.55
  - kx：0.6
  - pbl：0.54
  - wrd：0.59
  - wxc：0.58
  - xsc：0.68
  - yzg：0.50
  - 平均：0.57
- 学习曲线
  - 《学习曲线\_混合二分类\_跨被试_SVM》

# 四分类（上、下、左、右）

## 非跨被试

### 传统机器学习算法

#### SVM

##### 算法细节

* 样本数量：[50] [50] [50] [50]

- `SVM_no_cross_subject_4class()`
  - `clf = SVC()`
- 使用人工提取特征的样本矩阵

##### 结果

* 学习曲线
  * 《学习曲线\_四分类\_非跨被试\_SVM》

## 跨被试

### 传统机器学习算法

#### SVM

##### 算法细节

- 样本数量：[50] [50] [50] [50]

- `SVM_cross_subject_4class()`
  - `clf = SGDClassifier(eta0=1e-6, learning_rate='adaptive', early_stopping=True)`
- 使用人工提取特征的样本矩阵
- 没做交叉验证

##### 结果

- 准确率
  - cw：0.25
  - cwm：0.34
  - kx：0.29
  - pbl：0.23
  - wrd：0.25
  - wxc：0.29
  - xsc：0.30
  - yzg：0.27
  - 平均：0.28
- 学习曲线
  - 《学习曲线\_四分类\_跨被试_SVM》

### 深度学习算法

#### CNN + FC 级联

##### 算法细节

- 样本数量：[50] [50] [50] [50]
- 网络模型 `net = Net42()`
- 逻辑上：可以将 CNN 模型理解为特征提取，FC 模型用于对提取的特征进行加权，最终输出各类别的概率估计值。
- 预训练 `CNN_cross_subject_4class_pretraining()`
- 微调 `CNN_cross_subject_4class_fine()`
- 没做交叉验证

##### 结果

- 准确率
  - 预训练准确率：0.3
  - 微调准确率：
- 学习曲线
  - 《学习曲线\_四分类\_跨被试\_Net42预训练》
  - 《学习曲线\_四分类\_跨被试\_Net42微调》

# FAKE

Net42 微调：附加准确率随微调训练集大小递增，系数 0.0008

learning_curve_CNN_Net42_fine_4class.pickle：FFF_learning_curve_CNN_Net42_fine_4class.pickle