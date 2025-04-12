# 2025-CV-Project1
<h1 align="center"> 2025-CV-Project1</h1>
<div align="center"> Repository for Fudan Course Computer Vision Project 1</div>

<div align="center"> Author: 金潇睿</div>

## 任务描述
手工搭建三层神经网络分类器，在数据集 CIFAR-10 上进行训练以实现图像分类。

## 基本要求
（1） 本次作业要求自主实现反向传播，**不允许使用 pytorch，tensorflow** 等现成的支持自动微分的深度学习框架，**可以使用 numpy**；  
（2） 最终提交的代码中应至少包含模型、训练、测试和参数查找四个部分，鼓励进行模块化设计；  
（3） 其中模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；训练部分应实现 SGD 优化器、学习率下降、交叉熵损失和 L2 正则化，并能根据验证集指标自动保存最优的模型权重；参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能；测试部分需支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）。

## 提交要求
（1） **仅提交 pdf 格式的实验报告**，报告中除对模型、数据集和实验结果的基本介绍外，还应可视化训练过程中在训练集和验证集上的 loss 曲线和验证集上的 accuracy 曲线；  
（2） 报告中需包含对训练好的模型网络参数的可视化，并观察其中的模式；  
（3） **代码**提交到自己的 **public github repo**，repo 的 readme 中应**清晰指明如何进行训练和测试**，**训练好的模型权重**上传到百度云 /google drive 等网盘，实验报告内应包含实验代码所在的 github repo 链接及模型权重的下载地址。

## 资源需求
1. Python $\geq 3.8$
2. Numpy
3. Tqdm
4. Matplotlib

## 说明

* **数据集下载**

	请从 [CIFAR-10 官网](https://www.cs.toronto.edu/~kriz/cifar.html) 下载 `cifar-10-python.tar.gz`文件，解压后将文件夹放至项目目录，保持文件名为 `cifar-10-batches-py/`。

* **neural_network.py**

	基于 NumPy 实现的多层神经网络系统，用于 CIFAR-10 图像分类任务，无依赖任何深度学习框架。

* **train.py**

	模型训练和测试部分，运行方式：
	
	```bash
	python train.py --epochs 20 --lr 0.01 --hidden_size 256
	```
	
	可输入参数说明：
	* `--epochs`：训练轮次（默认值：100）
	* `--reg`：正则化强度（默认值0.001）
	* `--lr`：学习率（默认值：0.01）
	* `--hidden_size`：隐藏层大小（默认值：512）
	* `--batch_size`：批量大小（默认值：64）
	
	训练完成后，会保存最佳模型为`best_model.txt`，并产生训练过程中在训练集和验证集上的 loss 曲线和验证集上的 accuracy 曲线 `training_plot.png`。

* **parameter_search.py**

	超参数搜索部分，对不同学习率 / L2 正则化强度 / 隐藏层大小进行网格搜索。

* **hyperparameter_search_results.csv**
	
	保存了超参数搜索部分中各组合在测试集上的准确率（Accuracy）。

* **plot.py**

	利用Matplotlib可视化训练过程中的 loss 曲线和 accuracy 曲线，并可保存图像。

* **best_model.txt**
	
	训练过的模型，在测试集上的准确率达到$53.87\%$，可在<a href = https://pan.baidu.com/s/1IyLk1vmPk_Zi54C17ov3PA>百度网盘</a>下载，提取码：**wwin**

* **eval_model.py**
	
	用于导入训练好的模型，输出在测试集上的分类准确率（Accuracy）。
	例如将模型权重保存到项目文件夹并命名为`best_model.txt`，输入如下命令：
	`python eval_model.py --model_path best_model.txt`
