# 使用CRNN进行字符识别



## 文件介绍

这是一个CRNN进行OCR识别的学习代码，模型基于CNN+RNN+CTCLoss结构

文件包含一个data的数据集文件，以及四个py程序主体文件。

data文件夹：包含train_data、valid_data及其对应label，以及生成data的函数

dataset.py：用于整理数据集，使得模型输入Size为torch.Size([3, 32, 200])，输出torch.Size([M])，M为字符长度

model.py：包含模型的主体文件，torch.Size([batch, 3, 32, 200])->torch.Size([51, batch, 28])

train.py：模型训练函数，训练过程中分别保存train和valid的loss变化

predict.py：模型的预测函数，使用验证集的数据进行正向推理，直观展示模型结果



## 学习心得

针对无法对齐的场景，可以使用CTCLoss进行loss计算，CTCLoss涉及到空白blank和beta变换。

CTCLoss可以理解为使得模型输出beta变换为目标结果的所有情况的概率和最大，详见参考链接。



## 参考链接

https://www.bilibili.com/video/BV1fX4y1D7VR

https://blog.csdn.net/ooooocj/article/details/117366227