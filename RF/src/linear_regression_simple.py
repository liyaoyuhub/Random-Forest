import numpy as np
import torch
from torch.utils import data
# nn是神经网络的缩写
from torch import nn


# ##############################################数据###################################################
# 随机数据生成器
def synthetic_data(w, b, num_example):
	X = torch.normal(0, 1, (num_example, len(w)))
	y = torch.matmul(X, w) + b
	y += torch.normal(0, 0.01, y.shape)
	return X, y.reshape(-1, 1)


# 生成数据集
def data_set():
	true_w = torch.tensor([2., -3.4, 4.6])  # 必须为float型，不然会报错
	true_b = 4
	features, labels = synthetic_data(true_w, true_b, 1000)
	return features, labels


# 读取数据集
def load_array(data_arrays, batch_size, is_train = True):
	"""
	将features和labels作为API的参数传递
	:param data_arrays: (features, labels)的list
	:param batch_size:批量大小
	:param is_train:是否打乱训练
	:return:批量大小的数据集
	"""
	dataSet = data.TensorDataset(*data_arrays)  # 将list传入函数后得到数据集
	return data.DataLoader(dataSet, batch_size, shuffle = is_train)  # 从dataSet中抽batch_size个数据并打乱


# 测试-读取数据集
def test_load():
	batch_size = 10
	features, labels = data_set()
	data_iter = load_array((features, labels), batch_size)
	return data_iter


# test_load()
# print(next(iter(data_iter)))  #转成python的iterator，通过next函数得到一个(X, y)，再打印到console

# #############################################模型###################################################
# 定义模型

# Sequential(): list of layers
net = nn.Sequential(nn.Linear(3, 1))  # 指定：输入的维度为3，输出的维度为1
# 初始化模型
net[0].weight.data.normal_(0, 0.01)  # 访问第一层的权重的值，并使用正态分布替换data的值
net[0].bias.data.fill_(0)  # 访问第一层的偏置的值，并设置为0


# 定义损失函数
def set_loss():
	loss = nn.MSELoss()
	return loss


# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr = 0.01)


# #############################################训练###################################################
def training():
	num_epochs = 10
	data_iter = test_load()
	loss = set_loss()
	features, labels = data_set()

	for epoch in range(num_epochs):
		for X, y in data_iter:
			L = loss(net(X), y)
			trainer.zero_grad()  # trainer优化器调用梯度清0函数
			L.backward()  # pytorch已经sum()过了
			trainer.step()  # step()函数：对模型进行更新
		L = loss(net(features), labels)
		print(f'epoch{epoch + 1},loss{L:f}')
	return net[0].weight.data, net[0].bias.data


# training()


# #############################################预测###################################################
def test_func():
	w, b = training()
	print(f'{w}\n{b}')


test_func()
