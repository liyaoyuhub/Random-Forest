"""
需要修改：
1、文件路径：15行
2、写入txt的类别：87,102行
3、输出彩色图像：214,241行
"""

from osgeo import gdal
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import model_selection
import pickle

# ###########################################路径#############################################
Dir_Path = r"E:\random_f\data_test1\\"  # 绝对路径
Image_Path = Dir_Path + "data_test.tif"  # 原始图像
Label_Path = Dir_Path + "data_label.tif"  # 标签图像
Sample_Path = Dir_Path + "data_sample.txt"  # 样本数据
RFModel_Path = Dir_Path + "model.pickle"  # 训练模型
Predicate_Path = Dir_Path + "Data8.tif"  # 预测图像
Result_Path = Dir_Path + "result.tif"  # 结果图像


# ##########################################数据集############################################
def read_tif(fileName):
	"""
	读取tif数据。
	:param fileName: 文件路径
	:return: 读取到的数据
	"""
	DataSet = gdal.Open(fileName)
	if DataSet is None:
		print(fileName + "文件无法打开")
	return DataSet


def read_data():
	"""
	获取原始图像和标签图像的信息，并将tif数据格式转化为Array数据格式。
	:return: 长、宽、波段数、仿射矩阵、原始图像列表、标签图像列表
	"""
	# 读原始图像相关信息
	DataSet = read_tif(Image_Path)
	Tif_width = DataSet.RasterXSize  # 栅格矩阵的列数
	Tif_height = DataSet.RasterYSize  # 栅格矩阵的行数
	Tif_bands = DataSet.RasterCount  # 波段数
	Tif_geotrans = DataSet.GetGeoTransform()  # 获取仿射矩阵信息
	Image_data = DataSet.ReadAsArray(0, 0, Tif_width, Tif_height)  # 数据格式转化
	# 读标签图像相关信息
	DataSet = read_tif(Label_Path)
	Label_data = DataSet.ReadAsArray(0, 0, Tif_width, Tif_height)
	return Tif_width, Tif_height, Tif_bands, Tif_geotrans, Image_data, Label_data


def write_data_set():
	"""
	遍历标签图像中的所有像素，并将（band1, band2... bandMAX, 像素所属类别）写入data_sample.txt。
	:return:无
	"""
	# 写之前，先检验文件是否存在，存在就删掉。如果文件不存在，就会自动创建
	file_write_obj = None
	if os.path.exists(Sample_Path):
		os.remove(Sample_Path)
	else:
		file_write_obj = open(Sample_Path, 'w')
	# 获取原始图像数据和标签图像信息
	Tif_width, Tif_height, Tif_bands, Tif_geotrans, Image_data, Label_data = read_data()
	# 遍历所有像素，收集每种类别的样本数据
	count = 0
	for i in range(Label_data.shape[0]):
		for j in range(Label_data.shape[1]):
			# 写入500个未知类别像素的波段信息和类别名称
			if Label_data[i][j] == 0 and count < 500:
				var = ""
				for k in range(Tif_bands):
					var = var + str(Image_data[k][i][j]) + ","
				var = var + "unclassified"
				file_write_obj.writelines(var)
				file_write_obj.write('\n')
				count += 1
			# 写入其他类别像素的波段信息和类别名称
			elif Label_data[i][j] != 0:
				var = ""
				for k in range(Tif_bands):
					var = var + str(Image_data[k][i][j]) + ","
				# 判断所属类别并追加类别名称
				if Label_data[i][j] == 1:
					var = var + "grass"
				if Label_data[i][j] == 2:
					var = var + "water"
				if Label_data[i][j] == 3:
					var = var + "road"
				# 写入txt文件
				file_write_obj.writelines(var)
				file_write_obj.write('\n')
	# 关闭txt文件
	file_write_obj.close()


# ###########################################训练#############################################
def iris_label(s):
	"""
	定义字典，解析样本数据集txt。
	:param s:类别对应的编号
	:return:类别的名称
	"""
	it = {b'unclassified': 0, b'grass': 1, b'water': 2, b'road': 3}
	return it[s]


def make_model_file():
	"""
	读取原始影像，获得波段数。
	读取写好的data_sample.txt数据集，将标签转换为数字，选取合适的波段数量和决策树数量来训练随机森林模型。
	计算按9:1的方式划分训练集精度和测试集，并测试两个数据集各自的精度。
	保存训练后的模型文件model.pickle。
	:return:无
	"""
	# 标签所在的列的序号 == 波段个数
	DataSet = read_tif(Image_Path)
	band_num = DataSet.RasterCount  # 波段数
	#  1.读取数据集
	#  “band_num”指的是第band_num+1列：将第band_num+1列的标签（str）转化为label(number)
	sample_data = np.loadtxt(Sample_Path, dtype = float, delimiter = ',', converters = {band_num: iris_label})
	#  2.划分数据与标签
	x, y = np.split(sample_data, indices_or_sections = (band_num,), axis = 1)  # x为数据，y为标签
	x = x[:, 0:band_num]  # 选取前band_num个波段作为特征
	# 获取训练数据、测试数据、训练标签、测试标签
	train_data, test_data, train_label, test_label = \
		model_selection.train_test_split(x, y, random_state = 1, train_size = 0.9, test_size = 0.1)
	#  3.用100个树来创建随机森林模型，训练随机森林
	classifier = RandomForestClassifier(n_estimators = 100, bootstrap = True, max_features = 'sqrt')
	classifier.fit(train_data, train_label.ravel())  # ravel函数拉伸到一维
	#  4.计算随机森林的准确率
	print("训练集：", classifier.score(train_data, train_label))
	print("测试集：", classifier.score(test_data, test_label))
	#  5.保存模型
	# 以二进制的方式打开文件
	file = open(RFModel_Path, "wb")
	# 将模型写入文件
	pickle.dump(classifier, file)
	# 最后关闭文件
	file.close()


# ###########################################预测#############################################
def write_tiff(im_data, im_geotrans, im_proj, path):
	"""
	保存tif文件（预测结果图）。
	:param im_data: 预测后的数据，每个像素对应一个类别（数字形式）
	:param im_geotrans: 仿射矩阵信息
	:param im_proj: 投影信息
	:param path: 预测图像目录
	:return: 无
	"""
	# 选择与输入相同的数据类型，然后输出
	if 'int8' in im_data.dtype.name:
		datatype = gdal.GDT_Byte
	elif 'int16' in im_data.dtype.name:
		datatype = gdal.GDT_UInt16
	else:
		datatype = gdal.GDT_Float32
	# 初始化，然后判断输入数据是否为3维张量，然后获取预测后数据的波段数、行数、列数
	im_bands, im_height, im_width = 0, 0, 0
	if len(im_data.shape) == 3:
		im_bands, im_height, im_width = im_data.shape
	elif len(im_data.shape) == 2:
		im_data = np.array([im_data])  # 让2维矩阵升维，im_bands为1
		im_bands, im_height, im_width = im_data.shape
	# 创建tif图像写入引擎driver，依据预测图像的目录、列数、行数、波段数、数据类型，创建写入工具dataset
	driver = gdal.GetDriverByName("GTiff")
	dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
	if dataset is not None:
		dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
		dataset.SetProjection(im_proj)  # 写入投影
	# 根据波段数写入文件，在输出图像的第i+1个波段，赋予输入数据集的第i个波段的全部数据
	for i in range(im_bands):
		dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
	# 删除写入工具
	del dataset


def predict_classify(output_dimension):
	predict_data = read_tif(Predicate_Path)
	Tif_width = predict_data.RasterXSize  # 栅格矩阵的列数
	Tif_height = predict_data.RasterYSize  # 栅格矩阵的行数
	Tif_geotrans = predict_data.GetGeoTransform()  # 仿射矩阵信息
	Tif_proj = predict_data.GetProjection()  # 获取投影信息
	Image_data = predict_data.ReadAsArray(0, 0, Tif_width, Tif_height)
	# 调用保存好的模型，以读二进制的方式打开文件
	file = open(RFModel_Path, "rb")
	rf_model = pickle.load(file)
	file.close()
	# 预测前要调整数据的格式（band，height*width）
	data = np.zeros((Image_data.shape[0], Image_data.shape[1] * Image_data.shape[2]))
	for i in range(Image_data.shape[0]):
		data[i] = Image_data[i].flatten()
	# 转置后，每个像素对应n个波段，然后线性排列
	data = data.swapaxes(0, 1)
	# 使用模型对调整好格式的数据进行预测
	pred = rf_model.predict(data)
	# 对预测好的数据还原为输入图像矩阵的格式
	pred = pred.reshape(Image_data.shape[1], Image_data.shape[2])
	pred = pred.astype(np.uint8)
	# 根据参数选择输出tif的格式
	if output_dimension == "2d":
		# 调用函数，将结果写到tif图像里
		write_tiff(pred, Tif_geotrans, Tif_proj, Result_Path)
	elif output_dimension == "3d":
		# 让输出的预测矩阵升到三维，将输出变为彩色图像
		threeD_data = np.zeros((3, Image_data.shape[1], Image_data.shape[2]))
		for i in range(Image_data.shape[1]):
			for j in range(Image_data.shape[2]):
				if pred[i][j] == 0:  # 0类对应的RGB值
					threeD_data[0][i][j] = 0  # R
					threeD_data[1][i][j] = 0  # G
					threeD_data[2][i][j] = 0  # B
				elif pred[i][j] == 1:
					threeD_data[0][i][j] = 255
					threeD_data[1][i][j] = 0
					threeD_data[2][i][j] = 0
				elif pred[i][j] == 2:
					threeD_data[0][i][j] = 0
					threeD_data[1][i][j] = 255
					threeD_data[2][i][j] = 0
				elif pred[i][j] == 3:
					threeD_data[0][i][j] = 0
					threeD_data[1][i][j] = 0
					threeD_data[2][i][j] = 255
		# 调用函数，将结果写到tif图像里
		write_tiff(threeD_data, Tif_geotrans, Tif_proj, Result_Path)


# ###########################################启动项#############################################
def boot_options():
	print("制作数据：1\n训练模型：2\n执行预测：3\n全部执行：4")
	c_num = input("请输入编号：")
	if c_num == 1:
		'''制作数据集'''
		write_data_set()
	elif c_num == 2:
		'''训练模型文件'''
		make_model_file()
	elif c_num == 3:
		'''预测，参数代表：输出3维or2维图像（"3d" or "2d"）'''
		predict_classify("2d")
	elif c_num == 4:
		'''执行完整流程'''
		write_data_set()
		make_model_file()
		predict_classify("2d")
	else:
		print("非法输入！")


if __name__ == "__main__":
	boot_options()
