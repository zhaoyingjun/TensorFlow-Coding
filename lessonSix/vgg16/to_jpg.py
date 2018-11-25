#coding=utf-8
import cv2
import numpy as np
import os
import pickle

 
#文件夹名
str_2 = 'train_data'
str_1 = 'test_data'
 
#判断文件夹是否存在，不存在的话创建文件夹
if os.path.exists(str_1) == False:
   os.mkdir(str_1)
if os.path.exists(str_2) == False:
   os.mkdir(str_2)
 
# 解压缩，返回解压后的字典，f,encoding='bytes'
def unpickle(file):

   fo = open(file, 'rb')
   dict = pickle.load(fo, encoding='bytes')
   fo.close()
   return dict
 
def cifar_jpg(dir_file):
# 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
   for j in range(1, 6):
      dataName = dir_file + '/' + "data_batch_" + str(j) # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径，本文和脚本文件在同一目录下。
      Xtr = unpickle(dataName)
      print(Xtr)
      print(dataName + " is loading...")
 
      for i in range(0, 10000):
        img = np.reshape(Xtr[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0) # 读取image
        picName = 'train_data/' + str(Xtr[b'labels'][i])  + str(i + (j - 1) * 10000) + '.jpg' # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
        cv2.imwrite(picName, img)
        print(dataName + " loaded.")
 
   print("test_batch is loading...")
 
# 生成测试集图片
   testName = dir_file + '/' + 'data_batch_6'
   testXtr = unpickle(testName)
   for i in range(0, 10000):
    img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'test_data/' + str(testXtr[b'labels'][i])  + str(i) + '.jpg'
    cv2.imwrite(picName, img)
    print("test_batch loaded.")
  
 
#标签与名字的对应关系
def label_name():
	label_name_dict ={
       'airplane': "0",
       'automobile': "1",
       'bird': "2",
       'cat': "3",
       'deer': "4",
       'dog': "5",
       'frog': "6",
       'horse': "7",
       'ship': "8",
       'truck': "9"
    }
	return label_name_dict
 
if __name__ == '__main__':
   dir_file = 'train_data'
   #cifar_jpg(dir_file)
try:
   cifar_jpg(dir_file)
except:
   print('出错了')
