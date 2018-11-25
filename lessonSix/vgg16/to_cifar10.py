# -*- coding:utf-8 -*-
import pickle,pprint
from PIL import Image
import numpy as np
import os

class DictSave(object):
    def __init__(self,filenames,file):
        self.filenames = filenames
        self.file=file
        self.arr = []
        self.all_arr = []
        self.label=[]

   
    def image_input(self,filenames,file):
           i=0
           for filename in filenames:
              self.arr,self.label = self.read_file(filename,file)
              if self.all_arr==[]:
                 self.all_arr = self.arr
              else:
                self.all_arr = np.concatenate((self.all_arr,self.arr))
              
              print(i)
              i=i+1
    def read_file(self,filename,file):
     
            im = Image.open(filename)#打开一个图像
             # 将图像的RGB分离
            r, g, b = im.split()
            # 将PILLOW图像转成数组
            r_arr = plimg.pil_to_array(r)
            g_arr = plimg.pil_to_array(g)
            b_arr = plimg.pil_to_array(b)

        # 将60*180二维数组转成1024的一维数组
        #r_arr1 = r_arr.reshape(10800)
        #g_arr1 = g_arr.reshape(10800)
        #b_arr1 = b_arr.reshape(10800)
        # 3个一维数组合并成一个一维数组,大小为32400
            arr = np.concatenate((r_arr, g_arr, b_arr))
        #print(file)
        #label=file[0]
            label=[]
            for i in file:
            #print(i[0])
               label.append(i[0])
        #print (label)
            return arr,label
    def pickle_save(self,arr,label):
           print ("正在存储")
        # 构造字典,所有的图像数据都在arr数组里,这里只存图像数据,没有存label
           contact = {'data': arr,'label':label}
           f = open('data_batch', 'wb')

           pickle.dump(contact, f)#把字典存到文本中去
           f.close()
           print ("存储完毕")
if __name__ == "__main__":
   file_dir='train_data'
   L=[]
   F=[]
   for root,dirs,files in os.walk(file_dir):
     for file in files:  
        if os.path.splitext(file)[1] == '.jpg':  
          L.append(os.path.join(root, file))  
          F.append(file)

   ds = DictSave(L,F)
   ds.image_input(ds.filenames,ds.file)
   print(ds.all_arr)
   ds.pickle_save(ds.all_arr,ds.label)
   print ("最终数组的大小:"+str(ds.all_arr.shape))