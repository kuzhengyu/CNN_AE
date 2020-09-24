from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from time import *
import os

# 获取用于训练的原图片
def get_data():
    data_train_path ='./raw_data/Place/data_256/'
    dataSet_num =5000

    for initial in os.listdir(data_train_path) :
        if(initial=='c'):# 先读取到i的部分
            break
        for place in os.listdir(data_train_path+'/'+initial):
            images = []
            for x in range(dataSet_num):
                if (not os.path.exists(data_train_path+'/'+initial+'/'+place+'/'+str(x+1).zfill(8)+'.jpg')):
                    break
                # 加载图片
                image = load_img(data_train_path+'/'+initial+'/'+place+'/'+str(x+1).zfill(8)+'.jpg'
                                 , target_size=(224, 224))

                image_data = img_to_array(image)
                # 其转化为VGG16能够接受的输入，实际上为每个像素减去均值
                image_data = preprocess_input(image_data)

                # 保存原图片数据
                images.append(image_data)

                print(place+'_'+str(x)+'is ok')

            if(len(images)!=0):
                # 将数据转换成array形式，保存起来
                images = np.array(images)
                np.save(data_train_path+'/'+initial+'/'+place+'/'+str(dataSet_num)+'_data.npy',images)
                print(place+'is ok')


if __name__ == '__main__':
    begin_time = time()
    get_data()
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, 's')