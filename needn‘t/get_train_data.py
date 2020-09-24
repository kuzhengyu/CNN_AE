from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import numpy as np
from time import *
import os

def get_train_data():
    dataSet_num = 12607 # 用于预测图片的数量
    # data_A_path ='C:/Users/11354/Desktop/Alderley dataset/FRAMESA/'
    # data_B_path = 'C:/Users/11354/Desktop/Alderley dataset/FRAMESB/'
    # data_A_path ='./raw_data/FRAMESA/'
    # data_B_path = './raw_data/FRAMESB/'

    if (os.path.exists( './data/VGG_feature_'+str(dataSet_num)+'.npy') and
        os.path.exists( './data/image_data_'+str(dataSet_num)+'.npy')):
        print('data exist!')
        return 0

    # 获取VGG 倒数第二个全连接层fc2的输出
    base_model = VGG16(include_top=True, weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    for time in [data_B_path ]:
    # for time in [data_A_path, data_B_path]:
        images = []  # 保存原来图片
        features = []  # 保存VGG预测得到的特征
        for x in range(dataSet_num):
            # 加载图片
            image = load_img(time+'Image'+str(x+1).zfill(5)+'.jpg', target_size=(224, 224))

            image_data = img_to_array(image)

            # 归一化
            image_data = image_data/255.0

            # 保存原图片数据
            images.append(image_data)

            # 添加维度,调节成适用于VGG的输入形式
            image_data = np.expand_dims(image_data, axis=0)

            # 图片通过预测,得到特征向量
            feature = model.predict(image_data)

            # 将feature添加到列表中
            features.append(feature[0])

        # 将数据转换成array形式，保存起来
        features = np.array(features)
        images = np.array(images)
        np.save('./data/VGG_feature_'+time[-2]+'_'+str(dataSet_num)+'.npy',features)
        np.save('./data/image_data_'+time[-2]+'_'+str(dataSet_num)+'.npy',images)

if __name__ == '__main__':
    begin_time = time()
    get_train_data()
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, 's')