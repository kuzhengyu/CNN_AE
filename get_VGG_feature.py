from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import numpy as np
from time import *
import os
from sklearn.decomposition import PCA

def get_VGG_f():
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
    data_train_path ='./raw_data/Place/data_256/'
    dataSet_num =5000
    # 获取VGG 倒数第二个全连接层fc2的输出
    base_model = VGG16(include_top=True, weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    for initial in os.listdir(data_train_path) :
        if(initial=='c'): #先处理目录'i'
            break
        for place in os.listdir(data_train_path+'/'+initial):
            features = []  # 保存VGG预测得到的特征
            for x in range(dataSet_num):
                # 确保有图片
                if (not os.path.exists(data_train_path+'/'+initial+'/'+place+'/'+str(5000).zfill(8)+'.jpg')):
                    break
                # 加载图片
                image = load_img(data_train_path+'/'+initial+'/'+place+'/'+str(x+1).zfill(8)+'.jpg'
                                 , target_size=(224, 224))

                image_data = img_to_array(image)
                # 其转化为VGG16能够接受的输入，实际上为每个像素减去均值
                image_data = preprocess_input(image_data)

                # 添加维度,调节成适用于VGG的输入形式
                image_data = np.expand_dims(image_data, axis=0)

                # 图片通过预测,得到特征向量
                feature = model.predict(image_data)

                # 将feature添加到列表中
                features.append(feature[0])
                print(place+'_'+str(x)+'is ok')

            if(len(features)!=0):
                # 将数据转换成array形式，保存起来
                features = np.array(features)

                #PCA降维 25088太大了
                # features = features.reshape(-1, 25088)
                # pca = PCA(n_components=4096)
                # features = pca.fit_transform(features)  # 5000*4096

                # 保存
                np.save(data_train_path+'/'+initial+'/'+place+'/'+str(dataSet_num)+'_block5_pool_VGG_feature.npy',features)
                print(place+'is ok')


if __name__ == '__main__':
    begin_time = time()
    get_VGG_f()
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, 's')