from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import numpy as np
from time import *
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def get_train_data():
    dataSet_num = 10000 # 用于预测图片的数量
    data_path ='C:/Users/11354/Desktop/Alderley dataset/FRAMESA/'
    images = [] # 保存原来图片
    features = [] # 保存VGG预测得到的特征

    if (os.path.exists( './data/VGG_feature_'+str(dataSet_num)+'.npy') and
        os.path.exists( './data/image_data_'+str(dataSet_num)+'.npy')):
        print('data exist!')
        return 0

    # 获取VGG 倒数第二个全连接层fc2的输出
    base_model = VGG16(include_top=True, weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    for x in range(dataSet_num):
        # 加载图片
        image = load_img(data_path+'Image'+str(x+1).zfill(5)+'.jpg', target_size=(224, 224))
        image_data = img_to_array(image)

        # 归一化
        image_data = image_data/255.0

        # 保存原图片数据
        images.append(image_data)

        # 添加维度,调节成适用于VGG的输入形式
        image_data = np.expand_dims(image_data, axis=0)

        # 图片通过预测,得到特征向量
        feature = model.predict(image_data)
        # print(feature,image_data)

        # 将feature添加到列表中
        features.append(feature[0])

    # 将数据转换成array形式，保存起来
    features = np.array(features)
    images = np.array(images)
    np.save('./data/VGG_feature_'+str(dataSet_num)+'.npy',features)
    np.save('./data/image_data_'+str(dataSet_num)+'.npy',images)


# fit KNN
# 找到与查询图片最接近的前三张图片
def get_top3_match(query_code,database_codes):

    n_neigh = 3 # top3
    nbrs = NearestNeighbors(n_neighbors=n_neigh).fit(database_codes)
    distances, indices = nbrs.kneighbors(np.array(query_code))

    # indices 下标+1 即图片编号，即第n帧图片
    return distances,indices


# 相差5帧之内均为匹配成功
def is_matched(query_frame,predict_frame):
    if(abs(query_frame-predict_frame)<5):
        return True
    else:
        return False

# 计算Precision-recall
# def cacl_ROC():


# 显示训练集和验证集的loss曲线
def loss_plot(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# 加载模型的权重
def loadWeights(autoencoder,checkpoint_save_path):
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        autoencoder.load_weights(checkpoint_save_path).expect_partial()


if __name__ == '__main__':
    begin_time = time()
    get_train_data()
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time,'s')