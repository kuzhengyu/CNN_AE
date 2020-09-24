from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# 读取出白天的数据，经过encoder编码，得到一组特征向量，作为数据库
# 将数据库的内容保存起来，用于图片匹配
if __name__ == '__main__':
    data_path ='C:/Users/11354/Desktop/Alderley dataset/FRAMESB/'
    dataSet_num = 10
    images =[]
    # 加载encoder
    encoder = tf.keras.models.load_model('./model/d_encoder_model.h5',compile=False)

    # 读取白天图片，进行预处理
    for x in range(dataSet_num):
        # 加载图片
        image = load_img(data_path + 'Image' + str(x + 1).zfill(5) + '.jpg', target_size=(224, 224))
        image_data = img_to_array(image)

        # 归一化
        image_data = image_data / 255.0

        # 保存原图片数据
        images.append(image_data)

    database = np.array(images)

    #数据库中所有图片进行特征提取，得到一组特征向量
    database_codes = []
    for index in range(dataSet_num):
        database_single =  np.expand_dims(database[index], axis=0)
        database_single_feature = encoder.predict(database_single)
        database_codes.append(database_single_feature)

    database_codes = np.array(database_codes)
    # 保存特征向量
    np.save('./feature/image_data_B_'+str(dataSet_num)+'.npy',database_codes)
