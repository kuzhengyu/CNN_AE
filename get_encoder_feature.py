from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import os
from keras.applications.vgg16 import  preprocess_input

# 读取出白天B的数据，经过encoder编码，得到一组特征向量，作为数据库
# 将数据库的内容保存起来，用于图片匹配
def get_one_encoder_feature(data_path,encoder_path,model_type,data_range,day_feature_path):
    images =[]
    # 加载encoder
    encoder = tf.keras.models.load_model(encoder_path,compile=False)

    # 读取B白天图片，进行预处理
    for x in range(data_range[0],data_range[1]):
        # 加载图片
        image_data = load_img(data_path + 'Image' + str(x + 1).zfill(5) + '.jpg', target_size=(224, 224))
        image_data = img_to_array(image_data)
        # image_data = preprocess_input(image_data)
        # 保存原图片数据
        images.append(image_data)

    database = np.array(images)

    #数据库中所有图片进行特征提取，得到一组特征向量
    database_codes = []
    for index in range(data_range[1]-data_range[0]):
        database_single =  np.expand_dims(database[index], axis=0)
        database_single_feature = encoder.predict(database_single)
        database_codes.append(database_single_feature)

    database_codes = np.array(database_codes)
    # 保存特征向量
    np.save(day_feature_path ,database_codes)

def get_encoder_feature(seqA_path,seqB_path,encoder_path,model_type,seqA_data_range,seqB_data_range,seqA_feature_path,seqB_feature_path):
    get_one_encoder_feature(seqA_path, encoder_path, model_type, seqA_data_range,seqA_feature_path)
    get_one_encoder_feature(seqB_path, encoder_path, model_type, seqB_data_range, seqB_feature_path)

