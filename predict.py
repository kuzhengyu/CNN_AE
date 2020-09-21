import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

encoder = tf.keras.models.load_model('./model/my_encoder_model.h5',compile=False)
images = np.load('./data/image_data_1000.npy')
label = load_img('C:/Users/11354/Desktop/Alderley dataset/FRAMESB/Image00398.jpg', target_size=(224, 224))

image = images[389]
image =  np.expand_dims(image, axis=0)
print(image.shape)

label = img_to_array(label)
label = label / 255.0
label =  np.expand_dims(label, axis=0)
print(label.shape)


result = encoder.predict(image).flatten()
result2 = encoder.predict(label).flatten()
print(result.shape)
print(result2.shape)


# 计算欧氏距离
feature_1 = result
feature_2 = result2
def embedding_distance(feature_1, feature_2):
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist
dis = embedding_distance(feature_1, feature_2)

print('feature_1,feature_2的欧式距离为',dis)

def cosim(x,y):
    a=np.sum(x*y.T)
    b=np.sqrt(np.sum(x*x.T))*np.sqrt(np.sum(y*y.T))
    return  (a/b)
print('feature_1,feature_2的余弦相似度为',cosim(feature_1,feature_2))

