import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# 显示数据库中查询到最佳匹配的图片
def show(time,index):
    im = plt.imread('C:/Users/11354/Desktop/Alderley dataset/FRAMES'+time+'/'+'Image'+str(index+1).zfill(5)+'.jpg')
    plt.imshow(im)
    plt.show()

# 相差5帧之内均为匹配成功
def is_matched(query_frame,predict_frame):
    if(abs(query_frame-predict_frame)<5):
        return True
    else:
        return False


encoder = tf.keras.models.load_model('./model/my_encoder_model.h5',compile=False)
dataSet_num = 1000
query_index = 700

images = np.load('./data/VGG_'+str(dataSet_num)+'.npy')

#查询的图片
query_img = images[query_index]

# 添加维度用于预测
query_img = np.expand_dims(query_img, axis=0)

# 提取查询的图片(黑夜)的特征，并且拉直
query_code = encoder.predict(query_img)
query_code = query_code.reshape(-1,784)

# 加载用于查询的数据库（白天的特征向量）
database_codes = np.load('./feature/image_data_B_'+str(dataSet_num)+'.npy')
database_codes = database_codes.reshape(-1,784)


# fit KNN
# 找到与查询图片最接近的前三张图片
def get_top3_match(query_code,database_codes):
    n_neigh = 3 # top3
    nbrs = NearestNeighbors(n_neighbors=n_neigh).fit(database_codes)
    distances, indices = nbrs.kneighbors(np.array(query_code))
    # indices 下标+1 即图片编号，即第n帧图片
    return distances,indices

distances,indices = get_top3_match(query_code,database_codes)

print(distances,indices)

# 显示查询的图片
show('A',query_index)
# 显示匹配到的图片
for x in range(3):
    show('B',indices[0][x])
