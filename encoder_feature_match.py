import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import  preprocess_input

def cosim(x,y):
    # a=np.sum(x*y.T)
    # b=np.sqrt(np.sum(x*x.T))*np.sqrt(np.sum(y*y.T))
    d1 = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return  d1


# 显示图片 A黑夜 B白天
def show(time,index):
    im = plt.imread('./raw_data/FRAMES'+time+'/'+'Image'+str(index+1).zfill(5)+'.jpg')
    print('./raw_data/FRAMES'+time+'/'+'Image'+str(index+1).zfill(5)+'.jpg')
    plt.imshow(im)
    plt.show()


# fit KNN
# 找到与查询图片最接近的前三张图片
def get_top3_match(query_code,database_codes):
    # n_neigh = 3 # top3
    # nbrs = NearestNeighbors(n_neighbors=n_neigh).fit(database_codes)
    # distances, indices = nbrs.kneighbors(np.array(query_code))
    distances =[]
    for dc in database_codes:
        dist = cosim(query_code,dc)
        distances.append(dist[0])
    print("distances:",distances)
    print("sorted_distances:",sorted(distances,reverse=True))
    print("sorted_distances_index:", np.array(distances).argsort()[::-1])

    # indices 下标+1 即图片编号，即第n帧图片
    # return distances,indices
    return sorted(distances,reverse=True)[:3],np.array(distances).argsort()[::-1][:3]

def one_feature_match(encoder_path,day_feature_path,night_path,query_index,data_range):

    encoder = tf.keras.models.load_model(encoder_path,compile=False)

    #读取查询图片，并处理成输入encoder的形式
    query_img = load_img(night_path + 'Image' + str(query_index+1).zfill(5) + '.jpg', target_size=(224, 224))
    query_img = img_to_array(query_img)
    # query_img = preprocess_input(query_img)
    query_img = np.expand_dims(query_img, axis=0)

    # 提取查询的图片A(黑夜)的特征，并且拉直
    query_code = encoder.predict(query_img)
    query_code = query_code.reshape(-1,3136)

    # 加载用于查询的数据库（白天的特征向量）
    database_codes = np.load(day_feature_path)
    database_codes = database_codes.reshape(-1,3136)


    # 找到与查询图片最接近的前三张图片
    distances,indices = get_top3_match(query_code,database_codes)
    print('【距离，下标】',distances,indices+data_range[0])


    # 显示查询的图片
    show('A',query_index)

    # 显示匹配到的前三个图片
    for x in range(3):
        show('B',indices[x]+data_range[0])


def get_sim_matrix(seqA_feature_path,seqB_feature_path,seqA_data_range,seqB_data_range,sim_path):
    seqA_codes = np.load(seqA_feature_path)
    seqA_codes = seqA_codes.reshape(-1, 3136)

    seqB_codes = np.load(seqB_feature_path)
    seqB_codes = seqB_codes.reshape(-1, 3136)

    # savepath = xxxx
    f1 = open(sim_path, 'wb')

    for seqA_index in range(seqA_data_range[1]-seqA_data_range[0]):
        distances =[]
        for seqB_index in range(seqB_data_range[1]-seqB_data_range[0]):
            dist = cosim(seqA_codes[seqA_index], seqB_codes[seqB_index])
            distances.append(dist)

        # 保存seqA一个图片对应seqB所有图片的余弦值
        np.save(f1,np.array(distances))

    f1.close()
