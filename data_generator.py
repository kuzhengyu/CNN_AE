import os
import numpy as np
import tensorflow.keras
import math

class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, datas,labels, batch_size=64, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.label =labels
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas[k] for k in batch_indexs]
        batch_labels = [self.label[k] for k in batch_indexs]
        return np.array(batch_datas), np.array(batch_labels)

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



