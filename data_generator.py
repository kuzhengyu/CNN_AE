from skimage.io import imread
from skimage.transform import resize
import numpy as np
import tensorflow.keras
import math

class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, datas, batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
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

        # 生成数据
        X, y = self.data_generation(batch_datas)

        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        images = []
        labels = []

        # 生成数据
        for i, data in enumerate(batch_datas):
            # x_train数据
            image = cv2.imread(data)
            image = list(image)
            images.append(image)
            # y_train数据
            right = data.rfind("\\", 0)
            left = data.rfind("\\", 0, right) + 1
            class_name = data[left:right]
            if class_name == "dog":
                labels.append([0, 1])
            else:
                labels.append([1, 0])
        # 如果为多输出模型，Y的格式要变一下，外层list格式包裹numpy格式是list[numpy_out1,numpy_out2,numpy_out3]
        return np.array(images), np.array(labels)

# generator 的使用
# my_training_batch_generator = MY_Generator(training_filenames,
#                                            GT_training,
#                                            batch_size)
# my_validation_batch_generator = MY_Generator(validation_filenames,
#                                              GT_validation,
#                                              batch_size)