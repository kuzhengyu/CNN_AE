from Autoencoder import *
from my_tool import *
import os

if __name__ == '__main__':
    checkpoint_save_path = "./checkpoint/AE.ckpt"

    autoencoder = Autoencoder()
    autoencoder.compile(optimizer='Adam', loss='mse')

    # 若有已经训练的权重,则读取权重,继续训练
    loadWeights(autoencoder, checkpoint_save_path)

    # 使用ModelCheckpoint回调函数,在训练过程中,将最好的模型保存到filepath
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    # 模型训练
    VGG_f = np.load('./data/VGG_feature_100.npy')# VGG输出向量
    images = np.load('./data/image_data_100.npy') # 输入图片
    # print(images.shape,VGG_f.shape)

    history = autoencoder.fit(images, VGG_f, batch_size=8, epochs=5,
                        callbacks=[cp_callback])


    # 打印模型的概述信息，通过模型的概述信息知道模型的基本结构
    autoencoder.summary()






