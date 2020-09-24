from Autoencoder import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# 加载模型的权重
def loadWeights(autoencoder,checkpoint_save_path):
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        autoencoder.load_weights(checkpoint_save_path)


if __name__ == '__main__':
    # 设置使用指定GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    checkpoint_save_path = "./checkpoint/AE.ckpt"

    autoencoder = get_ae_model()

    autoencoder.compile(optimizer='Adam', loss='mse',metrics=['acc'])

    # 若有已经训练的权重,则读取权重,继续训练
    # autoencoder.load_weights(checkpoint_save_path)

    # 使用ModelCheckpoint回调函数,在训练过程中,将最好的模型保存到filepath
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    # 加载训练用的图片
    dataSet_num = 5000 #训练图片的数目
    images = np.load('./data/image_data_'+str(dataSet_num)+'.npy') # 输入图片  x
    VGG_f = np.load('./data/VGG_feature_'+str(dataSet_num)+'.npy')# VGG输出向量  y

    # 后2000个作为验证集
    x_train = images[0:4000]
    y_train = VGG_f[0:4000]
    x_vaild = images[4000:]
    y_vaild = VGG_f[4000:]


    history = autoencoder.fit(x_train, y_train, batch_size=8, epochs=20,
                              validation_data=(x_vaild,y_vaild),
                              shuffle=True,
                              callbacks=[cp_callback])

    # 打印模型的概述信息，通过模型的概述信息知道模型的基本结构
    autoencoder.summary()

    # 显示训练集和验证集的acc和loss曲线
    print(history)

    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()