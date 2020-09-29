from Autoencoder import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from data_generator import DataGenerator
from tensorflow.python.keras import backend as K
import gc

def acc_loss_plot(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def model_config():
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
    checkpoint_save_path = "./checkpoint/{epoch:03d}_.h5"
    autoencoder = get_ae_model()

    Adam = optimizers.Adam(lr=0.0001)
    autoencoder.compile(optimizer=Adam, loss='mse')

    # 若有已经训练的权重,则读取权重,继续训练
    if os.path.exists('./checkpoint/002_.h5' ):
        autoencoder.load_weights('./checkpoint/002_.h5' )

    # 使用ModelCheckpoint回调函数,在训练过程中,将最好的模型保存到filepath
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    return autoencoder,cp_callback, early_stopping

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
    # ================================================================================================
    data_train_path = './raw_data/Place/data_256/'
    dataSet_num = 5000
    history = []
    count = 0  # 第几个场景
    test = 0
    for initial in os.listdir(data_train_path):  # [1:]已经训练m 从d开始
        # test 查看在m中的训练效果
        # if (initial != 'm'):
        #     break
        # 先训练['m', 'd', 'g', 'u', 'n', 'o', 's', 'a', 'b', 'k', 'i']的部分
        if (initial == 'c'):
            break
        for place in os.listdir(data_train_path + '/' + initial):
            print(initial + '_' + place + 'is start!')
            # 避免读到目录,确保有图片的数据在当前目录下
            if (os.path.exists(data_train_path + '/' + initial + '/' + place + '/' + str(5000).zfill(8) + '.jpg')):
                count = count + 1
                images = np.load(
                    data_train_path + '/' + initial + '/' + place + '/' + str(dataSet_num) + '_data.npy')  # 输入图片  x
                VGG_f = np.load(data_train_path + '/' + initial + '/' + place + '/' + str(
                    dataSet_num) + '_VGG_feature.npy')  # VGG输出向量  y

                # 后1000个作为验证集
                x_train = images[0:4000]
                y_train = VGG_f[0:4000]
                x_vaild = images[4000:]
                y_vaild = VGG_f[4000:]

                training_generator = DataGenerator(x_train, y_train)

                autoencoder, cp_callback, early_stopping = model_config()

                # autoencoder.fit(x_train, y_train, batch_size=64, epochs=3,
                #                           validation_data=(x_vaild, y_vaild),
                #                           shuffle=True,
                #                           callbacks=[cp_callback,early_stopping])
                autoencoder.fit_generator(training_generator, epochs=2, max_queue_size=20, workers=1,
                                          steps_per_epoch=32,
                                          validation_data=(x_vaild,y_vaild),
                                          callbacks=[cp_callback,early_stopping])

                print('No.' + str(count) + '_' + initial + '_' + place + 'is ok!')
                K.clear_session()
                del autoencoder,images,VGG_f,x_train,y_train,x_vaild,y_vaild,training_generator
                gc.collect()
                # acc_loss_plot(history)
                # break #test 只跑一个场景


    # 打印模型的概述信息，通过模型的概述信息知道模型的基本结构
    autoencoder.summary()
