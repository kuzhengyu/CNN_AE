from Autoencoder import *
import tensorflow as tf
import os
import numpy as np
from time import *
from matplotlib import pyplot as plt
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.keras import optimizers

def acc_loss_plot(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# 多GPU 检查点
class ParallelModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=True,#保存权重
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor,verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

def model_config():
    checkpoint_save_path = "./checkpoint/{epoch:03d}_.h5"
    autoencoder = get_ae_model()
    # 若有已经训练的权重,则读取权重,继续训练
    # 可行！
    if os.path.exists('./checkpoint/{002_.h5' ):
        autoencoder.load_weights('./checkpoint/002_.h5')

    paralle_model = multi_gpu_model(autoencoder, gpus=2)

    Adam = optimizers.Adam(lr=0.0001)
    paralle_model.compile(optimizer=Adam, loss='mse')

    # 使用ModelCheckpoint回调函数,在训练过程中,将最好的模型保存到filepath
    cp_callback = ParallelModelCheckpoint(autoencoder, filepath=checkpoint_save_path)

    return autoencoder,paralle_model,cp_callback

def train_model():
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
    data_train_path = './raw_data/Place/data_256/'


# ============================================train======================================
    dataSet_num = 5000
    history=[]
    count =0# 第几个场景
    test =0
    for initial in os.listdir(data_train_path):# [1:]已经训练m 从d开始
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
                count=count+1
                images = np.load(data_train_path+'/'+initial+'/'+place+'/'+str(dataSet_num) + '_data.npy')  # 输入图片  x
                VGG_f = np.load(data_train_path+'/'+initial+'/'+place+'/'+str(dataSet_num) + '_VGG_feature.npy')  # VGG输出向量  y

                # VGG_f = VGG_f.reshape(-1,25088)
                # 后1000个作为验证集
                x_train = images[0:4000]
                y_train = VGG_f[0:4000]
                x_vaild = images[4000:]
                y_vaild = VGG_f[4000:]

                autoencoder ,paralle_model ,cp_callback = model_config()

                paralle_model.fit(x_train, y_train, batch_size=64, epochs=2,
                                          validation_data=(x_vaild, y_vaild),
                                          shuffle=True,
                                          callbacks=[cp_callback])
                print( 'No.'+str(count)+'_'+initial +'_'+ place +'is ok!')
                # acc_loss_plot(history)
                # break #test 只跑一个场景


# =========================================plot===============================================
    # 打印模型的概述信息，通过模型的概述信息知道模型的基本结构
    autoencoder.summary()
    # 显示训练集和验证集的acc和loss曲线



if __name__ == '__main__':
    begin_time = time()
    train_model()
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, 's')