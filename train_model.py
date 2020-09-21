from Autoencoder import *
from my_tool import *
import tensorflow as tf


if __name__ == '__main__':
    # 设置使用指定GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    checkpoint_save_path = "./checkpoint/AE.ckpt"

    autoencoder = get_ae_model()

    autoencoder.compile(optimizer='Adam', loss='mse')

    # 若有已经训练的权重,则读取权重,继续训练
    # loadWeights(autoencoder, checkpoint_save_path)

    # 使用ModelCheckpoint回调函数,在训练过程中,将最好的模型保存到filepath
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    # 加载训练用的图片
    dataSet_num = 1000 #训练图片的数目
    images = np.load('./data/image_data_'+str(dataSet_num)+'.npy') # 输入图片  x
    VGG_f = np.load('./data/VGG_feature_'+str(dataSet_num)+'.npy')# VGG输出向量  y

    # 打散数据
    # x_train, y_train = shuffle(images, VGG_f, random_state=0)
    # print(images.shape,VGG_f.shape)

    history = autoencoder.fit(images, VGG_f, batch_size=64, epochs=3,
                              validation_split=0.20,
                              shuffle=True,
                              callbacks=[cp_callback])

    # 打印模型的概述信息，通过模型的概述信息知道模型的基本结构
    autoencoder.summary()

    # 显示训练集和验证集的acc和loss曲线
    print(history)
    loss_plot(history)

    # 保存编码器的部分,用于后期编码
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
    encoder.save('./model/my_encoder_model.h5')




