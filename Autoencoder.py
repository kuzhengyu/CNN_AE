import os
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense,MaxPooling2D
from tensorflow.python.keras import Model

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            Conv2D(64, (5, 5), activation='relu', padding='same',input_shape=(224,224,3),strides=2),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(128, (4, 4), activation='relu', padding='same',strides=2),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(4, (3, 3), activation='relu', padding='same')
            # 得到14*14*4=784的特征向量
        ])
        self.decoder = tf.keras.Sequential([
            Flatten(),
            Dense(784, activation='relu'),
            Dense(2048, activation='relu'),
            Dense(4096, activation='relu'),
            # 还原成4096
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 加载模型的权重
def loadWeights(autoencoder,checkpoint_save_path):
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        autoencoder.load_weights(checkpoint_save_path)

# 取出编码器的部分,用于后期编码
def get_Encoder(autoencoder):
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
    encoder.save('./data/my_encoder_model.h5')


