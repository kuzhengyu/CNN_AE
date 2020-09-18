import os
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras import Model

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            # Flatten(),
            # Dense(150528, activation='relu'),
            # Dense(4096, activation='relu'),
            # Dense(1024, activation='relu'),
            # 卷积的样式还没确定好，用全连接太大了
            # Conv2D(64, (5, 5), activation='relu', padding='same'),
            # Conv2D(128, (4, 4), activation='relu', padding='same'),
            # Conv2D(4, (3, 3), activation='relu', padding='same')
        ])
        self.decoder = tf.keras.Sequential([
            Dense(1024, activation='relu'),
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


