from tensorflow.python.keras.layers import Conv2D, Flatten, Dense,MaxPooling2D,Input,Dropout
from tensorflow.python.keras import Model

# 2.0
def get_ae_model():
    input_img = Input(shape=(224, 224, 3))
    x = Conv2D(64, (5, 5), activation='relu', padding='same',strides=2)(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (4, 4), activation='relu', padding='same',strides=2)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(8, (3, 3), activation='relu', padding='same', name='encoder')(x)

    x = Flatten()(encoded)
    x = Dense(1568, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    decoded = Dense(4096, activation='relu')(x)

    return Model(input_img, decoded)



# 1.1
# class Autoencoder(Model):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = tf.keras.Sequential([
#             Conv2D(64, (5, 5), activation='relu', padding='same',strides=2),
#             MaxPooling2D((2, 2), padding='same'),
#             Conv2D(128, (4, 4), activation='relu', padding='same',strides=2),
#             MaxPooling2D((2, 2), padding='same'),
#             Conv2D(4, (3, 3), activation='relu', padding='same')
#             # 得到14*14*4=784的特征向量
#         ],name='enco_layer')
#         self.decoder = tf.keras.Sequential([
#             Flatten(),
#             Dense(784, activation='relu'),
#             Dense(2048, activation='relu'),
#             Dense(4096, activation='relu'),
#             # 还原成4096
#         ])
#
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded


# 1.0
# class Autoencoder(Model):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.c1 = Conv2D(64, (5, 5), activation='relu', padding='same', input_shape=(224,224,3),strides=2)
#         self.p1 = MaxPooling2D((2, 2), padding='same')
#         self.c2 = Conv2D(128, (4, 4), activation='relu', padding='same', strides=2)
#         self.p2 = MaxPooling2D((2, 2), padding='same')
#         self.c3 = Conv2D(4, (3, 3), activation='relu', padding='same', name='enco_layer')
#         self.flatten = Flatten()
#         self.fc1 = Dense(784, activation='relu')
#         self.fc2 =  Dense(2048, activation='relu')
#         self.fc3 =  Dense(4096, activation='relu')
#
#     def call(self, x):
#         x = self.c1(x)
#         x = self.p1(x)
#         x = self.c2(x)
#         x = self.p2(x)
#         x = self.c3(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x
#



