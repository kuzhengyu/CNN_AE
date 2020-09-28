import numpy as np
from keras.layers import Conv2D, Flatten, Dense,MaxPooling2D,Input,Dropout
from keras.models import Model
from keras import optimizers
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "3"


images = np.load('./raw_data/Place/data_256' + '/' + 'm' + '/' + 'mountain' + '/' + str(5000) + '_data.npy')
VGG_f = np.load('./raw_data/Place/data_256' + '/' + 'm' + '/' + 'mountain' + '/' + str(5000) + '_VGG_feature.npy')

x_train = images[0:4000]
y_train = VGG_f[0:4000]
x_vaild = images[4000:]
y_vaild = VGG_f[4000:]


input_img = Input(shape=(224,224,3))
x = Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(4,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(x)
encoded = MaxPooling2D((2,2), padding='same', name='encoder')(x)

x = Flatten()(encoded)
x = Dense(3136, activation='relu')(x)
decoded = Dense(4096, activation='relu')(x)

autoencoder = Model(input_img, decoded)

autoencoder.load_weights('./model/a_model.h5')

Adam = optimizers.Adam(lr=0.0001)

autoencoder.compile(optimizer=Adam, loss='mse')

autoencoder.fit(x_train, y_train, batch_size=64, epochs=2,
                                          validation_data=(x_vaild, y_vaild))

autoencoder.save('./model/a_model.h5')
autoencoder.summary()