import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,Flatten,Dense
from keras.models import Model

input_img = Input(shape=(224,224,3))
x = Conv2D(64,(5,5), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(128,(4,4), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
encoded = Conv2D(4,(3,3), activation='relu', padding='same',name='encoder')(x)

x = Flatten()(encoded)
x = Dense(784, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
decoded = Dense(4096, activation='relu')(x)

autoencoder = Model(input_img, decoded)