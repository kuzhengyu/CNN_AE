from Autoencoder import *
import os

autoencoder = get_ae_model()
checkpoint_save_path = "./checkpoint/015_.h5"

autoencoder.load_weights(checkpoint_save_path)

# 保存编码器的部分,用于后期编码
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
encoder.save('./model/d_encoder_model.h5')