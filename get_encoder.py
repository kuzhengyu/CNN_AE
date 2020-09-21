from Autoencoder import *
import os

autoencoder = get_ae_model()
checkpoint_save_path = "./checkpoint/AE.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    autoencoder.load_weights(checkpoint_save_path).expect_partial()

# 保存编码器的部分,用于后期编码
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
encoder.save('./model/my_encoder_model.h5')