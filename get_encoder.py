from Autoencoder import *
import os


def get_encoder(AE_path,model_type):

    autoencoder =get_ae_model()

    checkpoint_save_path = AE_path

    autoencoder.load_weights(checkpoint_save_path)

    # 保存编码器的部分,用于后期编码
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
    # encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=8).output)
    encoder.save('./model/'+model_type+'_encoder.h5')