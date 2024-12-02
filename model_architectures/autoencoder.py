from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, LayerNormalization, ZeroPadding2D, \
    BatchNormalization, Flatten, Dense, Reshape, DepthwiseConv2D, Add, Dropout, MultiHeadAttention, Rescaling, Activation
from keras.models import Model
import numpy as np
import tensorflow as tf
from keras.applications import imagenet_utils
import keras

import sys
sys.path.insert(0, '..')
from losses.custom_losses import focal_loss_for_regression, root_mean_squared_error

class SingleAutoencoder():

    def __init__(self, input_dimension: int, summarize_model: bool, patch_size: int = 4, expansion_factor: int = 2, pre_trained_path: str = None):
        self._input_dimension = input_dimension
        self._patch_size = patch_size
        self._expansion_factor = expansion_factor

        if pre_trained_path is not None:
            self.model = keras.models.load_model(pre_trained_path, custom_objects={"root_mean_squared_error": root_mean_squared_error, 'focal_loss': focal_loss_for_regression(gamma=1.1, alpha=0.5)})
        else:
            self.model = self._create_model2(summarize_model=summarize_model)

    ## layer to compose the mobileViT
    def conv_block(self, x, filters=16, kernel_size=3, strides=2):
        conv_layer = Conv2D(filters, kernel_size, strides=strides, activation=tf.nn.swish, padding='same')
        return conv_layer(x)

    def inverted_residual_block(self, x, expanded_channels, output_channels, strides=1):
        m = Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
        m = BatchNormalization()(m)
        m = Activation('swish')(m)

        if strides == 2:
            m = ZeroPadding2D(padding=correct_pad(m, 3))(m)
        m = DepthwiseConv2D(3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False)(m)
        m = BatchNormalization()(m)
        m = Activation('swish')(m)

        m = Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
        m = BatchNormalization()(m)

        if tf.math.equal(x.shape[-1], output_channels) and strides == 1:
            return Add()([m, x])

        return m

    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = Dense(units, activation=tf.nn.swish)(x)
            x = Dropout(dropout_rate)(x)
        return x

    def transformer_block(self, x, transformer_layers, projection_dim, num_heads=2):
        for _ in range(transformer_layers):
            x1 = LayerNormalization(epsilon=1e-6)(x)

            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
            x2 = Add()([attention_output, x])
            x3 = LayerNormalization(epsilon=1e-6)(x2)
            x3 = self.mlp(x3, hidden_units=[x.shape[-1]*2, x.shape[-1]], dropout_rate=0.1,)
            x = Add()([x3, x2])

        return x

    def mobilevit_block(self, x, num_blocks, projection_dim, strides=1):
        local_features = self.conv_block(x, filters=projection_dim, strides=strides)
        local_features = self.conv_block(local_features, filters=projection_dim, kernel_size=1, strides=strides)

        num_patches = int((local_features.shape[1] * local_features.shape[2]) / self._patch_size)
        non_overlapping_patches = Reshape((self._patch_size, num_patches, projection_dim))(local_features)

        global_features = self.transformer_block(non_overlapping_patches, num_blocks, projection_dim)

        folded_feature_map = Reshape((*local_features.shape[1:-1], projection_dim))(global_features)

        folded_feature_map = self.conv_block(folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides)

        local_global_features = Concatenate(axis=-1)([x, folded_feature_map])

        local_global_features = self.conv_block(local_global_features, filters=projection_dim, strides=strides)

        return local_global_features


    def _create_model(self, summarize_model: bool = False):
        model_input = Input((self._input_dimension, self._input_dimension, 43))
        x = Rescaling(scale=1.0/255)(model_input)

        ## initial conv-stem -> MV2 block
        x = self.conv_block(x, filters=24)
        x = self.inverted_residual_block(x, expanded_channels=16*self._expansion_factor, output_channels=24)

        ## Downsampling with MV2 block

        x = self.inverted_residual_block(x, expanded_channels=24 * self._expansion_factor, output_channels=24, strides=2)
        x = self.inverted_residual_block(x, expanded_channels=24 * self._expansion_factor, output_channels=24)
        x = self.inverted_residual_block(x, expanded_channels=24 * self._expansion_factor, output_channels=24)

        ## First MV2 -> MobileViT block
        x = self.inverted_residual_block(x, expanded_channels=24 * self._expansion_factor, output_channels=48, strides=2)
        x = self.mobilevit_block(x, num_blocks=2, projection_dim=64)

        ## Second MV2 -> MobileViT block
        x = self.inverted_residual_block(x, expanded_channels=64 * self._expansion_factor, output_channels=64, strides=2)
        x = self.mobilevit_block(x, num_blocks=4, projection_dim=80)

        ## Second MV2 -> MobileViT block
        x = self.inverted_residual_block(x, expanded_channels=80 * self._expansion_factor, output_channels=80, strides=2)
        x = self.mobilevit_block(x, num_blocks=3, projection_dim=96)

        x = self.conv_block(x, filters=128, kernel_size=1, strides=1)

        decoding = Reshape((8, 8, 128))(x)
        decoding = Conv2D(128, (3, 3), activation='relu', padding='same', name='intermediate_layer')(decoding)
        decoding = Conv2D(128, (3, 3), activation='relu', padding='same', name='intermediate_layer_2')(decoding)
        decoding = UpSampling2D(size=(2, 2), name='up1')(decoding)

        decoding = Conv2D(128, (3, 3), activation='relu', padding='same', name='intermediate_layer_3')(decoding)
        decoding = Conv2D(128, (3, 3), activation='relu', padding='same', name='intermediate_layer_4')(decoding)
        decoding = UpSampling2D(size=(2, 2), name='up2')(decoding)

        decoding = Conv2D(64, (3, 3), activation='relu', padding='same', name='intermediate_layer_5')(decoding)
        decoding = Conv2D(64, (3, 3), activation='relu', padding='same', name='intermediate_layer_6')(decoding)
        decoding = UpSampling2D(size=(2, 2), name='up3')(decoding)

        decoding = Conv2D(32, (3, 3), activation='relu', padding='same', name='intermediate_layer_7')(decoding)
        decoding = Conv2D(32, (3, 3), activation='relu', padding='same', name='intermediate_layer_8')(decoding)
        #decoding = UpSampling2D(size=(2, 2), name='up4')(decoding)

        decoding = Conv2D(16, (3, 3), activation='relu', padding='same', name='intermediate_layer_9')(decoding)
        decoding = Conv2D(16, (3, 3), activation='relu', padding='same', name='intermediate_layer_10')(decoding)
        decoding = UpSampling2D(size=(4, 4), name='up5')(decoding)

        output = Conv2D(48, (3, 3), activation='relu', padding='same')(decoding)

        autoencoder = Model([model_input], [output])
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        if summarize_model:
            autoencoder.summary()

        return autoencoder

    def fit_model(self, input_data: np.ndarray, validation_data: np.ndarray, number_of_epochs: int, dataset: str,
                  removed_attack:str):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'trained_models/{dataset}/best_autoencoder_single_autoencoder_minus_{removed_attack}.hdf5',
                                                        monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        self.model.fit_generator(epochs=number_of_epochs,
                                 shuffle=True,
                                 generator=input_data,
                                 validation_data=validation_data,
                                 use_multiprocessing=True,
                                 workers=12,
                                 callbacks=[checkpoint])

    def _create_model2(self, summarize_model: bool = False):
        # defining the model inputs
        model_input = Input(shape=(self._input_dimension, self._input_dimension, 1), name='input_bsif')
        model_input2 = Input(shape=(self._input_dimension, self._input_dimension, 3), name='input_reflectance')
        model_input3 = Input(shape=(self._input_dimension, self._input_dimension, 1), name='input_elbp')

        ## first input
        encoding1 = Conv2D(8, (3, 3), activation='relu', padding='same')(model_input)
        encoding1 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoding1)
        encoding1 = LayerNormalization()(encoding1)

        ## second input
        encoding2 = Conv2D(8, (3, 3), activation='relu', padding='same')(model_input2)
        encoding2 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoding2)
        encoding2 = LayerNormalization()(encoding2)

        ## third input
        encoding3 = Conv2D(8, (3, 3), activation='relu', padding='same')(model_input3)
        encoding3 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoding3)
        encoding3 = LayerNormalization()(encoding3)

        encoding = Concatenate()([encoding1, encoding2, encoding3])
        encoding = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool_full')(encoding)

        encoding = Conv2D(32, (3, 3), activation='relu', padding='same')(encoding)
        encoding = Conv2D(32, (3, 3), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same', name='pool1')(encoding)

        # encoding input into a higher level representation - 1st group
        encoding = Conv2D(128, (3, 3), activation='relu', padding='same')(encoding)
        encoding = Conv2D(128, (3, 3), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same', name='pool4')(encoding)

        encoding = ZeroPadding2D(padding=(3, 3))(encoding)
        encoding = Conv2D(256, (7, 7), activation='relu', padding='valid')(encoding)

        encoding = Flatten()(encoding)
        encoding = Dense(256, activation='sigmoid', name='intermediate_layer')(encoding)
        encoding = Reshape((1, 1, 256))(encoding)

        encoding = UpSampling2D(size=(7, 7), name='up_intermediate')(encoding)

        # decoding step -> getting back to original representation
        decoding = Conv2D(128, (3, 3), activation='relu', padding='same')(encoding)
        decoding = UpSampling2D(size=(4, 4), name='up1')(decoding)

        decoding = Conv2D(64, (3, 3), activation='relu', padding='same')(decoding)
        decoding = Conv2D(64, (3, 3), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D(size=(4, 4), name='up3')(decoding)

        decoding = Conv2D(32, (3, 3), activation='relu', padding='same')(decoding)
        decoding = Conv2D(8, (3, 3), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D(size=(2, 2), name='up4')(decoding)

        output = Conv2D(5, (3, 3), activation='relu', padding='same')(decoding)
        ## BSIF - 3, Reflectance - 1, LBP - 1

        autoencoder = Model([model_input, model_input2, model_input3], [output])
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        if summarize_model:
            autoencoder.summary()

        return autoencoder


def correct_pad(inputs, kernel_size):
        """Returns the correct padding for a layer, based on input shape and kernel size."""
        input_size = inputs.shape[1:3]  # Only consider spatial dimensions (height, width)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        adjust = [1 if dim is None else (dim % stride) for dim, stride in zip(input_size, kernel_size)]
        correct = [(k // 2, k // 2) if a == 0 else (k // 2, k // 2 + 1) for k, a in zip(kernel_size, adjust)]
        return correct
