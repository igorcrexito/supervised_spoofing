from keras.layers import Input, Conv2D, Conv3D, MaxPooling3D, UpSampling2D, Concatenate, LayerNormalization, ZeroPadding2D, \
    BatchNormalization, Flatten, Dense, Reshape, DepthwiseConv2D, Add, Dropout, MultiHeadAttention, Rescaling, Activation, Conv3DTranspose, UpSampling3D
from keras.models import Model
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
import sys
sys.path.insert(0, '..')

class AutoencoderModel:

    def __init__(self, summarize_model: bool, pre_trained_path: str = None):
        self.summarize_model = summarize_model
        if pre_trained_path is not None:
            self.model = keras.models.load_model(pre_trained_path)
        else:
            self.model = self._create_model()


    def _create_model(self):
        # defining model input shape
        input_layer = Input(shape=(4, 224, 224, 3), name='model_input')

        # Encoder: Conv3D block
        x = Conv3D(filters=4, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer)
        x = Conv3D(filters=4, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=4, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = Conv3D(filters=4, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=4, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=4, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)

        x = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)

        # Bottleneck (latent space)
        encoded = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same', name='endoded')(x)

        # Decoder: Conv3DTranspose block to reconstruct the input
        x = Conv3DTranspose(filters=8, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same', name='i_1')(
            encoded)
        x = Conv3DTranspose(filters=8, kernel_size=(3, 3, 3), activation='relu', padding='same', name='i_2')(x)
        x = Conv3DTranspose(filters=8, kernel_size=(3, 3, 3), activation='relu', padding='same', name='i_3')(x)
        x = UpSampling3D(size=(1, 2, 2))(x)

        x = Conv3DTranspose(filters=8, kernel_size=(3, 3, 3), activation='relu', padding='same', name='i_4')(x)
        x = Conv3DTranspose(filters=8, kernel_size=(3, 3, 3), activation='relu', padding='same', name='i_5')(x)
        x = Conv3DTranspose(filters=8, kernel_size=(3, 3, 3), activation='relu', padding='same', name='i_6')(x)
        x = UpSampling3D(size=(1, 2, 2))(x)

        x = Conv3DTranspose(filters=4, kernel_size=(3, 3, 3), activation='relu', padding='same', name='i_7')(x)
        x = Conv3DTranspose(filters=4, kernel_size=(3, 3, 3), activation='relu', padding='same', name='i_8')(x)
        x = Conv3DTranspose(filters=4, kernel_size=(3, 3, 3), activation='relu', padding='same', name='i_9')(x)
        x = UpSampling3D(size=(2, 2, 2))(x)

        x = Conv3DTranspose(filters=4, kernel_size=(3, 3, 3), activation='relu', padding='same', name='i_10')(x)
        x = Conv3DTranspose(filters=4, kernel_size=(3, 3, 3), activation='relu', padding='same', name='i_11')(x)
        x = Conv3DTranspose(filters=4, kernel_size=(3, 3, 3), activation='relu', padding='same', name='i_12')(x)

        # Output layer to match the input shape
        decoded = Conv3DTranspose(filters=7, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', name='decoded')(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=decoded)

        # Compile the model (using mean squared error for reconstruction tasks)
        model.compile(optimizer=Adam(), loss='mae')

        # Summarize the model if needed
        if self.summarize_model:
            model.summary()

        return model


    def fit_model(self, input_data_head: np.ndarray, input_data_forehead: np.ndarray, input_data_left_cheek: np.ndarray,
                  input_data_right_cheek: np.ndarray, input_data_mouth: np.ndarray, labels: np.ndarray, number_of_epochs: int):
        """
        Fit the model with multiple inputs.

        Args:
            input_data_head (np.ndarray): Input data for the head region.
            input_data_forehead (np.ndarray): Input data for the forehead region.
            input_data_left_cheek (np.ndarray): Input data for the left cheek region.
            input_data_right_cheek (np.ndarray): Input data for the right cheek region.
            input_data_mouth (np.ndarray): Input data for the mouth region.
            labels (np.ndarray): Corresponding labels for the input data.
            number_of_epochs (int): Number of epochs to train.
        """
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='trained_models/single_model.keras', monitor='loss',
            verbose=1, save_best_only=True)

        inputs = {
            'input_face': input_data_head,
            'input_forehead': input_data_forehead,
            'input_left_cheek': input_data_left_cheek,
            'input_right_cheek': input_data_right_cheek,
            'input_mouth': input_data_mouth
        }

        self.model.fit(
            inputs,
            labels,
            epochs=number_of_epochs,
            batch_size=1,
            shuffle=True,
            class_weight={0: 1, 1:2},
            callbacks=[checkpoint]
        )


    def predict_data(self, input_data_head: np.ndarray, input_data_forehead: np.ndarray, input_data_left_cheek: np.ndarray,
                  input_data_right_cheek: np.ndarray, input_data_mouth: np.ndarray, labels: np.ndarray):
        """
        Fit the model with multiple inputs.

        Args:
            input_data_head (np.ndarray): Input data for the head region.
            input_data_forehead (np.ndarray): Input data for the forehead region.
            input_data_left_cheek (np.ndarray): Input data for the left cheek region.
            input_data_right_cheek (np.ndarray): Input data for the right cheek region.
            input_data_mouth (np.ndarray): Input data for the mouth region.
            labels (np.ndarray): Corresponding labels for the input data.
            number_of_epochs (int): Number of epochs to train.
        """
        inputs = {
            'input_face': input_data_head,
            'input_forehead': input_data_forehead,
            'input_left_cheek': input_data_left_cheek,
            'input_right_cheek': input_data_right_cheek,
            'input_mouth': input_data_mouth
        }

        predictions = self.model.predict(inputs)

        return predictions