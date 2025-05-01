from keras.layers import Input, Conv2D, Conv3D, MaxPooling3D, UpSampling2D, Concatenate, LayerNormalization, ZeroPadding2D, \
    BatchNormalization, Flatten, Dense, Reshape, DepthwiseConv2D, Add, Dropout, MultiHeadAttention, Rescaling, Activation
from keras.models import Model
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
import sys
sys.path.insert(0, '..')

class SingleModel:

    def __init__(self, summarize_model: bool, pre_trained_path: str = None):
        self.summarize_model = summarize_model
        if pre_trained_path is not None:
            self.model = keras.models.load_model(pre_trained_path)
        else:
            self.model = self._create_model()


    def _create_model(self):
        # Number of inputs
        num_inputs = 5
        inputs = []
        processed_inputs = []

        # Define Conv3D and Flatten layers for each input
        for i in range(num_inputs):
            if i == 0:
                input_layer = Input(shape=(6, 80, 80, 7), name=f'input_face')
                inputs.append(input_layer)
            elif i == 1:
                input_layer = Input(shape=(6, 32, 32, 7), name=f'input_forehead')
                inputs.append(input_layer)
            elif i == 2:
                input_layer = Input(shape=(6, 24, 24, 7), name=f'input_left_cheek')
                inputs.append(input_layer)
            elif i == 3:
                input_layer = Input(shape=(6, 24, 24, 7), name=f'input_right_cheek')
                inputs.append(input_layer)
            else:
                input_layer = Input(shape=(6, 20, 20, 7), name=f'input_mouth')
                inputs.append(input_layer)

            # Conv3D block
            x = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer)
            x = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
            x = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)

            x = MaxPooling3D(pool_size=(2, 2, 2))(x)

            x = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
            x = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
            x = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)

            x = MaxPooling3D(pool_size=(2, 2, 2))(x)

            # Flatten the output
            x = Flatten()(x)
            processed_inputs.append(x)

        # Concatenate the flattened outputs from all branches
        concatenated = Concatenate()(processed_inputs)

        # Fully connected layers
        x = Dense(units=128, activation='relu')(concatenated)
        x = Dropout(0.25)(x)
        x = Dense(units=32, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(units=16, activation='relu')(x)

        # Output layer for categorical classification
        output = Dense(units=2, activation='softmax', name='output')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=output)

        # Compile the model
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

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
            verbose=1, save_best_only=True, mode='min')

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