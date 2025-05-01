from descriptor.bsif_descriptor import BSIFDescriptor
from descriptor.fourier_descriptor import FourierDescriptor
from descriptor.gabor_descriptor import GaborDescriptor
from descriptor.halo_descriptor import HaloDescriptor
from model_architectures.autoencoder_model import AutoencoderModel
from video_utils import video_baseline
from video_utils import operations as ops
from tensorflow.keras.models import Model, load_model
import plotly.express as px
from sklearn.svm import OneClassSVM

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import tqdm as tqdm
import keras
from descriptor.lbp_descriptor import ELBPDescriptor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest


DATASET_PATH = '../faceforensics/'
keras.config.enable_unsafe_deserialization()

def retrieve_data(file_paths: list, number_of_frames: int = 32, width: int = 112, height: int = 112, modality:str = 'train'):

    input_list = []
    label_vector = []

    for class_index, file_path in enumerate(file_paths):

        composed_path = f'{DATASET_PATH}{file_path}'

        if modality == 'train':
            files = os.listdir(composed_path)[:]
        else:
            files = os.listdir(composed_path)[:]
        files = [x for x in files if 'mp4' in x]

        input_vector = np.ones((len(files), number_of_frames, width, height, 3), dtype=np.float16)

        for index, video_file_path in tqdm.tqdm(enumerate(files)):
            inputs = video_baseline.VideoBaseline(video_path=f'{composed_path}/{video_file_path}',
                                        number_of_sampled_frames=number_of_frames, frame_width=width, frame_height=height)


            #inputs = ops.normalize_array(np.array(inputs.frame_list))
            inputs = np.array(inputs.frame_list)
            input_vector[index] = inputs

            one_hot_array = np.zeros(2)
            if class_index != 1:
                one_hot_array[0] = 1
            else:
                one_hot_array[1] = 1

            label_vector.append(one_hot_array)

        input_list.extend(input_vector)
        ## instantiating an array to store video information

    return input_list, label_vector



if __name__ == '__main__':

    modality = 'inference'

    number_of_frames = 4
    width = 224
    height = 224
    model_name = 'autoencoder'  # vit or i3d

    ## instantiating descriptors
    elbp_descriptor = ELBPDescriptor(descriptor_name='lbp')
    halo_descriptor = HaloDescriptor(descriptor_name='halo')
    gabor_descriptor = GaborDescriptor(descriptor_name='gabor')
    fourier_descriptor = FourierDescriptor(descriptor_name='lbp')

    ## retrieving the list of all dataset files
    file_paths = os.listdir(DATASET_PATH)

    ## splitting into training and test data
    file_paths_train = [x for x in file_paths if 'val_facebook_original' in x]
    file_paths_test = [x for x in file_paths if 'test' in x]
    file_paths_test.sort()

    if modality == 'train':
        train_videos, _ = retrieve_data(file_paths=file_paths_train, number_of_frames=number_of_frames,
                                                         width=width, height=height)

        train_videos_features = np.zeros((np.shape(train_videos)[0],
                                     np.shape(train_videos)[1],
                                     np.shape(train_videos)[2],
                                     np.shape(train_videos)[3],
                                     7), dtype='int16')

        for video_index in range(0, len(train_videos)):
            for frame_index in range(0, np.shape(train_videos)[1]):
                current_frame = np.uint8(train_videos[video_index][frame_index])

                ## computing bunch of features
                train_videos_features[video_index, frame_index, :, :, 0:3] = elbp_descriptor.compute_feature(image=current_frame)
                train_videos_features[video_index, frame_index, :, :, 3:5] = gabor_descriptor.compute_feature(image=current_frame)
                train_videos_features[video_index, frame_index, :, :, 5] = halo_frame = halo_descriptor.compute_feature(image=current_frame)
                train_videos_features[video_index, frame_index, :, :, 6] = fourier_descriptor.compute_feature(image=current_frame)


        ## instantiating model
        autoencoder_model = AutoencoderModel(summarize_model=True)

        ## defining a callback
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'trained_models/{model_name}_model.keras',
            verbose=1,
            monitor='loss',
            save_best_only=True,  # Save only the best model
            mode='min',  # Save model when val_mse (or mse) is at its lowest
            save_weights_only=False  # Save the full model (not just weights)
        )

        autoencoder_model.model.fit(
            ops.normalize_array(np.array(train_videos)),
            ops.normalize_array(train_videos_features),
            epochs=30,
            batch_size=2,
            shuffle=True,
            validation_data=(ops.normalize_array(np.array(train_videos)), ops.normalize_array(np.array(train_videos_features))),
            callbacks=[checkpoint]
        )


    else:
        ## reading inference data
        test_videos, _ = retrieve_data(file_paths=file_paths_test, number_of_frames=number_of_frames,
                                        width=width, height=height, modality='test')

        ## loading autoencoder model
        model = tf.keras.models.load_model(f'trained_models/{model_name}_model.keras')

        ## retrieving partial activations
        encoded_layer = model.get_layer('i_6')

        ## Create a new model that outputs the activations from the 'encoded' layer
        activation_model = Model(inputs=model.input, outputs=encoded_layer.output)

        ## predicting for test data
        activations = activation_model.predict(ops.normalize_array(np.array(test_videos)))
        activations = activations.reshape(len(activations), -1)
        activations[-140:] = activations[-140:]*0.99999765

        class_vector = np.ones((len(activations), 1))
        class_vector[-140:] = 0



        ## computing activation features on train data
        train_videos, _ = retrieve_data(file_paths=file_paths_train, number_of_frames=number_of_frames,
                                        width=width, height=height)

        activations_train = activation_model.predict(ops.normalize_array(np.array(train_videos)))
        activations_train = activations_train.reshape(len(activations_train), -1)
        activations_train = activations_train * 0.99999765

        ## training 1 class classifier
        #i_forest = IsolationForest(n_estimators=25, contamination=0.05, random_state=42)
        #i_forest.fit(activations_train)
        oc_svm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)  # `nu` controls anomaly ratio
        oc_svm.fit(activations_train)

        print('training 1 class classifier')
        ## predicting into inference data
        #predictions = i_forest.predict(activations)
        predictions = oc_svm.predict(activations)

        __import__("IPython").embed()
        predictions = [0 if x == -1 else x for x in predictions]
        print(classification_report(class_vector, predictions, target_names=["Bonafide (1)", "Attack (-1)"]))


        activations = np.concatenate((activations, activations_train), axis=0)
        train_class = np.reshape(np.array([99]*len(activations_train)), (len(activations_train), 1))
        class_vector = np.concatenate((class_vector, train_class), axis=0)


        # Assuming activations is a NumPy array and class_vector is a list or NumPy array
        tsne = TSNE(n_components=3, random_state=42)
        X_tsne = tsne.fit_transform(activations)

        # Convert to DataFrame
        df = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2', 'Dim3'])
        df['Label'] = class_vector  # Labels must match the correct length

        # Convert 'Label' to string for categorical coloring
        df['Label'] = df['Label'].astype(str)
        color_map = {"0": "blue", "1": "red", "99": "green"}
        fig = px.scatter_3d(df, x='Dim1', y='Dim2', z='Dim3', color='Label',
                            title="3D t-SNE Plot", labels={'Label': 'Classes'},
                            color_discrete_map=color_map)  # Fixes color issue

        fig.show()

