from layers.custom_layers import CustomBroadcastLayer
from model_architectures.vivit_model import VivitModel
from video_utils import video_baseline
import os
import numpy as np
from model_architectures.i3d_inception import Inception_Inflated3d
from model_architectures.vit_model import VitModel
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from video_utils import operations as ops
import tqdm as tqdm
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import keras

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


            inputs = ops.normalize_array(np.array(inputs.frame_list))
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

    # ViViT ARCHITECTURE
    patch_size = (4, 4, 4)  # 3D patches (volumes)
    num_patches = (224 // 224) ** 2  # We assume square input
    layer_norm_eps = 1e-6
    projection_dim = 128  # Embedding Dim
    num_heads = 2
    num_layers = 2
    weight_decay = 1e-5
    learning_rate = 1e-4
    model_name = 'vivit'

    ## retrieving the list of all dataset files
    file_paths = os.listdir(DATASET_PATH)

    ## splitting into training and test data
    file_paths_train = [x for x in file_paths if 'val' in x]
    file_paths_test = [x for x in file_paths if 'test' in x]

    if modality == 'train':
        train_videos, train_label_vector = retrieve_data(file_paths=file_paths_train, number_of_frames=number_of_frames,
                                                         width=width, height=height)
        ## instantiating the vivit model
        vivit_model = VivitModel(learning_rate=learning_rate, num_layers=num_layers, num_heads=num_heads, input_shape=(number_of_frames, width, height, 3),
                               projection_dim=projection_dim, patch_size=patch_size, layer_norm_eps=layer_norm_eps,
                               num_classes=2, summarize_model=True)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'trained_models/{model_name}_model.keras',
            monitor='loss',  # or 'val_accuracy' if using validation data
            verbose=1,
            save_best_only=True,  # Save only if the monitored metric improves
            mode='min',
            save_weights_only=False
        )

        X_train, X_val, y_train, y_val = train_test_split(
            np.array(train_videos),
            np.array(train_label_vector),
            test_size=0.5,
            random_state=42,
            stratify=np.array(train_label_vector)  # Maintains class balance (if classification)
        )

        vivit_model.model.fit(
            X_train,
            y_train,
            epochs=60,
            batch_size=7,
            shuffle=True,
            class_weight={0: 1, 1: 6},
            callbacks=[checkpoint]
        )

    elif modality == 'inference':
        test_videos, test_label_vector = retrieve_data(file_paths=file_paths_test, number_of_frames=number_of_frames,
                                                       width=width, height=height)

        model = tf.keras.models.load_model(f'trained_models/{model_name}_model.keras')

        predictions = model.predict(np.array(test_videos))

        y_true = np.argmax(test_label_vector, axis=1)
        y_pred = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        report = classification_report(y_true, y_pred)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print('Classification Report:\n', report)

