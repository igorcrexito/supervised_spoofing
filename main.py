from model_architectures.single_model import SingleModel
from video_utils import video
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from video_utils import operations as ops

DATASET_PATH = '../faceforensics/'
DESCRIPTOR_LIST = ['gabor', 'lbp', 'halo', 'fourier']


def retrieve_data(file_paths: list):

    input_face_list = []
    input_forehead_list = []
    input_left_cheek_list = []
    input_right_cheek_list = []
    input_mouth_list = []

    label_vector = []

    num_categories = len(file_paths)

    for class_index, file_path in enumerate(file_paths):

        composed_path = f'{DATASET_PATH}{file_path}'
        files = os.listdir(composed_path)

        ## instantiating an array to store video information
        number_of_frames = 6
        input_head = np.ones((len(files), number_of_frames, 80, 80, 7))
        input_forehead = np.ones((len(files), number_of_frames, 32, 32, 7))
        input_left_cheek = np.ones((len(files), number_of_frames, 24, 24, 7))
        input_right_cheek = np.ones((len(files), number_of_frames, 24, 24, 7))
        input_mouth = np.ones((len(files), number_of_frames, 20, 20, 7))

        for index, video_file_path in enumerate(files):
            print(f'reading video: {composed_path}/{video_file_path}')
            video_regions = video.Video(video_path=f'{composed_path}/{video_file_path}', descriptor_list=DESCRIPTOR_LIST,
                                        number_of_sampled_frames = number_of_frames, frame_width=480, frame_height=480)

            ## composing model inputs
            face, forehead, left_cheek, right_cheek, mouth = video_regions.retrieve_face_areas()

            # Normalize each region
            face = ops.normalize_array(face)
            forehead = ops.normalize_array(forehead)
            left_cheek = ops.normalize_array(left_cheek)
            right_cheek = ops.normalize_array(right_cheek)
            mouth = ops.normalize_array(mouth)

            ## Composing the data structures
            input_head[index] = face
            input_forehead[index] = forehead
            input_left_cheek[index] = left_cheek
            input_right_cheek[index] = right_cheek
            input_mouth[index] = mouth

            # Create a one-hot encoded array
            one_hot_array = np.zeros(2)
            if class_index != 1:
                one_hot_array[0] = 1
            else:
                one_hot_array[1] = 1

            label_vector.append(one_hot_array)

        input_face_list.append(input_head)
        input_forehead_list.append(input_forehead)
        input_left_cheek_list.append(input_left_cheek)
        input_right_cheek_list.append(input_right_cheek)
        input_mouth_list.append(input_mouth)

    return input_face_list, input_forehead_list, input_left_cheek_list, input_right_cheek_list, input_mouth_list, label_vector



if __name__ == '__main__':

    ## retrieving the list of all dataset files
    file_paths = os.listdir(DATASET_PATH)

    ## splitting into training and test data
    file_paths_train = [x for x in file_paths if 'val' in x]
    file_paths_test = [x for x in file_paths if 'test' in x]

    ## retrieving data
    train_face, train_forehead, train_left_cheek, train_right_cheek, train_mouth, train_labels = retrieve_data(file_paths_train)
    test_face, test_forehead, test_left_cheek, test_right_cheek, test_mouth, test_labels = retrieve_data(file_paths_test)

    ## creating a single structure
    train_face = np.concatenate([train_face[0], train_face[1],  train_face[2],  train_face[3],  train_face[4],  train_face[5]], axis=0)
    train_forehead = np.concatenate([train_forehead[0], train_forehead[1],  train_forehead[2],  train_forehead[3],  train_forehead[4],  train_forehead[5]], axis=0)
    train_left_cheek = np.concatenate([train_left_cheek[0], train_left_cheek[1],  train_left_cheek[2],  train_left_cheek[3],  train_left_cheek[4],  train_left_cheek[5]], axis=0)
    train_right_cheek = np.concatenate([train_right_cheek[0], train_right_cheek[1],  train_right_cheek[2],  train_right_cheek[3],  train_right_cheek[4],  train_right_cheek[5]], axis=0)
    train_mouth = np.concatenate([train_mouth[0], train_mouth[1],  train_mouth[2],  train_mouth[3],  train_mouth[4],  train_mouth[5]], axis=0)

    test_face = np.concatenate(
        [test_face[0], test_face[1], test_face[2], test_face[3], test_face[4], test_face[5]], axis=0)
    test_forehead = np.concatenate(
        [test_forehead[0], test_forehead[1], test_forehead[2], test_forehead[3], test_forehead[4],
         test_forehead[5]], axis=0)
    test_left_cheek = np.concatenate(
        [test_left_cheek[0], test_left_cheek[1], test_left_cheek[2], test_left_cheek[3], test_left_cheek[4],
         test_left_cheek[5]], axis=0)
    test_right_cheek = np.concatenate(
        [test_right_cheek[0], test_right_cheek[1], test_right_cheek[2], test_right_cheek[3], test_right_cheek[4],
         test_right_cheek[5]], axis=0)
    test_mouth = np.concatenate(
        [test_mouth[0], test_mouth[1], test_mouth[2], test_mouth[3], test_mouth[4], test_mouth[5]], axis=0)

    ## instantiating the classification model
    model = SingleModel(summarize_model=True)

    ## training the model
    model.fit_model(input_data_head=train_face,
                    input_data_forehead=train_forehead,
                    input_data_right_cheek=train_right_cheek,
                    input_data_left_cheek=train_left_cheek,
                    input_data_mouth=train_mouth,
                    labels=np.array(train_labels),
                    number_of_epochs=20)

    predictions = model.predict_data(input_data_head=test_face,
                    input_data_forehead=test_forehead,
                    input_data_right_cheek=test_right_cheek,
                    input_data_left_cheek=test_left_cheek,
                    input_data_mouth=test_mouth,
                    labels=np.array(test_labels))


    ## model evaluation
    y_true = np.argmax(test_labels, axis=1)
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