from model_architectures.single_model import SingleModel
from video_utils import video
import os
import numpy as np

DATASET_PATH = '../faceforensics/'
DESCRIPTOR_LIST = ['gabor', 'lbp', 'halo', 'fourier']

if __name__ == '__main__':

    ## retrieving the list of all dataset files
    file_paths = os.listdir(DATASET_PATH)

    ## instantiating an array to store video information
    input_head = np.ones((len(file_paths), 32, 80, 80, 7))
    input_forehead = np.ones((len(file_paths), 32, 32, 32, 7))
    input_left_cheek = np.ones((len(file_paths), 32, 24, 24, 7))
    input_right_cheek = np.ones((len(file_paths), 32, 24, 24, 7))
    input_mouth = np.ones((len(file_paths), 32, 20, 20, 7))

    ## iterating over dataset files and creating data structures
    for index, video_file_path in enumerate(file_paths):
        video_regions = video.Video(video_path=f'{DATASET_PATH}{video_file_path}', descriptor_list=DESCRIPTOR_LIST,
                                    number_of_sampled_frames = 32, frame_width=480, frame_height=480)

        ## composing model outputs
        face, forehead, left_cheek, right_cheek, mouth = video_regions.retrieve_face_areas()
        input_head[index] = face
        input_forehead[index] = forehead
        input_left_cheek[index] = left_cheek
        input_right_cheek[index] = right_cheek
        input_mouth[index] = mouth


    ## instantiating the classification model
    model = SingleModel(summarize_model=True)

    labels = np.array([[0, 1], [0, 1], [1, 0]])

    ## training the model
    model.fit_model(input_data_head=input_head,
                    input_data_forehead=input_forehead,
                    input_data_right_cheek=input_right_cheek,
                    input_data_left_cheek=input_left_cheek,
                    input_data_mouth=input_mouth,
                    labels=labels,
                    number_of_epochs=5)