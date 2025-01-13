from video_utils import video
import os

DATASET_PATH = '../faceforensics/'
DESCRIPTOR_LIST = ['gabor', 'lbp', 'halo', 'fourier']

if __name__ == '__main__':

    ## retrieving the list of all dataset files
    file_paths = os.listdir(DATASET_PATH)

    ## iterating over dataset files
    for video_file_path in file_paths:
        video_regions = video.Video(video_path=f'{DATASET_PATH}{video_file_path}', descriptor_list=DESCRIPTOR_LIST,
                                    number_of_sampled_frames = 32, frame_width=480, frame_height=480)

        a, b, c, d, e = video_regions.retrieve_face_areas()

    __import__("IPython").embed()