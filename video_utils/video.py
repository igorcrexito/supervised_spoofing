import cv2
from skimage import img_as_ubyte
import dlib
import numpy as np

from descriptor.bsif_descriptor import BSIFDescriptor
from descriptor.fourier_descriptor import FourierDescriptor
from descriptor.gabor_descriptor import GaborDescriptor
from descriptor.halo_descriptor import HaloDescriptor
from descriptor.lbp_descriptor import ELBPDescriptor

## instantiating descriptors
halo_descriptor = HaloDescriptor(descriptor_name='halo')
lbp_descriptor = ELBPDescriptor(descriptor_name='lbp')
gabor_descriptor = GaborDescriptor(descriptor_name='gabor')
fourier_descriptor = FourierDescriptor(descriptor_name='fourier')
bsif_descriptor1 = BSIFDescriptor(descriptor_name='bsif_3x3x5', filter_size='3x3_5', base_path='descriptor/filters/texturefilters/')

class Video:

    def __init__(self, video_path: str, descriptor_list, number_of_sampled_frames: int = 32,
                 frame_width: int = 240, frame_height: int = 240):
        self.number_of_sampled_frames = number_of_sampled_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.descriptor_list = descriptor_list
        self.frame_list = self.read_and_sample(video_path)

    def read_and_sample(self, video_path: str):

        frame_list = []

        capture_object = cv2.VideoCapture(video_path)
        if not capture_object.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return []

        frame_count = int(capture_object.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count < self.number_of_sampled_frames:
            return []

        sample_indices = [
            round(i * frame_count / self.number_of_sampled_frames)
            for i in range(self.number_of_sampled_frames)
        ]

        frame_number = 0
        sampled_count = 0

        while True:
            ret, frame = capture_object.read()
            if not ret:
                break


            if frame_number in sample_indices:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                frame_list.append(img_as_ubyte(frame))
                sampled_count += 1

            frame_number += 1

            if sampled_count == self.number_of_sampled_frames:
                break

        capture_object.release()

        return frame_list

    def retrieve_face_areas(self):
        # Load the pre-trained face detector and facial landmarks predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        face_crop_array = np.ones((self.number_of_sampled_frames ,80, 80, len(self.descriptor_list) + 3))
        forehead_crop_array = np.ones((self.number_of_sampled_frames ,32, 32, len(self.descriptor_list) + 3))
        left_cheek_crop_array = np.ones((self.number_of_sampled_frames ,24, 24, len(self.descriptor_list) + 3))
        right_cheek_crop_array = np.ones((self.number_of_sampled_frames ,24, 24, len(self.descriptor_list) + 3))
        mouth_crop_array = np.ones((self.number_of_sampled_frames ,20, 20, len(self.descriptor_list) + 3))

        for index, frame in enumerate(self.frame_list):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                # Get landmarks
                landmarks = predictor(gray, face)

                # Define facial regions using landmark indices
                left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
                both_eyes_points = left_eye_points + right_eye_points
                mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(49, 68)]

                # Calculate the bounding box for the forehead
                x_min = min([point[0] for point in both_eyes_points])
                x_max = max([point[0] for point in both_eyes_points])
                y_min = min([point[1] for point in both_eyes_points])

                # Define forehead height as a fraction of the distance between the eyes
                forehead_height = int((x_max - x_min) * 0.5)
                y_max = y_min - forehead_height

                # Forehead crop
                forehead_crop = frame[max(0, y_max):max(0, y_min), max(0, x_min):max(0, x_max)]

                # Calculate cheek and mouth regions
                mouth_center_x = (max([point[0] for point in mouth_points]) + min(
                    [point[0] for point in mouth_points])) // 2
                mouth_center_y = (max([point[1] for point in mouth_points]) + min(
                    [point[1] for point in mouth_points])) // 2

                # Left cheek crop
                left_cheek_x_min = x_min - int((x_max - x_min) * 0.15)
                left_cheek_x_max = (x_min + mouth_center_x) // 2
                left_cheek_y_min = y_min + int((y_min - y_max) * 0.1)
                left_cheek_y_max = mouth_center_y - int((mouth_center_y - y_min) * 0.3)
                left_cheek_crop = frame[max(0, left_cheek_y_min):max(0, left_cheek_y_max),
                                  max(0, left_cheek_x_min):max(0, left_cheek_x_max)]

                # Right cheek crop
                right_cheek_x_min = (x_max + mouth_center_x) // 2
                right_cheek_x_max = x_max + int((x_max - x_min) * 0.15)
                right_cheek_y_min = y_min + int((y_min - y_max) * 0.1)
                right_cheek_y_max = mouth_center_y - int((mouth_center_y - y_min) * 0.3)
                right_cheek_crop = frame[max(0, right_cheek_y_min):max(0, right_cheek_y_max),
                                   max(0, right_cheek_x_min):max(0, right_cheek_x_max)]

                # Mouth crop
                mouth_x_min = min([point[0] for point in mouth_points])
                mouth_x_max = max([point[0] for point in mouth_points])
                mouth_y_min = min([point[1] for point in mouth_points])
                mouth_y_max = max([point[1] for point in mouth_points])
                mouth_crop = frame[max(0, mouth_y_min):max(0, mouth_y_max),
                             max(0, mouth_x_min):max(0, mouth_x_max)]

                # Whole face crop (using the bounding box from the face detector)
                x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
                face_crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

                # Applying descriptors for each part of the face (You can apply descriptors like you already do)
                crops = apply_descriptors(descriptor_list=self.descriptor_list,
                                          face_crop=face_crop, forehead_crop=forehead_crop,
                                          left_cheek_crop=left_cheek_crop, right_cheek_crop=right_cheek_crop,
                                          mouth_crop=mouth_crop)

                face_crop_array[index] = crops[0]
                forehead_crop_array[index] = crops[1]
                left_cheek_crop_array[index] = crops[2]
                right_cheek_crop_array[index] = crops[3]
                mouth_crop_array[index] = crops[4]

        return face_crop_array, forehead_crop_array, left_cheek_crop_array, right_cheek_crop_array, mouth_crop_array


def apply_descriptors(descriptor_list: list, face_crop,
                      forehead_crop, left_cheek_crop, right_cheek_crop, mouth_crop):

    ## standardizing crop dimensions
    face_crop = cv2.resize(face_crop, (80, 80))
    forehead_crop = cv2.resize(forehead_crop, (32, 32))
    left_cheek_crop = cv2.resize(left_cheek_crop, (24, 24))
    right_cheek_crop = cv2.resize(right_cheek_crop, (24, 24))
    mouth_crop = cv2.resize(mouth_crop, (20, 20))

    ## instantiating arrays containing descriptor activations --> 1 extra dimension due to gabor's 2 channels, LBP 3 channels
    face_crop_array = np.ones((80, 80, len(descriptor_list) + 3))
    forehead_crop_array = np.ones((32, 32, len(descriptor_list) + 3))
    left_cheek_crop_array = np.ones((24, 24, len(descriptor_list) + 3))
    right_cheek_crop_array = np.ones((24, 24, len(descriptor_list) + 3))
    mouth_crop_array = np.ones((20, 20, len(descriptor_list) + 3))

    for descriptor in descriptor_list:
        if descriptor == 'halo':
            face_crop_array[:,:,0] = halo_descriptor.compute_feature(face_crop)
            forehead_crop_array[:,:,0] = halo_descriptor.compute_feature(forehead_crop)
            left_cheek_crop_array[:,:,0] = halo_descriptor.compute_feature(left_cheek_crop)
            right_cheek_crop_array[:,:,0] = halo_descriptor.compute_feature(right_cheek_crop)
            mouth_crop_array[:,:,0] = halo_descriptor.compute_feature(mouth_crop)
        elif descriptor == 'lbp':
            face_crop_array[:,:,1:4] = lbp_descriptor.compute_feature(face_crop)
            forehead_crop_array[:,:,1:4] = lbp_descriptor.compute_feature(forehead_crop)
            left_cheek_crop_array[:,:,1:4] = lbp_descriptor.compute_feature(left_cheek_crop)
            right_cheek_crop_array[:,:,1:4] = lbp_descriptor.compute_feature(right_cheek_crop)
            mouth_crop_array[:,:,1:4] = lbp_descriptor.compute_feature(mouth_crop)
        elif descriptor == 'gabor':
            face_crop_array[:,:,4:6] = gabor_descriptor.compute_feature(face_crop)
            forehead_crop_array[:,:,4:6] = gabor_descriptor.compute_feature(forehead_crop)
            left_cheek_crop_array[:,:,4:6] = gabor_descriptor.compute_feature(left_cheek_crop)
            right_cheek_crop_array[:,:,4:6] = gabor_descriptor.compute_feature(right_cheek_crop)
            mouth_crop_array[:,:,4:6] = gabor_descriptor.compute_feature(mouth_crop)
        elif descriptor == 'fourier':
            face_crop_array[:,:,6] = fourier_descriptor.compute_feature(face_crop)
            forehead_crop_array[:,:,6] = fourier_descriptor.compute_feature(forehead_crop)
            left_cheek_crop_array[:,:,6] = fourier_descriptor.compute_feature(left_cheek_crop)
            right_cheek_crop_array[:,:,6] = fourier_descriptor.compute_feature(right_cheek_crop)
            mouth_crop_array[:,:,6] = fourier_descriptor.compute_feature(mouth_crop)

    return face_crop_array, forehead_crop_array, left_cheek_crop_array, right_cheek_crop_array, mouth_crop_array