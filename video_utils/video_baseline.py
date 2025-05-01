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

class VideoBaseline:

    def __init__(self, video_path: str, number_of_sampled_frames: int = 32,
                 frame_width: int = 224, frame_height: int = 224):
        self.number_of_sampled_frames = number_of_sampled_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
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