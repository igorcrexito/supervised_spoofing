from descriptor.image_descriptor import ImageDescriptor
import numpy as np
import cv2
from skimage import filters


class GaborDescriptor(ImageDescriptor):

    def __init__(self, descriptor_name: str, frequency: float, theta: float):
        super().__init__(descriptor_name)
        self.frequency = frequency
        self.theta = theta

    def compute_feature(self, image: np.ndarray):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        height, width = image.shape

        gabor_image = np.ones((height, width, 2), dtype=np.uint8)
        filtered_real, filtered_imaginary = filters.gabor(image, frequency=self.frequency, theta=self.theta)

        gabor_image[:, :, 0] = filtered_real
        gabor_image[:, :, 1] = filtered_imaginary
        return gabor_image