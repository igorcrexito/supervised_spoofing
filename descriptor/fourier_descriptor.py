from descriptor.image_descriptor import ImageDescriptor
import numpy as np
import cv2


class FourierDescriptor(ImageDescriptor):

    def __init__(self, descriptor_name: str, decibels_scaling: bool=True):
        super().__init__(descriptor_name)
        self.decibels_scaling = decibels_scaling


    def compute_feature(self, image: np.ndarray):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        decibels_factor = 20 if self.decibels_scaling else 1
        magnitude_spectrum = decibels_factor * np.log(np.abs(fshift) + 1)

        return magnitude_spectrum