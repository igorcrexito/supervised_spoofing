from descriptor.image_descriptor import ImageDescriptor
import numpy as np
import cv2


class HaloDescriptor(ImageDescriptor):

    def __init__(self, descriptor_name: str, threshold_1: int, threshold_2: int,
                 gaussian_filter_size: int, dilation_epochs: int = 1):
        super().__init__(descriptor_name)
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.gaussian_filter_size = gaussian_filter_size
        self.dilation_epochs = dilation_epochs


    def compute_feature(self, image: np.ndarray):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(image, self.threshold_1, self.threshold_2)

        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=self.dilation_epochs)

        blurred_image = cv2.GaussianBlur(image, (self.gaussian_filter_size, self.gaussian_filter_size), 0)
        intensity_diff = cv2.absdiff(image, blurred_image)

        halo_map = cv2.bitwise_and(intensity_diff, intensity_diff, mask=dilated_edges)

        return halo_map