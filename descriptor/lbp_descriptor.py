from descriptor.image_descriptor import ImageDescriptor
import numpy as np
import cv2
from skimage.feature import local_binary_pattern


class ELBPDescriptor(ImageDescriptor):

    def __init__(self, descriptor_name: str, method: str = 'uniform', radius: int = 1, neighbors: int = 8):
        super().__init__(descriptor_name)
        self.radius = radius
        self.neighbors = neighbors
        self.method = method

    def compute_feature(self, image: np.ndarray):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        height, width = image.shape

        elbp_image_final = np.ones((height, width, 3), dtype=np.uint8)

        lbp = local_binary_pattern(image, self.neighbors, self.radius, self.method)*255
        elbp_image_final[:, :, 0] = lbp
        elbp_image_final[:, :, 1] = lbp
        elbp_image_final[:, :, 2] = lbp
        return elbp_image_final