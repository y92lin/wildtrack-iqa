import cv2
import numpy as np
from numpy.linalg import norm


# https://stackoverflow.com/questions/14243472/estimate-brightness-of-an-image-opencv/22020098#22020098
def image_brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)


# https://towardsdatascience.com/measuring-enhancing-image-quality-attributes-234b0f250e10
def blurry(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

