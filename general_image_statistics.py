import math
import cv2


def pixel_brightness(pixel):
    assert 3 == len(pixel)
    r, g, b = pixel
    return math.sqrt(0.299 * r ** 2 + 0.587 * g ** 2 + 0.114 * b ** 2)


def image_brightness(img):
    nr_of_pixels = len(img) * len(img[0])
    return sum(pixel_brightness(pixel) for row in img for pixel in row ) / nr_of_pixels


def blurry(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

