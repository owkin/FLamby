from __future__ import division

import cv2
import numpy


def color_constancy(img, power=6, gamma=None):
    """
    Preprocessing step to make sure that the images appear with similar brightness
    and contrast.
    See this [link}(https://en.wikipedia.org/wiki/Color_constancy) for an explanation.
    Thank you to [Aman Arora](https://github.com/amaarora) for this
    [implementation](https://github.com/amaarora/melonama)
    Parameters
    ----------
    img: 3D numpy array, the original image
    power: int, degree of norm
    gamma: float, value of gamma correction
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype("uint8")
        look_up_table = numpy.ones((256, 1), dtype="uint8") * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype("float32")
    img_power = numpy.power(img, power)
    rgb_vec = numpy.power(numpy.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = numpy.sqrt(numpy.sum(numpy.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * numpy.sqrt(3))
    img = numpy.multiply(img, rgb_vec)

    return img.astype(img_dtype)
