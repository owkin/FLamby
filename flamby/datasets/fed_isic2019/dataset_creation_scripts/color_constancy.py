from __future__ import division
import numpy
import cv2

def color_constancy(img, power=6, gamma=None):
    """
    Parameters
    ----------
    img: 3D numpy array
        The original image with format of (h, w, c)
    power: int
        The degree of norm, 6 is used in reference paper
    gamma: float
        The value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = numpy.ones((256,1), dtype='uint8') * 0
        for i in range(256): look_up_table[i][0] = 255*pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = numpy.power(img, power)
    rgb_vec = numpy.power(numpy.mean(img_power, (0,1)), 1/power)
    rgb_norm = numpy.sqrt(numpy.sum(numpy.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*numpy.sqrt(3))
    img = numpy.multiply(img, rgb_vec)
    
    return img.astype(img_dtype)