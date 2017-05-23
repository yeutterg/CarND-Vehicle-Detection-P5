import cv2

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def grayscale(img):
    """
    Converts a BGR image to grayscale
    :param img: The image in BGR color format
    :return The grayscale image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)