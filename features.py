import cv2
import numpy as np

import matplotlib.image as mpimg


def bin_spatial(img, size=(32, 32)):
    """

    :param img:
    :param size:
    :return:
    """
    print(img)
    print(size)
    return cv2.resize(img, size).ravel()


def color_hist(img, nbins=32):
    """

    :param img:
    :param nbins:
    :return:
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features_img(img, cspace='BGR', spatial_size=(32, 32),
                         hist_bins=32, hist_range=(0, 256)):
    """
    Extracts features using bin_spatial() and color_hist()

    :param img:
    :param cspace:
    :param spatial_size:
    :param hist_bins:
    :param hist_range:
    :return:
    """
    feature_image = np.copy(img)

    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, spatial_size)

    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, hist_bins)

    # Append the new feature vector to the features list
    features = (np.concatenate((spatial_features, hist_features)))

    # Return list of feature vectors
    return features


def extract_features(imgs, cspace='BGR', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    """
    Extracts features using bin_spatial() and color_hist()

    :param imgs:
    :param cspace:
    :param spatial_size:
    :param hist_bins:
    :param hist_range:
    :return:
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        img = mpimg.imread(file)
        features_img = extract_features_img(img, cspace, spatial_size, hist_bins, hist_range)
        features.append(features_img)
    # Return list of feature vectors
    return features
