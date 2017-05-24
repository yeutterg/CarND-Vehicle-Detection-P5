import cv2
import numpy as np
import matplotlib.pyplot as mpimg
from skimage.feature import hog


def bin_spatial(img, size=(32, 32)):
    """

    :param img:
    :param size:
    :return:
    """
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
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
        img = (img*255.).astype(np.uint8)
        features.append(extract_features(img, cspace, spatial_size, hist_bins, hist_range))
    # Return list of feature vectors
    return features


def extract_features_img(image, cspace='BGR', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    """

    :param image:
    :param cspace:
    :param spatial_size:
    :param hist_bins:
    :param hist_range:
    :return:
    """
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    else:
        feature_image = np.copy(image)
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins)
    # Append the new feature vector to the features list
    return np.concatenate((spatial_features, hist_features))


def get_hog_features_img(img, orient, pix_per_cell, cell_per_block,
                         vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def get_features_img(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
    """

    :param img:
    :param color_space:
    :param spatial_size:
    :param hist_bins:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel:
    :return:
    """
    features = []

    # If color space is other than RGB, convert
    if color_space == 'RGB':
        feature_img = np.copy(img)
    else:
        if color_space == 'HSV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    # Calculate spatial features
    spatial_features = bin_spatial(feature_img, size=spatial_size)
    features.append(spatial_features)

    # Calculate histogram features
    hist_features = color_hist(feature_img, nbins=hist_bins)
    features.append(hist_features)

    # Calculate HOG features
    if hog_channel == 'ALL':
        hog_features = []

        for ch in range(feature_img.shape[2]):
            hog_features.extend(get_hog_features_img(feature_img[:, :, ch],
                                                     orient, pix_per_cell, cell_per_block, vis=False,
                                                     feature_vec=True))

    else:
        hog_features = get_hog_features_img(feature_img[:, :, hog_channel],
                                            orient, pix_per_cell, cell_per_block, vis=False,
                                            feature_vec=True)

    features.append(hog_features)

    # Return the concatenated array of features
    return np.concatenate(features)
