import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import time

from skimage.feature import hog
from img_processing import convert_color, grayscale
from features import bin_spatial, color_hist
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def load_data_pickle(filename='data.p'):
    """
    Loads the pickle file and returns training, validation, and test sets

    :param filename: The filename
    :return: cars_train, cars_test, cars_valid, noncars_train, noncars_test,
             noncars_valid
    """
    with open(filename, mode='rb') as file:
        data = pickle.load(file)

    cars_train = data['cars_train']
    cars_test = data['cars_test']
    cars_valid = data['cars_valid']
    noncars_train = data['noncars_train']
    noncars_test = data['noncars_test']
    noncars_valid = data['noncars_valid']

    return cars_train, cars_test, cars_valid, noncars_train, noncars_test, noncars_valid


def save_hog_pickle(color_space, svc, stack, scaler, orient, hist_bins, spatial_size, pix_per_cell,
                    cells_per_block, hog_channel, filename='hog.p'):
    print('Saving hog output.')
    try:
        with open(filename, 'wb') as file:
            pickle.dump(
                {
                    'svc': svc,
                    'color_space': color_space,
                    'stack': stack,
                    'scaler': scaler,
                    'orient': orient,
                    'hist_bins': hist_bins,
                    'spatial_size': spatial_size,
                    'pix_per_cell': pix_per_cell,
                    'cells_per_block': cells_per_block,
                    'hog_channel': hog_channel
                },
                file, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Error saving hog output to', filename, ':', e)
        raise


def get_hog_features_img(img, orient, pix_per_cell, cell_per_block,
                         vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
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


def get_features_dataset(dataset, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                         orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=2):
    """

    :param dataset:
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

    t0 = time.time()

    # Extract features from each image in the dataset and append
    for file in dataset:
        img = mpimg.imread(file)
        features_img = get_hog_features_img(img, orient, pix_per_cell, cell_per_block, hog_channel, vis=True)
        features.append(features_img)

    t1 = time.time()
    print('Extracted %s features in %s seconds.' % (len(features_img), t1 - t0))

    return features


def get_features_all_datasets(color_space, orient, hist_bins, spatial_size, pix_per_cell,
                              cell_per_block, hog_channel):
    """
    Gets features for all datasets

    :return: cars_train_feat, cars_test_feat, cars_valid_feat, noncars_train_feat, noncars_test_feat, noncars_valid_feat
    """

    # Load the pickle file
    cars_train, cars_test, cars_valid, noncars_train, noncars_test, noncars_valid = load_data_pickle()

    # Get features for each dataset
    cars_train_feat = get_features_dataset(cars_train, color_space, spatial_size, hist_bins,
                                           orient, pix_per_cell, cell_per_block, hog_channel)
    cars_test_feat = get_features_dataset(cars_test, color_space, spatial_size, hist_bins,
                                          orient, pix_per_cell, cell_per_block, hog_channel)
    cars_valid_feat = get_features_dataset(cars_valid, color_space, spatial_size, hist_bins,
                                           orient, pix_per_cell, cell_per_block, hog_channel)
    noncars_train_feat = get_features_dataset(noncars_train, color_space, spatial_size, hist_bins,
                                              orient, pix_per_cell, cell_per_block, hog_channel)
    noncars_test_feat = get_features_dataset(noncars_test, color_space, spatial_size, hist_bins,
                                             orient, pix_per_cell, cell_per_block, hog_channel)
    noncars_valid_feat = get_features_dataset(noncars_valid, color_space, spatial_size, hist_bins,
                                              orient, pix_per_cell, cell_per_block, hog_channel)

    return cars_train_feat, cars_test_feat, cars_valid_feat, noncars_train_feat, noncars_test_feat, noncars_valid_feat, \
           cars_train, cars_test, cars_valid, noncars_train, noncars_test, noncars_valid


def scale_features(cars_train_feat, cars_test_feat, cars_valid_feat, noncars_train_feat,
                   noncars_test_feat, noncars_valid_feat):
    """
    Standardizes features using sklearn's StandardScaler

    :param cars_train_feat:
    :param cars_test_feat:
    :param cars_valid_feat:
    :param noncars_train_feat:
    :param noncars_test_feat:
    :param noncars_valid_feat:
    :return: The scaled feature stack
    """
    # Stack the features into a matrix
    stack = np.vstack((cars_train_feat, cars_valid_feat, cars_test_feat, noncars_train_feat,
                       noncars_valid_feat, noncars_test_feat)).astype(np.float64)

    # Fit a StandardScaler based on the shape of the stack
    scaler = StandardScaler().fit(stack)

    # Apply the scaler to the stack and return
    return scaler.transform(stack), scaler


def generate_train_valid_test(cars_train_feat, cars_test_feat, cars_valid_feat,
                              noncars_train_feat, noncars_test_feat, noncars_valid_feat):
    """
    Generates X and y datasets for training, validation, and testing

    :param stack:
    :param cars_train_feat:
    :param cars_test_feat:
    :param cars_valid_feat:
    :param noncars_train_feat:
    :param noncars_test_feat:
    :param noncars_valid_feat:
    :return:
    """
    # Get the length of each dataset
    c_train_len = len(cars_train_feat)
    c_valid_len = len(cars_valid_feat)
    c_test_len = len(cars_test_feat)
    n_train_len = len(noncars_train_feat)
    n_valid_len = len(noncars_valid_feat)
    n_test_len = len(noncars_test_feat)

    # Set array index points
    br1 = c_train_len
    br2 = br1 + c_valid_len
    br3 = br2 + c_test_len
    br4 = br3 + n_train_len
    br5 = br4 + n_valid_len

    # Get the scaled feature stack
    stack, scaler = scale_features(cars_train_feat, cars_test_feat, cars_valid_feat, noncars_train_feat,
                           noncars_test_feat, noncars_valid_feat)

    # Reindex features from the scaled stack
    cars_train_feat = stack[:br1]
    cars_valid_feat = stack[br1:br2]
    cars_test_feat = stack[br2:br3]
    noncars_train_feat = stack[br3:br4]
    noncars_valid_feat = stack[br4:br5]
    noncars_test_feat = stack[br5:]

    # Get X_ and y_ values for training, validation, and testing
    X_train = np.vstack((cars_train_feat, noncars_train_feat))
    X_valid = np.vstack((cars_valid_feat, noncars_valid_feat))
    X_test = np.vstack((cars_test_feat, noncars_test_feat))
    y_train = np.hstack((np.ones(c_train_len), np.zeros(n_train_len)))
    y_valid = np.hstack((np.ones(c_valid_len), np.zeros(n_valid_len)))
    y_test = np.hstack((np.ones(c_test_len), np.zeros(n_test_len)))

    # Shuffle the values
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_valid, y_valid = shuffle(X_valid, y_valid, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, stack, scaler


def sv_classifier(X_train, X_valid, X_test, y_train, y_valid, y_test):
    """
    Runs the support vector classifier on X_ and y_ datasets

    :param X_train:
    :param X_valid:
    :param X_test:
    :param y_train:
    :param y_valid:
    :param y_test:
    :return:
    """
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    valid_acc = svc.score(X_valid, y_valid)
    test_acc = svc.score(X_test, y_test)
    return svc, valid_acc, test_acc


def visualize(cars_train, noncars_train, cars_valid_feat, noncars_valid_feat, cars_valid,
              noncars_valid, svc, orient, pix_per_cell, cells_per_block):
    # Plot hog features for cars and noncars
    f, ax = plt.subplots(6, 7, figsize=(20, 10))
    f.subplots_adjust(hspace=0.2, wspace=0.05)
    colorspace = cv2.COLOR_RGB2HLS

    for i, j, in enumerate([60, 800, 1800]):
        img = plt.imread(cars_train[j])
        feat_img = cv2.cvtColor(img, colorspace)

        ax[i, 0].imshow(img)
        ax[i, 0].set_title('car {0}'.format(j))
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])

        for ch in range(3):
            ax[i, ch + 1].imshow(feat_img[:, :, ch], cmap='gray')
            ax[i, ch + 1].set_title('img ch {0}'.format(ch))
            ax[i, ch + 1].set_xticks([])
            ax[i, ch + 1].set_yticks([])

            feat, h_img = get_hog_features_img(feat_img[:, :, ch], orient, pix_per_cell, cells_per_block, vis=True)
            ax[i, ch + 4].imshow(h_img, cmap='gray')
            ax[i, ch + 4].set_title('HOG ch {0}'.format(ch))
            ax[i, ch + 4].set_xticks([])
            ax[i, ch + 4].set_yticks([])

        img = plt.imread(noncars_train[j])
        feat_img = cv2.cvtColor(img, colorspace)

        i += 3

        ax[i, 0].imshow(img)
        ax[i, 0].set_title('noncar {0}'.format(j))
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])

        for ch in range(3):
            ax[i, ch + 1].imshow(feat_img[:, :, ch], cmap='gray')
            ax[i, ch + 1].set_title('img ch {0}'.format(ch))
            ax[i, ch + 1].set_xticks([])
            ax[i, ch + 1].set_yticks([])

            feat, h_img = get_hog_features_img(feat_img[:, :, ch], orient, pix_per_cell, cells_per_block, vis=True)
            ax[i, ch + 4].imshow(h_img, cmap='gray')
            ax[i, ch + 4].set_title('HOG ch {0}'.format(ch))
            ax[i, ch + 4].set_xticks([])
            ax[i, ch + 4].set_yticks([])

    plt.savefig('./output_images/hog_features.png')


def apply_hog():
    """

    :return:
    """
    # Parameters
    color_space = 'HLS'
    spatial_size = (32, 32)
    hist_bins = 32
    orient = 9
    pix_per_cell = 8
    cells_per_block = 2
    hog_channel = 'ALL'

    cars_train_feat, cars_test_feat, cars_valid_feat, noncars_train_feat, noncars_test_feat, noncars_valid_feat, \
    cars_train, cars_test, cars_valid, noncars_train, noncars_test, noncars_valid = \
        get_features_all_datasets(color_space, orient, hist_bins, spatial_size, pix_per_cell, cells_per_block, hog_channel)

    X_train, X_valid, X_test, y_train, y_valid, y_test, stack, scaler = \
        generate_train_valid_test(cars_train_feat, cars_test_feat, cars_valid_feat, noncars_train_feat,
                                  noncars_test_feat, noncars_valid_feat)

    svc, test_acc, valid_acc = sv_classifier(X_train, X_valid, X_test, y_train, y_valid, y_test)

    print('Validation Accuracy:', valid_acc)
    print('Test Accuracy:', test_acc)

    save_hog_pickle(color_space, svc, stack, scaler, orient, hist_bins, spatial_size, pix_per_cell, cells_per_block, hog_channel)

    orient = 9
    pix_per_cell = 8
    cells_per_block = 2

    visualize(cars_train, noncars_train, cars_valid_feat, noncars_valid_feat, cars_valid,
              noncars_valid, svc, orient, pix_per_cell, cells_per_block)


# Uncomment this to run:
apply_hog()
