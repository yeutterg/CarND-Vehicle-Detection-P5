import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from skimage.feature import hog
from img_processing import convert_color
from features import bin_spatial, color_hist
from sklearn.preprocessing import StandardScaler

# dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
# svc = dist_pickle["svc"]
# X_scaler = dist_pickle["scaler"]
# orient = dist_pickle["orient"]
# pix_per_cell = dist_pickle["pix_per_cell"]
# cell_per_block = dist_pickle["cell_per_block"]
# spatial_size = dist_pickle["spatial_size"]
# hist_bins = dist_pickle["hist_bins"]
#
# img = mpimg.imread('test_image.jpg')

def load_pickle(filename='data.p'):
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


def get_hog_features_img(img, orient, pix_per_cell, cell_per_block,
                         vis=False, feature_vec=True):
    """
    Gets HOG features for an image

    :param img:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param vis:
    :param feature_vec:
    :return:
    """
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features



def get_features_img(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=2):
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

    # Extract features from each image in the dataset and append
    for file in dataset:
        img = mpimg.imread(file)
        features_img = get_features_img(img, color_space, spatial_size, hist_bins, orient, pix_per_cell,
                                        cell_per_block, hog_channel)
        features.append(features_img)

    return features


def get_features_all_datasets():
    """
    Gets features for all datasets

    :return: cars_train_feat, cars_test_feat, cars_valid_feat, noncars_train_feat, noncars_test_feat, noncars_valid_feat
    """
    # Parameters
    color_space = 'HLS'
    spatial_size = (32, 32)
    hist_bins = 32
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'

    # Load the pickle file
    cars_train, cars_test, cars_valid, noncars_train, noncars_test, noncars_valid = load_pickle()

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

    return cars_train_feat, cars_test_feat, cars_valid_feat, noncars_train_feat, noncars_test_feat, noncars_valid_feat


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
                       noncars_valid_feat, noncars_test_feat))

    # Fit a StandardScaler based on the shape of the stack
    scaler = StandardScaler().fit(stack)

    # Apply the scaler to the stack and return
    return scaler.transform(stack)


def generate_train_valid_test(cars_train_feat, cars_test_feat, cars_valid_feat,
                              noncars_train_feat, noncars_test_feat, noncars_valid_feat,
                              random_state=42):
    """
    Generates X and y datasets for training, validation, and testing

    :param stack:
    :param cars_train_feat:
    :param cars_test_feat:
    :param cars_valid_feat:
    :param noncars_train_feat:
    :param noncars_test_feat:
    :param noncars_valid_feat:
    :param random_state:
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
    stack = scale_features(cars_train_feat, cars_test_feat, cars_valid_feat, noncars_train_feat,
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
    X_train, y_train = shuffle(X_train, y_train, random_state)
    X_valid, y_valid = shuffle(X_valid, y_valid, random_state)
    X_test, y_test = shuffle(X_test, y_test, random_state)

    return X_train, X_valid, X_test, y_train, y_valid, y_test




cars_train_feat, cars_test_feat, cars_valid_feat, noncars_train_feat, noncars_test_feat, \
noncars_valid_feat = get_features_all_datasets()

X_train, X_valid, X_test, y_train, y_valid, y_test = generate_train_valid_test(cars_train_feat, cars_test_feat,
                                                                               cars_valid_feat, noncars_train_feat,
                                                                               noncars_test_feat, noncars_valid_feat)



### old ###



# Define a single function that can extract features using hog sub-sampling and make predictions
def hog_find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return draw_img


def apply_hog(img):
    """

    :param img: The image
    :return:
    """
    ystart = 400
    ystop = 656
    scale = 1.5

    out_img = hog_find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                            hist_bins)





# ystart = 400
# ystop = 656
# scale = 1.5
#
# out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
#                     hist_bins)
#
# plt.imshow(out_img)