import pickle
import glob
import cv2
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np

from features import extract_features_img, get_features_img


def load_hog_pickle(filename='hog.p'):
    """
    Loads the pickle file and returns HOG parameters

    :param filename: The filename
    :return: color_space, svc, stack, orient, hist_bins, spatial_size, pix_per_cell, \
                    cells_per_block, hog_channel
    """
    with open(filename, mode='rb') as file:
        data = pickle.load(file)

    svc = data['svc']
    color_space = data['color_space']
    stack = data['stack']
    scaler = data['scaler']
    orient = data['orient']
    hist_bins = data['hist_bins']
    spatial_size = data['spatial_size']
    pix_per_cell = data['pix_per_cell']
    cells_per_block = data['cells_per_block']
    hog_channel = data['hog_channel']

    return color_space, svc, stack, scaler, orient, hist_bins, spatial_size, pix_per_cell, \
           cells_per_block, hog_channel


def load_test_images():
    """
    Loads the images in the test folder

    :return: An array with the test images
    """
    return glob.glob('./test_images/*.jpg')


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draws the specified boxes on an image

    :param img: The image to draw boxes on
    :param bboxes: The bounding boxes
    :param color: The color (R, G, B) values 0-255
    :param thick: The thickness of the line
    :return: The image with drawn boxes
    """
    # Make a copy of the image
    imcopy = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    # Return the image copy with boxes drawn
    return imcopy


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
    """

    :param img:
    :param windows:
    :param clf:
    :param scaler:
    :param color_space:
    :param spatial_size:
    :param hist_bins:
    :param hist_range:
    :return:
    """
    # 1) Create an empty list to receive positive detection windows
    on_windows = []

    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # 4) Extract features for that window using single_img_features()
        # features = extract_features_img(test_img, color_space, spatial_size, hist_bins, hist_range)
        features = get_features_img(test_img, color_space, spatial_size, hist_bins, orient,
                                    pix_per_cell, cell_per_block, hog_channel)
        print("feat", features.shape)

        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(-1, 1)) # (1, -1)

        # print(features.shape)
        # features = features.reshape(1, -1)
        # scaler = StandardScaler().fit(features)
        # test_features = scaler.transform(features)

        # test_features = scaler.transform(features)

        # 6) Predict using your classifier
        prediction = clf.predict(test_features)

        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    # 8) Return windows for positive detections
    return on_windows


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def search_image(img, svc, scaler, color_space, spatial_size, hist_bins,
                 orient, pix_per_cell, cell_per_block, hog_channel):
    """
    Performs a search on the image and finds all "hot" windows

    :param img: The image to search
    :return: (hot_windows) An array containing the hot windows
             (all_windows) An array with all windows
    """
    hot_windows = []
    all_windows = []

    # Define window starting/stopping points, boundaries, and overlaps
    x_start_stop = [[None, None], [None, None], [None, None], [None, None]]
    w0, w1, w2, w3 = 240, 180, 120, 70
    xy_window = [(w0, w0), (w1, w1), (w2, w2), (w3, w3)]
    o = 0.75
    xy_overlap = [(o, o), (o, o), (o, o), (o, o)]
    y0, y1, y2, y3 = 380, 380, 395, 405
    y_start_stop = [[y0, y0 + w0 / 2], [y1, y1 + w1 / 2], [y2, y2 + w2 / 2], [y3, y3 + w3 / 2]]

    # Search all windows and identify hot sections
    for i in range(len(y_start_stop)):
        windows = slide_window(img, x_start_stop[i], y_start_stop[i], xy_window[i], xy_overlap[i])

        all_windows.extend(windows)

        hot_windows += search_windows(img, windows, svc, scaler, color_space, spatial_size,
                                      hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)

    return hot_windows, all_windows


def img_process_pipeline(filename, color_space, svc, scaler, orient, hist_bins, spatial_size,
                         pix_per_cell, cells_per_block, hog_channel, saveFig=False):
    """

    :param filename:
    :param color_space:
    :param svc:
    :param scaler:
    :param orient:
    :param hist_bins:
    :param spatial_size:
    :param pix_per_cell:
    :param cells_per_block:
    :param hog_channel:
    :param saveFig:
    :return:
    """
    # Load the image
    img = mpimg.imread(filename[0])
    img = img.astype(np.float32) / 255

    # Search for hot windows at all scales
    t0 = time.time()
    hot_windows, all_windows = search_image(img, svc, scaler, color_space, spatial_size, hist_bins,
                                            orient, pix_per_cell, cells_per_block, hog_channel)
    t1 = time.time()
    print('Searched all windows in %s seconds' % round(t1 - t0, 3))

    # Draw boxes on hot areas
    img_hot_windows = draw_boxes(img, hot_windows, (0, 0, 255), 4)

    # Draw all identified windows on a separate image
    img_all_windows = np.copy(img)
    for i, window in enumerate(all_windows):
        if i == 0:
            color = (0, 0, 255)
        elif i == 1:
            color = (0, 255, 0)
        elif i == 2:
            color = (255, 0, 0)
        elif i == 3:
            color = (255, 255, 255)

    img_all_windows = draw_boxes(img_all_windows, all_windows[i], color, 4)

    # Plot and save the output
    if saveFig:
        plt.figure()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        ax1.imshow(img_hot_windows)
        ax1.set_title('Hot Windows')
        ax2.imshow(img_all_windows)
        ax2.set_title('All Windows')
        plt.savefig('output_images/sliding_window.png')

    return img_hot_windows, img_all_windows


def sliding_window_search():
    # Load the HOG parameters
    color_space, svc, stack, scaler, orient, hist_bins, spatial_size, pix_per_cell, \
    cells_per_block, hog_channel = load_hog_pickle()

    # Load test images
    images = load_test_images()

    # Test: run the pipeline on a single image
    img_hot_windows, img_all_windows = img_process_pipeline(images[2:3], color_space, svc,
                                                            scaler, orient, hist_bins, spatial_size,
                                                            pix_per_cell, cells_per_block, hog_channel,
                                                            saveFig=True)


sliding_window_search()