import glob
import pickle
import time

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from features import single_img_features, get_hog_features, bin_spatial, color_hist


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
                   orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
    """

    :param img:
    :param windows:
    :param clf:
    :param scaler:
    :param color_space:
    :param spatial_size:
    :param hist_bins:
    :return:
    """
    # 1) Create an empty list to receive positive detection windows
    on_windows = []

    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel)

        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # 6) Predict using your classifier
        prediction = clf.predict(test_features)

        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    # 8) Return windows for positive detections
    return on_windows


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Generates a list of windows to search

    :param img: The image
    :param x_start_stop: The start and stop positions in the x axis. Can be [None, None] for full span
    :param y_start_stop: The start and stop positions in the y axis. Can be [None, None] for full span
    :param xy_window: The size of the window in x and y
    :param xy_overlap: The overlap between x and y
    :return: A list containing all windows
    """
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

        hot_windows += search_windows(img, windows, svc, scaler, color_space, spatial_size, hist_bins,
                                      orient, pix_per_cell, cell_per_block, hog_channel)

    return hot_windows, all_windows


def convert_color(img, conv='RGB2YCrCb'):
    """
    Converts an image between color profiles

    :param img: The image
    :param conv: The conversion to make
    :return: The converted image
    """
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def visualize(fig, rows, cols, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])


def img_process_pipeline(filename, color_space, svc, scaler, orient, hist_bins, spatial_size,
                         pix_per_cell, cells_per_block, hog_channel, saveFig0=False,
                         saveFig1=False):
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
    :param saveFig0:
    :param saveFig1:
    :return:
    """
    # Load the image
    img = mpimg.imread(filename[0])
    split_filename = filename[0].split('.')
    filenum = split_filename[1][-1]
    if split_filename[2] == 'jpg':
        img = img.astype(np.float32) / 255

    # Search for hot windows at all scales
    t0 = time.time()
    hot_windows, all_windows = search_image(img, svc, scaler, color_space, spatial_size, hist_bins,
                                            orient, pix_per_cell, cells_per_block, hog_channel)
    t1 = time.time()
    print('Searched all windows in %s seconds' % (t1 - t0))

    # Draw boxes on hot areas
    img_hot_windows = draw_boxes(img, hot_windows, (0, 0, 1), 4)

    # Plot and save the output
    if saveFig0:
        # Draw all identified windows on a separate image
        img_all_windows = np.copy(img)
        for i, window in enumerate(all_windows):
            window = [window]
            img_all_windows = draw_boxes(img_all_windows, window, (0, 0, 1), 4)

        plt.figure()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        ax1.imshow(img_hot_windows)
        ax1.set_title('Hot Windows')
        ax2.imshow(img_all_windows)
        ax2.set_title('All Windows')
        outfilename = 'output_images/sliding_window' + filenum + '.png'
        print('Saving file %s.' % outfilename)
        plt.savefig(outfilename)

    if saveFig1:
        plt.figure()
        plt.imshow(img_hot_windows)
        outfilename = 'output_images/hot' + filenum + '.png'
        print('Saving file %s.' % outfilename)
        plt.savefig(outfilename)

    return img_hot_windows


def img_process_pipeline_2(images, orient, pix_per_cell, cell_per_block, spatial_size,
                           hist_bins, scaler, svc, saveFig=False):
    """
    Simplified / faster image processing pipeline

    :param images:
    :return:
    """
    out_images = []
    out_maps = []
    out_titles = []
    out_boxes = []

    ystart = 400
    ystop = 656
    scale = 1

    # Iterate over test images
    for img_src in images:
        img_boxes = []
        t = time.time()
        count = 0

        # Read in the image
        img = mpimg.imread(img_src)
        draw_img = np.copy(img)

        # Make a heatmap of zeros
        heatmap = np.zeros_like(img[:, :, 0])
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale, np.int(imshape[0]/scale))))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps
        nxblocks = (ch1.shape[1] // pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - 1
        nfeat_per_block = orient * cell_per_block**2
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                count += 1
                xpos = xb * cells_per_step
                ypos = yb * cells_per_step

                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_pred = svc.predict(test_features)

                if test_pred == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)

                    # Draw a box
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart), (0, 0, 255))
                    img_boxes.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))

                    # Draw a heatmap
                    heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1

        print(time.time()-t, 'seconds to run, total windows =', count)

        out_images.append(draw_img)

        out_titles.append(img_src[-9:])
        out_titles.append(img_src[-9:])
        out_images.append(heatmap)
        out_maps.append(heatmap)
        out_boxes.append(img_boxes)

    fig = plt.figure(figsize=(12, 24))
    visualize(fig, 8, 2, out_images, out_titles)
    plt.savefig('output_images/streamlined.png')


def sliding_window_search():
    # Load the HOG parameters
    color_space, svc, stack, scaler, orient, hist_bins, spatial_size, pix_per_cell, \
    cells_per_block, hog_channel = load_hog_pickle()

    # Load test images
    images = load_test_images()

    # Test: run the pipeline on a single image
    # img_hot_windows = img_process_pipeline(images[2:3], color_space, svc,
    #                                                         scaler, orient, hist_bins, spatial_size,
    #                                                         pix_per_cell, cells_per_block, hog_channel,
    #                                                         saveFig0=True)

    # Test: run the pipeline on all test images
    for img in images:
        img_hot_windows = img_process_pipeline([img], color_space, svc,
                                                                scaler, orient, hist_bins, spatial_size,
                                                                pix_per_cell, cells_per_block, hog_channel,
                                                                saveFig1=True)


def streamlined_search():
    # Load the HOG parameters
    color_space, svc, stack, scaler, orient, hist_bins, spatial_size, pix_per_cell, \
    cells_per_block, hog_channel = load_hog_pickle()

    # Load test images
    images = load_test_images()

    # Run the streamlined pipeline
    img_process_pipeline_2(images, orient, pix_per_cell, cells_per_block, spatial_size,
                           hist_bins, scaler, svc, saveFig=True)


# sliding_window_search()
streamlined_search()


