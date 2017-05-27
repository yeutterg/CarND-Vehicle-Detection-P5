from collections import deque
import matplotlib.image as mpimg
from sliding_window import search_image, draw_boxes, load_test_images, load_hog_pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import cv2


class BoundingBoxes:
    def __init__(self, n=10):
        # Queue to store data
        self.n = n

        # Hot windows from the last n
        self.recent_boxes = deque([], maxlen=n)

        # Current boxes
        self.current_boxes = None
        self.allboxes = []

    def add_boxes(self):
        self.recent_boxes.appendleft(self.current_boxes)

    def pop_data(self):
        if self.n_buffered > 0:
            self.recent_boxes.pop()

    def set_current_boxes(self, boxes):
        self.current_boxes = boxes

    def get_all_boxes(self):
        allboxes = []
        for boxes in self.recent_boxes:
            allboxes += boxes
        if len(allboxes) == 0:
            self.allboxes = None
        else:
            self.allboxes = allboxes

    def update(self, boxes):
        self.set_current_boxes(boxes)
        self.add_boxes()
        self.get_all_boxes()


def add_heat(heatmap, bbox_list):
    if bbox_list:
        for box in bbox_list:
            # Add heat (+1) for all pixels inside each box
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        return heatmap


def apply_threshold(heatmap, threshold):
    # Set =0 pixels that fall below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap


def multi_img_process_pipeline(images, svc, scaler, color_space, spatial_size, hist_bins,
                               orient, pix_per_cell, cell_per_block, hog_channel, saveFig=False):
    boxes = BoundingBoxes(n=3)

    for i, file in enumerate(images):
        # Open the image
        img = mpimg.imread(file)
        img = img.astype(np.float32) / 255

        draw_image = np.copy(img)

        # Get hot windows
        t0 = time.time()
        hot_windows, all_windows = search_image(img, svc, scaler, color_space, spatial_size, hist_bins,
                                                orient, pix_per_cell, cell_per_block, hog_channel)
        t1 = time.time()
        print('Searched windows in %f seconds' % (t1-t0))

        boxes.update(hot_windows)

        # Draw boxes
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=4)

        # Get the heatmap
        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
        heatmap = add_heat(heatmap, boxes.allboxes)
        heatmap = apply_threshold(heatmap, 8)

        # Identify how many cars found
        labels = label(heatmap)
        print('Found', labels[1], 'cars in image', str(i))

        # Draw labeled boxes
        box_img = np.copy(img)
        for car in range(1, labels[1]+1):
            # Get pixels that contain cars
            nonzero = (labels[0] == car).nonzero()

            # Get the x and y values
            nonzerox = np.array(nonzero[1])
            nonzeroy = np.array(nonzero[0])

            # Draw a bounding box
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(box_img, bbox[0], bbox[1], (0, 0, 255), 6)

        if saveFig:
            # Plot
            plt.figure()
            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
            f.tight_layout()
            ax1.imshow(window_img)
            ax1.set_title('Identified Windows')
            ax2.imshow(heatmap)
            ax2.set_title('Heatmap')
            ax3.imshow(labels[0], cmap='gray')
            ax3.set_title('Labeled as Cars')
            ax4.imshow(box_img)
            ax4.set_title('Image w/ Labels')

            plt.savefig('./output_images/heat' + str(i))


def video_search():
    # Load the HOG parameters
    color_space, svc, stack, scaler, orient, hist_bins, spatial_size, pix_per_cell, \
    cell_per_block, hog_channel = load_hog_pickle()

    # Load test images
    images = load_test_images()

    multi_img_process_pipeline(images, svc, scaler, color_space, spatial_size, hist_bins,
                               orient, pix_per_cell, cell_per_block, hog_channel, saveFig=True)


video_search()
