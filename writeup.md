##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hist_cars]: ./output_images/preprocess_hist_cars.png
[hist_noncars]: ./output_images/preprocess_hist_noncars.png
[car_nocar]: ./output_images/preprocess_car_vs_noncar.png

[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Preprocessing

Udacity provided [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) datasets, which were downloaded and extracted into the project directory.

The preprocessing of the vehicle and non-vehicle datasets was performed in `dataset_preprocessing.py`. The images from various folders were combined into one of two arrays: `cars` and `noncars` using the function `load_images()`. 

An example of a car vs. non-car image:

![car vs. noncar][car_nocar]

The function `train_valid_test_split()` was used to shuffle and separate the cars and noncars datasets into training, validation, and test sets. The distributions were, respectively, 70%, 20% and 10%. The quantities for each dataset were similar:

`Cars: Total: 8792, Train: 6154, Test: 879, Valid: 1758`

`Noncars: Total: 8968, Train: 6277, Test: 896, Valid: 1794`

Bar graphs were also output to verify similar distribution between cars and non-cars:

![cars histogram][hist_cars]
![noncars_histogram][hist_noncars]

The training, test, and validation data were saved in a pickle file for ease of use later.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Extraction and exploration of HOG features is performed in the file `hog.py`.

I started by loading the training, validation, and test data for cars and non-cars in the method `load_data_pickle()`. 

For each of the 6 datasets (cars training, cars validation, cars test, noncars training, noncars validation, noncars test), I extracted HOG features. This is performed in `get_features_all_datasets()`. The settings I settled upon were:

```
# Parameters
color_space = 'HLS'
spatial_size = (32, 32)
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
```

I then processed the data into X_train, X_valid, X_test, y_train, y_valid, and y_test. This was performed in `generate_train_valid_test()`.


#### 2. Explain how you settled on your final choice of HOG parameters.

The parameters listed above seem to give the best performance.

For color space, it appears that a significant amount of information can be extracted using HLS. Specifically, the L channel is quite detailed, and the S channel provides some additional information. Subjetively, the H channel does not appear to provide much information.

Other color spaces do not provide consistent results. RGB performs inconsistently depending on the lighting, and YCrCb does not seem to provide as much information as HLS. LUV and HSV seem to provide similar results to HLS, so these could probably be used interchangeably.

Modifying the other values did not lead to results as good as those suggested in the project into. So, I ended up keeping the defaults:

```
orient = 9
pix_per_cell = 8
cell_per_block = 2
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear support vector classifier from the scikit-learn package, as found in the function `sv_classifier()` in `hog.py`. Using the training, validation, and test sets generated earlier, I obtained a validation accuracy of 99.3% and a test accuracy of 99.3%. This was performed using all three HLS channels, although I suspect that the L and S channels were contributing most of the information. These were the settings used:

```
spatial_size = (32, 32)
hist_bins = 32
hog_channel = 'ALL'
```


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

