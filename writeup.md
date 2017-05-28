**Vehicle Detection Project**

### By Greg Yeutter
### 2017-05-28

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
[hog_feat]: ./output_images/hog_features.png

[hot1]: ./output_images/hot1.png
[hot2]: ./output_images/hot2.png
[hot3]: ./output_images/hot3.png
[hot4]: ./output_images/hot4.png
[hot5]: ./output_images/hot5.png
[hot6]: ./output_images/hot6.png

[heat0]: ./output_images/heat0.png
[heat1]: ./output_images/heat1.png
[heat2]: ./output_images/heat2.png
[heat3]: ./output_images/heat3.png
[heat4]: ./output_images/heat4.png
[heat5]: ./output_images/heat5.png
[heat6]: ./output_images/heat6.png
[heat7]: ./output_images/heat7.png
[heat8]: ./output_images/heat8.png
[heat9]: ./output_images/heat9.png

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

The image below shows training set car and noncar images, followed by their HLS channels and HOG for each HLS channel. This confirms that the L and S channels both appear useful for HOG classification.

![hog_feat][hog_feat]


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

Sliding window search in implemented in the file `sliding_window.py`. After the settings, classifier, and images are loaded, `img_process_pipeline()` is invoked to perform the sliding window search on the image. The function `search_image()` gets "hot" windows where cars are identified. Those windows are then drawn on the original image with `draw_boxes()`. 

A streamlined version of this pipeline was also built based on the [example video](https://www.youtube.com/watch?v=P2zwrTM8ueA&feature=youtu.be&utm_medium=email&utm_campaign=2017-05-24_carnd_projectwalkthroughs&utm_source=blueshift&utm_content=2017-05-24_carnd_projectwalkthroughs&bsft_eid=809c46b1-7b0f-4960-9cc1-459c102110d5&bsft_clkid=60b7549c-754f-4c13-bb41-a3f40319287e&bsft_uid=dffaba4f-1ae5-4b6f-b338-91e83a90894a&bsft_mid=bd525702-83fa-479a-a6b7-8700d0e789b5), but some performance issues led this to be abandoned. That version is implemented in the function `img_process_pipeline_2()`.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tweaked some parameters, including color space, spatial size, and orientation. The best results were found to be:

color_space = 'HLS'
spatial_size = (32, 32)
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'

Although some false positives and negatives are present, overall, it seems to identify cars most of the time. Some examples are displayed below:

![hot1][hot1]
![hot2][hot2]
![hot3][hot3]
![hot4][hot4]
![hot5][hot5]
![hot6][hot6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's my video result [in mp4 format](./output_images/out_project_video.mp4) and [on YouTube](https://youtu.be/qZwQKitp1zc). While there are a few false positives and the occassional identification of cars in the opposing lane, the nearest cars are always identified.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Most of the functionality after sliding window search was implemented in `video_pipeline.py`. A prototype of the heatmap to vehicle detection pipeline is found in `multi_img_process_pipeline()` and then later refined in `process_image()` within `video_search()`. The class `BoundingBoxes` is used to store frame history for a smoother output. 

For each image, "hot" windows are identified using the `search_image()` function. These boxes are then turned into a heatmap using `add_heat()`, and some low heatmap values are removed using `apply_threshold()`. 

Using `scipy.ndimage.measurements.label`, the number of cars predicted in the image can be extracted from the heatmap. Using this function, new bounding boxes are drawn at the predicted car locations. This drawing is then saved to the output video.

### Here are 10 frames, with their corresponding heatmaps, predicted labels, and drawn bounding boxes:

![heat0][heat0]
![heat1][heat1]
![heat2][heat2]
![heat3][heat3]
![heat4][heat4]
![heat5][heat5]
![heat6][heat6]
![heat7][heat7]
![heat8][heat8]
![heat9][heat9]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

To recap, the general approach of this project was as follows:
* Perform HOG on a labeled dataset of cars and noncars
* Train a Linear SVM classifier with the binned color, spatial, and HOG features
* Separate the dataset into training, testing, and validation sets

Then, for each frame of the video:
* Search for windows of possible cars with a sliding window technique
* Generate a heatmap that favros overlapping windows (multiple detections = more likely a car)
* From the heatmap, extract the highest-weighted windows as cars (label)
* Draw a bounding box in the labeled position on the original frame

With a lot of parameter tweaking along the way, I was able to identify true positives most of the time and eliminate most of the false positives. 

In the [video](https://youtu.be/qZwQKitp1zc), I noticed detections of cars on the opposing lane of traffic, which may be desirable. However, I also noticed more false positives on the concrete sections of the road than the asphalt sections.

A similar issue was noticed in the [advanced lane line detection project](https://github.com/yeutterg/CarND-Advanced-Lane-Lines-P4), where results tended to be more unstable on the lighter sections of the road. This was partially mitigated with history tracking and re-searching the frame in that project, as well as parameter tuning. 

Since this project re-searches every frame but keeps a history, the vehicle tracking was maintained. However, false positives could probably be reduced in these sections with a larger training set and better processing of images into HOG features (such as adjusting saturation).
