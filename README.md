# Pedestrian Detection using OpenCV

This repository contains a Jupyter Notebook (`Pedestrian_Detection.ipynb`) that demonstrates how to perform pedestrian detection using OpenCV. The code leverages OpenCV's pre-trained HOG (Histogram of Oriented Gradients) and SVM (Support Vector Machine) based pedestrian detection model to detect pedestrians in images and videos.

## Overview

The notebook is designed to work in a Google Colab environment, which allows you to run the code directly in the cloud without needing to set up a local environment. The code uses the `cv2_imshow` function from the `google.colab.patches` module to display images directly in the notebook.

### Key Features:
- **Pedestrian Detection**: The code uses OpenCV's pre-trained HOG + SVM model to detect pedestrians in images.
- **Image Display**: The `cv2_imshow` function is used to display images with detected pedestrians.
- **Google Colab Integration**: The notebook is optimized for Google Colab, making it easy to run and experiment with the code.

## Requirements

To run this notebook, you need the following:

- **Google Colab**: The notebook is designed to run in Google Colab. You can open it directly in Colab by uploading the notebook to your Google Drive or by using the provided link.
- **OpenCV**: The code uses OpenCV for image processing and pedestrian detection. OpenCV is pre-installed in Google Colab, so you don't need to install it separately.

## How to Use

1. **Open in Google Colab**: Click on the "Open in Colab" button (if available) or upload the notebook to your Google Colab environment.
2. **Run the Notebook**: Execute each cell in the notebook sequentially. The notebook will guide you through the process of loading an image, detecting pedestrians, and displaying the results.
3. **Experiment**: Feel free to modify the code, try different images, or adjust the parameters of the pedestrian detection model to see how it affects the results.

## Code Structure

The notebook is structured as follows:

1. **Importing Libraries**: The necessary libraries, including OpenCV and `cv2_imshow`, are imported.
2. **Loading the Pedestrian Detection Model**: The pre-trained HOG + SVM model is loaded using OpenCV.
3. **Detecting Pedestrians**: The code processes an image to detect pedestrians and draws bounding boxes around them.
4. **Displaying Results**: The image with detected pedestrians is displayed using `cv2_imshow`.

## Example

Here’s a brief example of how the pedestrian detection works:

```python
from google.colab.patches import cv2_imshow
import cv2

# Load the pre-trained HOG + SVM pedestrian detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load an image
image = cv2.imread('path_to_image.jpg')

# Detect pedestrians in the image
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

# Draw bounding boxes around detected pedestrians
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the image with detected pedestrians
cv2_imshow(image)
