# Brain Tumor Detection & Localization

An end-to-end AI-powered system for analyzing brain MRI images using a
two-stage deep learning pipeline. The system first determines whether a
brain tumor is present, then localizes the predicted tumor region using
image segmentation.

## Live Demo

You can try the deployed application here:

[Brain Tumor Detection & Localization App]((https://brain-tumor-detection-and-localization.streamlit.app/))

------------------------------------------------------------------------

## Project Overview

Brain tumor analysis from MRI images is an important application of
artificial intelligence in medical imaging. This project implements a
two-stage deep learning pipeline designed to analyze brain MRI scans and
provide both tumor detection and localization.

The system performs the following tasks:

1.  Classifies the MRI image as **Tumor** or **No Tumor**
2.  If a tumor is detected, a segmentation model predicts the tumor
    region
3.  Displays the predicted tumor mask and visualizes the localized tumor
    area on the original MRI image

The project combines deep learning, computer vision, image
preprocessing, classification, and semantic segmentation into a complete
interactive application.

------------------------------------------------------------------------

## System Pipeline

### Stage 1: Tumor Detection

A ResNet50-based deep learning model analyzes the input brain MRI image
and predicts whether a tumor is present.

**Output Classes:**

-   Tumor
-   No Tumor

### Stage 2: Tumor Localization

If a tumor is detected, a ResUNet-based segmentation model is used to
identify the predicted tumor region at the pixel level.

The predicted segmentation mask is then processed and overlaid on the
original MRI image to visualize the tumor location.

------------------------------------------------------------------------

## Application Workflow

``` text
Brain MRI Image
       |
       v
Image Preprocessing
       |
       v
ResNet50 Classifier
       |
       ├────────────── No Tumor
       |
       v
     Tumor
       |
       v
ResUNet Segmentation Model
       |
       v
Predicted Tumor Mask
       |
       v
Tumor Localization Visualization
```

------------------------------------------------------------------------

## Models Used

### Classification Model

The first stage uses a ResNet50-based deep learning architecture for
tumor detection.

-   Input Size: `256 × 256 × 3`
-   Architecture: ResNet50-based CNN
-   Task: Binary Classification
-   Output: Tumor / No Tumor

### Segmentation Model

The second stage uses a ResUNet-based segmentation architecture.

-   Input Size: `256 × 256 × 3`
-   Architecture: ResUNet
-   Task: Image Segmentation
-   Output: Binary Tumor Segmentation Mask

------------------------------------------------------------------------

## Features

-   Brain MRI image upload
-   Support for JPG, JPEG, and PNG images
-   AI-based tumor detection
-   Tumor probability prediction
-   Tumor localization using image segmentation
-   Predicted segmentation mask visualization
-   Tumor overlay visualization on the original MRI
-   Interactive web interface built with Streamlit

------------------------------------------------------------------------

## Technologies Used

-   Python
-   TensorFlow
-   Keras
-   Streamlit
-   OpenCV
-   NumPy
-   Pillow
-   Scikit-image
-   Matplotlib
-   Deep Learning
-   Computer Vision

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone https://github.com/amrahmedabdelrehem/Brain-Tumor-Detection-and-Localization.git
```

Move to the project directory:

``` bash
cd Brain-Tumor-Detection-and-Localization
```

Install the required dependencies:

``` bash
pip install -r requirements.txt
```

Run the Streamlit application:

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## Deployment

The application is deployed using Streamlit Community Cloud, allowing
the system to be accessed through a public web link without requiring
local installation.

------------------------------------------------------------------------

## Disclaimer

This project is developed for educational and research purposes only. It
is not intended to replace professional medical diagnosis, clinical
evaluation, or treatment decisions.

------------------------------------------------------------------------

## Author

**Amr Ahmed**

AI Engineer

-   GitHub: [amrahmedabdelrehem](https://github.com/amrahmedabdelrehem)
