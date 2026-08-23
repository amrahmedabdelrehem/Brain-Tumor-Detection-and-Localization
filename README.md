# Brain Tumor Detection & Localization

An end-to-end AI-powered system for analyzing brain MRI images using a two-stage deep learning pipeline. The system first determines whether a brain tumor is present, then localizes the predicted tumor region using image segmentation.

## Live Demo

You can try the deployed application here:

[Brain Tumor Detection & Localization App](YOUR_STREAMLIT-APP-LINK)

---

## Project Overview

Brain tumor analysis from MRI images is an important application of artificial intelligence in medical imaging. This project implements a two-stage deep learning pipeline designed to analyze brain MRI scans and provide both tumor detection and localization.

The system performs the following tasks:

1. Classifies the MRI image as **Tumor** or **No Tumor**
2. If a tumor is detected, a segmentation model predicts the tumor region
3. Displays the predicted tumor mask and visualizes the localized tumor area on the original MRI image

The project combines deep learning, computer vision, image preprocessing, classification, and semantic segmentation into a complete interactive application.

---

## System Pipeline

The system follows a two-stage AI pipeline:

### Stage 1: Tumor Detection

A ResNet50-based deep learning model analyzes the input brain MRI image and predicts whether a tumor is present.

**Output Classes:**

- Tumor
- No Tumor

### Stage 2: Tumor Localization

If a tumor is detected, a ResUNet-based segmentation model is used to identify the predicted tumor region at the pixel level.

The predicted segmentation mask is then processed and overlaid on the original MRI image to visualize the tumor location.

---

## Application Workflow

```text
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
