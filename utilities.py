import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K


# ============================================================
# CUSTOM FUNCTIONS FOR RESUNET MODEL
# ============================================================

def tversky(y_true, y_pred, smooth=1e-6):

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    true_positive = K.sum(y_true * y_pred)

    false_negative = K.sum(
        y_true * (1 - y_pred)
    )

    false_positive = K.sum(
        (1 - y_true) * y_pred
    )

    alpha = 0.7

    return (
        true_positive + smooth
    ) / (
        true_positive
        + alpha * false_negative
        + (1 - alpha) * false_positive
        + smooth
    )


def tversky_loss(y_true, y_pred):

    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):

    tversky_index = tversky(
        y_true,
        y_pred
    )

    gamma = 0.75

    return K.pow(
        1 - tversky_index,
        gamma
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    classifier = tf.keras.models.load_model(
        "models/classifier-resnet-compatible.h5",
        compile=False
    )

    segmentation_model = tf.keras.models.load_model(
        "models/ResUNet.keras",
        custom_objects={
            "tversky": tversky,
            "tversky_loss": tversky_loss,
            "focal_tversky": focal_tversky
        },
        compile=False
    )

    return classifier, segmentation_model


# ============================================================
# CLASSIFICATION PREPROCESSING
# ============================================================

def preprocess_for_classifier(image):

    image = cv2.resize(
        image,
        (256, 256)
    )

    image = np.array(
        image,
        dtype=np.float64
    )

    image = image * (1.0 / 255.0)

    image = np.reshape(
        image,
        (1, 256, 256, 3)
    )

    return image


# ============================================================
# SEGMENTATION PREPROCESSING
# MATCHES ORIGINAL PROJECT PIPELINE
# ============================================================

def preprocess_for_segmentation(image):

    image = cv2.resize(
        image,
        (256, 256)
    )

    image = np.array(
        image,
        dtype=np.float64
    )

    image -= image.mean()

    std = image.std()

    if std > 0:
        image /= std

    X = np.empty(
        (1, 256, 256, 3)
    )

    X[0,] = image

    return X


# ============================================================
# CLASSIFICATION PREDICTION
# ============================================================

def predict_tumor(image, classifier):

    processed_image = preprocess_for_classifier(
        image
    )

    prediction = classifier.predict(
        processed_image,
        verbose=0
    )

    probabilities = prediction[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_class]
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# SEGMENTATION MASK CLEANING
# ============================================================

def clean_segmentation_mask(predicted_mask):

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    predicted_mask = np.nan_to_num(
        predicted_mask,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    # --------------------------------------------------------
    # First, find only very high-confidence regions.
    # These regions are used as tumor seeds.
    # --------------------------------------------------------

    seed_threshold = 0.85

    seed_mask = (
        predicted_mask >= seed_threshold
    ).astype(np.uint8)


    # If there are no very confident pixels,
    # fall back to the standard threshold.
    if np.sum(seed_mask) == 0:

        seed_threshold = 0.70

        seed_mask = (
            predicted_mask >= seed_threshold
        ).astype(np.uint8)


    # --------------------------------------------------------
    # Main threshold for tumor boundaries
    # --------------------------------------------------------

    boundary_threshold = 0.60

    binary_mask = (
        predicted_mask >= boundary_threshold
    ).astype(np.uint8)


    # --------------------------------------------------------
    # Clean isolated noise
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )


    # --------------------------------------------------------
    # Find connected regions
    # --------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )


    # No detected region
    if num_labels <= 1:

        return np.zeros_like(
            binary_mask
        )


    # --------------------------------------------------------
    # Select the region that contains the highest-confidence
    # tumor prediction.
    # --------------------------------------------------------

    best_label = 0
    best_score = -1.0


    for label in range(
        1,
        num_labels
    ):

        component_mask = (
            labels == label
        )


        # Pixels from the component
        component_values = predicted_mask[
            component_mask
        ]


        # Number of strong seed pixels
        seed_count = np.sum(
            seed_mask[component_mask]
        )


        # Maximum confidence inside component
        max_confidence = np.max(
            component_values
        )


        # Mean confidence inside component
        mean_confidence = np.mean(
            component_values
        )


        # Component score prioritizes
        # high-confidence tumor regions
        score = (
            seed_count * 10
            + max_confidence * 5
            + mean_confidence
        )


        if score > best_score:

            best_score = score

            best_label = label


    # --------------------------------------------------------
    # Keep only the best region
    # --------------------------------------------------------

    cleaned_mask = np.zeros_like(
        binary_mask
    )

    if best_label != 0:

        cleaned_mask[
            labels == best_label
        ] = 1


    # --------------------------------------------------------
    # Smooth the final tumor region
    # --------------------------------------------------------

    cleaned_mask = cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )


    return cleaned_mask


# ============================================================
# SEGMENTATION PREDICTION
# ============================================================

def predict_segmentation(
    image,
    segmentation_model
):

    processed_image = preprocess_for_segmentation(
        image
    )

    prediction = segmentation_model.predict(
        processed_image,
        verbose=0
    )

    predicted_mask = prediction[
        0,
        :,
        :,
        0
    ]

    binary_mask = clean_segmentation_mask(
        predicted_mask
    )

    return (
        predicted_mask,
        binary_mask
    )


# ============================================================
# CREATE TUMOR OVERLAY
# ============================================================

def create_overlay(image, mask):

    resized_image = cv2.resize(
        image,
        (256, 256)
    )

    overlay = resized_image.copy()

    mask = mask.astype(bool)

    overlay[mask] = [
        0,
        255,
        0
    ]

    return overlay


# ============================================================
# COMPLETE ANALYSIS PIPELINE
# ============================================================

def analyze_mri(
    image,
    classifier,
    segmentation_model
):

    predicted_class, confidence, probabilities = predict_tumor(
        image,
        classifier
    )

    analysis = {
        "result": None,
        "confidence": confidence,
        "probabilities": probabilities,
        "mask": None,
        "overlay": None
    }


    # --------------------------------------------------------
    # CLASS 0 = NO TUMOR
    # CLASS 1 = TUMOR
    # --------------------------------------------------------

    if predicted_class == 0:

        analysis["result"] = "No Tumor Detected"

        return analysis


    # --------------------------------------------------------
    # RUN SEGMENTATION
    # --------------------------------------------------------

    _, binary_mask = predict_segmentation(
        image,
        segmentation_model
    )


    # --------------------------------------------------------
    # VERIFY RESULT
    # --------------------------------------------------------

    if np.sum(binary_mask) == 0:

        analysis["result"] = "No Tumor Detected"

        return analysis


    # --------------------------------------------------------
    # TUMOR DETECTED
    # --------------------------------------------------------

    analysis["result"] = "Tumor Detected"

    analysis["mask"] = binary_mask

    analysis["overlay"] = create_overlay(
        image,
        binary_mask
    )

    return analysis