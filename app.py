import streamlit as st
import numpy as np
from PIL import Image

from utilities import load_models, analyze_mri


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Detection & Localization",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   GLOBAL THEME
============================================================ */

.stApp {
    background:
        radial-gradient(
            circle at top center,
            #102a4d 0%,
            #08182f 35%,
            #030712 75%,
            #000000 100%
        );
    color: #f8fafc;
}

.block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ============================================================
   TEXT COLORS
============================================================ */

h1,
h2,
h3,
h4,
p,
span,
label {
    color: #f8fafc;
}


/* ============================================================
   HERO SECTION
============================================================ */

.hero-section {
    background:
        linear-gradient(
            135deg,
            rgba(3, 7, 18, 0.95),
            rgba(12, 40, 75, 0.95)
        );

    border: 1px solid rgba(96, 165, 250, 0.25);

    padding: 3rem;

    border-radius: 24px;

    margin-bottom: 2rem;

    box-shadow:
        0 10px 40px rgba(0, 0, 0, 0.35);
}

.hero-title {
    color: #ffffff;

    font-size: 2.8rem;

    font-weight: 700;

    margin-bottom: 0.8rem;

    letter-spacing: -0.5px;
}

.hero-subtitle {
    color: #b6c8df;

    font-size: 1.05rem;

    line-height: 1.8;

    max-width: 850px;
}


/* ============================================================
   SECTION TITLES
============================================================ */

.section-title {
    color: #f8fafc;

    font-size: 1.55rem;

    font-weight: 700;

    margin-top: 2.2rem;

    margin-bottom: 1rem;
}


/* ============================================================
   GENERAL CARDS
============================================================ */

.custom-card,
.result-card {
    background:
        linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.92),
            rgba(8, 20, 40, 0.92)
        );

    border:
        1px solid rgba(96, 165, 250, 0.18);

    box-shadow:
        0 8px 30px rgba(0, 0, 0, 0.28);

    border-radius: 18px;
}

.custom-card {
    padding: 1.5rem;
}


/* ============================================================
   RESULT CARDS
============================================================ */

.result-card {
    padding: 1.6rem;

    text-align: center;

    min-height: 135px;
}

.result-label {
    color: #94a3b8;

    font-size: 0.9rem;

    margin-bottom: 0.8rem;
}

.result-positive {
    color: #ff5c5c;

    font-size: 1.6rem;

    font-weight: 700;
}

.result-negative {
    color: #4ade80;

    font-size: 1.6rem;

    font-weight: 700;
}

.metric-value {
    color: #f8fafc;

    font-size: 1.9rem;

    font-weight: 700;
}


/* ============================================================
   STREAMLIT METRICS
============================================================ */

[data-testid="stMetric"] {
    background:
        rgba(15, 23, 42, 0.85);

    border:
        1px solid rgba(96, 165, 250, 0.15);

    border-radius: 16px;

    padding: 1.2rem;

    box-shadow:
        0 6px 25px rgba(0, 0, 0, 0.2);
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

[data-testid="stMetricValue"] {
    color: #f8fafc;
}


/* ============================================================
   PROGRESS BARS
============================================================ */

.stProgress > div > div > div > div {
    background:
        linear-gradient(
            90deg,
            #3b82f6,
            #60a5fa
        );
}


/* ============================================================
   FILE UPLOADER - MAIN CONTAINER
============================================================ */

[data-testid="stFileUploader"] {
    background:
        linear-gradient(
            145deg,
            rgba(8, 20, 40, 0.85),
            rgba(12, 30, 58, 0.85)
        );

    border:
        1px solid rgba(96, 165, 250, 0.22);

    border-radius: 20px;

    padding: 1rem;

    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.22);
}


/* ============================================================
   UPLOAD LABEL
============================================================ */

[data-testid="stFileUploader"] label {
    color: #dbeafe !important;

    font-size: 1rem;

    font-weight: 600;

    margin-bottom: 0.8rem;
}


/* ============================================================
   DRAG AND DROP AREA
============================================================ */

[data-testid="stFileUploaderDropzone"] {
    background:
        linear-gradient(
            135deg,
            rgba(15, 35, 65, 0.95),
            rgba(20, 28, 55, 0.95)
        ) !important;

    border:
        1px dashed rgba(96, 165, 250, 0.55) !important;

    border-radius: 16px !important;

    min-height: 150px;

    transition:
        border-color 0.25s ease,
        background 0.25s ease,
        box-shadow 0.25s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color:
        rgba(96, 165, 250, 0.9) !important;

    background:
        linear-gradient(
            135deg,
            rgba(18, 45, 80, 0.98),
            rgba(28, 35, 68, 0.98)
        ) !important;

    box-shadow:
        0 0 25px rgba(59, 130, 246, 0.12);
}


/* ============================================================
   UPLOADER TEXT
============================================================ */

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #cbd5e1 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: #aabbd1 !important;
}


/* ============================================================
   UPLOAD ICON
============================================================ */

[data-testid="stFileUploaderDropzone"] svg {
    color: #60a5fa !important;

    fill: #60a5fa !important;
}


/* ============================================================
   BROWSE FILES BUTTON
============================================================ */

[data-testid="stFileUploaderDropzone"] button {
    background:
        linear-gradient(
            135deg,
            #1d4ed8,
            #2563eb
        ) !important;

    color: #ffffff !important;

    border:
        1px solid rgba(96, 165, 250, 0.6) !important;

    border-radius: 10px !important;

    padding:
        0.65rem 1.3rem !important;

    font-weight: 600 !important;

    box-shadow:
        0 5px 18px rgba(37, 99, 235, 0.25);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background:
        linear-gradient(
            135deg,
            #2563eb,
            #3b82f6
        ) !important;

    transform:
        translateY(-1px);

    box-shadow:
        0 8px 24px rgba(59, 130, 246, 0.35);
}


/* ============================================================
   UPLOADED FILE CONTAINER
============================================================ */

[data-testid="stFileUploaderFile"] {
    background:
        linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.95),
            rgba(10, 30, 55, 0.95)
        ) !important;

    border:
        1px solid rgba(96, 165, 250, 0.18) !important;

    border-radius: 14px !important;

    padding:
        0.7rem 0.9rem !important;

    margin-top:
        0.8rem !important;
}


/* ============================================================
   UPLOADED FILE TEXT
============================================================ */

[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] small,
[data-testid="stFileUploaderFile"] div {
    color: #dbeafe !important;
}


/* ============================================================
   UPLOADED FILE ICON
============================================================ */

[data-testid="stFileUploaderFile"] svg {
    color: #60a5fa !important;

    fill: #60a5fa !important;
}


/* ============================================================
   REMOVE FILE BUTTON
============================================================ */

[data-testid="stFileUploaderFile"] button {
    background:
        rgba(30, 41, 59, 0.8) !important;

    border:
        1px solid rgba(148, 163, 184, 0.2) !important;

    border-radius: 10px !important;

    color: #cbd5e1 !important;
}

[data-testid="stFileUploaderFile"] button:hover {
    background:
        rgba(127, 29, 29, 0.45) !important;

    border-color:
        rgba(248, 113, 113, 0.4) !important;
}


/* ============================================================
   SUMMARY
============================================================ */

.summary-row {
    display: flex;

    justify-content: space-between;

    gap: 1rem;

    padding: 1rem 0;

    border-bottom:
        1px solid rgba(148, 163, 184, 0.15);
}

.summary-row:last-child {
    border-bottom: none;
}

.summary-label {
    color: #94a3b8;
}

.summary-value {
    color: #f8fafc;

    font-weight: 600;

    text-align: right;
}


/* ============================================================
   PIPELINE
============================================================ */

.pipeline-container {
    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 0.6rem;

    flex-wrap: wrap;
}

.pipeline-step {
    background:
        linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.9),
            rgba(8, 20, 40, 0.9)
        );

    border:
        1px solid rgba(96, 165, 250, 0.18);

    border-radius: 16px;

    padding: 1.1rem;

    text-align: center;

    min-width: 145px;

    flex: 1;

    color: #e2e8f0;

    font-weight: 600;
}

.pipeline-arrow {
    color: #60a5fa;

    font-size: 1.5rem;

    font-weight: 700;
}


/* ============================================================
   ABOUT CARDS
============================================================ */

.about-title {
    color: #ffffff;

    font-size: 1.25rem;

    font-weight: 700;

    margin-bottom: 1rem;
}

.about-text {
    color: #aabbd1;

    line-height: 1.8;

    margin-bottom: 1rem;
}


/* ============================================================
   SIDEBAR
============================================================ */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #08182f 0%,
            #030712 100%
        );

    border-right:
        1px solid rgba(96, 165, 250, 0.15);
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}


/* ============================================================
   IMAGE CONTAINERS
============================================================ */

[data-testid="stImage"] {
    border-radius: 16px;

    overflow: hidden;

    border:
        1px solid rgba(96, 165, 250, 0.18);

    background:
        rgba(15, 23, 42, 0.8);
}


/* ============================================================
   WARNING BOX
============================================================ */

[data-testid="stAlert"] {
    border-radius: 16px;

    border:
        1px solid rgba(251, 191, 36, 0.25);

    background:
        rgba(120, 53, 15, 0.25);

    color: #fde68a;
}


/* ============================================================
   FOOTER
============================================================ */

.footer {
    text-align: center;

    color: #64748b;

    font-size: 0.9rem;

    padding-top: 2.5rem;

    padding-bottom: 1rem;

    line-height: 1.7;
}


/* ============================================================
   STREAMLIT DEFAULT ELEMENTS
============================================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ============================================================
   SCROLLBAR
============================================================ */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #030712;
}

::-webkit-scrollbar-thumb {
    background: #1e3a5f;

    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #315b8a;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def get_models():
    return load_models()


with st.spinner("Loading AI models..."):
    classifier, segmentation_model = get_models()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("System Information")

    st.markdown("---")

    st.subheader("Classification Model")

    st.write("Architecture: ResNet50")
    st.write("Input Size: 256 × 256")
    st.write("Output: Tumor / No Tumor")

    st.markdown("---")

    st.subheader("Localization Model")

    st.write("Architecture: ResUNet")
    st.write("Input Size: 256 × 256")
    st.write("Output: Tumor Segmentation Mask")

    st.markdown("---")

    st.subheader("Technologies")

    st.write("TensorFlow")
    st.write("Keras")
    st.write("OpenCV")
    st.write("Streamlit")

    st.markdown("---")

    st.caption(
        "This application demonstrates AI-based brain MRI analysis "
        "using a two-stage deep learning pipeline."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<div class="hero-section"><div class="hero-title">Brain Tumor Detection & Localization</div><div class="hero-subtitle">AI-powered analysis of brain MRI images using a two-stage deep learning pipeline. The system first detects the presence of a tumor and then localizes the predicted tumor region.</div></div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Upload MRI Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image in JPG, JPEG, or PNG format.",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)

    image_width, image_height = image.size


    with st.spinner("Analyzing MRI image..."):

        analysis = analyze_mri(
            image_np,
            classifier,
            segmentation_model
        )


    result = analysis["result"]

    confidence = analysis["confidence"]

    probabilities = analysis["probabilities"]

    no_tumor_probability = float(probabilities[0]) * 100

    tumor_probability = float(probabilities[1]) * 100


    # ========================================================
    # ANALYSIS RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">Analysis Result</div>',
        unsafe_allow_html=True
    )


    result_col, confidence_col, probability_col = st.columns(3)


    with result_col:

        if result == "Tumor Detected":

            st.markdown(
                '<div class="result-card"><div class="result-label">Detection Result</div><div class="result-positive">Tumor Detected</div></div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="result-card"><div class="result-label">Detection Result</div><div class="result-negative">No Tumor Detected</div></div>',
                unsafe_allow_html=True
            )


    with confidence_col:

        st.markdown(
            f'<div class="result-card"><div class="result-label">Classification Confidence</div><div class="metric-value">{confidence * 100:.2f}%</div></div>',
            unsafe_allow_html=True
        )


    with probability_col:

        st.markdown(
            f'<div class="result-card"><div class="result-label">Tumor Probability</div><div class="metric-value">{tumor_probability:.2f}%</div></div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY BREAKDOWN
    # ========================================================

    st.markdown(
        '<div class="section-title">Probability Breakdown</div>',
        unsafe_allow_html=True
    )


    prob_col_1, prob_col_2 = st.columns(2)


    with prob_col_1:

        st.metric(
            "No Tumor Probability",
            f"{no_tumor_probability:.2f}%"
        )

        st.progress(
            min(
                max(no_tumor_probability / 100, 0.0),
                1.0
            )
        )


    with prob_col_2:

        st.metric(
            "Tumor Probability",
            f"{tumor_probability:.2f}%"
        )

        st.progress(
            min(
                max(tumor_probability / 100, 0.0),
                1.0
            )
        )


    # ========================================================
    # MRI ANALYSIS RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-title">MRI Analysis Results</div>',
        unsafe_allow_html=True
    )


    if analysis["mask"] is not None:

        image_col, overlay_col = st.columns(2)


        with image_col:

            st.image(
                image,
                caption="Original Brain MRI",
                use_container_width=True
            )


        with overlay_col:

            st.image(
                analysis["overlay"],
                caption="AI Predicted Tumor Localization",
                use_container_width=True
            )


        # ====================================================
        # PREDICTED MASK
        # ====================================================

        st.markdown(
            '<div class="section-title">Predicted Tumor Mask</div>',
            unsafe_allow_html=True
        )


        mask_col_1, mask_col_2, mask_col_3 = st.columns(
            [1, 2, 1]
        )


        with mask_col_2:

            st.image(
                analysis["mask"] * 255,
                caption="AI Predicted Segmentation Mask",
                use_container_width=True
            )


        # ====================================================
        # TUMOR AREA CALCULATION
        # ====================================================

        mask = analysis["mask"]

        tumor_pixels = np.sum(mask)

        total_pixels = mask.size

        tumor_area_percentage = (
            tumor_pixels / total_pixels
        ) * 100

        localization_status = "Tumor Region Identified"


    else:

        image_col_1, image_col_2, image_col_3 = st.columns(
            [1, 2, 1]
        )


        with image_col_2:

            st.image(
                image,
                caption="Uploaded Brain MRI",
                use_container_width=True
            )


        tumor_area_percentage = 0.0

        localization_status = "No Localization Required"


    # ========================================================
    # ANALYSIS SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">Analysis Summary</div>',
        unsafe_allow_html=True
    )


    summary_html = (
        '<div class="custom-card">'
        '<div class="summary-row">'
        '<span class="summary-label">Detection Result</span>'
        f'<span class="summary-value">{result}</span>'
        '</div>'
        '<div class="summary-row">'
        '<span class="summary-label">Classification Confidence</span>'
        f'<span class="summary-value">{confidence * 100:.2f}%</span>'
        '</div>'
        '<div class="summary-row">'
        '<span class="summary-label">Localization Status</span>'
        f'<span class="summary-value">{localization_status}</span>'
        '</div>'
        '<div class="summary-row">'
        '<span class="summary-label">Estimated Tumor Area</span>'
        f'<span class="summary-value">{tumor_area_percentage:.2f}%</span>'
        '</div>'
        '<div class="summary-row">'
        '<span class="summary-label">Original Image Size</span>'
        f'<span class="summary-value">{image_width} × {image_height}</span>'
        '</div>'
        '</div>'
    )

    st.markdown(
        summary_html,
        unsafe_allow_html=True
    )


# ============================================================
# DETECTION PIPELINE
# ============================================================

st.markdown(
    '<div class="section-title">How the System Works</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="pipeline-container">'
    '<div class="pipeline-step">Brain MRI Image</div>'
    '<div class="pipeline-arrow">→</div>'
    '<div class="pipeline-step">ResNet50 Classification</div>'
    '<div class="pipeline-arrow">→</div>'
    '<div class="pipeline-step">Tumor Detection</div>'
    '<div class="pipeline-arrow">→</div>'
    '<div class="pipeline-step">ResUNet Segmentation</div>'
    '<div class="pipeline-arrow">→</div>'
    '<div class="pipeline-step">Tumor Localization</div>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ABOUT THIS SYSTEM
# ============================================================

st.markdown(
    '<div class="section-title">About This System</div>',
    unsafe_allow_html=True
)


about_col_1, about_col_2 = st.columns(2)


with about_col_1:

    st.markdown(
        '<div class="custom-card"><div class="about-title">Detection Model</div><div class="about-text">The classification stage uses a ResNet50-based deep learning model to analyze brain MRI images and predict whether a tumor is present.</div><div class="about-text">Input Size: 256 × 256 × 3</div><div class="about-text">Output Classes: Tumor / No Tumor</div></div>',
        unsafe_allow_html=True
    )


with about_col_2:

    st.markdown(
        '<div class="custom-card"><div class="about-title">Localization Model</div><div class="about-text">The second stage uses a ResUNet-based segmentation model to identify the predicted tumor region at the pixel level.</div><div class="about-text">Input Size: 256 × 256 × 3</div><div class="about-text">Output: Binary Tumor Segmentation Mask</div></div>',
        unsafe_allow_html=True
    )


# ============================================================
# IMPORTANT NOTICE
# ============================================================

st.markdown(
    '<div class="section-title">Important Notice</div>',
    unsafe_allow_html=True
)


st.warning(
    """
    This application is developed for educational and research purposes.
    The results generated by this system should not be considered a medical
    diagnosis or used as a substitute for professional medical advice.
    Clinical decisions should always be made by qualified healthcare
    professionals.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">Brain Tumor Detection & Localization<br><br>AI-powered MRI analysis using ResNet50 and ResUNet</div>',
    unsafe_allow_html=True
)