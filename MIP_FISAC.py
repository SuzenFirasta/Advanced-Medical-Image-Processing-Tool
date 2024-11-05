import streamlit as st
import numpy as np
import cv2

# Quality measures calculation functions
def michelson_contrast(image):
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    return (max_intensity - min_intensity) / (max_intensity + min_intensity)

def rms_contrast(image):
    mean_intensity = np.mean(image)
    return np.sqrt(np.mean((image - mean_intensity) ** 2))

def entropy_contrast(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    histogram = histogram / histogram.sum()
    entropy = -np.sum([p * np.log2(p) for p in histogram if p > 0])
    return entropy

# Enhancement and edge detection functions
def histogram_equalization(image):
    return cv2.equalizeHist(image)

def clahe_enhancement(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def laplacian_edge_enhancement(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    edge_enhanced = cv2.convertScaleAbs(laplacian)
    return edge_enhanced

def sobel_edge_enhancement(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction
    sobel_combined = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
    return sobel_combined

# Noise reduction functions
def gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def median_blur(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

# Segmentation functions
def threshold_segmentation(image, threshold=128):
    _, segmented = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return segmented

def watershed_segmentation(image):
    # Watershed requires a marker-based segmentation technique
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
    segmented = np.where(markers == -1, 255, 0).astype(np.uint8)
    return segmented

# Streamlit app
st.title("Advanced Medical Image Processing Tool")

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload a medical image (e.g., X-ray, MRI) in grayscale", type=['png', 'jpg', 'jpeg'])

# ROI selection settings
st.sidebar.header("Region-of-Interest (ROI) Selection")
select_roi = st.sidebar.checkbox("Enable ROI Selection")
x_start = st.sidebar.slider("X Start", min_value=0, max_value=512, value=0, step=1)
x_end = st.sidebar.slider("X End", min_value=0, max_value=512, value=512, step=1)
y_start = st.sidebar.slider("Y Start", min_value=0, max_value=512, value=0, step=1)
y_end = st.sidebar.slider("Y End", min_value=0, max_value=512, value=512, step=1)

# Quality Measures
with st.expander("1. Image Quality Measures"):
    quality_option = st.selectbox(
        "Select a Quality Measure:",
        ("None", "Global Contrast (Michelson)", "RMS Contrast", "Entropy")
    )

# Enhancement Techniques
with st.expander("2. Image Enhancement Techniques"):
    enhancement_option = st.selectbox(
        "Select an Image Enhancement Technique:",
        ("None", "Histogram Equalization", "CLAHE Enhancement")
    )

# Noise Reduction Techniques
with st.expander("3. Noise Reduction Techniques"):
    noise_reduction_option = st.selectbox(
        "Select a Noise Reduction Technique:",
        ("None", "Gaussian Blur (Linear Filtering)", "Median Filtering")
    )

# Edge Enhancement Techniques
with st.expander("4. Edge Enhancement Techniques"):
    edge_option = st.selectbox(
        "Select an Edge Enhancement Technique:",
        ("None", "Laplacian Edge Enhancement", "Sobel Edge Enhancement")
    )

# Segmentation Techniques
with st.expander("5. Segmentation Techniques"):
    segmentation_option = st.selectbox(
        "Select a Segmentation Technique:",
        ("None", "Threshold Segmentation", "Watershed Segmentation")
    )

if uploaded_file is not None:
    # Load image as grayscale
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # Apply ROI selection if enabled
    if select_roi:
        image = image[y_start:y_end, x_start:x_end]
        st.image(image, caption="ROI Selected Image", use_column_width=True)

    # Display the original image
    st.image(image, caption='Original Image', use_column_width=True)

    # Quality Measure Processing
    if quality_option != "None":
        if quality_option == "Global Contrast (Michelson)":
            contrast_value = michelson_contrast(image)
        elif quality_option == "RMS Contrast":
            contrast_value = rms_contrast(image)
        elif quality_option == "Entropy":
            contrast_value = entropy_contrast(image)
        st.write(f"{quality_option}: {contrast_value:.4f}")

    # Image Enhancement
    enhanced_image = image
    if enhancement_option != "None":
        if enhancement_option == "Histogram Equalization":
            enhanced_image = histogram_equalization(image)
        elif enhancement_option == "CLAHE Enhancement":
            enhanced_image = clahe_enhancement(image)
        st.image(enhanced_image, caption=f'{enhancement_option} Result', use_column_width=True)

    # Noise Reduction
    noise_reduced_image = enhanced_image
    if noise_reduction_option != "None":
        if noise_reduction_option == "Gaussian Blur (Linear Filtering)":
            noise_reduced_image = gaussian_blur(enhanced_image)
        elif noise_reduction_option == "Median Filtering":
            noise_reduced_image = median_blur(enhanced_image)
        st.image(noise_reduced_image, caption=f'{noise_reduction_option} Result', use_column_width=True)

    # Edge Enhancement
    edge_enhanced_image = noise_reduced_image
    if edge_option != "None":
        if edge_option == "Laplacian Edge Enhancement":
            edge_enhanced_image = laplacian_edge_enhancement(noise_reduced_image)
        elif edge_option == "Sobel Edge Enhancement":
            edge_enhanced_image = sobel_edge_enhancement(noise_reduced_image)
        st.image(edge_enhanced_image, caption=f'{edge_option} Result', use_column_width=True)

    # Segmentation Techniques
    segmented_image = edge_enhanced_image
    if segmentation_option != "None":
        if segmentation_option == "Threshold Segmentation":
            segmented_image = threshold_segmentation(edge_enhanced_image)
        elif segmentation_option == "Watershed Segmentation":
            segmented_image = watershed_segmentation(edge_enhanced_image)
        st.image(segmented_image, caption=f'{segmentation_option} Result', use_column_width=True)
