# Advanced Medical Image Processing Tool 🏥🔍

Welcome to the **Advanced Medical Image Processing Tool**—an intuitive, powerful, and user-friendly application crafted to streamline medical image analysis for healthcare professionals, researchers, and enthusiasts alike. Using this tool, you can easily analyze and enhance medical images with just a few clicks, gaining deep insights and clarity for your images. Whether you’re working with X-rays, MRIs, or any grayscale medical imaging, this tool is designed to make complex processes accessible and interactive.

## 🌟 Features

The application offers a full suite of advanced image processing functionalities to measure, enhance, and segment images. Here’s what you can do:

### 1. Image Quality Measures 📏
Assess the quality of medical images using a variety of contrast measurement techniques. This can be useful for determining image clarity and suitability for analysis.
- **Global Contrast (Michelson)**: Measures the contrast based on intensity range, giving insight into overall image clarity.
- **RMS Contrast**: Computes the contrast using the Root Mean Square method, which helps to identify the spread of pixel intensity.
- **Entropy**: Calculates the complexity of pixel intensity distribution, an essential metric in assessing image information richness.

### 2. Image Enhancement Techniques ✨
Enhancing medical images can bring out subtle details that are crucial for diagnostic accuracy. Choose from:
- **Histogram Equalization**: Redistributes pixel intensities, making hidden details more visible by balancing image brightness.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: An advanced method that prevents excessive brightness in localized areas, ideal for bringing out fine textures in complex images.

### 3. Noise Reduction Techniques 🧹
Noise in medical images can obscure important details. The tool offers:
- **Gaussian Blur (Linear Filtering)**: Smooths the image, reducing noise while maintaining essential details.
- **Median Filtering**: Reduces noise by replacing each pixel with the median of surrounding pixels, preserving edges effectively.

### 4. Edge Enhancement Techniques 🖊️
Edge detection highlights structures and boundaries, aiding in medical image analysis:
- **Laplacian Edge Enhancement**: Emphasizes regions where pixel intensity changes, enhancing the edges.
- **Sobel Edge Enhancement**: Calculates gradients in both horizontal and vertical directions, allowing fine edge detection.

### 5. Segmentation Techniques ✂️
Image segmentation isolates regions of interest, making it easier to analyze specific structures:
- **Threshold Segmentation**: A quick, efficient way to separate foreground and background.
- **Watershed Segmentation**: A sophisticated technique ideal for separating overlapping structures by treating intensity as a topographic surface.

## 🚀 Getting Started

Ready to dive into image processing? Here’s how to set up the tool and get started.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Advanced-Medical-Image-Processing-Tool.git
   cd Advanced-Medical-Image-Processing-Tool
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

Once the setup is complete, launch the Streamlit app:

```bash
streamlit run MIP_FISAC.py
```

Upload a medical image in `.png`, `.jpg`, or `.jpeg` format and explore the options in the side panel. 

## 🎛️ How It Works

This tool provides a guided interface where you can:

1. **Upload an Image**: Start by uploading a grayscale medical image. The tool will display the image for easy reference.
2. **Select Region of Interest (ROI)**: Use the sliders to define a specific region to analyze, zooming in on areas with high diagnostic value.
3. **Choose Processing Techniques**: Select from a range of quality measures, enhancement options, noise reduction, edge detection, and segmentation methods to transform the image. The tool then processes the image in real time, displaying each transformation.
4. **Download Processed Images**: Save any version of the image that’s been processed for use in reports, presentations, or further analysis.

## 🧑‍🔬 Example Workflow

Here’s an example of how you might use this tool in a typical analysis workflow:

1. **Upload an MRI scan**: Open an MRI scan, adjust the region of interest to focus on a specific area.
2. **Measure Contrast**: Calculate the RMS contrast to assess the scan quality.
3. **Enhance the Image**: Apply CLAHE to bring out finer details.
4. **Reduce Noise**: Use Median Filtering to smooth out unwanted noise.
5. **Edge Detection**: Apply Sobel Edge Enhancement to highlight structural boundaries.
6. **Segmentation**: Finish with Watershed Segmentation to isolate key structures.

## 🛠️ Technology Stack

The following libraries power this tool:
- **Streamlit**: Provides a responsive and interactive user interface.
- **OpenCV**: Performs image processing and computer vision tasks.
- **NumPy**: Enables efficient numerical operations on image data.

## 🚀 Future Directions

We’re excited about the potential for this tool and have plans to introduce:
- **Color Image Support**: Extend functionality to color medical images, enabling analysis on more complex datasets.
- **Additional Processing Techniques**: Incorporate methods like Fourier Transform analysis and multi-scale image processing.
- **3D Visualization**: Explore support for viewing and processing volumetric (3D) medical imaging data.

## 📜 License

This project is open-source and available under the MIT License.

## 🙌 Acknowledgments

This tool was developed with a focus on simplicity, utility, and accuracy to help the medical community leverage digital tools in image analysis. Your feedback and suggestions are always welcome!

