# Image Segmentation using U-Net on DeepGlobe Dataset

This project implements a U-Net-based convolutional neural network for satellite image segmentation. It focuses on classifying each pixel of a satellite image into land cover categories like forest, water, urban area, and more using the DeepGlobe Land Cover Classification dataset.

## Project Highlights

- Built a U-Net model from scratch using TensorFlow and Keras
- Trained on DeepGlobe land cover dataset with 7 class labels
- Applied advanced data augmentations using Albumentations
- Used a combination of focal Tversky loss and weighted categorical crossentropy to address class imbalance
- Achieved mean IoU of approximately 0.67 on validation data
- Developed a simple and interactive Streamlit app for predictions on new satellite images

## Dataset

The dataset used is the DeepGlobe Land Cover Classification Dataset available on Kaggle.

[Kaggle Dataset Link](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)

Class details:

| ID | Class Name        | RGB Color         |
|----|-------------------|-------------------|
| 0  | Urban land        | (0, 255, 255)     |
| 1  | Agriculture land  | (255, 255, 0)     |
| 2  | Rangeland         | (255, 0, 255)     |
| 3  | Forest land       | (0, 255, 0)       |
| 4  | Water             | (0, 0, 255)       |
| 5  | Barren land       | (255, 255, 255)   |
| 6  | Unknown           | (0, 0, 0)         |

## Model Overview

The U-Net model consists of:

- Encoder: Series of convolution + pooling layers to capture context
- Bottleneck: Deepest layer with 1024 filters
- Decoder: Upsampling layers with skip connections to regain spatial information
- Final layer: 1×1 convolution to classify each pixel into one of the 7 classes using softmax

Custom loss and metrics:

- Combined loss: 60% Focal Tversky loss + 40% weighted categorical crossentropy
- Metric: Mean Intersection over Union (IoU) per class

Validation IoU per class:

- Urban: 0.71
- Agriculture: 0.71
- Rangeland: 0.35
- Forest: 0.72
- Water: 0.66
- Barren: 0.57
- Unknown: 0.98

**Mean IoU:** 0.67

## Deployment

A Streamlit web application was created to test the model on user-uploaded satellite images. The app provides options to apply morphological and median postprocessing for cleaner output.

Try it online:

[Streamlit App](https://4classifying-every-pixel-of-the-planets.streamlit.app/)

## Installation

Make sure Python 3.7 or higher is installed on your system. Then, install the required packages and launch the app:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py


