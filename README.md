# Plant Disease Detection using Image Segmentation

## Project Overview

This project implements a deep learning-based image classification system to detect 10 different tomato leaf diseases using a dataset of 16,012 images. The system employs Convolutional Neural Networks (CNN) for automated plant disease diagnosis, combining computer vision and deep learning techniques to assist farmers and researchers in monitoring crop health efficiently.

## Dataset

- **Source**: Plant Village dataset
- **Size**: 16,012 images of tomato leaves
- **Classes**: 10 different types of tomato leaf diseases:
  - Tomato Target Spot
  - Tomato Mosaic Virus
  - Tomato Yellow Leaf Curl Virus
  - Tomato Bacterial Spot
  - Tomato Early Blight
  - Tomato Healthy
  - Tomato Late Blight
  - Tomato Leaf Mold
  - Tomato Septoria Leaf Spot
  - Tomato Spider Mites (Two-spotted spider mite)

## Technical Implementation

### Preprocessing Pipeline
1. **Image Loading**: Load and process images from directory structure
2. **Grayscale Conversion**: Convert RGB images to single-channel grayscale
3. **Histogram Equalization**: Enhance contrast for better feature detection
4. **Pixel Normalization**: Scale pixel values to range [0, 1]
5. **Resizing**: Standardize all images to consistent dimensions (32x32 pixels)
6. **Train-Test Split**: 80-20 split for training and evaluation

### Model Architecture
- **Framework**: PyTorch with GPU acceleration support
- **Convolutional Layers**: Multiple Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling2D for dimensionality reduction
- **Regularization**: Dropout layers to prevent overfitting
- **Classification**: Fully connected Dense layers with softmax output
- **Optimization**: Adam optimizer with appropriate loss function

### Key Features
- End-to-end deep learning pipeline
- Image loading and preprocessing
- Label encoding and dataset management
- Model training and validation
- Comprehensive evaluation metrics

## Model Workflow

1. **Data Preparation**
   - Load dataset from directory structure
   - Resize images to consistent dimensions
   - Convert to appropriate tensor format
   - Normalize pixel values
   - Encode categorical labels

2. **Model Development**
   - Build CNN architecture using PyTorch
   - Configure training parameters
   - Implement training loop with validation

3. **Evaluation & Analysis**
   - Calculate model accuracy
   - Generate classification report (precision, recall, F1-score)
   - Create confusion matrix for performance visualization
   - Plot training/validation loss curves

## Evaluation Metrics

The project provides comprehensive evaluation including:
- **Model Accuracy**: Overall classification performance
- **Classification Report**: Detailed precision, recall, and F1-scores per class
- **Confusion Matrix**: Visual representation of classification performance
- **Loss Curves**: Training and validation loss progression

## Applications

- Automated plant disease diagnosis and classification
- Precision agriculture and crop management
- Agricultural research and development
- Farm monitoring systems
- Early disease detection and prevention

This project demonstrates a practical application of deep learning in agriculture, providing an effective solution for automated plant disease detection that can significantly improve crop management practices and agricultural productivity.

## Screenshots


<img width="1059" height="763" alt="image" src="https://github.com/user-attachments/assets/fce53dee-760f-445f-a2ad-96cf9aa29663" />

<img width="1143" height="734" alt="image" src="https://github.com/user-attachments/assets/3e364289-ba11-42e7-b0d7-5bc682a42421" />

<img width="1084" height="750" alt="image" src="https://github.com/user-attachments/assets/3cc1ba7f-1d52-46e6-9768-6defbb5c5dff" />
