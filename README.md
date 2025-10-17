# Brain Tumor Detection Using Deep Learning on MRI Images



## Overview
This project uses deep learning to classify brain tumors (Pituitary, Glioma, Meningioma, or No Tumor) from MRI images. Leveraging **VGG16 transfer learning** and a custom CNN architecture, the model achieves **95% accuracy**.

## Features
- VGG16-based transfer learning for accurate tumor classification
- Data preprocessing and augmentation for better generalization
- Custom CNN layers with dropout for regularization
- Visualization of training metrics using Matplotlib
- Detection interface to classify new MRI scans with confidence scores

## Technologies
Python, TensorFlow/Keras, OpenCV, Matplotlib

## Installation
```bash
git clone https://github.com/Kaushalraj27/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
pip install -r requirements.txt
