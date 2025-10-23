# Brain Tumor Detection Using VGG16 on MRI Images



## Overview
This project uses deep learning to classify brain tumors (Pituitary, Glioma, Meningioma, or No Tumor) from MRI images. Leveraging **VGG16 transfer learning** and a custom CNN architecture, the model achieves **95% accuracy**.

## Features
- VGG16-based transfer learning for accurate tumor classification
- Data preprocessing and augmentation for better generalization
- Custom CNN layers with dropout for regularization
- Visualization of training metrics using Matplotlib
- Detection interface to classify new MRI scans with confidence scores

## Technologies
Python, TensorFlow/Keras, Transfer Learning(VGG16), Matplotlib, Seaborn

## Installation
```bash
git clone https://github.com/Kaushalraj27/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
pip install -r requirements.txt

```
---
# 🧠 Brain Tumor Detection using CNN

This project detects brain tumors from MRI images using a **Convolutional Neural Network (CNN)** built with **Keras** and **TensorFlow**.  
It classifies MRI scans into four categories:
- **Pituitary**
- **Glioma**
- **No Tumor**
- **Meningioma**
<img width="1501" height="690" alt="image" src="https://github.com/user-attachments/assets/67ff0072-ab04-4f94-a932-caac68395632" />



```


## 📁 Project Structure
brain-tumor-detection/
│
├── brain-tumor-detection.ipynb     # Main Jupyter Notebook
├── model/
│   └── tumor_detection_model.h5    # Trained CNN model
├── test_images/                    # Folder for MRI test images
├── utils/
│   └── detect_and_display.py       # Tumor detection function
└── README.md                       # Project documentation

````


 **Create and activate virtual environment**

   ```
   python -m venv venv
   source venv/bin/activate    # Mac/Linux
   venv\Scripts\activate       # Windows
   ```
**Install dependencies**

   ```
   pip install -r requirements.txt
   ```

If you don’t have `requirements.txt`, install manually:

```
pip install tensorflow keras numpy matplotlib
```

---

## 🚀 Usage

1. **Run the Jupyter Notebook**

   ```
   jupyter notebook brain-tumor-detection.ipynb
   ```

2. **Use the function for detection**

   ```python
   from keras.models import load_model
   from utils.detect_and_display import detect_and_display

   model = load_model('model/tumor_detection_model.h5')
   img_path = 'test_images/sample1.jpg'

   detect_and_display(img_path, model)
   ```

---

## 🧩 Function Description

### `detect_and_display(img_path, model)`

Loads an MRI image, processes it, and predicts whether a tumor is present.

**Steps:**

1. Loads and resizes image to (128,128)
2. Normalizes pixel values to [0,1]
3. Predicts tumor type using CNN
4. Displays image with prediction and confidence score

**Example Output:**


<img width="637" height="524" alt="image" src="https://github.com/user-attachments/assets/ec9b1e24-b62f-42de-be57-f1d84ded6d6e" />


---

## 📊 Model Details

* **Architecture:** CNN (3 Conv layers + Dense layers)
* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam
* **loss:** sparse_categorical_crossentropy
* **metrics:** sparse_categorical_accuracy

---
## Train and Val Plots
<img width="1074" height="498" alt="image" src="https://github.com/user-attachments/assets/c7e46663-308c-46a8-85de-89037c2c121b" />

## Confusion Matrix
<img width="950" height="691" alt="image" src="https://github.com/user-attachments/assets/3d6594d5-552c-4f44-a08d-239bafdccdd9" />


## 👨‍💻 Author

**Kaushal Raj**
B.Tech (ECE), Gati Shakti Vishwavidyalaya
Focused on **AI & Computer Vision in Healthcare**


