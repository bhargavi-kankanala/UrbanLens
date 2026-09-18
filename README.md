


# UrbanLens: AI-Powered Bitemporal Satellite Change Detection

UrbanLens is an AI-powered web application for detecting changes in urban areas using satellite images captured at two different points in time.

The system uses a **Siamese U-Net** deep learning architecture to compare two satellite images of the same geographical area and generate a pixel-level change map. The detected changes are visualized through an overlay on the second satellite image, along with an approximate percentage of the image area predicted as changed.

Live : https://urbanlens.streamlit.app/

---

## 👥 Team

This is a group project developed by:

- **Dakuri Manasa**
- **Gotte KavyaSri**
- **Kankanala Bhargavi**
- **Badavath Anjali**

---

## 🌍 Project Overview

Urban areas continuously change due to construction, development, expansion, and other human activities. Monitoring these changes using satellite imagery can help provide useful information about urban development.

UrbanLens focuses on **bitemporal satellite image change detection**.

The user provides:

- **Time 1:** Satellite image captured earlier
- **Time 2:** Satellite image captured later

The system processes both images and predicts the regions where changes have occurred.

### Workflow

```text
Time 1 Satellite Image
          │
          ▼
    Shared Encoder
          │
          │
          ├──────────────┐
          │              │
          ▼              ▼
    Features A      Features B
          │              │
          └──────┬───────┘
                 │
          Feature Difference
                 │
                 ▼
           U-Net Decoder
                 │
                 ▼
        Predicted Change Map
                 │
                 ▼
          Change Overlay
                 │
                 ▼
      Approx. Changed Area (%)
````

---

## 🎯 Objectives

The main objectives of UrbanLens are:

1. Detect changes between two satellite images captured at different times.
2. Identify the spatial regions where changes have occurred.
3. Generate a pixel-level binary change map.
4. Visualize detected changes using an overlay.
5. Calculate the approximate percentage of image area predicted as changed.
6. Provide an easy-to-use web interface for demonstrating satellite change detection.
7. Evaluate the model using standard change-detection metrics.

---

## 🧠 Proposed Approach

UrbanLens uses a **Siamese U-Net** architecture.

### Siamese Network

The two satellite images are passed through the same encoder with shared weights.

This allows the model to extract comparable features from both time periods.

```text
Image A ──► Shared Encoder ──► Features A
                                      │
                                      │ Difference
                                      ▼
Image B ──► Shared Encoder ──► Features B
```

The absolute difference between corresponding feature maps is used to represent changes between the two images.

### U-Net Decoder

The extracted differences are passed to a U-Net-style decoder.

The decoder gradually reconstructs the spatial resolution and produces a pixel-level change map.

```text
Two Images
    ↓
Shared Siamese Encoder
    ↓
Feature Differences
    ↓
U-Net Decoder
    ↓
Binary Change Map
```

The final output contains:

* **Black pixels:** No predicted change
* **White pixels:** Predicted change

---

## 🏗️ Model Architecture

The current implementation uses a standard **Siamese U-Net** architecture without an attention mechanism.

### Encoder

The encoder contains four convolutional blocks:

```text
Input: 3 × 256 × 256

↓
DoubleConv
64 channels

↓ Max Pooling
128 channels

↓ Max Pooling
256 channels

↓ Max Pooling
512 channels
```

The same encoder is shared between the two input images.

### Feature Difference

For each encoder level:

```text
Difference = |Features_Time1 - Features_Time2|
```

The resulting feature differences are passed to the decoder.

### Decoder

The decoder uses:

* Transposed convolution for upsampling
* Skip connections
* Double convolution blocks

The final layer produces a single-channel binary change prediction.

---

## 📊 Dataset

UrbanLens was trained using the **LEVIR-CD dataset**.

LEVIR-CD is a bitemporal remote sensing change-detection dataset containing pairs of very-high-resolution satellite images.

The dataset contains:

* **637 image pairs**
* Image size: **1024 × 1024**
* Spatial resolution: approximately **0.5 m/pixel**
* RGB satellite imagery
* Binary change labels
* Images captured several years apart
* Building-related urban changes

### Dataset Structure

The dataset used for training follows this structure:

```text
LEVIR-CD/
│
├── train/
│   ├── A/
│   ├── B/
│   └── label/
│
├── val/
│   ├── A/
│   ├── B/
│   └── label/
│
└── test/
    ├── A/
    ├── B/
    └── label/
```

Where:

* `A` → Earlier image / Time 1
* `B` → Later image / Time 2
* `label` → Ground-truth change mask

The original 1024 × 1024 images are resized to **256 × 256** during model training and inference.

---

## 🧹 Dataset Validation

Before training, the dataset structure and image pairs were checked.

The validation included:

* Matching filenames between A, B, and label folders
* Missing image checks
* Image size checks
* Corrupt/unreadable image checks
* Label value verification

The dataset used for training contained no missing pairs or unreadable images during these checks.

---

## ⚙️ Training Configuration

### Input Size

```text
256 × 256
```

### Batch Size

```text
4
```

### Optimizer

```text
Adam
```

### Learning Rate

```text
0.0001
```

### Loss Function

```text
BCEWithLogitsLoss
```

### Prediction Threshold

```text
0.5
```

### Training Strategy

The model was trained with early stopping based on validation IoU.

The best checkpoint was selected based on **validation IoU** rather than simply using the final training epoch.

---

## 📈 Model Performance

The current best checkpoint was obtained at **Epoch 9**.

### Validation Performance

| Metric    |  Score |
| --------- | -----: |
| IoU       | 62.14% |
| Precision | 76.10% |
| Recall    | 76.85% |
| F1-score  | 76.29% |

### Test Performance

The final evaluation was performed on the unseen test set.

| Metric         |      Score |
| -------------- | ---------: |
| IoU            | **56.51%** |
| Precision      | **72.32%** |
| Recall         | **69.34%** |
| F1-score       | **69.93%** |
| Pixel Accuracy | **97.75%** |

> Pixel accuracy is reported for completeness. Since most pixels in a change-detection image are typically unchanged, IoU, Precision, Recall, and F1-score provide more informative measures of change detection performance.

---

## 📐 Evaluation Metrics

### IoU

Intersection over Union measures the overlap between the predicted change region and the ground-truth change region.

```text
IoU = Intersection / Union
```

A higher IoU indicates better overlap between prediction and ground truth.

### Precision

Precision measures how many pixels predicted as changed were actually changed.

```text
Precision = TP / (TP + FP)
```

### Recall

Recall measures how many of the actual changed pixels were successfully detected.

```text
Recall = TP / (TP + FN)
```

### F1-score

F1-score provides a balance between precision and recall.

```text
F1 = 2 × Precision × Recall
          ───────────────────
          Precision + Recall
```

### Pixel Accuracy

Pixel accuracy measures the percentage of correctly classified pixels.

However, because unchanged pixels usually dominate the image, accuracy alone can be misleading for change detection.

---

## 🖥️ Web Application

UrbanLens is deployed as a **Streamlit web application**.

The application allows users to upload two satellite images:

```text
┌─────────────────────┐
│       Time 1        │
│   Satellite Image   │
└─────────────────────┘

┌─────────────────────┐
│       Time 2        │
│   Satellite Image   │
└─────────────────────┘
```

The model then generates:

### 1. Predicted Change

A binary change map showing the regions predicted as changed.

### 2. Change Overlay

The predicted changed regions are highlighted on the Time 2 satellite image.

### 3. Approximate Changed Area

The application calculates:

```text
Changed Pixels
──────────────── × 100
Total Pixels
```

This represents the **approximate percentage of the image area predicted as changed**.

It should not be interpreted as an exact geographic land-area measurement.

---

## 🌐 Live Demo

**UrbanLens Live Application:**

[https://urbanlens.streamlit.app/](https://urbanlens.streamlit.app/)

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch

### Computer Vision / Image Processing

* Pillow
* NumPy

### Web Application

* Streamlit

### Dataset

* LEVIR-CD

### Development Environment

* Google Colab
* Visual Studio Code

### Version Control

* Git
* GitHub

---

## 📁 Project Structure

```text
UrbanLens/
│
├── app.py
├── model.py
├── requirements.txt
├── urbanlens_siamese_unet.pth
├── test_model.py
└── README.md
```

### File Description

| File                         | Description                        |
| ---------------------------- | ---------------------------------- |
| `app.py`                     | Streamlit web application          |
| `model.py`                   | Siamese U-Net model architecture   |
| `requirements.txt`           | Python dependencies                |
| `urbanlens_siamese_unet.pth` | Trained model weights              |
| `test_model.py`              | Local model loading/inference test |
| `README.md`                  | Project documentation              |

---

## 🚀 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/ManasaDakuri/UrbanLens.git
```

### 2. Navigate to the project

```bash
cd UrbanLens
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔍 How to Use

1. Open the UrbanLens application.
2. Upload the earlier satellite image as **Time 1**.
3. Upload the corresponding later satellite image as **Time 2**.
4. Run the change detection.
5. View the predicted change map.
6. View the change overlay.
7. Check the approximate changed-area percentage.

For meaningful predictions, the two images should represent the **same geographical area at different points in time**.

---

## 🏙️ Potential Applications

UrbanLens can support applications such as:

* Urban development monitoring
* Building change detection
* Monitoring urban expansion
* Infrastructure development analysis
* Satellite-based geographical change monitoring
* Supporting urban planning workflows

The system is intended as a **decision-support and visualization tool**, rather than a replacement for domain experts.

---

## ⚠️ Current Limitations

The current version has several limitations:

1. The model is trained on the LEVIR-CD dataset and focuses primarily on building-related changes.
2. The model performs binary change detection rather than identifying the semantic type of every change.
3. The input images are resized from 1024 × 1024 to 256 × 256 for inference.
4. The reported changed-area percentage is an image-pixel percentage, not an exact geographic area.
5. Performance can vary depending on image quality, scene characteristics, and similarity between the input image pair.
6. The current model is a baseline Siamese U-Net and is not claimed to be state-of-the-art.

---

## 🔮 Future Scope

Possible future improvements include:

* Training with larger and more diverse datasets
* Improving spatial resolution during inference
* Experimenting with advanced feature extraction techniques
* Exploring attention-based architectures
* Semantic change classification
* Improved visualization and reporting
* Integration with geographic information systems
* More robust handling of different satellite sensors and image conditions
* Cloud-based scalable inference

---

## 📚 References

### LEVIR-CD Dataset

Chen, Hao, et al.
**"Revisiting Change Detection in VHR Remote Sensing Images: A Large-Scale Dataset and a Deep Learning Benchmark."**

LEVIR-CD project repository:

[https://github.com/justchenhao/LEVIR](https://github.com/justchenhao/LEVIR)

### Siamese Attention U-Net

Cummings, Sol, Lukas Kondmann, and Xiao Xiang Zhu.
**"Siamese Attention U-Net for Multi-Class Change Detection."**
IGARSS 2022.

---

## 📜 Project Status

**Current Status: Deployed**

The current version includes:

* ✅ LEVIR-CD dataset preparation
* ✅ Siamese U-Net implementation
* ✅ Model training
* ✅ Validation and testing
* ✅ IoU, Precision, Recall and F1 evaluation
* ✅ Change-map visualization
* ✅ Change overlay
* ✅ Approximate changed-area calculation
* ✅ Streamlit web application
* ✅ GitHub repository
* ✅ Online deployment

---

## 👥 Team

**UrbanLens — Group Project**

* Dakuri Manasa
* Gotte KavyaSri
* Kankanala Bhargavi
* Badavath Anjali

