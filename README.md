# 🧠 Brain Tumor Detection & Confidence-Aware Follow-Up Analysis
## 📌 Project Overview

This project presents an  Brain Tumor Decision Support System that goes beyond basic tumor detection.
In addition to predicting tumor type and confidence score from MRI images, the system supports confidence-aware follow-up analysis and provides tumor-specific educational information and non-medical precautions.

Unlike typical brain tumor detection projects that focus only on single-image classification, this system is designed to simulate realistic clinical workflows, including initial screening and follow-up evaluation.

## 🎯 Key Features
### 1️⃣ Brain Tumor Classification

Predicts tumor type from MRI images:

No Tumor

Glioma

Meningioma

Pituitary Tumor

Uses a CNN model based on VGG architecture

Displays model confidence score for transparency

### 2️⃣ Prediction Confidence Score

Shows how confident the model is about the predicted tumor type

Helps assess prediction reliability

Confidence is treated as model certainty, not medical risk

### 3️⃣ Confidence-Aware Follow-Up Analysis (Unique Feature ⭐)

#### Allows comparison between:

Previous MRI scan

Current MRI scan

Compares prediction confidence only when tumor types match

#### Highlights:

Increase in confidence

Decrease in confidence

No significant change

⚠️ Important:
This comparison reflects changes in model certainty due to MRI appearance variation, not tumor improvement or worsening.

### 4️⃣ Tumor-Specific Information Panel

For each predicted tumor type, the system displays:

General educational information

Non-medical precautions and lifestyle guidance

This feature improves interpretability and user awareness while maintaining ethical boundaries.

### 5️⃣ Clear Medical Disclaimer

The system is explicitly designed as a decision support tool, not a diagnostic system.
All medical interpretations are avoided, and disclaimers are clearly shown in the UI.

## 🧠 System Workflow

Upload MRI image

Preprocess image (resize, normalization, VGG preprocessing)

CNN predicts tumor type and confidence

### Optional follow-up:

Upload previous and current MRI scans

Compare confidence scores

### Display:

Prediction results

Confidence trend

Tumor-specific information and precautions

## 🛠️ Technologies Used

Python

TensorFlow / Keras

VGG-based CNN

Streamlit (UI & deployment)

NumPy, Pandas

PIL (Image Processing)

## 🖥️ Application Pages
### 🔹 Prediction Page

Upload MRI image

View tumor type and confidence score

View tumor-specific educational information and precautions

### 🔹 Confidence Comparison Page

Upload previous and current MRI scans

Compare confidence scores

Observe prediction stability trends

Includes safety warnings and disclaimers

## ⚠️ Disclaimer

This application is intended for educational and research purposes only.
It does not provide medical diagnosis, treatment recommendations, or clinical decisions.
Final medical decisions should always be made by qualified healthcare professionals.
## 📈 Future Enhancements

Tumor segmentation for size-based progression analysis

Patient history–aware longitudinal tracking

Deployment as a web-based clinical support prototype
