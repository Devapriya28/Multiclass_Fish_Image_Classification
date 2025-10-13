# Multiclass_Fish_Image_Classification

## 📘 Project Overview

This project focuses on building and deploying a Multiclass Fish Image Classification system using deep learning.
The goal is to classify fish images into multiple categories using both a custom CNN model and transfer learning with pre-trained models.

The project also demonstrates how to:

Preprocess and augment image data.

Train, evaluate, and compare multiple deep learning models.

Deploy a Streamlit web app that predicts the fish category from user-uploaded images.

## 🌍 Domain

Image Classification

## 🧠 Skills Gained

Deep Learning, Python, TensorFlow/Keras, Streamlit, Data Preprocessing, Transfer Learning, Model Evaluation, Visualization, and Model Deployment

## 🧩 Approach
1️⃣ Data Preprocessing & Augmentation

Load dataset using TensorFlow’s ImageDataGenerator.

Rescale images to [0,1].

Apply augmentation (rotation, flipping, zooming, shifting).

2️⃣ Model Training

Train a custom CNN from scratch.

Fine-tune five pre-trained models:

VGG16

ResNet50

MobileNet

InceptionV3

EfficientNetB0

Save the best-performing model (.h5 or .pkl).

3️⃣ Model Evaluation

Compute metrics:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Visualize:

Training & Validation Accuracy

Training & Validation Loss

4️⃣ Deployment

Build a Streamlit web app that:

Allows users to upload images.

## 🧠 Technologies Used

Predicts and displays the fish category.
| Category      | Tools / Libraries   |
| ------------- | ------------------- |
| Programming   | Python              |
| Deep Learning | TensorFlow, Keras   |
| Visualization | Matplotlib, Seaborn |
| Deployment    | Streamlit           |
| Evaluation    | Scikit-learn        |

Shows confidence scores for each prediction.

## 🚀 Streamlit Deployment

App Features:

✅ Upload a fish image
✅ Predict the fish category
✅ View prediction confidence scores
✅ Simple and interactive UI

## 🏁 Conclusion

This project demonstrates how transfer learning significantly boosts performance in image classification tasks.
By integrating deep learning, data preprocessing, and deployment, it forms a complete end-to-end ML pipeline for real-world applications.

