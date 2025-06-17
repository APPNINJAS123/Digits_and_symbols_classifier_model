# 🧠 Handwritten Digit & Symbol Recognition using CNN
A beginner-friendly deep learning project that uses a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) and *mathematical symbols (+, -, , /) from 28x28 grayscale images.

# 🔍 Project Overview
This project was built to:

✅ Generate synthetic training data using OpenCV.
✅ Train a CNN model to classify digits and symbols.
✅ Predict characters from new test images.
⚠️ Experiment with evaluating full arithmetic expressions (like 3+4) — partial success, still under improvement.

# 📸 Dataset
Images: Generated programmatically using OpenCV.
Classes:
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, +, -, *, /
→ Total: 14 classes
Image Size: 28x28 pixels, grayscale.
Variations: Random font styles, thicknesses, and positions for better generalization.

# 🏗️ Model Architecture
Built using TensorFlow Keras, the CNN includes:

Convolutional layers
MaxPooling layers
Fully connected (dense) layers
Output layer with softmax activation
The model is trained to predict 1 of 14 classes.

# ⚙️ Workflow
## 1. Data Generation
Automatically creates thousands of labeled digit and symbol images.

## 2. Model Training
Trained for multiple epochs with validation.
Uses categorical_crossentropy loss and Adam optimizer.

## 3.Prediction (Single Character)
Load the model.
Preprocess test image: grayscale → threshold → resize → normalize.
Predict the class using the CNN.

## 4. (Experimental) Expression Evaluation
Attempted to segment and solve full math expressions like 3+4.
Works occasionally, but needs better segmentation logic.

# 🧠 What I Learned
Creating synthetic datasets with OpenCV.
Building and training CNNs from scratch.
Preprocessing images for ML input.
Basic image segmentation techniques using contours.

# 🚀 Future Improvements
Improve symbol segmentation from expressions.
Add more variation to training data.
Train on real handwritten samples (e.g., MNIST + symbol dataset).
Build a web interface or app for handwriting recognition.

# 📌 Note
This is a learning project. The arithmetic evaluator (expression solving) is still a work-in-progress, and the model may misclassify overlapping or tightly spaced characters.

# 💬 Acknowledgements
Thanks to the open-source Python, TensorFlow, and OpenCV communities.

## 🛠️ Built with curiosity and code by [AppNinjas123](https://github.com/AppNinjas123)
