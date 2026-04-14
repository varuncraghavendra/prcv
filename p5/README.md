# Project 5: Recognition using Deep Networks
Varun Raghavendra  
CS 5330 Computer Vision , Spring 2026  

---

## 1. Project Overview

This project focuses on building, training, analyzing, and modifying deep learning models for image recognition using the MNIST dataset. The objective is to understand the full deep learning pipeline, including convolutional neural networks, feature analysis, transfer learning, transformer-based models, and hyperparameter optimization.

---

## 2. Project Structure

- task1.py: Training a convolutional neural network (CNN) on MNIST and testing on a handwritten input image  
- task2.py: Analysis and visualization of learned CNN filters  
- task3.py: Transfer learning from MNIST to Greek letter classification  
- task4.py: Training and evaluation of a Vision Transformer on MNIST  
- task5.py: Systematic hyperparameter experimentation on CNN architecture  

---

## 3. Setup Instructions

Ensure Python 3 is installed.

Install required dependencies:

pip install -r requirements.txt

---

## 4. Execution Instructions

The tasks must be executed in the following order.

### Task 1: Train CNN Model (Required First Step)

python3 task1.py 5 64 IMG_0074.jpeg

This step trains the CNN on MNIST, saves the trained model, and performs inference on a handwritten input image.

---

### Task 2: Analyze CNN Filters

python3 task2.py

This step visualizes the learned convolutional filters and their effect on input images.

---

### Task 3: Transfer Learning (Greek Letters)

python3 task3.py greek_letters 100

This step adapts the pretrained MNIST model to classify Greek letters (alpha, beta, gamma) using transfer learning.

---

### Task 4: Vision Transformer

python3 task4.py 15

This step trains a Vision Transformer model on MNIST and evaluates its performance.

---

### Task 5: Hyperparameter Experimentation

python3 task5.py 5 3

This step performs a systematic exploration of CNN hyperparameters including filter size, dropout rate, and hidden layer size.

---

## 5. Output Files

All generated outputs are stored in the following directories:

outputs/  
saved_models/  

These include model weights, performance plots, and experiment results.

---

## 6. Key Concepts Covered

- Convolutional Neural Networks (CNNs)  
- Feature visualization and filter analysis  
- Transfer learning  
- Vision Transformers (ViT)  
- Hyperparameter tuning and experimental design  

---

## 7. Notes

- Task 1 must be executed before all other tasks, as it generates the pretrained model required by subsequent tasks.  
- The MNIST dataset will be automatically downloaded if not already present.  
- The implementation is designed to run on CPU; GPU acceleration is optional.  

---

## 8. Summary

This project provides a comprehensive workflow for understanding deep learning-based recognition systems. It progresses from training a baseline CNN to analyzing learned features, adapting models to new tasks, experimenting with transformer architectures, and performing systematic model optimization.
