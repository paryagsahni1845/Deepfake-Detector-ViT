# Deepfake vs. Real Image Detection: A Vision Transformer Approach

## Project Overview

This project presents a comprehensive, end-to-end solution for deepfake image detection using a fine-tuned Vision Transformer (ViT) model. The system is designed to be highly accurate, scalable, and user-friendly, culminating in a publicly accessible web application. The pipeline includes meticulous data preparation, advanced model training using transfer learning, robust evaluation, and a Dockerized deployment on Hugging Face Spaces.

## Core Objectives

- **Develop a High-Accuracy Model:** Create a robust deep learning model capable of distinguishing between real and deepfake images with exceptional performance.
- **Address Class Imbalance:** Systematically handle the imbalanced nature of the initial dataset through strategic oversampling to ensure a fair and effective training process.
- **Enable Scalable Deployment:** Deploy the final model as a containerized web application using Docker and Flask, hosted on Hugging Face Spaces for global accessibility.
- **Create an Intuitive User Interface:** Design a user-friendly web interface that allows for simple image uploads and provides clear, animated results with confidence scores.
- **Ensure Reproducibility:** Document and structure the project for reproducibility, including hosting the trained model on Hugging Face Hub.

## Dataset and Preprocessing

The dataset, sourced from Kaggle's "[Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)," was initially composed of 1,000 images (500 "Real" and 500 "Fake"). To mitigate class imbalance and enhance model generalization, the dataset was expanded to **76,161 images** (38,080 "Real" and 38,081 "Fake") using **manual random oversampling**.

Images were preprocessed to a uniform size of **224x224 pixels**. A critical distinction was made between training and test data augmentation:

- **Training Data:** Augmented with random rotations and sharpness adjustments to prevent overfitting.
- **Test Data:** Only underwent resizing and normalization to ensure consistent, unbiased evaluation.

## Methodology

### 1. Data Pipeline
The dataset was loaded and organized into a pandas DataFrame. Oversampling was applied, and labels were mapped to numerical values (0 for "Real," 1 for "Fake"). The dataset was then split into a 60% training set and a 40% test set using **stratification** to maintain the class distribution. A `ViTImageProcessor` was used to apply all transformations lazily via `set_transform`, optimizing memory usage.

### 2. Model Architecture and Training
The core of the system is a **Vision Transformer (ViT)** model, fine-tuned from the pretrained weights of `dima806/deepfake_vs_real_image_detection`. The model was trained with the Hugging Face `Trainer` class over **2 epochs** with a batch size of 32, a learning rate of 1e-6, and weight decay of 0.02. This transfer learning approach allowed the model to leverage existing knowledge of image features while adapting to the specific task of deepfake detection. The training process was completed in **2 hours and 6 minutes**.

### 3. Evaluation
The model's performance was rigorously evaluated on the extensive 76,161-image test set. Key metrics and insights include:

- **Accuracy:** **99.24%**
- **Loss:** **0.0229**
- **Precision:** 99.25% for "Real" and 99.22% for "Fake," indicating minimal false positives.
- **Recall:** 99.22% for "Real" and 99.25% for "Fake," indicating minimal false negatives.
- **Misclassifications:** Only ~576 images were misclassified, highlighting the model's robustness and balanced performance.

### 4. Deployment
The final model, saved as `model.safetensors`, `config.json`, and `preprocessor_config.json`, was uploaded to the Hugging Face Hub as `Sxhni/deepfake-detector-vit`. A web application was built using **Flask** and containerized with **Docker** for dependency management. The container was then deployed on Hugging Face Spaces using **Gunicorn** on port 7860, creating a scalable and accessible web service.

## Technology Stack

- **Programming Language:** Python 3.10
- **Machine Learning:** `transformers`, `torch`
- **Web Framework:** `flask`
- **Containerization:** `Docker`
- **Cloud Hosting:** Hugging Face Spaces & Hub
- **Data Handling:** `pandas`, `numpy`
- **Image Processing:** `Pillow`
- **Version Control:** `Git`
- **Development Environment:** Jupyter Notebook, VS Code

## Model Performance Results

| Metric | Real Images | Fake Images | Overall |
|--------|-------------|-------------|---------|
| **Precision** | 99.25% | 99.22% | 99.24% |
| **Recall** | 99.22% | 99.25% | 99.24% |
| **F1-Score** | 99.24% | 99.24% | 99.24% |


<img width="682" height="590" alt="image" src="https://github.com/user-attachments/assets/f7a70409-2611-4188-b3c1-dbdd59177e35" />



**Training Metrics:**
- Training Time: 2 hours 6 minutes
- Test Set Size: 76,161 images
- Misclassifications: ~582 images
- Model Parameters: 85K trainable parameters

## User Interface and Experience


<img width="1351" height="597" alt="image" src="https://github.com/user-attachments/assets/301c85a7-039f-43fd-9d22-bdcf7839b3e8" />


The web application's frontend is crafted with a modern, responsive design using HTML, CSS, and JavaScript.

### Key Features:
- **Aesthetics:** A dark gradient background with animated, interactive elements.
- **Image Upload:** A stylish, dashed-border upload area with hover effect.
- **Live Preview:** The uploaded image is displayed with zoom-on-hover effect.
- **Prediction:** A gradient "Analyze Image" button triggers prediction, showing a spinning loader.
- **Results:** The prediction and confidence score are displayed with smooth fade-in animation.

The application is fully responsive, ensuring a seamless experience across desktop and mobile devices. Predictions are returned within 5-10 seconds, providing quick feedback to the user.

## System Architecture

### Model Architecture
- **Base Model:** Vision Transformer (ViT)
- **Pre-trained Weights:** `dima806/deepfake_vs_real_image_detection`
- **Input Size:** 224x224 pixels
- **Output Classes:** 2 (Real, Fake)
- **Training Approach:** Transfer learning with fine-tuning

### Deployment Architecture
- **Backend:** Flask REST API
- **Frontend:** HTML/CSS/JavaScript responsive interface
- **Containerization:** Docker with Python 3.10-slim base image
- **Web Server:** Gunicorn
- **Hosting:** Hugging Face Spaces (port 7860)
- **Model Storage:** Hugging Face Hub

## Conclusion 

This project successfully demonstrates the power of transfer learning with Vision Transformers for deepfake detection, achieving a remarkable **99.24% accuracy**. The entire pipeline, from data sourcing to public deployment, serves as a practical example of a full-stack AI application.

## Acknowledgments
- Thanks to Kaggle for providing the "[Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)".
- Gratitude to the Hugging Face community for the `transformers` library and Spaces platform.
- Special thanks to open-source contributors for tools like Docker, Flask, and Jupyter Notebook
   
## Project Links

- **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/Sxhni/deepfakedetector-vit)
- **Model Repository:** [Hugging Face Hub](https://huggingface.co/Sxhni/deepfake-detector-vit)
- **Dataset Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
- **GitHub Repository:** [paryagsahni1845/deepfake-detector-vit](https://github.com/paryagsahni1845/deepfake-detector-vit)
