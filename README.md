# 🧠 Multimodal Brain Tumor Classifier: End-to-End Deep Learning Pipeline

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://brain-tumor-classifier-cv.streamlit.app/)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)

## 🚀 Project Overview

This project is an end-to-end machine learning operations (MLOps) pipeline that classifies brain tumors from both MRI and CT medical imaging scans. Built as a proof-of-concept for deploying medical imaging models in constrained compute environments, it demonstrates the full lifecycle of a deep learning project—from data ingestion and modular refactoring to cloud deployment, containerization, and automated CI/CD.

> **Note:** This is an educational prototype and portfolio project. It is **NOT** intended for clinical diagnostic use.

## 🔗 Live Demo

[**Test the application live on Streamlit Community Cloud**](https://brain-tumor-classifier-cv.streamlit.app/)

## 🔑 Key Features (Phase 2 Complete)

- **Multimodal Inference:** Dynamically supports 4-class MRI classification and binary CT scan classification via a unified Streamlit interface.
- **Production DevOps:** Fully containerized via Docker for reproducible local execution, integrated with a GitHub Actions CI pipeline for automated syntax and smoke testing.
- **Modular Architecture:** Core machine learning logic transitioned from exploratory Colab notebooks into a robust, production-grade script architecture (`src/`).
- **Transfer Learning Engine:** Utilizes fine-tuned ResNet50 architectures to achieve high recall on tumor classification despite limited training datasets.
- **Decoupled Architecture:** Model weights are stored externally (via Google Drive and `gdown`) to keep the GitHub repository lightweight and adhere to standard Git limits.

## 📊 Datasets

The models were trained on datasets sourced from Kaggle:

- **MRI Dataset:** ~3,264 images across 4 Classes (Glioma, Meningioma, Pituitary Tumor, and No Tumor).
- **CT Dataset:** Multimodal Image dataset processed for binary classification (Healthy vs. Tumor).
- **Preprocessing:** Images resized to 224x224, normalized to ImageNet standards, and augmented with random horizontal flips and rotations to prevent overfitting.

## 📈 Model Evaluation & Results (MRI Baseline)

The model's primary evaluation metric is **Recall (Sensitivity)**, as minimizing False Negatives is the most critical constraint in medical imaging contexts.

![Brain Tumor Classification - Confusion Matrix](assets/confusion_matrix.png) _(Note: Ensure your image path matches the assets folder)_

- **Best Validation Accuracy:** 97.45%
- **Optimizer:** Adam (lr=0.0001)
- **Loss Function:** CrossEntropyLoss

## 📁 Repository Structure

```text
brain-tumor-classifier/
├── .devcontainer/         # Dev container configurations for consistent local environments
├── .github/workflows/     # CI pipeline configurations (GitHub Actions)
├── assets/                # Documentation assets and evaluation graphics
├── data/                  # Local data storage directory
├── models/                # Local model weights storage directory
├── notebooks/             # Colab notebooks for EDA, initial training, and evaluation
├── src/                   # Production modular Python scripts (dataset.py, model.py, train.py)
├── .gitignore             # Git ignore rules
├── Dockerfile             # Production environment blueprint for containerization
├── README.md              # Project documentation
├── app.py                 # Multimodal Streamlit web application script
├── requirements.txt       # Python dependencies (includes gdown, torch, torchvision, etc.)
├── run_ct_training.sh     # Bash script to execute CT scan model training
└── run_training.sh        # Bash script to execute MRI scan model training
```

## 💻 How to Run Locally

You can run this application using either a standard Python virtual environment or Docker.

### Option A: Standard Python Setup

1. Clone the repository

```bash
git clone https://github.com/muhammadabdurrehmanmaqsood/brain-tumor-classifier.git
cd brain-tumor-classifier
```

2. Install dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app

```bash
streamlit run app.py
```

> Note: The first time you run the app, the `gdown` library will automatically download the heavy `.pth` model weights files from cloud storage.

### Option B: Docker Setup (Recommended)

Ensure you have Docker Desktop installed and running.

1. Build the image

```bash
docker build -t brain-tumor-app .
```

2. Run the container

```bash
docker run -p 8501:8501 brain-tumor-app
```

Navigate to http://localhost:8501 in your browser.

## 🔮 Future Work (Phase 3)

~~Multi-Modality: Expand the data pipeline to ingest and classify CT scans.~~ (Completed)

~~Modular Refactoring: Transition core notebook logic into modular Python scripts.~~ (Completed)

Explainable AI (XAI): Integrate Grad-CAM functionality into the Streamlit UI to highlight the specific spatial regions of the scans that triggered the model's predictions.
