
# Random Forest Classifier for Emotion Classification

This project implements a Random Forest classifier to predict emoticon-based sentiments from a dataset. The model evaluates performance using different proportions of the training data, enabling insights into how training data size affects model accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Performance Analysis](#performance-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The primary objective of this project is to classify emoticon sequences into binary labels (e.g., positive vs. negative emotions) using a Random Forest classifier. The model evaluates its performance across various training data proportions to analyze the relationship between the amount of training data and the validation accuracy.

## Data Preparation

### Dataset
The dataset consists of sequences of emoticons, where each sequence is paired with a corresponding binary label representing the emotion (e.g., happy or sad).

### Preprocessing
1. **Splitting the Data:** The dataset is split into training and validation sets.
2. **Feature Selection:** The emoticon sequences are processed and vectorized for model training.

## Training and Evaluation

The Random Forest model is trained on varying percentages of the training dataset (from 20% to 100%) to evaluate how the size of the training data affects model performance. The model's accuracy is calculated on a separate validation set after each training phase.

### Training Process:
- **Model:** RandomForestClassifier
- **Number of estimators:** 100
- **Random state:** 42
- **Validation Accuracy Measurement:** Calculated after each training phase.

The results are printed, including validation accuracy, confusion matrix, and classification report for each proportion of the training data used.

## Performance Analysis

The model's performance is assessed using the `accuracy_score`, `confusion_matrix`, and `classification_report` from the `sklearn` library. The validation accuracy is recorded for each proportion of the training data, allowing for a visual representation of accuracy against the percentage of training data used.

## Installation

### Requirements
- Python 3.7+
- Scikit-learn
- NumPy
- Matplotlib
- Pandas

### Setup
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Prepare your dataset and ensure it's in the correct format for training and validation.

## Usage

### Training the Model
To run the training and evaluation, use the following script:
```bash
python train_random_forest.py
