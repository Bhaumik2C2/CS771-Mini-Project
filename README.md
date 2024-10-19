# CS771-Mini-Project
# Emotion Classification using SimpleRNN

This project demonstrates an emotion classification system using SimpleRNN and binary classification techniques, implemented in Keras/TensorFlow. The model predicts emoticon-based sentiments from sequences of emoticons using a reduced number of trainable parameters to improve efficiency.

## Project Overview

The primary objective of this project is to classify emoticon sequences into binary labels (e.g., positive vs. negative emotions). A reduced SimpleRNN model is utilized to handle sequential emoticon data, minimizing the model's complexity while maintaining classification performance. Hyperparameter tuning was performed to find the best model configuration.

## Model Architecture

The architecture of the model includes:
1. **Embedding Layer:** To convert emoticon tokens into dense vectors.
2. **SimpleRNN Layer:** A simplified recurrent layer for sequential data processing. Unlike LSTM, SimpleRNN reduces the number of trainable parameters, which makes it more computationally efficient.
3. **Dropout Layers:** Used to avoid overfitting by randomly dropping neurons during training.
4. **Dense Layers:** Fully connected layers with ReLU activation for feature extraction.
5. **Sigmoid Output Layer:** A final layer for binary classification.

**Hyperparameters:**
- LSTM Units: 56
- Dense Units: 52
- Dropout Rate: 0.3

## Hyperparameter Tuning

The following parameters were tuned for performance:
- **LSTM Units:** The number of units in the SimpleRNN layer.
- **Dense Units:** The number of neurons in the dense layer.
- **Dropout Rate:** The dropout rate between layers.

The script iterates through these configurations, training the model on different subsets of the data, to find the best performing model.

## Data Preparation

### Dataset
The dataset consists of sequences of emoticons, where each sequence has a corresponding binary label representing the emotion (e.g., happy or sad).

### Preprocessing
1. **Emoticon Splitting:** The input emoticon sequence is split into individual characters.
2. **Tokenization:** Each character (emoticon) is tokenized into an index for the embedding layer.
3. **Padding:** Sequences are padded to a fixed length (13 characters) for uniformity.
4. **Label Encoding:** The labels are already binary, so no further encoding is necessary.

## Training and Evaluation

The model is trained on various percentages of the data (from 20% to 100%) to analyze how much data is required for optimal performance. Accuracy is measured on the validation set after each training cycle.

**Training Process:**
- Number of epochs: 10
- Batch size: 32
- Loss function: Binary Crossentropy
- Optimizer: Adam
- Metric: Accuracy

Validation accuracy is calculated after each training phase to identify the best model parameters.

## Performance Analysis

The model was evaluated using the `accuracy_score` metric from the `sklearn` library. The best-performing model was achieved using the following configuration:
- **LSTM Units:** 56
- **Dense Units:** 52
- **Dropout Rate:** 0.3
- **Data Percentage:** 100%

### Best Accuracy:
The model achieved a best accuracy of `<Best Accuracy>` on the validation set.


