# SONAR Rock vs Mine Prediction

## Overview
This project implements a machine learning model to classify SONAR signals as either **rock** or **mine** using **Logistic Regression**. The dataset contains 208 samples with 60 numerical features representing frequency response values.

## Dataset
- **Source**: SONAR dataset
- **Features**: 60 continuous numerical values representing frequency responses
- **Target Variable**:
  - `R` (Rock)
  - `M` (Mine)

## Steps in the Project
### 1. Data Loading and Preprocessing
- The dataset is loaded using **pandas**.
- The last column (target) is separated from the features.
- The dataset is checked for missing values and data types.

### 2. Training and Testing Split
- The dataset is split into **80% training data** and **20% test data** using `train_test_split`.
- Stratified sampling is used to maintain class balance.

### 3. Model Training
- **Logistic Regression** is used for classification.
- The model is trained using the training dataset.

### 4. Model Evaluation
- Predictions are made on the training set.
- Accuracy is calculated using `accuracy_score`.
- The model achieved an accuracy of **84.3%** on training data.

### 5. Making Predictions
- The model is tested with a single input instance.
- If the prediction is `'R'`, it classifies as **Rock**; if `'M'`, it classifies as **Mine**.

## Requirements
To run this project, you need the following libraries:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Running the Project
1. Load the dataset: `pd.read_csv('sonar.csv', header=None)`
2. Preprocess the data and split into train/test.
3. Train the model using:
```python
model = LogisticRegression()
model.fit(x_train, y_train)
```
4. Evaluate the model accuracy:
```python
y_pred = model.predict(x_train)
accuracy = accuracy_score(y_pred, y_train)
print(f"Accuracy: {accuracy}")
```
5. Predict for a new input:
```python
input_data = np.array([...])
input_data_reshaped = input_data.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(f"Prediction: {prediction}")
```

## Conclusion
This project successfully classifies SONAR signals with a simple **Logistic Regression** model. Further improvements can be made by testing with different algorithms like **SVM, Decision Trees, or Neural Networks**.

---
🚀 **Future Work:** Try implementing different feature selection methods or hyperparameter tuning for better accuracy!

