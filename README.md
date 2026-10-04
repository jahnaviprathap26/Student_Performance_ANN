# Student Performance Prediction Using Artificial Neural Network (ANN)

**Soft Computing Mini-Project**

---

## 📌 Project Overview

This project implements an end-to-end Machine Learning / Soft Computing application to predict student academic performance using an **Artificial Neural Network (ANN)** built with TensorFlow/Keras, Scikit-learn, Pandas, NumPy, and Streamlit.

The application uses the actual `student_performance.csv` dataset to perform:
1. **Binary Classification**: Predicts whether a student will `PASS` or `FAIL` (`passed` target: Yes/No).
2. **Continuous Score Regression**: Predicts the student's estimated `final_score` (out of 100).
3. **Interactive Streamlit Web Dashboard**: Allows users to enter student details, receive instant real-time predictions, view true model evaluation graphs, and study for Soft Computing viva exams.

---

## 📂 Project Structure

```
Student_Performance_ANN/
├── student_performance.csv  # Actual dataset (500 records, 11 attributes)
├── train_model.py           # Pipeline to preprocess data, train ANN, evaluate & save models
├── app.py                   # Streamlit web application
├── requirements.txt         # Dependencies list
├── README.md                # Project documentation & Viva Guide
└── models/                  # Directory containing trained model artifacts (generated upon training)
    ├── ann_classifier.keras # Trained TensorFlow ANN Classifier
    ├── ann_regressor.keras  # Trained TensorFlow ANN Regressor
    ├── preprocessor.joblib  # Scikit-Learn ColumnTransformer scaler/encoder
    └── metrics.json         # Genuine evaluation metrics & training history
```

---

## 📊 Dataset Features & Target Variables

- **Input Features ($X$)**:
  - `gender`: Categorical (`Male`, `Female`)
  - `age`: Discrete numeric (15–19 years)
  - `study_hours_per_week`: Continuous numeric (2.0–30.0 hrs)
  - `attendance_rate`: Continuous numeric percentage (50.2%–100.0%)
  - `parent_education`: Categorical ordinal (`None`, `High School`, `Bachelor`, `Master`, `PhD`)
  - `internet_access`: Categorical binary (`Yes`, `No`)
  - `extracurricular`: Categorical binary (`Yes`, `No`)
  - `previous_score`: Continuous numeric score (30–95)

- **Target Variables ($y$)**:
  - `passed` (Classification Target): `Yes` (1) / `No` (0)
  - `final_score` (Regression Target): Marks scored out of 100 (20–95)

- **Dropped Attribute**:
  - `student_id`: Primary key string (non-predictive metadata)

---

## 🚀 How to Run the Application

### Step 1: Install Dependencies
Open your terminal in this project directory and run:
```bash
pip install -r requirements.txt
```

### Step 2: Train the Artificial Neural Network
Run the model training script:
```bash
python train_model.py
```
This script will:
- Load `student_performance.csv`
- Preprocess data with `StandardScaler` and `OneHotEncoder`
- Train ANN classifier & regressor models using TensorFlow/Keras
- Evaluate test set performance and output real accuracy/metrics
- Save model files to the `models/` directory

### Step 3: Run the Streamlit Web App
Launch the web interface:
```bash
streamlit run app.py
```
Or with Python launcher:
```bash
py -3.10 -m streamlit run app.py
```

---

## 🎓 Soft Computing Viva Concepts & Q&A

### 1. What is an Artificial Neural Network (ANN)?
An ANN is a soft computing model inspired by biological neural networks. It consists of interconnected processing nodes (neurons) arranged in layers (Input, Hidden, and Output) that learn mathematical mappings between input features and target outputs using backpropagation.

### 2. Neural Network Architecture Used
- **Input Layer**: Dimension = 12 (4 standardized numerical features + 8 one-hot encoded categorical features).
- **Hidden Layer 1**: `Dense(32 neurons)` + `BatchNormalization()` + `Dropout(0.2)` + `ReLU`
- **Hidden Layer 2**: `Dense(16 neurons)` + `BatchNormalization()` + `Dropout(0.1)` + `ReLU`
- **Hidden Layer 3**: `Dense(8 neurons)` + `ReLU`
- **Output Layer**:
  - Classification: `Dense(1 neuron)` with `Sigmoid` activation function.
  - Regression: `Dense(1 neuron)` with `Linear` activation function.

### 3. Activation Functions Used
- **ReLU (Rectified Linear Unit)**: $f(z) = \max(0, z)$. Used in hidden layers to introduce non-linearity and prevent vanishing gradient problems.
- **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$. Used in the classification output layer to constrain values between 0 and 1 (probabilities).

### 4. Loss Functions & Optimization
- **Binary Cross-Entropy Loss**: Used for binary classification:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Mean Squared Error (MSE)**: Used for score regression:
  $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$
- **Adam Optimizer**: Adaptive Moment Estimation algorithm used for backpropagation weight updates.
