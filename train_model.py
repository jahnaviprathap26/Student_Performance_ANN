"""
Student Performance Prediction - ANN Model Training Script
Soft Computing Mini-Project

This script:
1. Loads student_performance.csv
2. Preprocesses input features (numerical & categorical)
3. Trains genuine Artificial Neural Networks (ANN):
   - Binary Classifier (predicting 'passed': Yes/No)
   - Continuous Regressor (predicting 'final_score': 0-100)
4. Evaluates genuine model performance on test set
5. Saves models, preprocessor, and metrics to disk
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def main():
    print("============================================================")
    print("Starting Student Performance ANN Model Training Pipeline...")
    print("============================================================")

    # 1. Load Dataset
    csv_path = "student_performance.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset '{csv_path}' not found in current directory!")

    df = pd.read_csv(csv_path)
    print(f"[+] Loaded dataset successfully: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 2. Select Features & Targets
    feature_cols = [
        'gender', 'age', 'study_hours_per_week', 'attendance_rate',
        'parent_education', 'internet_access', 'extracurricular', 'previous_score'
    ]

    X = df[feature_cols].copy()
    y_clf = (df['passed'].str.strip() == 'Yes').astype(int)
    y_reg = df['final_score'].astype(float)

    print(f"\nTarget Distributions:")
    print(f" - Classification ('passed'): Pass={y_clf.sum()} ({(y_clf.mean()*100):.1f}%), Fail={(len(y_clf)-y_clf.sum())}")
    print(f" - Regression ('final_score'): Min={y_reg.min()}, Max={y_reg.max()}, Mean={y_reg.mean():.2f}, Std={y_reg.std():.2f}")

    num_cols = ['age', 'study_hours_per_week', 'attendance_rate', 'previous_score']
    cat_cols = ['gender', 'parent_education', 'internet_access', 'extracurricular']

    # 3. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )
    print(f"\nTrain set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples.")

    # 4. Data Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Target Scaler for Regression
    y_scaler = StandardScaler()
    y_reg_train_scaled = y_scaler.fit_transform(y_reg_train.values.reshape(-1, 1)).ravel()

    input_dim = X_train_proc.shape[1]
    print(f"[+] Preprocessing completed. Processed feature dimension: {input_dim}")

    # 5. Build & Train ANN Classifier for Pass/Fail Prediction
    print("\n------------------------------------------------------------")
    print("Building & Training ANN Classifier (Target: passed)...")
    print("------------------------------------------------------------")
    
    clf_model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    clf_model.compile(
        optimizer=Adam(learning_rate=0.002),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop_clf = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    clf_history = clf_model.fit(
        X_train_proc, y_clf_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop_clf],
        verbose=1
    )

    # Evaluate ANN Classifier
    clf_probs = clf_model.predict(X_test_proc).ravel()
    clf_preds = (clf_probs >= 0.5).astype(int)

    acc = accuracy_score(y_clf_test, clf_preds)
    prec = precision_score(y_clf_test, clf_preds, zero_division=0)
    rec = recall_score(y_clf_test, clf_preds, zero_division=0)
    f1 = f1_score(y_clf_test, clf_preds, zero_division=0)
    cm = confusion_matrix(y_clf_test, clf_preds).tolist()
    auc = roc_auc_score(y_clf_test, clf_probs)

    print("\nGenuine ANN Classifier Test Metrics:")
    print(f" - Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f" - Precision: {prec:.4f}")
    print(f" - Recall:    {rec:.4f}")
    print(f" - F1 Score:  {f1:.4f}")
    print(f" - ROC AUC:   {auc:.4f}")
    print(f" - Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # 6. Build & Train ANN Regressor for Final Score Prediction
    print("\n------------------------------------------------------------")
    print("Building & Training ANN Regressor (Target: final_score)...")
    print("------------------------------------------------------------")

    reg_model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])

    reg_model.compile(
        optimizer=Adam(learning_rate=0.002),
        loss='mean_squared_error',
        metrics=['mae']
    )

    early_stop_reg = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    reg_history = reg_model.fit(
        X_train_proc, y_reg_train_scaled,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop_reg],
        verbose=1
    )

    # Evaluate ANN Regressor (inverse transform predictions back to 0-100 score)
    reg_preds_scaled = reg_model.predict(X_test_proc).ravel()
    reg_preds = y_scaler.inverse_transform(reg_preds_scaled.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_reg_test, reg_preds)
    mse = mean_squared_error(y_reg_test, reg_preds)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_reg_test, reg_preds)

    print("\nGenuine ANN Regressor Test Metrics:")
    print(f" - MAE:  {mae:.4f} marks")
    print(f" - MSE:  {mse:.4f}")
    print(f" - RMSE: {rmse:.4f}")
    print(f" - R2 Score: {r2:.4f}")

    # 7. Save Models and Metadata
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    clf_model_path = os.path.join(models_dir, "ann_classifier.keras")
    reg_model_path = os.path.join(models_dir, "ann_regressor.keras")
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    y_scaler_path = os.path.join(models_dir, "y_scaler.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")

    clf_model.save(clf_model_path)
    reg_model.save(reg_model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(y_scaler, y_scaler_path)

    metrics_payload = {
        "classifier": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm,
            "train_loss": [float(x) for x in clf_history.history['loss']],
            "val_loss": [float(x) for x in clf_history.history['val_loss']],
            "train_acc": [float(x) for x in clf_history.history['accuracy']],
            "val_acc": [float(x) for x in clf_history.history['val_accuracy']],
        },
        "regressor": {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2_score": float(r2),
            "train_loss": [float(x) for x in reg_history.history['loss']],
            "val_loss": [float(x) for x in reg_history.history['val_loss']],
            "train_mae": [float(x) for x in reg_history.history['mae']],
            "val_mae": [float(x) for x in reg_history.history['val_mae']],
        },
        "dataset_info": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_columns": feature_cols,
            "num_cols": num_cols,
            "cat_cols": cat_cols
        }
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"\n[+] Saved all artifacts successfully:")
    print(f" - Classifier Model: {clf_model_path}")
    print(f" - Regressor Model:  {reg_model_path}")
    print(f" - Preprocessor:     {preprocessor_path}")
    print(f" - Target Y Scaler:   {y_scaler_path}")
    print(f" - Metrics File:      {metrics_path}")

    print("\nTraining Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
