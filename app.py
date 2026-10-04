"""
Student Performance Prediction - Streamlit Web Application
Soft Computing Mini-Project using Artificial Neural Network (ANN)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Set Page Config for wide layout & mobile friendliness
st.set_page_config(
    page_title="Student Performance Prediction (ANN)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high contrast, theme compatibility, and mobile responsiveness
st.markdown("""
<style>
    /* Responsive Title Typography */
    .main-title {
        font-size: clamp(1.6rem, 4vw, 2.6rem);
        color: var(--text-color, #1E3A8A);
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: clamp(0.9rem, 2vw, 1.15rem);
        color: var(--text-color, #4B5563);
        text-align: center;
        margin-bottom: 1.75rem;
        opacity: 0.85;
    }

    /* High-contrast result cards (Works on both light & dark Streamlit themes) */
    .pass-card {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%) !important;
        border-left: 8px solid #34d399 !important;
        padding: 22px !important;
        border-radius: 12px !important;
        margin-top: 15px !important;
        margin-bottom: 15px !important;
        box-shadow: 0 10px 15px -3px rgba(6, 78, 59, 0.4) !important;
    }
    .pass-card, .pass-card h1, .pass-card h2, .pass-card h3, .pass-card h4, .pass-card p, .pass-card b, .pass-card span, .pass-card div {
        color: #FFFFFF !important;
    }

    .fail-card {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%) !important;
        border-left: 8px solid #f87171 !important;
        padding: 22px !important;
        border-radius: 12px !important;
        margin-top: 15px !important;
        margin-bottom: 15px !important;
        box-shadow: 0 10px 15px -3px rgba(127, 29, 29, 0.4) !important;
    }
    .fail-card, .fail-card h1, .fail-card h2, .fail-card h3, .fail-card h4, .fail-card p, .fail-card b, .fail-card span, .fail-card div {
        color: #FFFFFF !important;
    }

    .card-badge-pass {
        display: inline-block;
        background-color: #059669;
        color: #FFFFFF !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 10px;
    }

    .card-badge-fail {
        display: inline-block;
        background-color: #DC2626;
        color: #FFFFFF !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 10px;
    }

    /* Mobile friendly media queries */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.5rem !important;
        }
        .sub-title {
            font-size: 0.9rem !important;
        }
        .stButton > button {
            width: 100% !important;
            font-size: 1.1rem !important;
            padding: 12px !important;
        }
        .pass-card, .fail-card {
            padding: 16px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load artifacts safely
@st.cache_resource
def load_models_and_preprocessor():
    models_dir = "models"
    clf_path = os.path.join(models_dir, "ann_classifier.keras")
    reg_path = os.path.join(models_dir, "ann_regressor.keras")
    prep_path = os.path.join(models_dir, "preprocessor.joblib")
    y_scale_path = os.path.join(models_dir, "y_scaler.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")

    if not (os.path.exists(clf_path) and os.path.exists(prep_path) and os.path.exists(metrics_path)):
        return None, None, None, None, None

    clf_model = tf.keras.models.load_model(clf_path)
    reg_model = tf.keras.models.load_model(reg_path) if os.path.exists(reg_path) else None
    preprocessor = joblib.load(prep_path)
    y_scaler = joblib.load(y_scale_path) if os.path.exists(y_scale_path) else None
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return clf_model, reg_model, preprocessor, y_scaler, metrics

def main():
    st.markdown('<div class="main-title">🎓 Student Performance Prediction using ANN</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Soft Computing Mini-Project | Artificial Neural Network (TensorFlow/Keras)</div>', unsafe_allow_html=True)

    clf_model, reg_model, preprocessor, y_scaler, metrics = load_models_and_preprocessor()

    if clf_model is None or preprocessor is None:
        st.error("⚠️ Trained ANN models or preprocessor files were not found in the `models/` directory!")
        st.info("💡 Please run `python train_model.py` first to train the Artificial Neural Network and save trained artifacts.")
        st.stop()

    # Sidebar Navigation
    st.sidebar.title("📌 Navigation")
    app_mode = st.sidebar.radio(
        "Select Tab",
        ["🔮 Predict Performance", "📊 Model Performance", "📖 Soft Computing Viva Guide"]
    )

    # =========================================================================
    # TAB 1: PREDICTION
    # =========================================================================
    if app_mode == "🔮 Predict Performance":
        st.header("📝 Enter Student Details for ANN Prediction")
        st.write("Fill in the student's demographic and academic details below to get a real-time prediction from the trained Artificial Neural Network.")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age (Years)", min_value=15, max_value=19, value=17)
            parent_education = st.selectbox("Parent Education Level", ["None", "High School", "Bachelor", "Master", "PhD"])

        with col2:
            study_hours = st.slider("Study Hours / Week", min_value=1.0, max_value=35.0, value=15.0, step=0.5)
            attendance_rate = st.slider("Attendance Rate (%)", min_value=40.0, max_value=100.0, value=75.0, step=0.5)
            previous_score = st.slider("Previous Exam Score (0-100)", min_value=20.0, max_value=100.0, value=65.0, step=1.0)

        with col3:
            internet_access = st.radio("Has Internet Access at Home?", ["Yes", "No"])
            extracurricular = st.radio("Participates in Extracurriculars?", ["Yes", "No"])

        st.markdown("---")

        if st.button("🚀 Predict Student Performance", type="primary", use_container_width=True):
            # Form DataFrame matching preprocessor expectations
            input_df = pd.DataFrame([{
                'gender': gender,
                'age': age,
                'study_hours_per_week': study_hours,
                'attendance_rate': attendance_rate,
                'parent_education': parent_education,
                'internet_access': internet_access,
                'extracurricular': extracurricular,
                'previous_score': previous_score
            }])

            # Preprocess input using saved preprocessor
            processed_input = preprocessor.transform(input_df)

            # Predict Pass/Fail using ANN Classifier
            pass_prob = float(clf_model.predict(processed_input)[0][0])
            is_passed = pass_prob >= 0.5

            # Predict Exact Score using ANN Regressor
            if reg_model is not None:
                raw_reg_pred = float(reg_model.predict(processed_input)[0][0])
                if y_scaler is not None:
                    pred_score = float(y_scaler.inverse_transform([[raw_reg_pred]])[0][0])
                else:
                    pred_score = raw_reg_pred
                pred_score_clipped = max(0.0, min(100.0, pred_score))
            else:
                pred_score_clipped = None

            st.subheader("🎯 ANN Model Prediction Results")

            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.metric("Pass Probability", f"{pass_prob * 100:.1f}%")
                st.progress(pass_prob)

            with res_col2:
                if is_passed:
                    st.success("🟢 **Status: PASSED**")
                else:
                    st.error("🔴 **Status: FAILED**")

            with res_col3:
                if pred_score_clipped is not None:
                    st.metric("Predicted Final Score", f"{pred_score_clipped:.1f} / 100")
                else:
                    st.metric("Predicted Final Score", "N/A")

            # Result Card Display with High Contrast Text Guarantee
            if is_passed:
                st.markdown(f"""
                <div class="pass-card">
                    <span class="card-badge-pass">✅ PASSED</span>
                    <h3 style="margin-top: 5px; font-weight: 700;">Prediction Outcome: Student is likely to PASS</h3>
                    <p style="font-size: 1.05rem; margin-bottom: 0;">The Artificial Neural Network predicts a <b>{pass_prob * 100:.1f}% confidence</b> of passing with an estimated final score of <b>{pred_score_clipped:.1f} marks</b> out of 100.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="fail-card">
                    <span class="card-badge-fail">❌ AT RISK / FAILED</span>
                    <h3 style="margin-top: 5px; font-weight: 700;">Prediction Outcome: Student is at risk of FAILING</h3>
                    <p style="font-size: 1.05rem; margin-bottom: 0;">The Artificial Neural Network predicts only <b>{pass_prob * 100:.1f}% probability</b> of passing with an estimated final score of <b>{pred_score_clipped:.1f} marks</b> out of 100.</p>
                </div>
                """, unsafe_allow_html=True)

            # Academic Insights & Recommendations
            st.subheader("💡 Recommendations & Academic Insights")
            recommendations = []
            if study_hours < 12:
                recommendations.append("📚 **Increase Study Time**: The student currently studies less than 12 hours/week. Increasing study time by 5-10 hours can significantly improve scores.")
            if attendance_rate < 75:
                recommendations.append("⏰ **Improve Attendance**: Attendance is below 75%. Regular classroom participation is crucial for performance.")
            if previous_score < 50:
                recommendations.append("✍️ **Foundational Review**: Previous score is low. Remedial sessions on core concepts are recommended.")

            if not recommendations:
                st.success("🌟 The student shows strong academic habits (good attendance, study hours, and prior score). Keep maintaining this routine!")
            else:
                for rec in recommendations:
                    st.write(rec)

    # =========================================================================
    # TAB 2: MODEL PERFORMANCE
    # =========================================================================
    elif app_mode == "📊 Model Performance":
        st.header("📈 ANN Model Evaluation & Metrics")
        st.write("These metrics reflect the actual test performance achieved by the trained Artificial Neural Network on unseen student test data.")

        clf_metrics = metrics.get("classifier", {})
        reg_metrics = metrics.get("regressor", {})

        st.subheader("1. ANN Binary Classifier Metrics (Target: Passed)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{clf_metrics.get('accuracy', 0)*100:.2f}%")
        m2.metric("Precision", f"{clf_metrics.get('precision', 0):.4f}")
        m3.metric("Recall", f"{clf_metrics.get('recall', 0):.4f}")
        m4.metric("F1-Score", f"{clf_metrics.get('f1_score', 0):.4f}")
        m5.metric("ROC AUC", f"{clf_metrics.get('roc_auc', 0):.4f}")

        st.subheader("2. ANN Regressor Metrics (Target: Final Score)")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("MAE (Mean Abs Error)", f"{reg_metrics.get('mae', 0):.2f} marks")
        r2.metric("MSE (Mean Sq Error)", f"{reg_metrics.get('mse', 0):.2f}")
        r3.metric("RMSE (Root Mean Sq Error)", f"{reg_metrics.get('rmse', 0):.2f}")
        r4.metric("R² Score", f"{reg_metrics.get('r2_score', 0):.4f}")

        st.markdown("---")
        st.subheader("📉 Training History & Confusion Matrix Plots")

        p_col1, p_col2 = st.columns(2)

        with p_col1:
            st.write("#### Classifier Loss Curve (Binary Crossentropy)")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(clf_metrics.get('train_loss', []), label='Train Loss', color='#2563EB', linewidth=2)
            ax.plot(clf_metrics.get('val_loss', []), label='Val Loss', color='#DC2626', linestyle='--', linewidth=2)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_title('Training vs Validation Loss (Classifier)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with p_col2:
            st.write("#### Classifier Accuracy Curve")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(clf_metrics.get('train_acc', []), label='Train Accuracy', color='#059669', linewidth=2)
            ax.plot(clf_metrics.get('val_acc', []), label='Val Accuracy', color='#D97706', linestyle='--', linewidth=2)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training vs Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        p_col3, p_col4 = st.columns(2)

        with p_col3:
            st.write("#### Confusion Matrix (Test Set)")
            cm = np.array(clf_metrics.get('confusion_matrix', [[0,0],[0,0]]))
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Pred Fail', 'Pred Pass'],
                        yticklabels=['Actual Fail', 'Actual Pass'])
            ax.set_title("Confusion Matrix Heatmap")
            st.pyplot(fig)

        with p_col4:
            st.write("#### Regressor Loss Curve (MSE)")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(reg_metrics.get('train_loss', []), label='Train Loss (Scaled)', color='#7C3AED', linewidth=2)
            ax.plot(reg_metrics.get('val_loss', []), label='Val Loss (Scaled)', color='#EC4899', linestyle='--', linewidth=2)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss (MSE)')
            ax.set_title('Training vs Validation Loss (Regressor)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # =========================================================================
    # TAB 3: VIVA GUIDE & ABOUT
    # =========================================================================
    elif app_mode == "📖 Soft Computing Viva Guide":
        st.header("📚 Soft Computing Viva Preparation Guide")
        st.write("Use this guide to confidently explain the concepts, mathematical formulas, architecture, and code logic during your Soft Computing project viva.")

        st.subheader("1. Project Summary & Soft Computing Context")
        st.markdown("""
        - **Project Title**: Student Performance Prediction Using Artificial Neural Network (ANN)
        - **Domain**: Soft Computing / Machine Learning / Deep Learning
        - **What is Soft Computing?**: Soft computing tolerant of imprecision, uncertainty, partial truth, and approximation to achieve tractability, robustness, and low solution cost. Artificial Neural Networks (ANN) are a primary subfield of Soft Computing inspired by biological brain neural networks.
        """)

        st.subheader("2. Artificial Neural Network (ANN) Architecture")
        st.markdown("""
        Our trained model uses a Multi-Layer Perceptron (MLP) Artificial Neural Network with the following structure:
        - **Input Layer**: Takes preprocessed feature vector (Standardized numerical values + One-Hot encoded categoricals).
        - **Hidden Layers**:
          - `Dense(32 neurons)` + `BatchNormalization()` + `Dropout(0.2)` + `ReLU`
          - `Dense(16 neurons)` + `BatchNormalization()` + `Dropout(0.1)` + `ReLU`
          - `Dense(8 neurons)` + `ReLU`
        - **Output Layer**:
          - **Classification**: `Dense(1 neuron)` with **Sigmoid** activation function for binary pass/fail output $y \\in [0, 1]$.
          - **Regression**: `Dense(1 neuron)` with **Linear** activation function for continuous score prediction.
        """)

        st.subheader("3. Key Mathematical Functions & Formulas")
        st.markdown(r"""
        * **ReLU (Rectified Linear Unit)**:
          $$f(z) = \max(0, z)$$
          *Used in hidden layers to introduce non-linearity and avoid vanishing gradient problems.*

        * **Sigmoid Activation Function**:
          $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
          *Used in output layer of binary classification to output a probability between 0 and 1.*

        * **Binary Cross-Entropy Loss Function**:
          $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

        * **Mean Squared Error (MSE) Loss Function (Regression)**:
          $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

        * **Backpropagation & Adam Optimizer**:
          Backpropagation computes gradients of the loss function with respect to weights using the chain rule ($\frac{\partial \mathcal{L}}{\partial w}$). Adam (Adaptive Moment Estimation) updates weights dynamically using first and second moments of gradients.
        """)

        st.subheader("4. Frequently Asked Viva Questions (FAQ)")
        with st.expander("Q1: Why use an Artificial Neural Network (ANN) instead of simple linear regression or decision tree?"):
            st.write("ANNs can learn complex, non-linear relationships between student attributes (e.g., interaction between study hours, attendance, and prior scores) without needing manual feature engineering.")

        with st.expander("Q2: Why is feature scaling (StandardScaler) necessary before feeding data to ANN?"):
            st.write("ANNs use gradient descent optimization. If feature scales differ wildly (e.g., age [15-19] vs attendance [50-100%]), gradient descent becomes unstable and slow. Standardization scales features to zero mean and unit variance.")

        with st.expander("Q3: What is the role of Dropout and EarlyStopping?"):
            st.write("Dropout randomly deactivates neurons during training to prevent co-adaptation and overfitting. EarlyStopping monitors validation loss and stops training when performance stops improving.")

        with st.expander("Q4: Why drop `student_id` column?"):
            st.write("`student_id` is an arbitrary primary key index string (e.g. STU0001). It has zero correlation with student academic performance and would cause spurious overfitting if included.")

if __name__ == "__main__":
    main()
