# ❤️ Heart Stroke Prediction System

## 📌 Overview

The Heart Stroke Prediction System is a Machine Learning-based web application that predicts the likelihood of a patient experiencing a stroke based on various health and lifestyle factors. The project utilizes data preprocessing, feature engineering, and classification algorithms to provide accurate stroke risk predictions through an interactive Streamlit interface.

---

## 🚀 Features

- Data preprocessing and cleaning
- Missing value handling
- Categorical data encoding
- Feature scaling using StandardScaler
- Machine Learning model training
- Model evaluation using classification metrics
- Real-time stroke risk prediction
- User-friendly Streamlit web application

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

---

## 📂 Project Structure

```
Heart-Stroke-Prediction/
│
├── app.py                     # Streamlit application
├── model.pkl                  # Trained ML model
├── scaler.pkl                 # Saved scaler
├── dataset.csv                # Dataset
├── requirements.txt           # Required libraries
├── README.md                  # Project documentation
└── notebooks/                 # Jupyter notebooks (optional)
```

---

## 📊 Dataset Features

The model uses the following patient information:

- Age
- Gender
- Hypertension
- Heart Disease
- Ever Married
- Work Type
- Residence Type
- Average Glucose Level
- BMI
- Smoking Status

**Target Variable:**
- Stroke (0 = No Stroke, 1 = Stroke)

---

## 🤖 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Data Preprocessing
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Streamlit Deployment

---

## 📈 Machine Learning Algorithms

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

---

## 📉 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Heart-Stroke-Prediction.git
```

Move to the project directory:

```bash
cd Heart-Stroke-Prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 💻 Usage

1. Open the Streamlit application.
2. Enter the patient's health details.
3. Click the **Predict** button.
4. The application displays whether the patient has a **High Risk** or **Low Risk** of stroke.

---

## 📸 Application Preview

Add screenshots of your Streamlit application here.

Example:

```
Home Page Screenshot

Prediction Result Screenshot
```

---

## 🔮 Future Improvements

- Improve prediction accuracy using XGBoost
- Hyperparameter tuning
- Explain predictions using SHAP
- Deploy on Streamlit Cloud
- Add user authentication
- Integrate with healthcare databases

---

## 👩‍💻 Author

**Anusha Peyalla**

- GitHub: https://github.com/your-username
- LinkedIn: https://www.linkedin.com/in/your-profile

---

## 📜 License

This project is developed for educational and learning purposes.
