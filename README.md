# Parkinson's Disease Prediction App

A Machine Learning based web application that predicts the likelihood of Parkinson’s Disease using biomedical voice measurement data. The system analyzes various voice-related biomedical parameters and provides prediction results using trained machine learning models.

---

# Features

- Parkinson’s Disease prediction using Machine Learning
- Interactive and user-friendly web interface
- Real-time prediction system
- Data preprocessing and feature scaling
- Optimized and trained ML model
- Fast prediction results
- Easy setup and deployment

---

# Tech Stack

## Frontend
- HTML
- CSS

## Backend
- Python
- Flask

## Machine Learning Libraries
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

# Project Structure

```bash
parkinsons_app/
│
├── app.py                     # Main Flask application
├── utils.py                   # Utility/helper functions
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Ignored files and folders
│
├── models/                    # Trained ML models
│   ├── model.pkl
│   └── scaler.pkl
│
├── templates/                 # HTML templates
│   ├── index.html
│   └── result.html
│
├── static/                    # Static assets
│   ├── css/
│   ├── js/
│   └── images/
│
├── dataset/                   # Dataset files
│   └── parkinsons.csv
│
└── venv/                      # Virtual environment (ignored)
```

---

# Installation Guide

## 1. Clone the Repository

```bash
git clone https://github.com/Shikhar1330/parkinsons_app.git
cd parkinsons_app
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

---

## 3. Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 4. Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

```bash
python app.py
```

The application will start locally at:

```bash
http://127.0.0.1:5000
```

---

# Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Selection
4. Data Preprocessing
5. Feature Scaling
6. Model Training
7. Hyperparameter Tuning
8. Model Evaluation
9. Prediction Generation
10. Flask Web Deployment

---

# Model Information

The project uses supervised Machine Learning algorithms trained on biomedical voice measurement data for Parkinson’s Disease prediction.

Possible algorithms used:
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost

---

# Dataset Information

The dataset contains biomedical voice measurements collected from individuals with and without Parkinson’s Disease.

Example features:
- MDVP:Fo(Hz)
- MDVP:Fhi(Hz)
- MDVP:Flo(Hz)
- Jitter
- Shimmer
- NHR
- HNR
- RPDE
- DFA
- PPE

---

# API Workflow

```text
User Input → Flask Backend → Data Preprocessing → ML Model → Prediction Result
```

---

# Future Improvements

- Deep Learning implementation
- Voice recording based prediction
- Improved UI/UX design
- Cloud deployment
- Docker containerization
- Authentication system
- Real-time speech analysis
- Higher model accuracy

---


# Author

## Shikhar Choudhary

GitHub:
https://github.com/Shikhar1330

LinkedIn:
https://www.linkedin.com/in/shikhar-choudhary-b41aa328b/

