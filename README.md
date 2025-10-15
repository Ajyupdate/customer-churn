# 🧠 Customer Churn Prediction API

A simple **FastAPI** service that predicts whether a customer is likely to churn using a trained **Machine Learning model**.

---

## 🚀 Project Overview

This API uses a trained **Random Forest model** on a customer dataset to predict churn (0 = No, 1 = Yes).

It’s designed to demonstrate the full ML lifecycle:
- Data preprocessing  
- Model training  
- Model saving/loading  
- REST API for prediction  
- Cloud deployment on **Render**

---

## 🧩 Tech Stack

- **Python 3.10+**
- **FastAPI** for building the REST API
- **scikit-learn** for machine learning
- **pandas** for data manipulation
- **joblib** for model serialization
- **Uvicorn** as ASGI server

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/Ajyupdate/customer-churn.git
cd customer-churn
