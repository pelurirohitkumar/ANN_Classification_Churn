# 🚀 Customer Churn Prediction App (Deep Learning + Streamlit)

This project is a deep learning-based web application that predicts whether a customer is likely to churn or not.

The model is built using TensorFlow (ANN) and deployed using Streamlit, containerized with Docker for easy deployment.

---

## 🔥 Features

- 📊 Real-time churn prediction
- 🤖 Deep Learning model (Artificial Neural Network)
- 🎯 Probability-based output
- 🎨 Modern Streamlit UI
- 🐳 Fully Dockerized for deployment

---

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Streamlit
- Docker

---

## 📦 Model Details

- Input: Customer demographic & financial data
- Output: Probability of churn (0 to 1)
- Threshold: 0.5 for classification

---

## ▶️ Run with Docker

```bash
docker pull <your-username>/churn-app
docker run -p 8501:8501 <your-username>/churn-app
