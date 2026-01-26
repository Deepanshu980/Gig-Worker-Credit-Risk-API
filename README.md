# Gig Worker Credit Score Prediction 🚴📊

This project builds a **credit scoring system for gig economy workers** (e.g., Zomato, Swiggy delivery partners) using **machine learning (XGBoost)**.  
It helps financial institutions and fintech platforms assess **creditworthiness** of workers who lack traditional credit history.

---

## 📌 Problem Statement

Gig workers often do not have:
- Fixed salaries
- Formal employment records
- Traditional credit scores

This makes it difficult for banks and NBFCs to:
- Approve loans
- Set interest rates
- Estimate default risk

👉 This project solves the problem by predicting **default risk / credit score** using **delivery and income behavior data**.

---

## 🧠 Solution Overview

We use **XGBoost (Extreme Gradient Boosting)** to:
- Analyze worker performance & income stability
- Predict **default risk**
- Achieve **~96% model accuracy**

---

## 📂 Dataset Description

Each row represents a **gig worker** with the following features:

| Column Name | Description |
|------------|------------|
| `worker_id` | Unique ID of the delivery partner |
| `platform` | Platform name (Zomato / Swiggy) |
| `avg_daily_income` | Average daily earnings |
| `weekly_deliveries` | Number of deliveries per week |
| `active_days` | Days active per week |
| `rating` | Customer rating (1–5) |
| `income_std` | Income volatility (standard deviation) |
| `account_age_months` | Account age on platform |
| `cancel_rate` | Percentage of cancelled orders |
| `monthly_income` | Total monthly income |
| `default_risk` | Target variable (0 = Low Risk, 1 = High Risk) |

---

## ⚙️ Tech Stack

- **Python**
- **Pandas / NumPy**
- **Scikit-learn**
- **XGBoost**
- **Matplotlib / Seaborn**

---

## 🔍 Model Used

### XGBoost Classifier
Why XGBoost?
- Handles non-linear data well
- Works great with tabular datasets
- Robust to missing values
- High performance with minimal tuning

---

## 📈 Model Performance

| Metric | Score |
|------|------|
| Accuracy | **96%** |
| Precision | High |
| Recall | High |
| F1-Score | Balanced |

---

## 🛠️ Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Train-Test Split
5. XGBoost Model Training
6. Model Evaluation
7. Credit Risk Prediction

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/gig-worker-credit-score.git
cd gig-worker-credit-score
