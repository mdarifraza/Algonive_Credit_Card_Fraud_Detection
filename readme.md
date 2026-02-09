# 💳 Credit Card Fraud Detection

A machine learning–based web application to predict whether a credit card transaction is **Fraud** or **Legit**. The app is built using **XGBoost** for modeling and **Streamlit** for an interactive UI.

---

## 🚀 Project Overview

Credit card fraud is a critical problem in the financial industry. This project uses a supervised ML model trained on anonymized transaction data to identify fraudulent transactions in real time.

The dataset uses **PCA-transformed features (V1–V28)** to protect sensitive customer information while retaining important fraud patterns.

---

## 🧠 Machine Learning Details

* **Model:** XGBoost Classifier
* **Problem Type:** Binary Classification
* **Target Variable:**

  * `0` → Legit Transaction
  * `1` → Fraud Transaction

### Features Used

| Feature | Description                                                        |
| ------- | ------------------------------------------------------------------ |
| Time    | Seconds elapsed between this transaction and the first transaction |
| Amount  | Transaction amount                                                 |
| V1–V28  | Anonymized PCA-transformed features                                |

> ⚠️ **Note:** V1–V28 are not raw inputs. They are PCA components generated from original confidential features.

---

## 🛠️ Tech Stack

* **Python**
* **XGBoost**
* **Scikit-learn**
* **NumPy**
* **Joblib**
* **Streamlit**
* **Anaconda (Conda environment)**

---

## 📂 Project Structure

```text
credit_card_fraud_detection/
│
├── app.py                  # Streamlit application
├── model.pkl               # Trained XGBoost model
├── scaler.pkl              # Feature scaler
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── data/                   # (Optional) Dataset files
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/credit_card_fraud_detection.git
cd credit_card_fraud_detection
```

### 2️⃣ Create & Activate Conda Environment

```bash
conda create -n project python=3.10 -y
conda activate project
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> Recommended for Windows users:

```bash
conda install -c conda-forge xgboost streamlit
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open your browser and go to:

```
http://localhost:8501
```

---

## 🧪 How the App Works

1. User enters transaction details
2. Data is scaled using a pre-trained scaler
3. Model predicts the transaction class
4. Result is displayed as:

   * ✅ Legit Transaction
   * 🚨 Fraudulent Transaction

---

## 📊 Model Performance

The model was evaluated using:

* Accuracy
* Precision
* Recall
* ROC-AUC Score

XGBoost performs well on highly imbalanced datasets like fraud detection.

---

## 📌 Important Notes

* Manual input of V1–V28 is **only for demo purposes**
* In real-world systems, PCA features are generated automatically via pipelines
* This project is for **educational and learning purposes**

---

## 🔮 Future Improvements

* Add automatic PCA + preprocessing pipeline
* Improve UI with feature tooltips
* Deploy on cloud (Streamlit Cloud / AWS / Azure)
* Add transaction history & logs

---

## 👤 Author

**Md Arif Raza**
Aspiring Data Scientist | Machine Learning Enthusiast

---

## ⭐ Acknowledgements

* Kaggle Credit Card Fraud Dataset
* Scikit-learn & XGBoost documentation

---

> If you found this project helpful, don’t forget to ⭐ the repository!
