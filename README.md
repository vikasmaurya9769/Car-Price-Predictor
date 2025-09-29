# 🚗 Car Price Prediction Web App

An **end-to-end Machine Learning project** to predict the price of second-hand cars.  
The app is built with **Flask (backend)**, **HTML/CSS/JS (frontend)**, and **Linear Regression (ML model)**.

---

## 🔹 Features
- User-friendly **web interface** to enter car details.
- Predicts **car price instantly** using a trained ML model.
- **Dynamic dropdowns**:
  - Company → Model → Variant filtering
  - State → City filtering
- **Data preprocessing**:
  - Outlier removal
  - Feature engineering
  - Color normalization (e.g., "dark grey" → "Grey")
- Integrated **ML pipeline** with `ColumnTransformer` & `OneHotEncoder`.

---

## 🔹 Tech Stack
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask
- **Machine Learning**: Python (scikit-learn, pandas, numpy)
- **Visualization**: matplotlib, seaborn
- **Deployment Ready**: Can be hosted on Render/Heroku/Streamlit

---

## 🔹 Dataset
- Custom dataset of used cars (`My_Clean_Data_final.csv`)  
- Contains features like:
  - Company, Model, Variant
  - Year of Manufacture
  - Fuel, Transmission, Body Type
  - State, City, Owner Type
  - Kilometers Driven
  - Color
  - Price (Target Variable)

---

## 🔹 ML Model
- Algorithm: **Linear Regression**  
- Preprocessing:
  - Encoded categorical features using `OneHotEncoder`
  - Handled unseen categories via preprocessing
  - Removed outliers to improve accuracy
- Evaluation Metrics:
  - R² Score
  - Mean Squared Error (MSE)
  - Residual analysis

---

## 🔹 Challenges Faced
- **Messy dataset** (outliers, missing values, duplicate colors) → solved with cleaning & normalization.
- **Unseen categories** (like new colors not in training data) → handled with "Other" bucket.
- **Dynamic dropdowns** in frontend → solved using Flask APIs (`/get_models/<company>`, `/get_variants/<model>`).

---
