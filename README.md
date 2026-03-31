# Titanic Survival Prediction 🚢

A machine learning project to predict the likelihood of passenger survival during the Titanic shipwreck using classification algorithms.

## 📌 Project Overview
The goal of this project is to build a predictive model that answers the question: "what sorts of people were more likely to survive?" using passenger data (ie. name, age, gender, socio-economic class, etc.).

This project covers the full data science lifecycle:
* **Exploratory Data Analysis (EDA):** Visualizing correlations and missing data.
* **Feature Engineering:** Creating new features from Titles, Family Size, and Deck information.
* **Model Selection:** Comparing Random Forest, Logistic Regression, and XGBoost.
* **Evaluation:** Using Accuracy, Precision, Recall, and F1-Score.
* **API Deployment:** REST API using FastAPI for model inference.
* **Web Interface:** Interactive Streamlit application for predictions.

## 🛠️ Tech Stack
* **Language:** Python 3.8+
* **Data Science:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn
* **Machine Learning:** XGBoost, Random Forest, Logistic Regression
* **API Framework:** FastAPI, Uvicorn
* **Web App:** Streamlit
* **Environment:** Jupyter Notebook / Google Colab

## 📊 Dataset
The dataset used is the classic **Titanic: Machine Learning from Disaster** from Kaggle.
* `train.csv`: Contains the training data with the target variable `Survived`.
* `test.csv`: Contains the test data for which we predict survival.

## 📦 Requirements

### Installation
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

