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

# 1. Clone the repository:
git clone https://github.com/MujahidHussain-2005/titanic-survival-prediction.git
cd titanic-survival-prediction

# 2. Install dependencies:
pip install -r requirements.txt

# 3. Run the FastAPI backend:
uvicorn main:app --reload --port 8001

API will be available at: http://localhost:8001
Interactive API docs: http://localhost:8001/docs
Health check: http://localhost:8001/health

# 4. Run the Streamlit frontend:
streamlit run application.py

Web app will open automatically at: http://localhost:8501

### 📝 License
This project is open-source and available under the MIT License.

### 👨‍💻 Author
Mujahid Hussain

GitHub: @MujahidHussain-2005
