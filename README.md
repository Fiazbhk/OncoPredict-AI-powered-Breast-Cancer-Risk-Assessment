# Breast Cancer Prediction Web App

A Streamlit-based web application for predicting breast cancer types based on preprocessed input features. Users select values from predefined options, and the app outputs the predicted cancer type using a trained machine learning model.

---

## Features

- Interactive UI with `st.number_input`, `st.selectbox`, and other input widgets.
- Backend preprocessing already applied; user only selects values.
- Handles prediction with a trained ML model.
- Clean, user-friendly interface.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/breast-cancer-streamlit.git
cd breast-cancer-streamlit
```

## Create a virtual environment
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

## Install dependencies:
```bash
pip install -r requirements.txt
```

## Project
```bash
breast-cancer-streamlit/
│
├── main.py           # Streamlit app
├── model.pkl         # Trained ML model
├── requirements.txt  # Dependencies
├── README.md
└── data/             # Optional folder for input CSV or preprocessed data
```
## Usage
```bash
streamlit run main.py
```
Open the URL provided in the terminal (usually http://localhost:8501) to access the app.

## Dependencies
- Python 3.9+
- Streamlit
- scikit-learn
- pandas
- numpy

Install all dependencies via:
```bash
pip install -r requirements.txt
```

## Credit
Developed by Muhammad Fiaz
