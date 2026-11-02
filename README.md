# Creditworthiness Prediction Using Machine Learning

## Project Overview

This project demonstrates how machine learning classification algorithms can be used to predict credit application approval using historical application data.

The project uses Python, Pandas, and Scikit-learn to preprocess data, train classification models, and evaluate their performance. Streamlit provides a simple web interface for making predictions.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Machine Learning Algorithms

Three classification algorithms are compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Dataset

The project uses the UCI Credit Approval dataset.

The dataset predicts credit application approval and contains anonymized features. It does not directly provide a comprehensive assessment of an individual's creditworthiness.

## Model Results

In the current test run, Random Forest achieved:

- Accuracy: 92.75%
- Precision: 90.48%
- Recall: 93.44%
- F1-score: 91.94%
- ROC-AUC: 96.36%

These results are specific to the current dataset split and model configuration. They should not be interpreted as evidence of performance on real-world lending decisions.

## How to Run the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train_model.py
```

Start the web application:

```bash
streamlit run app.py
```

## Limitations

The dataset uses anonymized features, so the current application does not directly accept meaningful financial fields such as annual income, debt-to-income ratio, and repayment history.

The predictions are educational demonstrations and should not be used as the sole basis for real lending decisions.