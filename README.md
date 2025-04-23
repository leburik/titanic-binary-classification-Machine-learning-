# Titanic Binary Classification - Machine Learning using Bayesian Optimization

This project is a machine learning solution to predict passenger survival on the Titanic using various classification techniques. The model leverages Bayesian Optimization to fine-tune hyperparameters for better performance. The dataset used in this project is the Titanic dataset, which is available on Kaggle.

## Project Overview

The goal of this project is to build a binary classification model that predicts whether a passenger survived or not based on the available features. We aim to achieve this by utilizing **Bayesian Optimization** to fine-tune the hyperparameters of the machine learning models and improve prediction accuracy.

The workflow includes:
1. **Data Exploration and Preprocessing**
2. **Feature Engineering**
3. **Modeling**
4. **Hyperparameter Optimization with Bayesian Optimization**
5. **Model Evaluation**

## Dataset

The dataset is available from the Kaggle Titanic competition. It contains information about the passengers, including features like:
- `PassengerId`: Unique ID for each passenger
- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Name`: Name of the passenger
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings or spouses aboard
- `Parch`: Number of parents or children aboard
- `Ticket`: Ticket number
- `Fare`: Ticket fare
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/titanic-binary-classification.git
   cd titanic-binary-classification

run 
run_pipeline.sh

Hyperparameter Tuning with Bayesian Optimization

Bayesian Optimization is used to find optimal hyperparameters for the classifier. The key parameters tuned include:

    learning_rate
    n_estimators
    max_depth
    subsample
    colsample_bytree

This is achieved by using the BayesianOptimization library to perform a search over possible parameter ranges.
Evaluation Metrics

The performance of the model is evaluated using the following metrics:
    Accuracy: Proportion of correct predictions.
    Precision: The proportion of positive predictions that were actually positive.
    Recall: The proportion of actual positives that were correctly identified.
    F1-Score: Harmonic mean of precision and recall.
    Confusion Matrix: A table showing the performance of the classification model.

  pip install -r requirements.txt

This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For inquiries or collaboration, you can reach me at my email: [leburikplc@gmail.com].


