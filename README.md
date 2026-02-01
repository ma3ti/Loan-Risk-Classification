# Data Analytics 2024/2025 — Loan Risk Prediction
This project focuses on predicting the **credit risk grade** of a loan using the dataset provided in `train.csv`. This file can be used for both training and validation.

## Objective
Build a model that predicts the loan’s **risk grade**, provided in the column **`grade`**.

## Dataset
The dataset contains approximately **150 raw features**, including:

- **Numerical features:** loan amount, interest rate, income levels, and other financial indicators.
- **Categorical features:** employment category, loan purpose, housing status.
- **Date features:** timestamps related to the loan or borrower.

When preprocessing the data, ensure that categorical encodings properly handle:
- **High-cardinality features**
- **Missing values**

## Reproducibility
All models will be evaluated using the provided test environment. 
* **If the code does not run within this environment, the project will not be evaluated**.

If you have access to a GPU, you must install CUDA libraries for accelerated training.

For Deep Tabular Models (TabNet, TabTransformer) you can use the following libraries:
- `pytorch-tabnet`
- `tab-transformer-pytorch`
- `pytorch_tabular` (not included by default)

## Contacts
- filippo.bartolucci3@unibo.it
- leonardo.ciabattini@unibo.it