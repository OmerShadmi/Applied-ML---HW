# First Notebook: Quantile Regression with Python

This project demonstrates how to implement **Quantile Regression** and compare its performance against **Linear Regression** using Python. It leverages machine learning techniques to minimize the Mean Absolute Error (MAE) and identifies the best model parameters using **GridSearchCV**.

## Project Structure

- **Data Loading and Preprocessing**: Prepares the data for regression analysis.
- **Linear Regression**: Implements a simple linear regression model to estimate the mean of the response variable.
- **Quantile Regression**: Estimates different quantiles of the response variable (e.g., median).
- **Model Comparison**: Visualizes and compares the results of both regression models.
- **Hyperparameter Tuning**: Uses `GridSearchCV` to find the best alpha value for minimizing the MAE.

## Key Concepts

- **Quantile Regression**: Unlike ordinary least squares (OLS) regression, which predicts the mean, quantile regression predicts different quantiles, making it suitable for data with non-constant variance.
- **Grid Search Optimization**: Utilizes grid search to optimize the `alpha` parameter for minimizing the error.




