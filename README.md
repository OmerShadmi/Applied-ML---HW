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


## Results
- **Best Alpha**: 0.51
- **Best MAE**: 3.38
The notebook provides plots comparing quantile regression at different quantiles with linear regression, helping to visualize how each model fits the data.

## Conclusion
Quantile regression offers a more flexible approach than traditional linear regression, especially for modeling data with heterogeneous error distributions. The grid search optimization helps to fine-tune the model for the best performance.

-------------------

# Second Notebook: Gradient Boosting Classifier with Hyperparameter Tuning

In this task we get a dataset about the titanic. Our goal is to predict the destiny of the passenger (survived or not) based on his/her attributes.
This project demonstrates the implementation and optimization of a **Gradient Boosting Classifier** for a binary classification problem. The notebook showcases how to fine-tune hyperparameters to improve model performance.

## Project Structure

- **Data Preparation**: Preprocesses the dataset and splits it into training and testing sets.
- **Model Building**: Implements a basic Gradient Boosting model for initial testing.
- **Hyperparameter Tuning**: Uses techniques such as `GridSearchCV` to optimize the model's parameters.
- **Improved Model**: Trains and evaluates the optimized Gradient Boosting Classifier on new data.
- **Model Evaluation**: Provides performance metrics including precision, recall, and F1-score for both the original and improved models.

## Key Components

- **Gradient Boosting Classifier**: A powerful ensemble machine learning algorithm, used here for classification.
- **Hyperparameter Tuning**: Optimizes hyperparameters such as learning rate, max depth, min samples leaf, and number of estimators to improve the model's performance.
- **Classification Report**: Detailed metrics such as precision, recall, F1-score, and accuracy for evaluating model performance.

## Key Results

### Initial Model Performance:
- **Accuracy**: Initial testing of the Gradient Boosting model provides a good starting point for further optimization.

### Improved Model Performance (After Hyperparameter Tuning):
- **Best Parameters**:
  - `learning_rate`: 0.1
  - `max_depth`: 4
  - `min_samples_leaf`: 2
  - `min_samples_split`: 2
  - `n_estimators`: 100

- **Classification Report**:
  - **Precision (Class 0)**: 0.80
  - **Recall (Class 0)**: 0.73
  - **F1-Score (Class 0)**: 0.76
  - **Precision (Class 1)**: 0.61
  - **Recall (Class 1)**: 0.70
  - **F1-Score (Class 1)**: 0.65
  - **Overall Accuracy**: 0.72

## Conclusion

The **Gradient Boosting Classifier** was improved through hyperparameter tuning, achieving better classification performance on both training and testing datasets. The tuned model showed significant improvement in precision, recall, and overall accuracy compared to the baseline model.

## License

This project is licensed under the MIT License.
