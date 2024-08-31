# Linear Regression with Gradient Descent

This repository contains a simple implementation of a linear regression model using gradient descent in Python. The model is designed to predict values based on a single feature and an intercept term.

## Features

- **Prediction Function**: Calculates predictions based on the current model parameters.
- **Cost Function**: Computes the mean squared error between predicted and actual values.
- **Gradient Descent**: Optimizes the model parameters by minimizing the cost function over multiple iterations.

## Code Overview

### Files

- `linear_regression.py`: Contains the main code for linear regression, including data initialization, cost calculation, gradient descent optimization, and prediction.

### Functions

- **predict(X, theta)**: Predicts the output for input `X` based on the parameters `theta`.
- **computeCost(X, Y, theta)**: Calculates the mean squared error cost for the predicted values compared to the actual `Y`.
- **gradientDescent(X, Y, theta, learningRate, iterations)**: Optimizes the parameters `theta` using gradient descent.

### How It Works

1. **Data Preparation**:

   - The feature matrix `X` is augmented with a column of ones to include the intercept term.
   - The target vector `Y` contains the actual values we want to predict.

2. **Model Initialization**:
   - Randomly initializes the parameter vector `theta`.
3. **Training**:
   - The model is trained using the `gradientDescent` function, which iteratively updates `theta` to minimize the cost function.
4. **Prediction**:
   - Once trained, the model makes predictions using the optimized `theta` values.
