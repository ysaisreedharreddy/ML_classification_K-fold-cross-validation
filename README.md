K-Fold Cross Validation with Kernel SVM
This repository contains a Python script that demonstrates the implementation of K-Fold Cross Validation using Kernel SVM to predict user behavior from social network advertisement data. This model utilizes the SVM classifier from scikit-learn and applies K-Fold Cross Validation to estimate its accuracy.

Features
Data Preprocessing: Includes scaling features for better model performance.
SVM Classification: Utilizes the SVM classifier with a radial basis function (RBF) kernel.
K-Fold Cross Validation: Employs cross-validation to ensure the model's effectiveness across different subsets of the dataset.
Model Evaluation: Generates a confusion matrix and calculates the accuracy of the model.
Visualization: Provides functions to visualize decision boundaries for both training and test sets.

Dataset
The dataset (Social_Network_Ads.csv) includes user attributes such as Age and Estimated Salary, alongside a binary Purchase indicator. This dataset is ideal for demonstrating how machine learning can predict user behavior based on demographic variables.

Usage
Ensure Python and necessary libraries (numpy, pandas, matplotlib, sklearn) are installed. You can run the script by navigating to the directory containing k_fold_cross_validation.py and executing:


Results
Outputs include:
Confusion Matrix: Visualizing the accuracy of predictions.
Accuracy Score: Overall accuracy of the model on the test data.

