# CPSC 393 Project: Support Vector Machines

## Student Information

- **Name**: Lawrence Leymarie
- **Email**: leymarie@chapman.edu
- **ID**: 2370408
- **Course**: CPSC 393
- **File**: `Homework1_svm.py`

## Introduction

This project aims to predict the `Group` variable using all the X variables (X1, X2... X8) from the provided dataset. A thorough analysis was conducted to ensure the best model was chosen for deployment. The data was standardized using z-scores to normalize the feature scale across all variables.

## Analysis

### Data Preparation

The dataset underwent an 80/20 train-test split to ensure a robust evaluation of model performance. Initial exploratory analysis included summary statistics, correlation heatmaps, and distribution plots to understand the data's characteristics. Data cleaning involved handling missing values and outliers to ensure high data quality.

### Variable Description

The dataset consists of eight independent variables (X1 through X8), which are numerical and have been standardized. The dependent variable, `Group`, is categorical and represents the classification target.

## Methods

### Model Building

Three models were built using sklearn pipelines: SVM, Logistic Regression, and KNearest Neighbors. GridSearchCV was employed to fine-tune hyperparameters for each model, ensuring optimal performance.

- **SVM**: Explored linear and rbf kernels; C values of [0.001, 0.01, 1, 5, 25, 50]; and gamma values of [0.001, 0.01, 0.1, 0.5, 1, 2, 5].
- **Logistic Regression**: Standard implementation with regularization.
- **KNearest Neighbors**: GridSearch determined the optimal n_neighbors.

Hyperparameters such as C and gamma in SVM influence model complexity and how the model handles different degrees of separability between classes. In KNearest Neighbors, n_neighbors affects the model's sensitivity to noise in the training data.

### Model Evaluation

Models were evaluated based on train and test accuracies, ROC/AUC scores, and confusion matrices were plotted to assess performance visually.

## Results

The detailed performance of each model will be discussed, focusing on accuracy and ROC/AUC scores. The evaluation will help determine the most suitable model for production, taking into account the balance between overfitting and generalization.

## Reflection

This section will provide personal insights gained throughout the project, challenges encountered, and strategies for future projects. Reflections on model selection, data preprocessing, and the impact of hyperparameter tuning on model performance will be included.

## Credits

- OpenAI. (2024). ChatGPT (4) [Large language model]. https://chat.openai.com: Used to make this readme, responsible for computing sensitivity and specificity 
- C. Parlett (2023) https://github.com/cmparlettpelleriti/CPSC392ParlettPelleriti: Logistic Regression and KNN model code.