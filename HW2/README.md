# Homework 2 Technical Report - Feed Forward Neural Networks

## Student Information

- **Name**: Lawrence Leymarie
- **Email**: [leymarie@chapman.edu](mailto:leymarie@chapman.edu)
- **Student ID**: 2370408
- **Course**: CPSC-393, Machine Learning

## Introduction

This report explores the construction and evaluation of a Deep Feed Forward Neural Network to classify emails as spam or not, utilizing the Spambase dataset. The significance of accurately detecting spam lies in improving email filtering systems, thereby enhancing user experience and security. A comparison with a simpler machine learning model, specifically Logistic Regression, serves to validate the necessity and efficiency of deploying a deep learning approach for this task.

## Analysis

The Spambase dataset comprises 57 features representing the frequency of specific words and characters in emails, alongside continuous attributes describing capital letter usage. Initial exploratory data analysis revealed a balanced distribution of classes, with a slight predominance of non-spam emails. Correlation analysis highlighted moderate relationships among some features, indicating potential predictive value for spam detection.

## Methods

The neural network architecture was designed with an input layer corresponding to the 57 features, followed by three hidden layers with 64, 32, and 16 neurons, respectively. Each hidden layer incorporated dropout and L1/L2 regularization to mitigate overfitting. The output layer employed a sigmoid activation function for binary classification. Data preprocessing involved standard scaling of features to normalize the input distribution. The model was compiled with a binary cross-entropy loss function and optimized using the Adam optimizer.

## Results

The neural network achieved a validation accuracy of approximately 92%, illustrating its capability to effectively distinguish between spam and non-spam emails. In contrast, the Logistic Regression model, serving as the comparative simpler machine learning approach, attained a test accuracy near 91%. While both models performed commendably, the slight edge in accuracy by the neural network underscores its potential benefits, albeit at the cost of increased complexity and computational resources.

## Reflection

The process underscored the importance of regularization and proper data scaling in enhancing model performance. Despite the neural network's marginally superior accuracy, the logistic regression model's comparable performance raises questions about the necessity and trade-offs of employing deep learning for this specific task. Future investigations might explore feature engineering, alternative model architectures, and cost-sensitive learning to further optimize spam detection.

## Credits

- **Lawrence Leymarie**: Author, Data Analysis, Model Implementation, Report Writing
- **Data Source**: UCI Machine Learning Repository - Spambase Dataset
- **Libraries Used**: TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **README Prepared By**: OpenAI. (2024). ChatGPT (4) [Large language model]. https://chat.openai.com
