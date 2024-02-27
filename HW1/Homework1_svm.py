
import warnings
warnings.filterwarnings('ignore')

# data and plotting
import pandas as pd
import numpy as np
from plotnine import *

# preprocessing
from sklearn.preprocessing import StandardScaler #Z-score variables
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
import matplotlib.pyplot as plt

# metrics
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, ConfusionMatrixDisplay, roc_auc_score, recall_score, precision_score

# models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("https://raw.githubusercontent.com/cmparlettpelleriti/CPSC393ParlettPelleriti/main/Data/hw1.csv")
data = data.dropna()
print(data.head())

bounds = []
for feature in data:
  if(feature == "Group"):
    break
  print(feature, "", np.mean(data[feature]))

for feature in data:
  if(feature == "Group"):
    break
  bounds.append(np.min(data[feature]))
  bounds.append(np.max(data[feature]))

print("min:", min(bounds), "max:", max(bounds))

numa = 0
for pred in data["Group"]:
  if pred == 'A':
    numa += 1
print("percentage 'A'", numa/len(data["Group"]) * 100)
print("percentage 'B'", (1-numa/len(data["Group"])) * 100)

predictors = [c for c in data.columns if c != "Group"]

X = data[predictors]
y = data["Group"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

z = make_column_transformer((StandardScaler(), predictors),
                            remainder = "passthrough")
svm = SVC(probability=True)

# build pipeline
pipe = Pipeline([("pre", z), ("model", svm)])

# parameters dict
params = {"model__C": [0.001, 0.01, 1, 5, 25, 50],
          "model__gamma": [0.001,0.01, 0.1, 0.5, 1,2,5],
          "model__kernel": ["linear", "rbf"]}
# grid Search
grid = GridSearchCV(pipe, params, scoring = "roc_auc", cv = 5, refit = True)

grid.fit(X_train, y_train)

best_params = grid.best_params_

# Predict on training and testing sets
y_train_pred = grid.predict(X_train)
y_test_pred = grid.predict(X_test)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate ROC/AUC scores
y_train_prob = grid.predict_proba(X_train)[:, 1]
y_test_prob = grid.predict_proba(X_test)[:, 1]
train_roc_auc = roc_auc_score(y_train, y_train_prob)
test_roc_auc = roc_auc_score(y_test, y_test_prob)

print("best Parameters:", best_params)
print("\n-----Accuracy Scores-----")
print("Train:", train_accuracy)
print("Test", test_accuracy)
print("\n-----ROC AUC Scores-----")
print("Train", train_roc_auc)
print("Test", test_roc_auc)

# Compute confusion matrices
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

# Plotting the confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Training set confusion matrix
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=grid.classes_)
disp_train.plot(cmap='Blues', ax=ax[0])
ax[0].set_title('Training Set Confusion Matrix')

# Testing set confusion matrix
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=grid.classes_)
disp_test.plot(cmap='Blues', ax=ax[1])
ax[1].set_title('Testing Set Confusion Matrix')

# Function to calculate sensitivity and specificity from confusion matrix
def calculate_metrics(cm):
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sensitivity, specificity

# Calculate metrics for training and testing sets
sensitivity_train, specificity_train = calculate_metrics(cm_train)
sensitivity_test, specificity_test = calculate_metrics(cm_test)

print(sensitivity_train, specificity_train)
print(sensitivity_test, specificity_test)

plt.tight_layout()
plt.show()

# Create Empty Model
pre = make_column_transformer((StandardScaler(), predictors),
                              remainder = "passthrough")
lr = LogisticRegression()

pipe = Pipeline([("pre", pre), ("model", lr)])

# fit
pipe.fit(X_train, y_train)

# predict

y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)

# Probabilities
y_pred_train_prob = pipe.predict_proba(X_train)[:,1]
y_pred_test_prob = pipe.predict_proba(X_test)[:,1]

y_pred_train_prob = pipe.predict_proba(X_train)[:,1]
y_pred_test_prob = pipe.predict_proba(X_test)[:,1]

# assess

print("-----Accuracy Scores-----")
print("Train:", round(accuracy_score(y_train, y_pred_train), 3))
print("Test", round(accuracy_score(y_test, y_pred_test), 3, ))
print("-----ROC AUC Scores-----")
print("Train", round(roc_auc_score(y_train, y_pred_train_prob), 3))
print("Test", round(roc_auc_score(y_test, y_pred_test_prob), 3))

# Compute confusion matrices
cm_train_lr = confusion_matrix(y_train, y_pred_train)
cm_test_lr = confusion_matrix(y_test, y_pred_test)

# Plotting the confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Training set confusion matrix
disp_train_lr = ConfusionMatrixDisplay(confusion_matrix=cm_train_lr)
disp_train_lr.plot(cmap='Blues', ax=ax[0])
ax[0].set_title('Training Set Confusion Matrix')

# Testing set confusion matrix
disp_test_lr = ConfusionMatrixDisplay(confusion_matrix=cm_test_lr)
disp_test_lr.plot(cmap='Blues', ax=ax[1])
ax[1].set_title('Testing Set Confusion Matrix')

# Calculate sens/spec for training and testing sets
sensitivity_train, specificity_train = calculate_metrics(cm_train_lr)
sensitivity_test, specificity_test = calculate_metrics(cm_test_lr)

print(sensitivity_train, specificity_train)
print(sensitivity_test, specificity_test)

plt.tight_layout()
plt.show()

knn = KNeighborsClassifier()

# create z score object
z = make_column_transformer((StandardScaler(), predictors),
                            remainder = "passthrough")

# make pipeline
pipe = Pipeline([("pre", z), ("model", knn)])

# choose potential values of k
ks = {"model__n_neighbors": [i for i in range(1, 20)]}

# use grid search to find best parameters
grid_knn = GridSearchCV(pipe, ks, scoring = "precision", cv = 5, refit = True)

grid_knn.fit(X_train, y_train)

print("GridSearchCV chose: ", grid_knn.best_estimator_.get_params()["model__n_neighbors"])

# predict
y_pred_train = grid_knn.predict(X_train)
y_pred_test = grid_knn.predict(X_test)

y_pred_train_prob = grid_knn.predict_proba(X_train)[:,1]
y_pred_test_prob = grid_knn.predict_proba(X_test)[:,1]

# assess

print("-----Accuracy Scores-----")
print("Train:", round(accuracy_score(y_train, y_pred_train), 3))
print("Test", round(accuracy_score(y_test, y_pred_test), 3, ))
print("-----ROC AUC Scores-----")
print("Train", round(roc_auc_score(y_train, y_pred_train_prob), 3))
print("Test", round(roc_auc_score(y_test, y_pred_test_prob), 3))

# Compute confusion matrices
cm_train_knn = confusion_matrix(y_train, y_pred_train)
cm_test_knn = confusion_matrix(y_test, y_pred_test)

# Plotting the confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Training set confusion matrix
disp_train_knn = ConfusionMatrixDisplay(confusion_matrix=cm_train_knn, display_labels=grid.classes_)
disp_train_knn.plot(cmap='Blues', ax=ax[0])
ax[0].set_title('Training Set Confusion Matrix')

# Testing set confusion matrix
disp_test_knn = ConfusionMatrixDisplay(confusion_matrix=cm_test_knn, display_labels=grid.classes_)
disp_test_knn.plot(cmap='Blues', ax=ax[1])
ax[1].set_title('Testing Set Confusion Matrix')

sensitivity_train, specificity_train = calculate_metrics(cm_train_knn)
sensitivity_test, specificity_test = calculate_metrics(cm_test_knn)

print(sensitivity_train, specificity_train)
print(sensitivity_test, specificity_test)

plt.tight_layout()
plt.show()