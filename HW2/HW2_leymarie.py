# Standard libraries
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning and preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error, roc_auc_score

# Deep learning libraries and callbacks
import tensorflow as tf
import tensorflow.keras as kb
from tensorflow.keras import backend
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Datasets
from keras.datasets import mnist
from ucimlrepo import fetch_ucirepo

# Visualization
from plotnine import *

# Suppress warnings if needed
warnings.filterwarnings('ignore')

# fetch dataset
spambase = fetch_ucirepo(id=94)

# data (as pandas dataframes)
X = spambase.data.features
y = spambase.data.targets

X_df = X.copy()
data = X_df.copy()
data['target'] = y


preds = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
       'word_freq_our', 'word_freq_over', 'word_freq_remove',
       'word_freq_internet', 'word_freq_order', 'word_freq_mail',
       'word_freq_receive', 'word_freq_will', 'word_freq_people',
       'word_freq_report', 'word_freq_addresses', 'word_freq_free',
       'word_freq_business', 'word_freq_email', 'word_freq_you',
       'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000',
       'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
       'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',
       'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
       'word_freq_technology', 'word_freq_1999', 'word_freq_parts',
       'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
       'word_freq_original', 'word_freq_project', 'word_freq_re',
       'word_freq_edu', 'word_freq_table', 'word_freq_conference',
       'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
       'char_freq_$', 'char_freq_#', 'capital_run_length_average',
       'capital_run_length_longest', 'capital_run_length_total']


print(y['Class'].value_counts(normalize=True))
# print(X.describe())

corr_matrix = data.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

z = StandardScaler()
X_train[preds] = z.fit_transform(X_train[preds])
X_test[preds] = z.transform(X_test[preds])

model = kb.Sequential([
    kb.layers.Dense(64, input_shape=[57], activation='relu'),
    kb.layers.Dropout(0.2),
    kb.layers.Dense(32, activation='relu', kernel_regularizer=kb.regularizers.l1_l2(l1=0.0001, l2=0.0001)),
    kb.layers.Dropout(0.2),
    kb.layers.Dense(16, activation='relu', kernel_regularizer=kb.regularizers.l1_l2(l1=0.0001, l2=0.0001)),
    kb.layers.Dropout(0.2),
    kb.layers.Dense(8, activation='relu', kernel_regularizer=kb.regularizers.l1_l2(l1=0.0001, l2=0.0001)),
    kb.layers.Dropout(0.2),
    kb.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', metrics=['accuracy'],
              optimizer=kb.optimizers.Adam(learning_rate=0.001))

# Callbacks setup
early_stopping = kb.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

# Fit the model with callbacks
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32,
                    callbacks=[early_stopping])

best_epoch_loss = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
best_val_loss = min(history.history['val_loss'])
best_val_accuracy = history.history['val_accuracy'][best_epoch_loss - 1] # epoch is 1 indexed


print(f"Best Model Performance:")
print(f"- Validation Loss: {best_val_loss:.4f}")
print(f"- Validation Accuracy: {best_val_accuracy*100:.2f}%")
print(f"This was achieved at epoch {best_epoch_loss}.")

## Logistic Regression

lr = LogisticRegression()

lr.fit(X_train, y_train)

# training accuracy
training_accuracy = lr.score(X_train, y_train)

# Predict the labels of the test set
y_pred = lr.predict(X_test)

# Evaluate the model
test_accuracy = accuracy_score(y_test, y_pred)
prob_train = lr.predict_proba(X_train)[:, 1]
prob_test = lr.predict_proba(X_test)[:, 1]

# Calculate log loss (binary cross-entropy)
log_loss_train = log_loss(y_train, prob_train)
log_loss_test = log_loss(y_test, prob_test)

print("========Logistic Regression========")
print(f"Training Log Loss: {log_loss_train:.4f}")
print(f"Test Log Loss: {log_loss_test:.4f}")
print(f"Training Accuracy: {training_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")