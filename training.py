import pandas as pd
import numpy as np


print("\n\nLoading dataset")

df = pd.read_csv('voice_preprocessed.csv')


## Stratified K Fold Cross Validation

print("\n\nInitializing training function with Stratified K Fold")

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train(data,model, describe=True):
  X = data.drop('gender_encoded',axis=1)
  y = data[['gender_encoded']]
  rskf = RepeatedStratifiedKFold(n_splits=7,
                   n_repeats=3,
                   random_state=42)

  lst_accu_stratified = []
  best_accu = -1
  best_train_index = -1
  best_test_index = -1
  for train_index, test_index in rskf.split(X, y):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(x_train, np.ravel(y_train))
    score=model.score(x_test, y_test)
    lst_accu_stratified.append(score)
    if score > best_accu:
      best_train_index = train_index
      best_test_index = test_index

  x_train, x_test = X.iloc[best_train_index], X.iloc[best_test_index]
  y_train, y_test = y.iloc[best_train_index], y.iloc[best_test_index]
  model.fit(x_train, np.ravel(y_train))

  max_accuracy=max(lst_accu_stratified)*100
  mean_accuracy = np.mean(lst_accu_stratified)*100

  c_matrix = confusion_matrix(y_test, model.predict(x_test))

  if describe:
    print('\nMaximum Accuracy:', max_accuracy, '%')
    print('\nMean Accuracy:', mean_accuracy, '%')

    print("\nSelecting best accuracy split:")

    print("Train data shape:{}".format(x_train.shape))
    print("Test data shape:{}".format(x_test.shape))

  return model,max_accuracy, mean_accuracy, c_matrix


## Training with various ML Models

import joblib
import matplotlib.pyplot as plt

label_encode = joblib.load("label_encode.pkl")


print("\n\nTraining with Logistic regression:")

from sklearn.linear_model import LogisticRegression
model, best_accuracy, mean_accuracy, c_matrix = train(data=df,
             model=LogisticRegression())
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,
                              display_labels=list(label_encode.inverse_transform(model.classes_)))
disp.plot()
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()

joblib.dump(model, 'models/log_reg.pkl')
print('\nLogistic Regression model Saved Successfully')


print("\n\nTraining with SVM:")

from sklearn import svm
model, best_accuracy, mean_accuracy, c_matrix = train(data=df,
             model=svm.SVC(kernel='linear'))
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,
                              display_labels=list(label_encode.inverse_transform(model.classes_)))
disp.plot()
plt.title("Confusion Matrix (SVM)")
plt.show()

joblib.dump(model, 'models/svm.pkl')
print('\nSVM model Saved Successfully')


print("\n\nTraining with RandomForest:")

from sklearn.ensemble import RandomForestClassifier
model, best_accuracy, mean_accuracy, c_matrix = train(data=df,
             model=RandomForestClassifier())
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,
                              display_labels=list(label_encode.inverse_transform(model.classes_)))
disp.plot()
plt.title("Confusion Matrix (Random Forest)")
plt.show()

joblib.dump(model, 'models/random_forest.pkl')
print('\nRandomForest model Saved Successfully')


print("\n\nTraining with KNN:")

from sklearn.neighbors import KNeighborsClassifier
best_model = None
best_score = 0
best_mean_accuracy = 0
best_c_matrix = None
for n in range(5,20):
  model, best_accuracy, mean_accuracy, c_matrix = train(data=df,
              model=KNeighborsClassifier(n_neighbors = n),describe=False)
  if best_accuracy > best_score:
    best_model = model
    best_mean_accuracy = mean_accuracy
    best_score = best_accuracy
    best_c_matrix = c_matrix

disp = ConfusionMatrixDisplay(confusion_matrix=best_c_matrix,
                              display_labels=list(label_encode.inverse_transform(best_model.classes_)))
disp.plot()
plt.title("Confusion Matrix (KNN)")
plt.show()

print("\nSelecting best accuracy model for KNN")

print('\nMaximum Accuracy:', best_accuracy, '%')
print('\nMean Accuracy:', best_mean_accuracy, '%')

joblib.dump(best_model, 'models/knn.pkl')
print('\nKNN model Saved Successfully')