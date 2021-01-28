import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import random
import time

random.seed(100)

### Data Preprocessing ###
dataset = pd.read_csv('financial_data.csv')

# Feature Engineering
dataset = dataset.drop(columns = ['months_employed'])
dataset['personal_account_months'] = dataset.personal_account_m + (dataset.personal_account_y * 12)
dataset[['personal_account_m', 'personal_account_y', 'personal_account_months']].head()
dataset = dataset.drop(columns = ['personal_account_m', 'personal_account_y'])

# One Hot Encoding
dataset = pd.get_dummies(dataset)
dataset.columns
dataset = dataset.drop(columns = ['pay_schedule_semi-monthly'])

# Removing Extra Columns
response = dataset["e_signed"]
users = dataset['entry_id']
dataset = dataset.drop(columns = ['e_signed', 'entry_id'])

# Splitting into Train & Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset, response, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train2 = pd.DataFrame(sc_x.fit_transform(x_train))
x_test2 = pd.DataFrame(sc_x.transform(x_test))
x_train2.columns = x_train.columns.values
x_test2.columns = x_test.columns.values
x_train2.index = x_train.index.values
x_test2.index = x_test.index.values
x_train = x_train2
x_test = x_test2

### Model Building ###
# (Comapring Different Models)

## Logistic Regression
from sklearn.linear_model import LogisticRegression
cls = LogisticRegression(random_state = 0, solver = 'saga', penalty = 'l1')
cls.fit(x_train, y_train)

# Predicting Test Set
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
y_pred = cls.predict(x_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame(
    [['Logistic Regression (Lasso)', acc, prec, rec, f1]],
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
)

## Support Vector Machine (Linear)
from sklearn.svm import SVC
cls = SVC(random_state = 0, kernel = 'linear')
cls.fit(x_train, y_train)

# Predicting Test Set
y_pred = cls.predict(x_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame(
    [['SVM (Linear)', acc, prec, rec, f1]],
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
)
results = results.append(model_results, ignore_index = True)

## Support Vector Machine (RBF)
from sklearn.svm import SVC
cls = SVC(random_state = 0, kernel = 'rbf')
cls.fit(x_train, y_train)

# Predicting Test Set
y_pred = cls.predict(x_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame(
    [['SVM (RBF)', acc, prec, rec, f1]],
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
)
results = results.append(model_results, ignore_index = True)

## Random Forest Model
from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier(random_state = 0, n_estimators = 100, criterion = 'entropy')
cls.fit(x_train, y_train)

# Predicting Test Set
y_pred = cls.predict(x_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame(
    [['Random Forest', acc, prec, rec, f1]],
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
)
results = results.append(model_results, ignore_index = True)

## K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = cls, X = x_train, y = y_train, cv = 10)
print("Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)" % (accuracies.mean(), accuracies.std() * 2))

### Parameter Tuning ###

## Applying Grid Search
# pip install joblib
# conda install joblib
from sklearn.model_selection import GridSearchCV

# Round 1: Entropy
parameters = {
    "max_depth": [3, None],
    "max_features": [1, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 5, 10],
    "bootstrap": [True, False],
    "criterion": ["entropy"]    
}
grid_search = GridSearchCV(
    estimator = cls, 
    param_grid = parameters, 
    scoring = "accuracy", 
    cv = 10, 
    n_jobs = -1
)
t0 = time.time()
grid_search = grid_search.fit(x_train, y_train)
t1 = time.time()
print("Took %0.2f Seconds" % (t1-t0))

rf_best_accuracy = grid_search.best_score_
rf_best_params = grid_search.best_params_
print(rf_best_accuracy, rf_best_params)

# Results of Round 1
# rf_best_accuracy
# Out[47]: 0.6345122647725013

# rf_best_params
# Out[48]: 
# {'bootstrap': True,
#  'criterion': 'entropy',
#  'max_depth': None,
#  'max_features': 5,
#  'min_samples_leaf': 5,
#  'min_samples_split': 2}

# Round 2: Entropy
parameters = {
    "max_depth": [None],
    "max_features": [3, 5, 7],
    "min_samples_split": [2, 3, 4],
    "min_samples_leaf": [3, 5, 7],
    "bootstrap": [True],
    "criterion": ["entropy"]    
}
grid_search = GridSearchCV(
    estimator = cls, 
    param_grid = parameters, 
    scoring = "accuracy", 
    cv = 10, 
    n_jobs = -1
)
t0 = time.time()
grid_search = grid_search.fit(x_train, y_train)
t1 = time.time()
print("Took %0.2f Seconds" % (t1-t0))

rf_best_accuracy = grid_search.best_score_
rf_best_params = grid_search.best_params_
print(rf_best_accuracy, rf_best_params)

# Results of Round 2
# rf_best_accuracy
# Out[50]: 0.6345122647725013

# rf_best_params
# Out[51]: 
# {'bootstrap': True,
#  'criterion': 'entropy',
#  'max_depth': None,
#  'max_features': 5,
#  'min_samples_leaf': 5,
#  'min_samples_split': 2}

# Predicting Test Set
y_pred = grid_search.predict(x_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame(
    [['Random Forest (n=100, GSx2 + Entropy)', acc, prec, rec, f1]],
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
)
results = results.append(model_results, ignore_index = True)

# Round 1: Gini
parameters = {
    "max_depth": [3, None],
    "max_features": [1, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 5, 10],
    "bootstrap": [True, False],
    "criterion": ["gini"]    
}
grid_search = GridSearchCV(
    estimator = cls, 
    param_grid = parameters, 
    scoring = "accuracy", 
    cv = 10, 
    n_jobs = -1
)
t0 = time.time()
grid_search = grid_search.fit(x_train, y_train)
t1 = time.time()
print("Took %0.2f Seconds" % (t1-t0))

rf_best_accuracy = grid_search.best_score_
rf_best_params = grid_search.best_params_
print(rf_best_accuracy, rf_best_params)

# Results of Round 1
# rf_best_accuracy
# Out[55]: 0.6353512282315882

# rf_best_params
# Out[56]: 
# {'bootstrap': True,
#  'criterion': 'gini',
#  'max_depth': None,
#  'max_features': 10,
#  'min_samples_leaf': 5,
#  'min_samples_split': 2}

# Round 2: Gini
parameters = {
    "max_depth": [None],
    "max_features": [8, 10, 12],
    "min_samples_split": [2, 3, 4],
    "min_samples_leaf": [3, 5, 7],
    "bootstrap": [True],
    "criterion": ["gini"]    
}
grid_search = GridSearchCV(
    estimator = cls, 
    param_grid = parameters, 
    scoring = "accuracy", 
    cv = 10, 
    n_jobs = -1
)
t0 = time.time()
grid_search = grid_search.fit(x_train, y_train)
t1 = time.time()
print("Took %0.2f Seconds" % (t1-t0))

rf_best_accuracy = grid_search.best_score_
rf_best_params = grid_search.best_params_
print(rf_best_accuracy, rf_best_params)

# Results of Round 2
# rf_best_accuracy
# Out[58]: 0.6327675755437474

# rf_best_params
# Out[59]: 
# {'bootstrap': True,
#  'criterion': 'gini',
#  'max_depth': None,
#  'max_features': 5,
#  'min_samples_leaf': 7,
#  'min_samples_split': 2}

# Predicting Test Set
y_pred = grid_search.predict(x_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame(
    [['Random Forest (n=100, GSx2 + Gini)', acc, prec, rec, f1]],
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
)
results = results.append(model_results, ignore_index = True)

### End of Model ###

# Formatting Final Results
final_results = pd.concat([y_test, users], axis = 1).dropna()
final_results['predictions'] = y_pred
final_results = final_results[['entry_id', 'e_signed', 'predictions']]
