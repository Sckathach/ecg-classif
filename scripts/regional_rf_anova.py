import pickle

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

with open("data/processed/features.pkl", "rb") as f:
    data = pickle.load(f)

loo = LeaveOneOut()
model = RandomForestClassifier(n_estimators=100, random_state=33)

X = data["regional"]["ALPHA"]["X"]
y = data["regional"]["ALPHA"]["y"]


y_true = []
y_pred = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    scaler.fit(X_train)
    selector = SelectKBest(f_classif, k=10)
    selector.fit(X_train, y_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    y_true.append(y_test[0])
    y_pred.append(pred[0])

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
logger.info(f"{acc}, {cm}")
