import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


data = pd.read_csv("Titanic Dataset\Titanic-Dataset.csv")
#print(data.head())

target = "Survived"
X = data.drop([target,"Name", "Ticket", "Cabin"], axis=1)
y = data[target]

X["Sex"] = X["Sex"].map({"male":0, "female":1})
X = pd.get_dummies(X, columns = ["Embarked"], dtype = int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=100)

X_train.to_csv("X_train.csv")


model = XGBClassifier(
    n_estimators = 1000,
    learning_rate = 0.03,
    gamma = 0,
    max_depth=6,
    reg_alpha = 1,
    reg_lambda = 5,
    subsample = 0.8,
    colsample_by_tree=0.8,
    objective = 'binary:logistic',
    random_state=100
    )
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
scores = cross_val_score(model, X,y, cv=5, scoring="accuracy")
print(accuracy)
print(scores)
print(scores.mean())
print(scores.std())



