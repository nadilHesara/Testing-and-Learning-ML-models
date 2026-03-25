import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. Separate features and target
X = train.drop(columns=["SalePrice"])
y = train["SalePrice"]

# 3. Identify column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# 4. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 5. Model
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# 6. Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# 7. Validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train
pipeline.fit(X_train, y_train)

# 9. Validate
valid_preds = pipeline.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))
print("Validation RMSE:", rmse)

# 10. Train on full data
pipeline.fit(X, y)

# 11. Predict test set
test_preds = pipeline.predict(test)

# 12. Save submission
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved")