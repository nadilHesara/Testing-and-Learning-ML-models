import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


data = pd.read_csv("Titanic Dataset\Titanic-Dataset.csv")
print(data.head())
