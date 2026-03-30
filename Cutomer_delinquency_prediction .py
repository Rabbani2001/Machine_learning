import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_excel("Delinquency_prediction_dataset.xlsx")


#print(df.info())

df["Income"].fillna(df["Income"].median(), inplace=True)
df["Loan_Balance"].fillna(df["Loan_Balance"].median(), inplace=True)

months = ["Month_1","Month_2","Month_3","Month_4","Month_5","Month_6"]

df["missed_payments"] = df[months].apply(lambda x: sum(x == "Missed"), axis=1)
df["late_payments"] = df[months].apply(lambda x: sum(x == "Late"), axis=1)


X = df.drop(columns=["Delinquent_Account"])
y = df["Delinquent_Account"]

# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_cols = ["Employment_Status", "Location"]
numerical_cols = X.select_dtypes(include=np.number).columns

# Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_cols),
    ("cat", cat_pipeline, categorical_cols)
])

# Final pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced"))
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced"))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))