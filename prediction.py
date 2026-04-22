import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv(r"C:\Users\HP\Desktop\mlops\ml_salary_data.csv")

# Encode categorical column
le = LabelEncoder()
df["Education"] = le.fit_transform(df["Education"])

# Features & target
X = df[["Age", "Experience", "Education"]]
y = df["Salary"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)