import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("churn.csv")
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df['gender'].value_counts())

# Drop customerID as it's just an identifier
df = df.drop('customerID', axis=1)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, drop_first=True)
# print("\nFirst row after encoding:")
# print(df.iloc[0])

df = df.fillna(df.mean())

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})
# Sort by importance and show top 10
print("\nTop 10 Most Important Features for Churn Prediction:")
print(feature_importance.sort_values('importance', ascending=False).head(10))

# Evaluate models
models = {'Logistic Regression': log_reg, 'Random Forest': rf}
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Model Evaluation:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

sns.heatmap(confusion_matrix(y_test, rf.predict(X_test)), annot=True, fmt='d')
plt.title("Random Forest Confusion Matrix")
plt.show()

joblib.dump(rf, "churn_model.pkl")
joblib.dump(log_reg, "logreg_model.pkl")

model = joblib.load("churn_model.pkl")
# Save feature names
joblib.dump(list(X.columns), "model_columns.pkl")
