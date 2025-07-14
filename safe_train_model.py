import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Read CSV with error handling
df = pd.read_csv("train.csv", encoding='latin1', on_bad_lines='skip')
df.dropna(inplace=True)

# Encode categorical variables
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']:
    df[col] = LabelEncoder().fit_transform(df[col])

df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "loan_approval_model.pkl")
print("✅ Model saved as loan_approval_model.pkl")
