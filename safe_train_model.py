import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("train.csv")  # update the path if needed
df.dropna(inplace=True)

# Label encode
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']:
    df[col] = le.fit_transform(df[col])

df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model (pure sklearn, no custom code)
joblib.dump(model, "loan_approval_model.pkl")
