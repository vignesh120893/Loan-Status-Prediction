import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
train_data_path = '/path/to/Loan_Status_train.csv'
train_data = pd.read_csv(train_data_path)

# Fill missing values for categorical columns with the mode
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Education', 'Property_Area']
for col in categorical_cols:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)

# Fill missing values for numerical columns with the median
numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'ApplicantIncome', 'CoapplicantIncome']
for col in numerical_cols:
    train_data[col].fillna(train_data[col].median(), inplace=True)

# Encoding categorical variables using one-hot encoding
train_encoded = pd.get_dummies(train_data, columns=categorical_cols, drop_first=True)

# Handling outliers by capping using the 99th percentile
for col in numerical_cols:
    cap_value = train_encoded[col].quantile(0.99)
    train_encoded[col] = train_encoded[col].clip(upper=cap_value)

# Normalizing numerical features using MinMaxScaler
scaler = MinMaxScaler()
train_encoded[numerical_cols] = scaler.fit_transform(train_encoded[numerical_cols])

# Splitting the dataset into training and test sets
X = train_encoded.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = train_encoded['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)  # Convert to binary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train models
log_reg = LogisticRegression(max_iter=300)
dec_tree = DecisionTreeClassifier()
rand_forest = RandomForestClassifier()

# Train Logistic Regression
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Train Decision Tree
dec_tree.fit(X_train, y_train)
y_pred_dec_tree = dec_tree.predict(X_test)

# Train Random Forest
rand_forest.fit(X_train, y_train)
y_pred_rand_forest = rand_forest.predict(X_test)

# Evaluate all models
results = {
    'Logistic Regression': {
        'Accuracy': accuracy_score(y_test, y_pred_log_reg),
        'Precision': precision_score(y_test, y_pred_log_reg),
        'Recall': recall_score(y_test, y_pred_log_reg),
        'F1 Score': f1_score(y_test, y_pred_log_reg)
    },
    'Decision Tree': {
        'Accuracy': accuracy_score(y_test, y_pred_dec_tree),
        'Precision': precision_score(y_test, y_pred_dec_tree),
        'Recall': recall_score(y_test, y_pred_dec_tree),
        'F1 Score': f1_score(y_test, y_pred_dec_tree)
    },
    'Random Forest': {
        'Accuracy': accuracy_score(y_test, y_pred_rand_forest),
        'Precision': precision_score(y_test, y_pred_rand_forest),
        'Recall': recall_score(y_test, y_pred_rand_forest),
        'F1 Score': f1_score(y_test, y_pred_rand_forest)
    }
}

print(results)
