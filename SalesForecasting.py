import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the datasets
stores = pd.read_csv('path_to_your/stores.csv')
features = pd.read_csv('path_to_your/features.csv')
train = pd.read_csv('path_to_your/train.csv')

# Merge the datasets on 'Store' and 'Date'
train = pd.merge(train, features, on=['Store', 'Date', 'IsHoliday'], how='left')
train = pd.merge(train, stores, on='Store', how='left')

# Fill missing values
train.fillna(0, inplace=True)

# Convert 'Date' into datetime format and extract more features
train['Date'] = pd.to_datetime(train['Date'])
train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month
train['Week'] = train['Date'].dt.isocalendar().week
train['Weekday'] = train['Date'].dt.weekday

# Select features for the regression model
features = ['Store', 'Dept', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday', 'Year', 'Month', 'Week', 'Weekday']
X = train[features]
y = train['Weekly_Sales']

# Convert 'IsHoliday' to integers
X['IsHoliday'] = X['IsHoliday'].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate and print performance metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Sales')
plt.show()
