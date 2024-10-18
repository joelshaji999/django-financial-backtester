import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample stock data (replace with your real data from your database)
# Assuming StockData has 'date' and 'close_price' columns
# Example: df = pd.DataFrame(list(StockData.objects.values('date', 'close_price')))
df = pd.DataFrame({
    'date': pd.date_range(start='2022-01-01', periods=200, freq='D'),
    'close_price': pd.Series(range(200)) + pd.Series([i * 0.5 for i in range(200)])
})

# Create a feature 'days' to represent the number of days since the first date
df['days'] = (df['date'] - df['date'].min()).dt.days

# Features (X) are the number of days; target (y) is the closing price
X = df[['days']]
y = df['close_price']

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model and train it on the data
model = LinearRegression()
model.fit(X_train, y_train)

# Check how well the model fits the data (optional step)
print(f"Model R^2 score: {model.score(X_test, y_test)}")

# Save the trained model to a pickle file
with open('linear_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully.")