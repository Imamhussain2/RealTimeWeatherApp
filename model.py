import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Example training data
data = {
    'humidity': [30, 45, 60, 75],
    'city_bengaluru': [1, 0, 0, 0],
    'city_delhi': [0, 1, 0, 0],
    'city_mumbai': [0, 0, 1, 1],
}
X = pd.DataFrame(data)
y = [25, 28, 35, 33]  # example temperatures

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')
