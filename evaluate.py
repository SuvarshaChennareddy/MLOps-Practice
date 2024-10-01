from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the test data
test_data = pd.read_csv('test_data.csv', index_col='Datetime', parse_dates=True)
test_data = test_data['Close'].astype(float)

# Load both models for evaluation
try:
    old_model = load_model('stock_price_model.h5')
except:
    old_model = None  # If no previous model exists

new_model = load_model('new_stock_price_model.h5')


# Prepare the data for evaluation
n_timesteps = 10
X_eval, y_eval = [], []

for i in range(n_timesteps, len(test_data)):
    X_eval.append(test_data[i-n_timesteps:i].values)
    y_eval.append([test_data[i]])

X_eval = np.array(X_eval)
y_eval = np.array(y_eval)

# Reshape data for LSTM input
X_eval = X_eval.reshape((X_eval.shape[0], X_eval.shape[1], 1))

# Evaluate the new model
new_predictions = new_model.predict(X_eval)

new_mse = mean_squared_error(y_eval, new_predictions)

# If there's no old model, consider the new model as better
if old_model is None:
    old_mse = float('inf')  # Assume an infinitely bad old model score
else:
    old_predictions = old_model.predict(X_eval)
    old_mse = mean_squared_error(y_eval, old_predictions)

# Compare the new model's performance with the old model
if new_mse < old_mse:
    with open('better.txt', 'w') as f:
        f.write('true')
else:
    with open('better.txt', 'w') as f:
        f.write('false')
