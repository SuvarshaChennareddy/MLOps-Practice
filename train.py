import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the training data
train_data = pd.read_csv('train_data.csv', index_col='Datetime', parse_dates=True)
train_data = train_data['Close'].astype(float)

# Prepare the data for LSTM model
n_timesteps = 10
X_train, y_train = [], []

for i in range(n_timesteps, len(train_data)):
    X_train.append(train_data[i-n_timesteps:i].values)
    y_train.append(train_data[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape data for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))  # Predict the next stock price

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Save the model
model.save('new_stock_price_model.h5')
print("Model trained and saved as new_stock_price_model.h5")
