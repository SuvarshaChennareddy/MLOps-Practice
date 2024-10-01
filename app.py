from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('stock_price_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Preprocess the data to match the model's input shape
    input_data = np.array(data['input']).reshape(1, -1, 1)  # Reshape to (1, n_timesteps, n_features)
    prediction = model.predict(input_data)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
