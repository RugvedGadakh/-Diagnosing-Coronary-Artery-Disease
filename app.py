from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained models
model = load_model(r'D:\ADCET_HACKATHON\New folder\categorical_model.h5')
binary_model = load_model(r'D:\ADCET_HACKATHON\New folder\binary_model.h5')

# Load the scaler (used during training)
scaler = StandardScaler()
scaler.mean_ = np.array([54.43, 0.68, 3.17, 131.69, 246.69, 0.15, 0.53, 149.61, 0.33, 1.05, 1.6, 0.67, 4.78])
scaler.scale_ = np.array([9.05, 0.47, 0.95, 17.54, 52.16, 0.36, 0.53, 22.86, 0.47, 1.16, 0.61, 0.93, 1.95])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [float(data[f'feature{i}']) for i in range(1, 15)]

        # Preprocess the input data using the scaler
        features_scaled = scaler.transform([features])

        # Make predictions using both models
        categorical_pred = int(np.argmax(model.predict(features_scaled), axis=1)[0])
        binary_pred = int(np.round(binary_model.predict(features_scaled))[0])

        return jsonify({'categorical_pred': categorical_pred, 'binary_pred': binary_pred})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
