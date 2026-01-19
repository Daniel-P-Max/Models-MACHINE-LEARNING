from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# -------------------------------------
# 1. LOAD TRAINED MODEL
# -------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear_model.pkl')

try:
    linear_model = pickle.load(open(MODEL_PATH, 'rb'))
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the model file at {MODEL_PATH}. Make sure 'linear_model.pkl' exists.")

# -------------------------------------
# 2. CREATE FLASK APP
# -------------------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return "Linear Regression Model API is Running!"

# -------------------------------------
# 3. PREDICTION ENDPOINT
# -------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400

        features = np.array(data['features']).reshape(1, -1)
        prediction = linear_model.predict(features)[0]

        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------
# 4. RUN THE APP
# -------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
