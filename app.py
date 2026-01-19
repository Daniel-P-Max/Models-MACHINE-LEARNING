from flask import Flask, request, jsonify
import pickle
import numpy as np

# -------------------------------------
# 1. LOAD TRAINED MODEL(S)
# -------------------------------------
# Example: you saved your model as a pickle file
# pickle.dump(linear_model, open('linear_model.pkl', 'wb'))
# pickle.dump(ridge, open('ridge_model.pkl', 'wb'))
# pickle.dump(lasso, open('lasso_model.pkl', 'wb'))

linear_model = pickle.load(open('linear_model.pkl', 'rb'))
ridge_model = pickle.load(open('ridge_model.pkl', 'rb'))
lasso_model = pickle.load(open('lasso_model.pkl', 'rb'))

# -------------------------------------
# 2. CREATE FLASK APP
# -------------------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return "Regression Model API is Running!"

# -------------------------------------
# 3. PREDICTION ENDPOINT
# -------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Expecting input like: {"features": [value1, value2, ...], "model": "linear"}
    features = np.array(data['features']).reshape(1, -1)
    model_choice = data.get('model', 'linear').lower()
    
    if model_choice == 'linear':
        prediction = linear_model.predict(features)[0]
    elif model_choice == 'ridge':
        prediction = ridge_model.predict(features)[0]
    elif model_choice == 'lasso':
        prediction = lasso_model.predict(features)[0]
    else:
        return jsonify({"error": "Invalid model name. Choose 'linear', 'ridge', or 'lasso'."})
    
    return jsonify({"prediction": float(prediction)})

# -------------------------------------
# 4. RUN THE APP
# -------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
