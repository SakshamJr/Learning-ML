from flask import Flask, request, render_template
import pandas
import numpy as np
import pickle as pkl

try:
    model = pkl.load(open('Projects-Vacation/01_Breast Cancer/model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: model.pkl file not found. Please run train_model.py first.")
    exit(1)
# Flask App
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['features']
    features = [float(x) for x in features.split(',')]
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)
    prediction_proba = prediction_proba[0][1] * 100
    if prediction[0] == 1:
        result = f"Prediction: Malignant \n (Probability: {prediction_proba:.2f}%)"
    else:
        result = f"Prediction: Benign \n (Probability: {100-prediction_proba:.2f}%)"
    return render_template('index.html', prediction=result)

# Python Main
if __name__ == '__main__':
    app.run(debug=True)