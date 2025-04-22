from flask import Flask, request, render_template
import pandas
import numpy as np
import pickle as pkl

model = pkl.load(open('model.pkl', 'rb'))

# Flask App
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pass

# Python Main
if __name__ == '__main__':
    app.run(debug=True)