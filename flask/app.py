from flask import Flask, render_template, request
import numpy as np
import pickle

# Load model
model = pickle.load(open(r"C:\Online Payments Fraud Detection\flask\payments.pkl", 'rb'))

app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("home.html")

# Predict Page
@app.route("/predict")
def predict_page():
    return render_template("predict.html")

# Prediction Route
@app.route("/submit", methods=['POST'])
def submit():

    data = list(request.form.values())
    data = [float(i) for i in data]

    # add missing feature
    data.append(0)

    x = np.array([data])

    pred = model.predict(x)

    # ⭐ CONVERT 0/1 → TEXT LABEL
    if pred[0] == 1:
        result = "is Fraud"
    else:
        result = "is not Fraud"

    return render_template("submit.html", prediction_text=result)




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

