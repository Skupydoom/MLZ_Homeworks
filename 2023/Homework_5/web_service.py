from flask import Flask, request
from ml_script import load_data


app = Flask("credit")

@app.route("/predict", methods=["POST"])
def predict():
    client = request.get_json()
    dv, model = load_data()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    give_credit = y_pred >= 0.5

    result = {
        "probability": float(y_pred),
        "give_credit": bool(give_credit)
    }

    return result

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
