import numpy as np
import pickle


def load_data():
    # The docker container has 2 different models.
    # So, write "model1.bin" instead of the current
    # string below to change the model to the first one:
    # ------------------------------------------------------------------------------
    model_file = "model2.bin"
    # ------------------------------------------------------------------------------
    dv_file = "dv.bin"

    with open(model_file, "rb") as f_in:
        model = pickle.load(f_in)

    with open(dv_file, "rb") as f_in:
        dv = pickle.load(f_in)

    print(f"Dv and model equal respectively: {dv}, {model}")

    return dv, model


def test_model(data):
    X = dv.transform([client])
    accuracy = model.predict_proba(X)[0, 1]
    print("Model's accuracy:", accuracy)


if __name__ == "__main__":
    dv, model = load_data()
    client = {"job": "retired", "duration": 445, "poutcome": "success"}
    test_model(client)
