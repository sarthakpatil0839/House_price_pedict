from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/house_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", result="")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    rooms = int(request.form["rooms"])
    age = int(request.form["age"])

    prediction = model.predict([[area, rooms, age]])

    return render_template(
        "index.html",
        result=f"Predicted House Price: ₹{int(prediction[0])}"
    )

if __name__ == "__main__":
    app.run(debug=True)
