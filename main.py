from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import datetime

app = Flask(__name__)

# Load trained model
regressor = pickle.load(open("E:\Car Price Prediction\models\LinearRegressorModel.pkl", "rb"))

# Load dataset
car = pd.read_csv("E:\Car Price Prediction\data\My_Clean_Data_final.csv")

# ---------- Utility: Normalize Colors ----------
def normalize_color(color):
    color = str(color).lower()
    if "silver" in color: return "Silver"
    elif "grey" in color or "gray" in color: return "Grey"
    elif "white" in color: return "White"
    elif "black" in color: return "Black"
    elif "blue" in color: return "Blue"
    elif "red" in color: return "Red"
    elif "green" in color: return "Green"
    elif "brown" in color: return "Brown"
    elif "orange" in color: return "Orange"
    elif "yellow" in color: return "Yellow"
    else: return "Other"

car["Color"] = car["Color"].apply(normalize_color)

@app.route('/')
def index():
    min_year = int(car["myear"].min())
    max_year = datetime.datetime.now().year
    myear = sorted(range(min_year, max_year + 1), reverse=True)

    fuel = sorted(car['fuel'].dropna().unique())
    transmission = sorted(car['transmission'].dropna().unique())
    body = sorted(car['body'].dropna().unique())
    oem = sorted(car['oem'].dropna().unique())
    variant = sorted(car['variant'].dropna().unique())
    state = sorted(car['state'].dropna().unique())
    City = sorted(car['City'].dropna().unique())
    owner_type = sorted(car['owner_type'].dropna().unique())
    Color = sorted(car['Color'].dropna().unique())

    state_city_map = car.groupby("state")["City"].unique().apply(list).to_dict()

    return render_template(
        "index.html",
        myear=myear,
        fuel=fuel,
        transmission=transmission,
        body=body,
        oem=oem,
        variant=variant,
        City=City,
        state=state,
        owner_type=owner_type,
        Color=Color,
        state_city_map=state_city_map
    )

@app.route("/get_models/<company>")
def get_models(company):
    models = sorted(car[car["oem"] == company]["model"].dropna().unique().tolist())
    return jsonify(models)

@app.route("/get_variants/<model>")
def get_variants(model):
    variants = sorted(car[car["model"] == model]["variant"].dropna().unique().tolist())
    return jsonify(variants)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            "oem": request.form.get("oem"),
            "model": request.form.get("model"),
            "myear": int(request.form.get("myear")),
            "fuel": request.form.get("fuel"),
            "transmission": request.form.get("transmission"),
            "body": request.form.get("body"),
            "variant": request.form.get("variant"),
            "City": request.form.get("City"),
            "state": request.form.get("state"),
            "owner_type": request.form.get("owner_type"),
            "Color": request.form.get("Color"),
            "km": int(request.form.get("km")),
        }

        input_df = pd.DataFrame([data])
        prediction = regressor.predict(input_df)[0]

        return jsonify({"price": round(float(prediction), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
