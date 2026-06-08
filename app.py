import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# High-quality embedded layout using modern Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Property Price Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-slate-50 min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">

    <div class="max-w-xl w-full space-y-8 bg-white p-8 rounded-2xl shadow-xl border border-slate-100">
        <div class="text-center">
            <h2 class="text-3xl font-extrabold text-slate-900 tracking-tight">Property Price Estimator</h2>
            <p class="mt-2 text-sm text-slate-500">Enter property details below to obtain an AI-driven valuation estimate.</p>
        </div>

        <form action="/" method="POST" class="mt-8 space-y-6">
            <div class="grid grid-cols-1 gap-y-6 sm:grid-cols-2 sm:gap-x-4">
                
                <div>
                    <label for="beds" class="block text-sm font-medium text-slate-700">Bedrooms</label>
                    <input type="number" step="any" name="beds" id="beds" required value="{{ input_values.beds if input_values else '' }}"
                           class="mt-1 block w-full px-4 py-3 bg-white border border-slate-300 rounded-lg text-sm shadow-sm placeholder-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500" 
                           placeholder="e.g. 3">
                </div>

                <div>
                    <label for="baths" class="block text-sm font-medium text-slate-700">Bathrooms</label>
                    <input type="number" step="any" name="baths" id="baths" required value="{{ input_values.baths if input_values else '' }}"
                           class="mt-1 block w-full px-4 py-3 bg-white border border-slate-300 rounded-lg text-sm shadow-sm placeholder-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500" 
                           placeholder="e.g. 2.5">
                </div>

                <div class="sm:col-span-2">
                    <label for="size" class="block text-sm font-medium text-slate-700">Property Size (sqft)</label>
                    <input type="number" step="any" name="size" id="size" required value="{{ input_values.size if input_values else '' }}"
                           class="mt-1 block w-full px-4 py-3 bg-white border border-slate-300 rounded-lg text-sm shadow-sm placeholder-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500" 
                           placeholder="e.g. 1850">
                </div>

                <div class="sm:col-span-2">
                    <label for="lot_size" class="block text-sm font-medium text-slate-700">Lot Size (sqft)</label>
                    <input type="number" step="any" name="lot_size" id="lot_size" required value="{{ input_values.lot_size if input_values else '' }}"
                           class="mt-1 block w-full px-4 py-3 bg-white border border-slate-300 rounded-lg text-sm shadow-sm placeholder-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500" 
                           placeholder="e.g. 5000">
                </div>

                <div class="sm:col-span-2">
                    <label for="zip_code" class="block text-sm font-medium text-slate-700">Zip Code</label>
                    <input type="number" step="1" name="zip_code" id="zip_code" required value="{{ input_values.zip_code if input_values else '' }}"
                           class="mt-1 block w-full px-4 py-3 bg-white border border-slate-300 rounded-lg text-sm shadow-sm placeholder-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500" 
                           placeholder="e.g. 90210">
                </div>
            </div>

            <div>
                <button type="submit" 
                        class="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-semibold text-white bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150">
                    Predict Valuation
                </button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="mt-6 p-6 bg-emerald-50 border border-emerald-200 rounded-xl text-center">
            <span class="text-sm font-medium text-emerald-800 uppercase tracking-wider">Estimated Price Prediction</span>
            <div class="mt-2 text-4xl font-extrabold text-emerald-700">
                ${{ prediction }}
            </div>
        </div>
        {% endif %}

        {% if error %}
        <div class="mt-6 p-4 bg-rose-50 border border-rose-200 rounded-xl text-center text-sm font-medium text-rose-700">
            {{ error }}
        </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    input_values = None

    if request.method == "POST":
        try:
            # Extract inputs matching the model features
            beds = float(request.form["beds"])
            baths = float(request.form["baths"])
            size = float(request.form["size"])
            lot_size = float(request.form["lot_size"])
            zip_code = float(request.form["zip_code"])
            
            input_values = {
                "beds": beds, "baths": baths, "size": size, 
                "lot_size": lot_size, "zip_code": zip_code
            }

            # Prepare the array shape (1, 5) for inference
            features = np.array([[beds, baths, size, lot_size, zip_code]])
            
            # Predict output values
            raw_prediction = model.predict(features)[0]
            prediction = f"{raw_prediction:,.2f}"

        except Exception as e:
            error = f"Error processing prediction input: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error, input_values=input_values)

# REST Endpoint capability if needed for external system consumption
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        features = np.array([[
            float(data["beds"]),
            float(data["baths"]),
            float(data["size"]),
            float(data["lot_size"]),
            float(data["zip_code"])
        ]])
        prediction = model.predict(features)[0]
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
