from flask import Flask, render_template, request, url_for
import pandas as pd
import joblib
import requests

# ------------------------
# Unsplash image search
# ------------------------
def search_unsplash(query):
    access_key = "erpDGYi8VmZG7JPdrKRK3S16u-bM6K5mfblV4VQftn4"
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query,
        "client_id": access_key,
        "per_page": 10
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
    except Exception:
        return []

    image_urls = []
    for result in data.get('results', []):
        desc = result.get('description') or ""
        alt = result.get('alt_description') or ""
        if "interior" not in desc.lower() and "interior" not in alt.lower():
            image_urls.append(result['urls']['regular'])
            if len(image_urls) >= 5:
                break
    return image_urls

# ------------------------
# Flask app setup
# ------------------------
app = Flask(__name__)

# Load model + columns
model = joblib.load('./notebooks/model.joblib')
X_columns = joblib.load('./notebooks/X_columns.joblib')

# ------------------------
# Landing Page
# ------------------------
@app.route('/')
def start_page():
    return render_template('startPage.html')

# ------------------------
# Valuation Page
# ------------------------
@app.route('/valuation', methods=['GET', 'POST'])
def valuation():
    prediction_text = None
    prev_input = None
    image_urls = []

    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            motor_volume = float(request.form['motor_volume'])
            running_km = float(request.form['running_km'])
        except ValueError:
            prediction_text = "Invalid numeric input. Please enter valid numbers."
            prev_input = request.form.to_dict()
            return render_template('index.html',
                                   prediction_text=prediction_text,
                                   prev_input=prev_input,
                                   image_urls=image_urls)

        # Categorical inputs
        model_input = request.form['model'].lower()
        motor_type_input = request.form['motor_type'].lower()
        wheel_input = request.form['wheel'].lower()
        color_input = request.form['color'].lower()
        type_input = request.form['type'].lower()
        status_input = request.form['status'].lower()

        prev_input = {
            'year': year,
            'motor_volume': motor_volume,
            'running_km': running_km,
            'model': model_input,
            'motor_type': motor_type_input,
            'wheel': wheel_input,
            'color': color_input,
            'type': type_input,
            'status': status_input
        }

        # Prepare input DataFrame
        input_df = pd.DataFrame([0] * len(X_columns), index=X_columns).T
        input_df.at[0, 'year'] = year
        input_df.at[0, 'motor_volume'] = motor_volume
        input_df.at[0, 'running_km'] = running_km

        dummy_cols = [
            f"model_{model_input}",
            f"motor_type_{motor_type_input}",
            f"wheel_{wheel_input}",
            f"color_{color_input}",
            f"type_{type_input}",
            f"status_{status_input}"
        ]

        for col in dummy_cols:
            if col in input_df.columns:
                input_df.at[0, col] = 1

        # Luxury brand flag
        luxury_brands = ['mercedes-benz', 'bmw', 'audi']
        if 'is_luxury' in input_df.columns:
            input_df.at[0, 'is_luxury'] = 1 if model_input in luxury_brands else 0

        # Predict
        predicted_price = model.predict(input_df)[0]
        prediction_text = f"Predicted Car Price: ${predicted_price:,.2f}"

        # Image search
        car_query = f"{year} {model_input} {type_input} {color_input} car exterior"
        image_urls = search_unsplash(car_query)

    return render_template('index.html',
                           prediction_text=prediction_text,
                           prev_input=prev_input,
                           image_urls=image_urls)

# ------------------------
# Run the app
# ------------------------
if __name__ == '__main__':
    app.run(debug=True)
