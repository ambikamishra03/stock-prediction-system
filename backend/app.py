from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_stock():
    file = request.files['file']

    data = pd.read_csv(file, skiprows=2)

    data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    # Remove empty rows
    data = data.dropna()

    # Convert types
    data['Date'] = pd.to_datetime(data['Date'])
    data['Close'] = data['Close'].astype(float)

    # Prepare Linear Regression
    data['Days'] = np.arange(len(data))
    X = data[['Days']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    # Predict next 10 days
    future_days = np.arange(len(data), len(data) + 10).reshape(-1, 1)
    predictions = model.predict(future_days)

    # Generate future date values
    last_date = data['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=10, freq='D')

    predictions_list = [round(float(p), 2) for p in predictions]

    return jsonify({
        "predictions": predictions_list,
        "historical": {
            "dates": data['Date'].astype(str).tolist(),
            "prices": data['Close'].tolist()
        },
        "future": {
            "dates": future_dates.astype(str).tolist(),
            "prices": predictions_list
        }
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
