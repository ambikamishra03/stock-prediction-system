import React, { useState } from "react";
import axios from "axios";
import "./App.css";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ResponsiveContainer
} from "recharts";

function App() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [historical, setHistorical] = useState([]);
  const [future, setFuture] = useState([]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const uploadFile = async () => {
    if (!file) return alert("Upload a CSV!");

    const formData = new FormData();
    formData.append("file", file);

    const res = await axios.post("http://localhost:5000/predict", formData);

    setPredictions(res.data.predictions);

    // Combine for chart
    const formattedHistorical = res.data.historical.dates.map((d, i) => ({
      date: d,
      price: res.data.historical.prices[i],
    }));

    const formattedFuture = res.data.future.dates.map((d, i) => ({
      date: d,
      predicted: res.data.future.prices[i],
    }));

    setHistorical(formattedHistorical);
    setFuture(formattedFuture);
  };

  return (
    <div className="app-container">

      {/* Title */}
      <div className="title">📈 STOCK PRICE PREDICTION</div>

      {/* Upload */}
      <div className="upload-box">
        <input type="file" accept=".csv" onChange={handleFileChange} />
        <button className="predict-btn" onClick={uploadFile}>
          Predict
        </button>
      </div>

      {/* Historical Chart */}
      {historical.length > 0 && (
        <div className="chart-card">
          <h2>Historical Stock Price</h2>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={historical}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" hide />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="price"
                stroke="#4aa8ff"
                strokeWidth={2}
                name="Actual Price"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Prediction Chart */}
      {future.length > 0 && (
        <div className="chart-card">
          <h2>Predicted Stock Prices</h2>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={future}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" hide />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#00ffcc"
                strokeWidth={2}
                name="Predicted"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Predictions List */}
      {predictions.length > 0 && (
        <div className="result-card">
          <h3>Predicted Prices for Next 10 Days</h3>
          <ul>
            {predictions.map((p, i) => (
              <li key={i}>
                <span>Day {i + 1}</span>
                <span className="price">${p.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

    </div>
  );
}

export default App;
