import React, { useState } from "react";
import Header from "../components/header";
import Footer from "../components/footer";

function Scanner() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
      setResult({ error: "Prediction failed" });
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setResult(null);
  };

  return (
    <div className="scanner-page">
      <Header />

      <main className="scanner-container">
        <h1>DeepScan Detector</h1>
        <p className="subtitle">Identify AI-generated faces in images</p>

        <div className="upload-box">
          <img src="/images.png" alt="upload icon" className="upload-icon" />
          <p>Choose an image to check</p>
          <input type="file" accept="image/*" onChange={handleFileChange} />
        </div>

        <div className="button-group">
          <button
            className="detect-btn"
            onClick={handleUpload}
            disabled={!file || loading}
          >
            {loading ? "Detecting..." : "Detect"}
          </button>
          <button
            className="clear-btn"
            onClick={handleClear}
            disabled={loading && !result}
          >
            Clear
          </button>
        </div>

        {loading && <div className="spinner" />}

        {result && result.prediction && (
          <div className="result">
            <h3>Result: {result.prediction.toUpperCase()}</h3>
            {result.confidence && <p>Confidence: {result.confidence}</p>}
            <p>
              <strong>File:</strong> {result.filename}
            </p>
          </div>
        )}
        {result?.error && <div className="error">{result.error}</div>}
      </main>

      <Footer />
    </div>
  );
}

export default Scanner;
