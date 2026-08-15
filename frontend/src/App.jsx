import { useState } from "react";
import "./App.css";

function App() {
  const [drugName, setDrugName] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const predictDrug = async () => {
    if (!drugName.trim()) {
      setError("Please enter a drug name.");
      return;
    }

    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const response = await fetch("https://drug-repurposing-ai-backend.onrender.com/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          drug_name: drugName.trim(),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Prediction failed. Please try again."
        );
      }

      setPrediction(data);
    } catch (err) {
      setError(err.message || "Unable to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter") {
      predictDrug();
    }
  };

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <div className="logo-section">
          <div className="logo-icon">Rx</div>

          <div>
            <h1>Drug Repurposing AI</h1>
            <p>AI-powered drug repurposing prediction</p>
          </div>
        </div>
      </header>

      {/* MAIN */}
      <main className="main-container">
        {/* SEARCH SECTION */}
        <section className="search-section">
          <h2>Drug Repurposing Prediction</h2>

          <p className="description">
            Enter a drug name to view its current use, potential repurposed
            use, prediction category, and confidence.
          </p>

          <div className="search-box">
            <input
              type="text"
              value={drugName}
              onChange={(event) => setDrugName(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter drug name e.g. Abacavir"
            />

            <button onClick={predictDrug} disabled={loading}>
              {loading ? "Predicting..." : "Predict"}
            </button>
          </div>

          {error && <div className="error-message">{error}</div>}
        </section>

        {/* PREDICTION RESULT */}
        {prediction && (
          <>
            <section className="result-section">
              <h2>Prediction Result</h2>

              <div className="result-grid">
                <div className="result-card">
                  <span>Drug Name</span>
                  <strong>{prediction.result.drug_name}</strong>
                </div>

                <div className="result-card">
                  <span>Current Use</span>
                  <strong>{prediction.result.current_use}</strong>
                </div>

                <div className="result-card highlight">
                  <span>Repurposed Use</span>
                  <strong>{prediction.result.repurposed_use}</strong>
                </div>

                <div className="result-card">
                  <span>Category</span>
                  <strong>{prediction.result.category}</strong>
                </div>

                <div className="result-card confidence-card">
                  <span>Confidence</span>
                  <strong>{prediction.result.confidence.toFixed(2)}%</strong>
                </div>
              </div>
            </section>

            {/* ALGORITHM COMPARISON */}
            <section className="comparison-section">
              <h2>Algorithm Performance Comparison</h2>

              <p className="description">
                Accuracy comparison of the four machine-learning algorithms
                used in the project.
              </p>

              <div className="chart">
                {Object.entries(prediction.model_comparison).map(
                  ([algorithm, accuracy]) => (
                    <div className="chart-row" key={algorithm}>
                      <div className="algorithm-name">
                        {algorithm}
                      </div>

                      <div className="bar-container">
                        <div
                          className="bar"
                          style={{ width: `${accuracy}%` }}
                        >
                          <span>{accuracy.toFixed(2)}%</span>
                        </div>
                      </div>
                    </div>
                  )
                )}

                {/* SCALE */}
                <div className="scale">
                  <span>0%</span>
                  <span>10%</span>
                  <span>20%</span>
                  <span>30%</span>
                  <span>40%</span>
                  <span>50%</span>
                  <span>60%</span>
                  <span>70%</span>
                  <span>80%</span>
                  <span>90%</span>
                  <span>100%</span>
                </div>
              </div>

              <div className="best-model">
                <span>Best Accuracy</span>

                <strong>
                  {Object.entries(prediction.model_comparison).reduce(
                    (best, current) =>
                      current[1] > best[1] ? current : best
                  )[0]}
                  {" — "}
                  {Math.max(
                    ...Object.values(prediction.model_comparison)
                  ).toFixed(2)}
                  %
                </strong>
              </div>

            </section>
          </>
        )}
      </main>

      {/* FOOTER */}
      <footer className="footer">
        <p>Drug Repurposing AI • Machine Learning Project</p>
      </footer>
    </div>
  );
}

export default App;