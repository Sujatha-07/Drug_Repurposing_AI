import { useState } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  NavLink,
} from "react-router-dom";

import logo from "./assets/drug-repurposing-logo.png";
import "./App.css";


/* =========================================================
   NAVBAR
========================================================= */

function Navbar() {
  return (
    <nav className="navbar">

      <div className="nav-logo">
        <img
          src={logo}
          alt="Drug Repurposing AI"
        />
      </div>

      <div className="nav-links">

        <NavLink to="/" end>
          Home
        </NavLink>

        <NavLink to="/about">
          About
        </NavLink>

        <NavLink to="/how-it-works">
          How It Works
        </NavLink>

        <NavLink to="/technology">
          Technology
        </NavLink>

      </div>

    </nav>
  );
}


/* =========================================================
   HOME
========================================================= */

function Home() {
  return (
    <section className="page home-page">

      <div className="hero-content">

        <span className="tag">
          AI • MACHINE LEARNING • HEALTHCARE
        </span>

        <h1>
          Discover New Possibilities
          <br />
          for <span>Existing Drugs</span>
        </h1>

        <p>
          Drug Repurposing AI is an intelligent platform
          that uses machine learning to identify potential
          new therapeutic applications for existing drugs.
        </p>

        <div className="hero-buttons">

          <NavLink
            to="/prediction"
            className="primary-btn"
          >
            Start Prediction →
          </NavLink>

          <NavLink
            to="/how-it-works"
            className="secondary-btn"
          >
            Explore How It Works
          </NavLink>

        </div>

      </div>

    </section>
  );
}


/* =========================================================
   ABOUT
========================================================= */

function About() {
  return (
    <section className="page content-page">

      <span className="tag">
        ABOUT THE PROJECT
      </span>

      <h1>
        About Drug Repurposing AI
      </h1>

      <p>
        Drug Repurposing AI is an artificial intelligence
        based platform designed to explore new therapeutic
        possibilities for existing medicines.
      </p>

      <p>
        Instead of developing a completely new drug from
        the beginning, drug repurposing investigates whether
        an already known drug can be useful for another
        disease or medical condition.
      </p>

      <div className="info-grid">

        <div className="info-card">
          <h3>Our Goal</h3>

          <p>
            To support faster and more intelligent exploration
            of potential drug applications using machine learning.
          </p>
        </div>

        <div className="info-card">
          <h3>Our Approach</h3>

          <p>
            Drug-related data is processed and analyzed to
            generate meaningful predictions about possible
            therapeutic uses.
          </p>
        </div>

        <div className="info-card">
          <h3>Our Vision</h3>

          <p>
            To demonstrate how artificial intelligence can
            contribute to modern drug discovery and research.
          </p>
        </div>

      </div>

    </section>
  );
}


/* =========================================================
   HOW IT WORKS
========================================================= */

function HowItWorks() {
  return (
    <section className="page content-page">

      <span className="tag">
        PROCESS
      </span>

      <h1>
        How It Works
      </h1>

      <p className="intro">
        The platform follows a structured machine-learning
        workflow to transform drug information into useful
        predictions.
      </p>

      <div className="steps">

        <div className="step-card">
          <span>01</span>
          <h3>Drug Data</h3>

          <p>
            Relevant drug properties and information are
            collected from the available dataset.
          </p>
        </div>

        <div className="step-card">
          <span>02</span>
          <h3>Preprocessing</h3>

          <p>
            The data is cleaned, transformed and prepared
            for the machine-learning model.
          </p>
        </div>

        <div className="step-card">
          <span>03</span>
          <h3>Prediction</h3>

          <p>
            The trained model analyzes the input and predicts
            a potential repurposing category.
          </p>
        </div>

        <div className="step-card">
          <span>04</span>
          <h3>Result</h3>

          <p>
            The prediction and confidence information are
            presented through the application.
          </p>
        </div>

      </div>

    </section>
  );
}


/* =========================================================
   TECHNOLOGY
========================================================= */

function Technology() {
  return (
    <section className="page content-page">

      <span className="tag">
        TECHNOLOGY STACK
      </span>

      <h1>
        Technology
      </h1>

      <p className="intro">
        Drug Repurposing AI combines modern web technologies,
        machine learning and backend services.
      </p>

      <div className="tech-grid">

        <div className="tech-card">
          <h3>Machine Learning</h3>

          <p>
            Machine-learning models analyze drug-related
            features and generate predictions.
          </p>
        </div>

        <div className="tech-card">
          <h3>Python</h3>

          <p>
            Python is used for data processing, model
            development and prediction logic.
          </p>
        </div>

        <div className="tech-card">
          <h3>FastAPI</h3>

          <p>
            FastAPI provides the backend API layer for
            communicating with the machine-learning system.
          </p>
        </div>

        <div className="tech-card">
          <h3>React</h3>

          <p>
            React is used to create the interactive
            frontend interface.
          </p>
        </div>

      </div>

    </section>
  );
}


/* =========================================================
   PREDICTION PAGE
========================================================= */

function Prediction() {

  const [drugName, setDrugName] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");


  /* =======================================================
     HANDLE PREDICTION
  ======================================================= */

  const handlePrediction = async () => {

    if (!drugName.trim()) {
      setError("Please enter a drug name.");
      setPrediction(null);
      setModelComparison(null);
      return;
    }

    setLoading(true);
    setError("");
    setPrediction(null);
    setModelComparison(null);

    try {

      const response = await fetch(
  "https://drug-repurposing-ai-backend.onrender.com/predict",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            drug_name: drugName.trim(),
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Prediction failed."
        );
      }

      setPrediction(data.result);
      setModelComparison(data.model_comparison);

    } catch (err) {

      console.error("Prediction error:", err);

      setError(
        err.message ||
        "Unable to connect to prediction server."
      );

    } finally {

      setLoading(false);

    }
  };


  /* =======================================================
     ENTER KEY
  ======================================================= */

  const handleKeyDown = (event) => {

    if (event.key === "Enter") {
      handlePrediction();
    }

  };


  /* =======================================================
     FIND BEST MODEL
  ======================================================= */

  const getBestModel = () => {

    if (!modelComparison) {
      return null;
    }

    return Object.entries(modelComparison).reduce(
      (best, current) => {

        if (Number(current[1]) > Number(best[1])) {
          return current;
        }

        return best;

      }
    );

  };


  const bestModel = getBestModel();


  /* =======================================================
     PAGE
  ======================================================= */

  return (

    <section className="page content-page prediction-page">

      <span className="tag">
        AI PREDICTION
      </span>


      <h1>
        Drug Prediction
      </h1>


      <p className="intro">
        Enter a drug name to explore its current use,
        potential repurposed use and prediction confidence.
      </p>


      {/* ===================================================
          INPUT
      =================================================== */}

      <div className="prediction-box">

        <label htmlFor="drug-name">
          Drug Name
        </label>


        <input
          id="drug-name"
          type="text"
          value={drugName}
          onChange={(event) =>
            setDrugName(event.target.value)
          }
          onKeyDown={handleKeyDown}
          placeholder="Enter drug name e.g. Abacavir"
        />


        <button
          onClick={handlePrediction}
          disabled={loading}
        >
          {loading ? "Analyzing..." : "Predict"}
        </button>


        {error && (
          <div className="prediction-error">
            {error}
          </div>
        )}

      </div>


      {/* ===================================================
          PREDICTION RESULT
      =================================================== */}

      {prediction && !error && (

        <div className="prediction-result">


          {/* RESULT HEADER */}

          <div className="result-header">

            <span className="result-tag">
              PREDICTION RESULT
            </span>

            <h2>
              Analysis Complete
            </h2>

            <p>
              AI analysis for{" "}
              <strong>
                {prediction.drug_name}
              </strong>
            </p>

          </div>


          {/* =================================================
              RESULT CARDS
              ALL SAME STYLE
          ================================================= */}

          <div className="result-grid">


            {/* 1. DRUG NAME */}

            <div className="result-card">

              <span className="result-label">
                Drug Name
              </span>

              <h3>
                {prediction.drug_name}
              </h3>

            </div>


            {/* 2. CURRENT USE */}

            <div className="result-card">

              <span className="result-label">
                Current Use
              </span>

              <h3>
                {prediction.current_use}
              </h3>

            </div>


            {/* 3. REPURPOSED USE */}

            <div className="result-card">

              <span className="result-label">
                Repurposed Use
              </span>

              <h3>
                {prediction.repurposed_use}
              </h3>

            </div>


            {/* 4. CATEGORY */}

            <div className="result-card">

              <span className="result-label">
                Prediction Category
              </span>

              <h3>
                {prediction.category}
              </h3>

            </div>


            {/* 5. CONFIDENCE */}

            <div className="result-card">

              <span className="result-label">
                Confidence
              </span>

              <h3>
                {Number(prediction.confidence).toFixed(2)}%
              </h3>

            </div>

          </div>


          {/* =================================================
              MODEL PERFORMANCE GRAPH
          ================================================= */}

          {modelComparison && (

            <div className="model-comparison">


              <div className="comparison-header">

                <span className="result-tag">
                  MODEL EVALUATION
                </span>

                <h2>
                  Algorithm Performance Comparison
                </h2>

                <p>
                  Accuracy comparison of the machine-learning
                  algorithms evaluated for this project.
                </p>

              </div>


              {/* GRAPH */}

              <div className="accuracy-chart">

                {Object.entries(modelComparison).map(
                  ([model, accuracy]) => (

                    <div
                      className="chart-row"
                      key={model}
                    >

                      <div className="chart-label">

                        <span>
                          {model}
                        </span>

                        <strong>
                          {Number(accuracy).toFixed(2)}%
                        </strong>

                      </div>


                      <div className="bar-background">

                        <div
                          className="bar-fill"
                          style={{
                            width: `${Number(accuracy)}%`,
                          }}
                        />

                      </div>

                    </div>

                  )
                )}

              </div>


              {/* BEST MODEL */}

              {bestModel && (

                <div className="best-model">

                  <div>

                    <span className="best-model-label">
                      BEST PERFORMING MODEL
                    </span>

                    <h3>
                      {bestModel[0]}
                    </h3>

                  </div>

                  <strong>
                    {Number(bestModel[1]).toFixed(2)}%
                  </strong>

                </div>

              )}

            </div>

          )}

        </div>

      )}

    </section>

  );
}


/* =========================================================
   MAIN APP
========================================================= */

function App() {

  return (

    <BrowserRouter>

      <Navbar />

      <main>

        <Routes>

          <Route
            path="/"
            element={<Home />}
          />

          <Route
            path="/about"
            element={<About />}
          />

          <Route
            path="/how-it-works"
            element={<HowItWorks />}
          />

          <Route
            path="/technology"
            element={<Technology />}
          />

          <Route
            path="/prediction"
            element={<Prediction />}
          />

        </Routes>

      </main>

    </BrowserRouter>

  );
}


export default App;