import React from "react";
import { Link } from "react-router-dom";
import Header from "../components/header";
import Footer from "../components/footer";

function Home() {
  return (
    <div className="home-page">
      <Header />
      <div className="hero-white-bg" />

      <main className="hero-section">
        <h1 className="hero-title">Scan & Detect Deepfakes Instantly</h1>
        <p className="hero-description">
          Upload an image and let our AI-powered detector tell you if it's real
          or fake.
        </p>
        <Link to="/scanner" className="cta-button">
          Go to Scanner
        </Link>
      </main>
      <section className="impact-section">
        <h2>The Impact of Deepfakes on Society</h2>
        <div className="impact-stats">
          <div className="stat-card">
            <h3>$500,000</h3>
            <p>
              Estimated Loss Per Business
              <br />
              Due to Deepfake-Driven Fraud
            </p>
          </div>
          <div className="stat-card">
            <h3>99%</h3>
            <p>of Deepfake Pornography Targets Women</p>
          </div>
        </div>
      </section>
      <section className="cta-highlight">
        <p className="cta-text">
          DeepScan Ensures <strong>96% Prevention</strong>
          <br />
          From <strong>Deepfakes</strong>!
        </p>
        <Link to="/scanner" className="cta-button">
          Go to Scanner
        </Link>
      </section>

      <Footer />
    </div>
  );
}

export default Home;
