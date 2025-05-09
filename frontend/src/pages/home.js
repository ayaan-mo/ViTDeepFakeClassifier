import React from "react";
import { Link } from "react-router-dom";
import Header from "../components/header";
import Footer from "../components/footer";

function Home() {
  return (
    <div className="home-page">
      <Header />

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

      <Footer />
    </div>
  );
}

export default Home;
