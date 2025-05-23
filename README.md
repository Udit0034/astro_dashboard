# 🪐 Astronomy Dashboard (v0.1)

An interactive Streamlit-based astronomy dashboard that pulls weather forecasts, moon data, and eclipse predictions — along with experimental AI tools like a constellation finder and astronomy chatbot.

---

## 🌌 Key Features

### ✅ Astronomy Dashboard (Complete)
- Pulls **real-time weather data** using [Open-Meteo API](https://open-meteo.com/)
- Retrieves astronomical data using **AstroPy** and **Skyfield**
- Uses **geopy + timezonefinder** to localize based on user coordinates
- Predicts:
  - **Night sky condition** using an XGBoost classifier (accuracy: ~85%)
  - **Next Full and New Moon**
  - **Upcoming solar and lunar eclipses**

> The dashboard frontend is built using **Streamlit**

---

## 🧠 Model Details

- **XGBoost Classifier** trained on weather and astronomical features
- Predicts average night sky condition:
  - `Bad`, `Fair`, `Good`, `Excellent`
- Accuracy score: **~85%**
- Predicts hourly sky condition (excluding sunlight hours)

---

## 🤖 Experimental Features

### 🗣️ Astronomy Chatbot (Prototype)
- Embedding-based QA using a small JSON corpus
- SentenceTransformer + cosine similarity
- Currently limited and prone to confusion

### ✨ Constellation Finder (Experimental)
- Detects constellations from night sky photos using OpenCV + DBSCAN
- Matches patterns to known constellations
- Current accuracy: **~5%** — to be improved

### 📆 Upcoming Event Tracker (Hardcoded)
- Displays upcoming astronomy events
- In future, will pull data via web scraping or APIs

---

## 📁 Project Structure

📦 astronomy-dashboard/
├── app.py # Main Streamlit app
├── model.pkl # Trained XGBoost model
├── astronomy.json # Knowledge base for chatbot
├── astronomy_embeddings/ # Sentence embeddings (for chatbot)
├── constellation_lines.json # Known constellation data (from GitHub, see credits)
├── eclipse_data.csv # Upcoming eclipse data (from Kaggle, see credits)
├── media/ # UI mockups / constellation references
├── requirements.txt # Project dependencies
└── README.md # This file


---

## 📦 Installation

### 1. Clone the repository

-git clone https://github.com/Udit0034/astronomy-dashboard.git
-cd astronomy-dashboard


-pip install -r requirements.txt


-run streamlit app.py

###🗓️ Future Roadmap

 -Improve chatbot with larger corpus and retrieval model (e.g., RAG)

 -Enhance constellation matching (rotation, scale invariance)

 -Use API/scraping for real-time astronomy events

 -Add mobile responsiveness

###🙏 Credits & Acknowledgments
-constellation_lines.json used from @ofrohn's GitHub repository

-eclipse_data.csv downloaded from Kaggle Eclipse Dataset (credit to uploader)

-Weather data powered by Open-Meteo API

-Astronomy calculations via AstroPy and Skyfield

-Classification via XGBoost

-Sentence embeddings via SentenceTransformers


###🔖 Version
-v0.1 – Stable base release with working dashboard and prototype AI features

###🧠 Developer Notes
-This project was built with limited compute resources (no GPU, 16GB RAM) and is the first working version. It will be improved in future updates.
