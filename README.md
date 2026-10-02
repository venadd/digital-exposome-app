# Digital Exposome — Biosensing & Urban Environmental Intelligence

A Streamlit web application providing physiological and urban environmental inference via multimodal sensor fusion and Random Forest classification.

## Features
- **Multimodal Inference**: Evaluates physiological biometric markers (IBI, HR, EDA, BVP) alongside urban micro-climate metrics (NO2, Noise, NH3, PM10, CO, PM2.5, PM1).
- **Likert Scale Prediction**: Computes class probabilities and subjective environmental impact scores (Scale 1–5).
- **Feature Importance & Analytics**: Inspects real-time model feature importance and dataset distribution comparisons.

## Tech Stack
- **Framework**: Streamlit
- **Machine Learning**: Scikit-Learn (Pipeline with MinMaxScaler & RandomForestClassifier)
- **Data & Visualization**: Pandas, NumPy, Matplotlib, Seaborn

## Getting Started Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/venadd/digital-exposome-app.git
   cd digital-exposome-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```
