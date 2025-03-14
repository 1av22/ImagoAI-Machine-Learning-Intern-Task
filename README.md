# Corn Mycotoxin Level Prediction using Hyperspectral Imaging

This project implements a machine learning model to predict mycotoxin (DON) concentration levels in corn samples using hyperspectral imaging data. The project includes both the model development notebook and a Streamlit web application for making predictions.

## Project Structure
```
├── app/
│   └── app.py          # Streamlit application
├── data/
│   └── data.csv        # Hyperspectral data
├── models/
│   ├── RF_Model.joblib # Trained Random Forest model
│   ├── Scaler.joblib   # Standard scaler
│   └── PCA.joblib      # PCA transformation
├── notebooks/
│   └── Task.ipynb      # Model development notebook
└── requirements.txt     # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ImagoAI-Machine-Learning-Intern-Task.git
cd ImagoAI-Machine-Learning-Intern-Task
```

2. Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

1. Navigate to the app directory:
```bash
cd app
```

2. Start the Streamlit server:
```bash
streamlit run app.py
```

3. Open your web browser and go to `http://localhost:8501`

### Model Development

- The `notebooks/Task.ipynb` contains the complete workflow including:
  - Data preprocessing
  - Exploratory Data Analysis
  - PCA dimensionality reduction
  - Model training and evaluation

## Features

- Interactive web interface for predictions
- Pre-trained Random Forest model
- PCA-based dimensionality reduction
- Standardized data processing

## Technologies Used

- Python 3.x
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Matplotlib
- Seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details.
