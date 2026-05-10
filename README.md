# ROGII - Wellbore Geology Prediction

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://kaggle.com/competitions/rogii-wellbore-geology-prediction)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Build machine learning models to predict geology along horizontal wellbores and help automate drilling operations in the oil and gas industry.

## 🎯 Competition Overview

**Goal**: Predict the continuous target `tvt` (True Vertical Thickness) for every row in the test dataset using horizontal wellbore sensor data and well-type metadata.

**Evaluation**: Root Mean Squared Error (RMSE)

**Prize Pool**: $50,000
- 🥇 1st Place: $25,000
- 🥈 2nd Place: $13,000
- 🥉 3rd Place: $7,000
- 4th Place: $5,000

**Timeline**:
- Start: May 5, 2026
- Entry Deadline: July 29, 2026
- Final Submission: August 5, 2026

## 📊 Dataset

- **Training Data**: 773 wells with paired CSV files
  - `{well_id}__horizontal_well.csv`: Sensor readings along wellbore
  - `{well_id}__typewell.csv`: Well metadata
  - `{well_id}.png`: Trajectory visualization
  
- **Test Data**: 3 wells with same structure (no target values)

- **Features**: Trajectory coordinates (X, Y, Z), sensor readings (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, GR), geology type

- **Target**: `tvt` (True Vertical Thickness) - continuous regression target

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rogii-wellbore-geology-prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Project Structure

```
rogii-wellbore-geology-prediction/
├── README.md                    # This file
├── ARCHITECTURE.md              # Detailed architecture documentation
├── requirements.txt             # Python dependencies
├── sample_submission.csv        # Submission format reference
│
├── train/                       # Training data (773 wells)
├── test/                        # Test data (3 wells)
├── models/                      # Saved models and scalers
│
├── notebooks/                   # Jupyter notebooks
│   └── eda.ipynb               # Exploratory data analysis
│
└── src/                         # Source code modules
    ├── data.py                 # Data loading
    ├── preprocess.py           # Preprocessing & feature engineering
    ├── model.py                # Model training
    └── predict.py              # Prediction generation
```

## 🔧 Usage

### 1. Exploratory Data Analysis

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/eda.ipynb
```

### 2. Train Model

```bash
# Train baseline LightGBM model
python src/model.py

# Output: models/lgb_model.pkl, CV RMSE scores
```

### 3. Generate Predictions

```bash
# Generate submission file
python src/predict.py

# Output: submission.csv
```

### 4. Validate Submission

```bash
# Check submission format
python -c "import pandas as pd; df = pd.read_csv('submission.csv'); print(df.head()); print(f'Shape: {df.shape}')"
```

## 🧠 Tech Stack

### Core Technologies
- **Python 3.8+**: Primary language
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Preprocessing, CV, metrics

### Machine Learning Models
- **LightGBM**: Primary gradient boosting model
- **XGBoost**: Alternative boosting model
- **CatBoost**: Categorical boosting model

### Visualization
- **matplotlib**: Static plots
- **seaborn**: Statistical visualizations
- **Jupyter**: Interactive notebooks

### Utilities
- **joblib**: Model serialization
- **tqdm**: Progress bars

## 📈 Model Pipeline

```
Raw Data → Data Loading → Preprocessing → Feature Engineering → Model Training → Prediction → Submission
```

### Key Features
1. **Data Loading**: Merge horizontal well and typewell data
2. **Preprocessing**: Handle missing values, normalize features, encode categoricals
3. **Feature Engineering**: 
   - Trajectory features (depth differences, distances)
   - Sensor derivatives and rolling statistics
   - Well-level aggregations
4. **Model Training**: GroupKFold CV to prevent well-level leakage
5. **Prediction**: Generate submission in Kaggle format

## 🎓 Key Techniques

### Cross-Validation Strategy
- **GroupKFold (5 splits)**: Keeps wells separate to prevent data leakage
- **Scoring**: RMSE (Root Mean Squared Error)

### Feature Engineering
- Trajectory-based: cumulative distance, depth changes, curvature
- Sensor-based: differences, rolling means/stds, lag features
- Well-level: per-well normalization, aggregations

### Model Optimization
- Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- Ensemble methods: averaging, stacking, blending
- Feature selection based on importance

## 📝 Development Workflow

1. **Setup**: Install dependencies, explore data structure
2. **EDA**: Understand distributions, correlations, patterns
3. **Baseline**: Train simple model, establish CV framework
4. **Feature Engineering**: Create domain-specific features
5. **Model Tuning**: Optimize hyperparameters, try different models
6. **Ensemble**: Combine multiple models for better performance
7. **Submission**: Generate predictions, validate format, submit to Kaggle

## 🏆 Competition Tips

### Do's ✅
- Use GroupKFold to prevent well-level leakage
- Start with simple baseline, iterate gradually
- Trust CV scores over single train/test split
- Document experiments and track results
- Leverage domain knowledge about drilling and geology

### Don'ts ❌
- Don't use future information to predict past
- Don't ignore well context (wells are not independent)
- Don't overfit to training data
- Don't forget to validate submission format
- Don't exceed 9-hour runtime limit on Kaggle

## 📚 Resources

- [Competition Page](https://kaggle.com/competitions/rogii-wellbore-geology-prediction)
- [Discussion Forum](https://kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## 🤝 Contributing

This is a competition project. Feel free to fork and experiment with your own approaches!

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- ROGII for hosting the competition
- Kaggle for the platform
- The data science community for shared knowledge

---

**Good luck and happy modeling! 🚀**
