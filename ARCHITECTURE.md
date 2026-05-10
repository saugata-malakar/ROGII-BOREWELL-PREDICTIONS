# ROGII Wellbore Geology Prediction - Complete Architecture

## Project Overview

This project is built for the Kaggle competition "ROGII - Wellbore Geology Prediction". The goal is to predict the continuous target `tvt` (True Vertical Thickness) for every row in the test dataset using horizontal wellbore data and well-type metadata.

**Competition Goal**: Build machine learning models to predict geology along horizontal wellbores, helping automate drilling operations in the oil and gas industry.

**Evaluation Metric**: Root Mean Squared Error (RMSE)

**Prize Pool**: $50,000 (1st: $25k, 2nd: $13k, 3rd: $7k, 4th: $5k)

## Data Organization

- `train/` contains training data files for each well:
  - `XXXX__horizontal_well.csv` — horizontal well trajectory and sensor data
  - `XXXX__typewell.csv` — well type or metadata for the same well
- `test/` contains matching test data files
- `sample_submission.csv` shows the required `id,tvt` format

## Proposed Architecture

### 1. Data Ingestion

- Load all paired CSV files from `train/` and `test/`
- Merge horizontal and typewell data on shared identifiers or indices
- Build a unified dataset with one row per sample and a global `id`

### 2. Preprocessing

- Clean and normalize numeric features
- Handle missing values by imputation
- Convert categorical fields to encoded representations
- Add derived features such as:
  - well trajectory metrics
  - depth/distance-based aggregates
  - sensor rate-of-change
  - well-type indicators

### 3. Feature Engineering

- Build features from both horizontal well logs and metadata
- Use temporal or ordered relationships along the wellbore
- Add statistical summaries, differences, and interaction terms
- Optionally perform feature selection or importance filtering

### 4. Model Training

- Use a robust regression model architecture
- Typical choices for competition baseline:
  - LightGBM
  - XGBoost
  - CatBoost
  - scikit-learn ensemble models
- Train with cross-validation to estimate RMSE reliably
- Use grouped or time-series-aware CV if needed to avoid leakage by well

### 5. Evaluation

- Score using root mean squared error (RMSE)
- Validate on hold-out splits or grouped folds across wells
- Log the training and validation scores clearly

### 6. Prediction / Submission

cd c:\Users\trina\Downloads\PROJECTS\rogii-wellbore-geology-prediction
kaggle competitions submit -c rogii-wellbore-geology-prediction -f submission.csv -m "LGB with 48 features, 5-fold CV"- Generate predictions for all rows in `test/`
- Assemble the submission file with columns `id,tvt`
- Save output as `submission.csv`

## Recommended Folder Structure

```
rogii-wellbore-geology-prediction/
  ARCHITECTURE.md
  requirements.txt
  sample_submission.csv
  train/
  test/
  models/
  notebooks/
    eda.ipynb
  src/
    data.py
    preprocess.py
    model.py
    predict.py
```

## Complete Tech Stack

### Core Technologies

#### Programming Language
- **Python 3.8+**: Primary language for data science and machine learning

#### Data Processing & Analysis
- **pandas (>=1.5.0)**: DataFrame operations, CSV loading, data manipulation
- **numpy (>=1.21.0)**: Numerical computations, array operations
- **scikit-learn (>=1.2.0)**: Preprocessing, feature scaling, cross-validation, metrics

#### Machine Learning Models
- **LightGBM (>=3.3.0)**: Gradient boosting framework (primary model)
- **XGBoost (>=1.7.0)**: Extreme gradient boosting (ensemble model)
- **CatBoost (>=1.1.0)**: Categorical boosting (alternative model)

#### Visualization & EDA
- **matplotlib (>=3.5.0)**: Static plots, charts, visualizations
- **seaborn (>=0.11.0)**: Statistical data visualization
- **plotly** (optional): Interactive visualizations

#### Development Tools
- **Jupyter (>=1.0.0)**: Interactive notebooks for EDA and experimentation
- **ipykernel (>=6.0.0)**: Jupyter kernel for Python
- **joblib (>=1.1.0)**: Model serialization and parallel processing
- **tqdm (>=4.64.0)**: Progress bars for long-running operations

#### Version Control & Experiment Tracking
- **Git**: Source code version control
- **DVC** (optional): Data version control
- **MLflow** (optional): Experiment tracking and model registry

---

## Backend Architecture (Data Pipeline & Model Training)

The backend consists of modular Python scripts that form a reproducible ML pipeline from raw data to predictions.

### Module Structure

#### 1. **Data Module (`src/data.py`)**
**Purpose**: Load and merge wellbore data files

**Key Functions**:
- `load_well_data(data_dir, well_id, data_type)`: Load single well's horizontal and typewell data
- `load_all_data(data_dir, data_type)`: Load all wells and combine into unified dataset

**Features**:
- Handles paired CSV files (horizontal_well + typewell)
- Merges on common keys (TVT)
- Adds well_id for grouping
- Creates unique IDs for test submissions

**Dependencies**: pandas, pathlib

---

#### 2. **Preprocessing Module (`src/preprocess.py`)**
**Purpose**: Clean, normalize, and prepare data for modeling

**Key Functions**:
- `preprocess_data(df, is_train)`: Handle missing values, encode categoricals, normalize features
- `add_features(df)`: Generate derived features from raw data

**Preprocessing Steps**:
1. **Missing Value Imputation**: Median imputation for numeric features
2. **Categorical Encoding**: Label encoding for geology types
3. **Feature Scaling**: StandardScaler for numeric features (excluding target)
4. **Feature Engineering**:
   - Trajectory features: depth differences, 3D distances
   - Sensor derivatives: differences, rolling statistics (mean, std)
   - Well-level aggregations: per-well means and standard deviations

**Dependencies**: scikit-learn (StandardScaler, SimpleImputer, LabelEncoder), numpy

---

#### 3. **Model Module (`src/model.py`)**
**Purpose**: Train regression models with cross-validation

**Key Functions**:
- `train_model(X, y, groups, model_type)`: Train with GroupKFold CV
- `save_model(model, filepath)`: Serialize trained model
- `load_model(filepath)`: Load saved model

**Model Options**:
- **LightGBM** (default): Fast, efficient, handles large datasets
- **XGBoost**: Robust, widely used in competitions
- **CatBoost**: Handles categorical features natively

**Cross-Validation Strategy**:
- **GroupKFold (5 splits)**: Prevents data leakage by keeping wells separate
- **Scoring**: Negative RMSE (converted to positive for reporting)

**Hyperparameters** (baseline):
```python
n_estimators=1000
learning_rate=0.1
max_depth=6
random_state=42
```

**Dependencies**: lightgbm, xgboost, catboost, scikit-learn, joblib

---

#### 4. **Prediction Module (`src/predict.py`)**
**Purpose**: Generate predictions for test data and create submission file

**Key Functions**:
- `generate_predictions(data_dir, model_path, scaler)`: End-to-end prediction pipeline

**Pipeline Steps**:
1. Load test data using `load_all_data()`
2. Apply same preprocessing as training (using saved scaler)
3. Generate features using `add_features()`
4. Load trained model
5. Predict TVT values
6. Format as submission CSV (id, tvt)

**Output**: `submission.csv` in Kaggle format

**Dependencies**: All modules above

---

## Frontend Architecture (Notebooks & Visualization)

Interactive Jupyter notebooks for exploration, experimentation, and submission preparation.

### Notebook Structure

#### 1. **EDA Notebook (`notebooks/eda.ipynb`)**
**Purpose**: Exploratory Data Analysis

**Contents**:
- Data loading and initial inspection
- Statistical summaries (describe, info, missing values)
- Distribution plots (histograms, box plots)
- Correlation analysis (heatmaps)
- Wellbore trajectory visualization
- Sensor data time series plots
- Target variable (TVT) distribution
- Well-level comparisons

**Visualizations**:
- Matplotlib/Seaborn for static plots
- Plotly for interactive 3D trajectory plots (optional)

---

#### 2. **Modeling Notebook (`notebooks/modeling.ipynb`)** (To be created)
**Purpose**: Model development and hyperparameter tuning

**Contents**:
- Feature importance analysis
- Cross-validation results visualization
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Model comparison (LightGBM vs XGBoost vs CatBoost)
- Learning curves
- Residual analysis
- Feature selection experiments

---

#### 3. **Submission Notebook (`notebooks/submission.ipynb`)** (To be created)
**Purpose**: Final prediction generation for Kaggle

**Contents**:
- Load best model
- Generate predictions on test set
- Create submission.csv
- Validation checks (format, missing values, ID uniqueness)
- Quick statistics on predictions

**Note**: This notebook must run offline with internet disabled for Kaggle submission.

---

## Full-Stack Integration

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Development Workflow

```
1. Setup Environment
   └─> Install Python 3.8+, create venv, install requirements.txt

2. Data Exploration (EDA)
   └─> Run notebooks/eda.ipynb
   └─> Understand data structure, distributions, correlations

3. Pipeline Development
   └─> Implement/refine src/data.py, src/preprocess.py
   └─> Test data loading and preprocessing

4. Feature Engineering
   └─> Enhance add_features() in src/preprocess.py
   └─> Create domain-specific features (trajectory, sensors)

5. Model Training
   └─> Run src/model.py with different model types
   └─> Tune hyperparameters in modeling notebook
   └─> Validate with GroupKFold CV

6. Evaluation & Iteration
   └─> Analyze CV scores, feature importance
   └─> Iterate on features and models
   └─> Ensemble multiple models if needed

7. Prediction & Submission
   └─> Run src/predict.py to generate submission.csv
   └─> Validate format matches sample_submission.csv
   └─> Submit to Kaggle

8. Kaggle Notebook Submission
   └─> Create Kaggle notebook with full pipeline
   └─> Ensure runtime < 9 hours
   └─> Disable internet access
   └─> Commit and submit
```

### File Organization

```
rogii-wellbore-geology-prediction/
├── ARCHITECTURE.md              # This file
├── requirements.txt             # Python dependencies
├── sample_submission.csv        # Submission format reference
├── README.md                    # Project overview (optional)
│
├── train/                       # Training data (773 wells)
│   ├── {well_id}.png           # Well trajectory images
│   ├── {well_id}__horizontal_well.csv
│   └── {well_id}__typewell.csv
│
├── test/                        # Test data (3 wells)
│   ├── {well_id}__horizontal_well.csv
│   └── {well_id}__typewell.csv
│
├── models/                      # Saved models and scalers
│   ├── lgb_model.pkl
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── notebooks/                   # Jupyter notebooks
│   ├── eda.ipynb               # Exploratory analysis
│   ├── modeling.ipynb          # Model development (to create)
│   └── submission.ipynb        # Final submission (to create)
│
├── src/                         # Source code modules
│   ├── __init__.py             # Package initialization
│   ├── data.py                 # Data loading
│   ├── preprocess.py           # Preprocessing & features
│   ├── model.py                # Model training
│   ├── predict.py              # Prediction generation
│   └── utils.py                # Helper functions (optional)
│
└── submission.csv               # Generated submission file
```

### Dependency Management

**requirements.txt** contains all necessary packages with version constraints:
- Core libraries pinned to stable versions
- Compatible with Kaggle's Python environment
- No external API dependencies (offline-compatible)

### Containerization (Optional)

For reproducible local development:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
```

### Version Control Strategy

```bash
# Initialize Git
git init
git add .
git commit -m "Initial project structure"

# .gitignore should exclude:
# - venv/
# - __pycache__/
# - *.pyc
# - .ipynb_checkpoints/
# - models/*.pkl (large files)
# - train/ and test/ (data files)
```

### Experiment Tracking (Optional)

Use MLflow for tracking experiments:

```python
import mlflow

mlflow.start_run()
mlflow.log_params({"model": "lgb", "n_estimators": 1000})
mlflow.log_metrics({"cv_rmse": 0.1234})
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()
```

---

## Kaggle Submission Constraints

### Technical Requirements
- **CPU Notebook**: <= 9 hours run-time
- **GPU Notebook**: <= 9 hours run-time (GPU optional for this competition)
- **Internet Access**: Must be disabled for submission
- **Submission File**: Must be named `submission.csv`
- **External Data**: Freely & publicly available data allowed (including pre-trained models)

### Submission Format
```csv
id,tvt
000d7d20_1442,0.00
000d7d20_1443,0.00
000d7d20_1444,0.00
...
```

### Code Competition Requirements
- All code must run in Kaggle Notebooks
- No external API calls during execution
- All dependencies must be pre-installed or pip-installable
- Reproducible results with fixed random seeds

---

## Advanced Features & Optimization Strategies

### Feature Engineering Ideas

#### 1. **Trajectory-Based Features**
- Cumulative distance along wellbore
- Angle of inclination and azimuth
- Curvature (rate of angle change)
- Distance from well start/end
- Depth rate of change

#### 2. **Sensor-Based Features**
- First and second derivatives of all sensors
- Rolling statistics (mean, std, min, max) over windows (3, 5, 10, 20)
- Exponential moving averages
- Sensor ratios and interactions (e.g., GR/ANCC)
- Lag features (previous N measurements)

#### 3. **Well-Level Features**
- Per-well normalization (z-scores within well)
- Well-specific statistics (mean, std, quantiles)
- Well type indicators from typewell data
- Well trajectory characteristics (total length, depth range)

#### 4. **Geological Context Features**
- Distance to known geological boundaries
- Sequence position within well (normalized)
- Clustering-based features (k-means on sensor data)
- PCA components of sensor readings

### Model Optimization

#### Hyperparameter Tuning
```python
# LightGBM tuning space
param_grid = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8, 10],
    'num_leaves': [31, 63, 127],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0]
}
```

#### Ensemble Strategies
1. **Simple Averaging**: Average predictions from LightGBM, XGBoost, CatBoost
2. **Weighted Averaging**: Weight by CV performance
3. **Stacking**: Train meta-model on out-of-fold predictions
4. **Blending**: Combine models with different feature sets

#### Cross-Validation Strategies
- **GroupKFold**: Prevent well-level leakage (current approach)
- **TimeSeriesSplit**: If temporal ordering matters within wells
- **Stratified GroupKFold**: Ensure balanced target distribution across folds

### Performance Optimization

#### Memory Efficiency
- Use `dtype` optimization (float32 instead of float64)
- Process data in chunks for large datasets
- Delete intermediate DataFrames

#### Computation Speed
- Parallel processing with `joblib` or `multiprocessing`
- Use GPU for XGBoost/LightGBM if available
- Cache preprocessed data with `joblib.Memory`

---

## Troubleshooting & Common Issues

### Data Loading Issues
**Problem**: Mismatched columns between horizontal and typewell data  
**Solution**: Use `how='left'` in merge, handle missing columns gracefully

**Problem**: Memory errors with large datasets  
**Solution**: Load data in chunks, use `dtype` optimization

### Preprocessing Issues
**Problem**: Scaler not saved/loaded correctly  
**Solution**: Save scaler with `joblib.dump()` after training, load before test preprocessing

**Problem**: New categories in test data  
**Solution**: Use `handle_unknown='ignore'` in encoders, or map unknown to special value

### Model Training Issues
**Problem**: Overfitting (large gap between train and CV scores)  
**Solution**: Increase regularization, reduce model complexity, add more features

**Problem**: Underfitting (high CV error)  
**Solution**: Add more features, increase model complexity, tune hyperparameters

**Problem**: Data leakage (CV score much better than leaderboard)  
**Solution**: Ensure GroupKFold is used, check for target leakage in features

### Submission Issues
**Problem**: Submission format error  
**Solution**: Verify columns are exactly `id,tvt`, no index column, correct number of rows

**Problem**: Runtime exceeds 9 hours  
**Solution**: Reduce model complexity, use fewer CV folds, optimize feature engineering

---

## Development Roadmap

### Phase 1: Foundation (Week 1)
- [x] Set up project structure
- [x] Implement data loading module
- [x] Implement preprocessing module
- [x] Create baseline model (LightGBM)
- [x] Establish CV framework (GroupKFold)
- [ ] Complete EDA notebook

### Phase 2: Feature Engineering (Week 2-3)
- [ ] Implement trajectory features
- [ ] Implement sensor derivative features
- [ ] Implement rolling statistics
- [ ] Implement well-level aggregations
- [ ] Feature selection and importance analysis

### Phase 3: Model Optimization (Week 4-5)
- [ ] Hyperparameter tuning for LightGBM
- [ ] Train XGBoost and CatBoost models
- [ ] Implement model ensembling
- [ ] Cross-validation refinement

### Phase 4: Advanced Techniques (Week 6-8)
- [ ] Deep learning models (optional)
- [ ] Feature interactions and polynomial features
- [ ] Pseudo-labeling or semi-supervised learning
- [ ] External data integration (if available)

### Phase 5: Final Submission (Week 9-12)
- [ ] Create Kaggle submission notebook
- [ ] Optimize for runtime constraints
- [ ] Final model selection and validation
- [ ] Submit and iterate based on leaderboard feedback

---

## Key Success Factors

### 1. **Prevent Data Leakage**
- Always use GroupKFold with well_id as groups
- Never use future information to predict past
- Validate that test wells are completely separate from training

### 2. **Domain Knowledge**
- Understand wellbore drilling physics
- Recognize geological patterns in sensor data
- Leverage trajectory information effectively

### 3. **Robust Validation**
- Trust CV scores over single train/test split
- Ensure CV mimics leaderboard evaluation
- Monitor for overfitting throughout development

### 4. **Reproducibility**
- Set random seeds everywhere (numpy, sklearn, model libraries)
- Document all preprocessing steps
- Version control code and track experiments

### 5. **Iterative Improvement**
- Start simple, add complexity gradually
- Validate each change with CV
- Keep a log of experiments and results

---

## Resources & References

### Competition Resources
- [Competition Page](https://kaggle.com/competitions/rogii-wellbore-geology-prediction)
- [Discussion Forum](https://kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion)
- [Code Notebooks](https://kaggle.com/competitions/rogii-wellbore-geology-prediction/code)

### Technical Documentation
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Domain Knowledge
- Wellbore drilling fundamentals
- Geological logging and interpretation
- Geosteering techniques
- Oil and gas industry best practices

---

## Notes for This Competition

### Competition-Specific Considerations
- **Small Test Set**: Only 3 wells in test set vs 773 in training
- **Multimodal Data**: CSV data + PNG trajectory images (images not currently used)
- **Continuous Target**: TVT is a continuous regression target
- **Well Heterogeneity**: Each well has unique characteristics

### Potential Pitfalls
- **Overfitting to Training Wells**: With 773 training wells, ensure model generalizes
- **Ignoring Well Context**: Wells are not independent; use well-level features
- **Feature Leakage**: Be careful with rolling statistics at well boundaries
- **Submission Format**: Double-check ID format matches exactly

### Advanced Ideas to Explore
1. **Image Data**: Use PNG trajectory images with CNN or vision transformers
2. **Sequence Models**: Treat wellbore as sequence, use LSTM/GRU/Transformer
3. **Graph Neural Networks**: Model well relationships as graph
4. **Transfer Learning**: Pre-train on public wellbore datasets
5. **Uncertainty Quantification**: Predict confidence intervals, not just point estimates

---

## Implementation Status

✅ **Completed**:
- Project architecture and design documentation
- Data loading and merging pipeline (`src/data.py`)
- Preprocessing and feature engineering (`src/preprocess.py`)
- Model training with cross-validation (`src/model.py`)
- Prediction and submission generation (`src/predict.py`)
- Exploratory data analysis notebook (`notebooks/eda.ipynb`)
- Dependency management (`requirements.txt`)
- Complete project structure and documentation

🔄 **Next Steps**:
- Run full EDA to understand data distributions and correlations
- Train baseline models and establish CV scores
- Implement hyperparameter tuning with Optuna
- Generate initial submission for Kaggle
- Iterate on features and models for better performance

---

## Conclusion

This architecture provides a complete, production-ready framework for the ROGII Wellbore Geology Prediction competition. The modular design allows for rapid experimentation while maintaining code quality and reproducibility.

**Next Steps**:
1. Complete EDA notebook to understand data patterns
2. Enhance feature engineering with domain-specific features
3. Tune hyperparameters and ensemble models
4. Create Kaggle submission notebook
5. Iterate based on leaderboard feedback

**Success Metrics**:
- CV RMSE < 0.10 (target for top 10%)
- Leaderboard RMSE close to CV RMSE (no overfitting)
- Runtime < 6 hours (buffer for Kaggle constraints)
- Clean, documented, reproducible code

Good luck! 🚀
