# Quick Start Guide - ROGII Wellbore Geology Prediction

This guide will get you up and running in 5 minutes.

## 🚀 Installation (2 minutes)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 📊 Explore Data (5 minutes)

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/eda.ipynb
```

Run all cells to see:
- Data structure and statistics
- Feature distributions
- Correlation analysis
- Missing value patterns

## 🎯 Train Baseline Model (10 minutes)

### Option 1: Using Python script (recommended)

```bash
# Train LightGBM model (default)
python train.py

# Or train XGBoost
python train.py --model xgb

# Or train CatBoost
python train.py --model cat
```

### Option 2: Using Makefile

```bash
make train
```

### Option 3: Using source modules directly

```bash
python src/model.py
```

**Expected output:**
```
CV RMSE Scores: [0.0234, 0.0245, 0.0238, 0.0241, 0.0239]
Mean CV RMSE: 0.0239 ± 0.0004
Model saved to models/lgb_model.pkl
```

## 🔮 Generate Predictions (2 minutes)

```bash
# Generate submission file
python src/predict.py

# Or use Makefile
make predict
```

This creates `submission.csv` in the project root.

## ✅ Validate Submission (1 minute)

```bash
# Validate format
python -c "from src.utils import validate_submission; validate_submission('submission.csv', 'sample_submission.csv')"

# Or use Makefile
make submit
```

## 📈 Improve Your Model

### 1. Experiment with Features

Edit `src/preprocess.py` and modify the `add_features()` function:

```python
def add_features(df):
    # Add your custom features here
    df['my_feature'] = df['X'] * df['Y']
    return df
```

### 2. Tune Hyperparameters

Edit `config.py` and modify `MODEL_PARAMS`:

```python
MODEL_PARAMS = {
    'lgb': {
        'n_estimators': 2000,  # Increase trees
        'learning_rate': 0.05,  # Lower learning rate
        'max_depth': 8,         # Deeper trees
        # ... other params
    }
}
```

### 3. Try Model Ensemble

Open `notebooks/modeling.ipynb` and run the ensemble section to combine multiple models.

## 📁 Project Structure

```
rogii-wellbore-geology-prediction/
├── train.py                 # Main training script ⭐
├── config.py                # Configuration settings
├── requirements.txt         # Dependencies
├── Makefile                 # Convenient commands
│
├── src/                     # Source code
│   ├── data.py             # Data loading
│   ├── preprocess.py       # Preprocessing & features ⭐
│   ├── model.py            # Model training ⭐
│   ├── predict.py          # Prediction generation
│   └── utils.py            # Helper functions
│
├── notebooks/               # Jupyter notebooks
│   ├── eda.ipynb           # Exploratory analysis
│   ├── modeling.ipynb      # Model development
│   └── submission.ipynb    # Final submission ⭐
│
└── models/                  # Saved models
    ├── lgb_model.pkl
    ├── scaler.pkl
    └── lgb_results.json
```

⭐ = Most important files to modify

## 🎓 Common Workflows

### Workflow 1: Quick Baseline
```bash
make install
python train.py
python src/predict.py
make submit
```

### Workflow 2: Experiment with Features
```bash
# 1. Edit src/preprocess.py
# 2. Retrain
python train.py
# 3. Generate new predictions
python src/predict.py
```

### Workflow 3: Hyperparameter Tuning
```bash
# 1. Edit config.py
# 2. Train multiple models
python train.py --model lgb
python train.py --model xgb
python train.py --model cat
# 3. Compare results in models/*_results.json
```

### Workflow 4: Model Ensemble
```bash
# 1. Train all models
python train.py --model lgb
python train.py --model xgb
python train.py --model cat
# 2. Open notebooks/modeling.ipynb
# 3. Run ensemble section
# 4. Generate submission from notebook
```

## 🐛 Troubleshooting

### Issue: Import errors
**Solution**: Make sure you're in the project root and virtual environment is activated.

### Issue: Memory errors
**Solution**: Reduce batch size or use data chunking in `src/data.py`.

### Issue: CV score doesn't match leaderboard
**Solution**: Ensure GroupKFold is used to prevent well-level leakage.

### Issue: Submission format error
**Solution**: Run validation: `make submit` or check with `src/utils.py`.

## 📚 Next Steps

1. **Read ARCHITECTURE.md** for detailed technical documentation
2. **Explore notebooks/eda.ipynb** to understand the data
3. **Experiment with features** in `src/preprocess.py`
4. **Tune hyperparameters** in `config.py`
5. **Try ensembling** in `notebooks/modeling.ipynb`
6. **Submit to Kaggle** using `notebooks/submission.ipynb`

## 🏆 Tips for Top Performance

1. **Feature Engineering**: Focus on domain-specific features (trajectory, geology)
2. **Cross-Validation**: Always use GroupKFold to prevent leakage
3. **Ensemble**: Combine LightGBM, XGBoost, and CatBoost
4. **Hyperparameter Tuning**: Use Optuna or GridSearchCV
5. **Validation**: Trust CV scores over single train/test split

## 📞 Need Help?

- Check [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation
- Review [README.md](README.md) for project overview
- Visit [Kaggle Discussion Forum](https://kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion)

---

**Good luck! 🚀**
