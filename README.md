# 🌍 ROGII Wellbore Geology Prediction

![Build Status](https://img.shields.io/badge/Algorithm-LightGBM-blue) ![Environment](https://img.shields.io/badge/Environment-Kaggle-blue)

## 📌 Overview
This repository contains the end-to-end Machine Learning pipeline to predict wellbore geology (Target Variable: TVT) for the **ROGII Kaggle Code Competition**. The pipeline dynamically aligns train and test features, handles missing values, engineers spatial differences, and trains a highly efficient LightGBM Regressor.

## 🚀 Deploy / Submission Link
- **Live Website:** [Rogers Wellbore Geology on Vercel](https://website-beta-teal-60.vercel.app)
- **Kaggle Competition:** [ROGII Wellbore Geology Prediction Submit Page](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
- **Deployment Strategy:** Upload and run inal_kaggle_submission.ipynb directly in the Kaggle notebooks environment. It will auto-detect the datasets, train the LightGBM model, and generate submission.csv successfully!

## 📂 Project Structure
- inal_kaggle_submission.ipynb: The final Notebook containing the entire robust data loading, imputation, modeling, and output.
- ast_pipeline.py: The local end-to-end python script for fast local iteration.
- 
otebooks/eda.ipynb: Exploratory Data Analysis plotting and metrics.

## 💻 How to Run Locally
1. Clone the repository.
2. Ensure Python and dependencies (pandas, scikit-learn, lightgbm) are installed.
3. Run python fast_pipeline.py.

