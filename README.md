otebooks/eda.ipynb: Exploratory Data Analysis plotting and metrics.
# ROGII GeoAI

Commercial-style project site and Kaggle-ready machine learning pipeline for the ROGII wellbore geology prediction competition.

## Live links

- Website: https://website-beta-teal-60.vercel.app
- GitHub: https://github.com/saugata-malakar/ROGII-BOREWELL-PREDICTIONS
- Kaggle competition: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction

## What this project does

The repository predicts TVT from well log and trajectory data. It loads the well files, merges the pair for each well, creates safe shared features, trains a LightGBM model, and writes submission-ready output for the Kaggle notebook workflow.

## Website

The `website/` folder contains the commercial landing page used for Vercel deployment. It explains:

- what the product does,
- how the pipeline works,
- why the approach is useful,
- and how to reach the GitHub and Kaggle resources.

## Main files

- `final_kaggle_submission.ipynb`: final Kaggle notebook used for the competition workflow.
- `fast_pipeline.py`: local training and prediction script.
- `website/index.html`: landing page.
- `website/styles.css`: site styling.
- `website/script.js`: lightweight UI animation.

## Run locally

1. Install dependencies from `requirements.txt`.
2. Run `python fast_pipeline.py` for the model workflow.
3. Open `website/index.html` in a browser, or deploy the `website/` folder to Vercel.

