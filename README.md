# ROGII GeoAI

Commercial-style project site and Kaggle-ready machine learning pipeline for the ROGII wellbore geology prediction competition.

## Live links

- Website: https://website-beta-teal-60.vercel.app
- GitHub repository: https://github.com/saugata-malakar/ROGII-BOREWELL-PREDICTIONS
- Kaggle competition: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction

## What this project does

The repository predicts TVT from well log and trajectory data. It loads the well files, merges the pair for each well, creates safe shared features, trains a LightGBM model, and writes submission-ready output for the Kaggle notebook workflow.

## Website

The `website/` folder contains the commercial landing page used for Vercel deployment. It explains:

- what the product does,
- how the pipeline works,
- what problem it solves,
- and where to find the source code and competition link.

## Repository files

- `README.md`: project overview and deployment links.
- `final_kaggle_submission.ipynb`: final Kaggle notebook used for the competition workflow.
- `kaggle_submission_notebook.ipynb`: alternate notebook version for Kaggle submission.
- `fast_pipeline.py`: local training and prediction script.
- `pipeline.py`: pipeline orchestration entry point.
- `run_pipeline.py`: helper script to run the pipeline.
- `create_submission.py`: helper to build a submission file.
- `quick_submit.py`: quick submission utility.
- `check_features.py`: feature inspection helper.
- `src/`: reusable Python modules for data handling, preprocessing, modeling, prediction, and utilities.
- `notebooks/`: exploration and modeling notebooks.
- `website/`: Vercel-deployed landing page.

## Run locally

1. Install dependencies from `requirements.txt`.
2. Run `python fast_pipeline.py` for the model workflow.
3. Open `website/index.html` in a browser, or deploy the `website/` folder to Vercel.

