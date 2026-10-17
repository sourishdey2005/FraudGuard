FRAUD DETECTION PROJECT
Generated: 20261016_1701

CONTENTS
--------
01_datasets/
    cleaned_fraud_data.csv          Final cleaned dataset (502,307 rows, 28 columns)
    messy_fraud_data_raw.csv        Raw messy dataset before cleaning

02_models/
    models/
        lgbm_fraud.txt              Trained LightGBM booster
        isotonic_calibrator.pkl     Isotonic regression probability calibrator
        scaler.pkl                  StandardScaler for logistic regression
        features.pkl                Ordered feature list
        metadata.pkl                Thresholds and training parameters
        model_comparison.csv        Final model comparison table
    models_dl/
        ann_model.keras             Trained ANN (Keras)
        scaler_dl.pkl               StandardScaler for ANN inputs
        metadata.pkl                ANN metadata

03_figures/
    figures/                        EDA figures (distribution plots, correlations)
    figures_model/                  Model figures (PR curves, ROC, confusion matrix)
    figures_dl/                     Deep learning figures (comparison charts)

04_reports/
    model_comparison.csv            Test set performance for all models
    operating_points.csv            Precision/recall trade-off across thresholds
    segment_performance.csv         Performance by merchant category

FINAL RESULTS (Test Set)
------------------------
Model                  ROC-AUC   PR-AUC    Precision  Recall   F1
Logistic Regression    0.7286    0.2405    0.2904     0.2776   0.2839
LightGBM               0.7435    0.2595    0.2950     0.3090   0.3018
LightGBM (Calibrated)  0.7435    0.2595    0.2950     0.3090   0.3018
ANN (MLP)              0.6824    0.1853    0.1893     0.2576   0.2182

REPRODUCIBILITY
---------------
Random seed: 42
Time-aware split: 70/15/15
SMOTE applied on training fold only
Isotonic calibration fit on validation set
F1-optimal thresholds computed per model on calibrated validation scores
