# ML Projects Portfolio: Elevvo Tasks

![ML Projects Banner](https://via.placeholder.com/1200x300/4A90E2/FFFFFF?text=ML+Projects:+Elevvo+Tasks) <!-- Replace with an actual image if available -->

Welcome to my collection of machine learning projects completed as part of the Elevvo tasks! This repository showcases hands-on implementations across various domains, including computer vision, time series forecasting, classification, and predictive modeling. Each notebook demonstrates end-to-end workflows: data loading, exploration, modeling, evaluation, and insights.

These projects use Python, TensorFlow/Keras, Scikit-learn, XGBoost, and more, with a focus on practical problem-solving, hyperparameter tuning, and interpretability.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Projects](#projects)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This repo contains 5 key ML tasks, each tackling real-world challenges:
- **Task 3**: Multi-class classification on forest cover types using ensemble methods.
- **Task 4**: Binary classification for loan approval prediction, emphasizing feature importance.
- **Task 7**: Time series forecasting for retail sales with seasonality handling.
- **Task 8 (Custom CNN)**: Traffic sign recognition with a lightweight CNN achieving 96.7% accuracy.
- **Task 8 (MobileNet Transfer Learning)**: Analysis of transfer learning pitfalls (e.g., mode collapse) on the same dataset.

Key themes: Handling imbalanced data, domain adaptation, interpretability (e.g., feature importance, residuals), and model comparisons (e.g., RF vs. XGBoost).

| Task | Domain | Key Techniques | Performance Highlights |
|------|--------|----------------|-------------------------|
| 3: Forest Covertype | Classification | RF, XGBoost, GridSearchCV | RF: 95.2% Test Acc; Elevation as top feature |
| 4: Loan Approval | Classification | DT, RF, SMOTE for imbalance | DT: 96.0% Acc, 95.4% F1 (CIBIL-only baseline) |
| 7: Sales Forecasting | Regression/Time Series | XGBoost, Lags/Rolling Features | MAE: ~$1,200; Captures holiday peaks |
| 8: Traffic Signs (Custom) | Computer Vision | CNN (Custom), Augmentation | 96.7% Weighted F1; Handles 43 classes |
| 8: Traffic Signs (MobileNet) | Computer Vision | Transfer Learning, Analysis | 6% Acc (failure case study: domain shift) |

## Projects

### Task 3: Forest Covertype Classification
- **Notebook**: [elevvo_task3_forest_covertype_classification.ipynb](elevvo_task3_forest_covertype_classification.ipynb)
- **Description**: Predict 7 forest cover types using 54 geospatial features (e.g., elevation, soil type). Explores Random Forest vs. XGBoost, with feature engineering and confusion matrix analysis.
- **Dataset**: UCI Forest Covertype (581K samples).
- **Insights**: Elevation dominates importance (40%+); RF outperforms XGBoost on test set due to bagging's variance reduction.
- **Run Time**: ~5-10 min on CPU.

### Task 4: Loan Approval Prediction
- **Notebook**: [elevvo_task4_load_approval_prediction_description.ipynb](elevvo_task4_load_approval_prediction_description.ipynb)
- **Description**: Binary classification to approve/reject loans based on CIBIL scores, income, and assets. Includes SMOTE for imbalance, model comparison (LR, DT, RF), and CIBIL-only ablation study.
- **Dataset**: Custom loan dataset (4K+ samples).
- **Insights**: CIBIL category alone retains 95.4% F1-score with 96% fewer features—proves proxy power for risk assessment.
- **Run Time**: ~2-5 min.

### Task 7: Sales Forecasting
- **Notebook**: [elevvo_task7_sales_forecasting_description.ipynb](elevvo_task7_sales_forecasting_description.ipynb)
- **Description**: Forecast Walmart store sales using time series features (lags, rolling means, holidays). Compares Linear Regression, RF, and XGBoost; includes decomposition and residual analysis.
- **Dataset**: Walmart Recruiting (historical sales, stores, features).
- **Insights**: Lags capture autocorrelation; residuals show minor holiday under-prediction (MAE ~$1,200 on $17K mean).
- **Run Time**: ~10-15 min (with cross-validation).

### Task 8: Traffic Sign Recognition (Custom CNN)
- **Notebook**: [elevvo_task8_traffic_sign_recognition.ipynb](elevvo_task8_traffic_sign_recognition.ipynb)
- **Description**: 43-class classification on GTSRB dataset using a custom CNN with augmentation (rotation, brightness). Evaluates with confusion matrix and per-class F1.
- **Dataset**: German Traffic Sign Recognition Benchmark (50K+ images).
- **Insights**: 96.7% accuracy; struggles with similar classes (e.g., speed limits: F1 0.79-0.99).
- **Run Time**: ~20-30 min on GPU (T4).

### Task 8: Traffic Sign Recognition (MobileNet Transfer Learning) - Failure Analysis
- **Notebook**: [elevvo_task8_trafiic_sign_recognition_mobilenet.ipynb](elevvo_task8_trafiic_sign_recognition_mobilenet.ipynb) <!-- Note: Typo in filename? -->
- **Description**: Attempts transfer learning with MobileNetV2 (ImageNet pretrained) on upscaled 32x32 images. Documents mode collapse (predicts 99% as Class 2) and lessons (e.g., domain shift, frozen layers).
- **Dataset**: Same as above.
- **Insights**: Only 6% accuracy—highlights risks of upscaling artifacts and no fine-tuning; custom CNN wins for low-res symbols.
- **Run Time**: ~15-20 min on GPU.

## 🛠 Tech Stack

- **Languages/Frameworks**: Python 3.12, TensorFlow/Keras 2.15, Scikit-learn 1.3, XGBoost 2.0, Pandas, NumPy.
- **Visualization**: Matplotlib, Seaborn.
- **Environments**: Google Colab (GPU: T4), Jupyter Notebooks.
- **Data Handling**: UCI/ML repos, CSV/Zip datasets.
- **Metrics**: Accuracy, F1-Score, MAE, Confusion Matrix, ROC-AUC.

