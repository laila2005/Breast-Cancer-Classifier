# Breast Cancer Classification Project

## Overview
This project develops a machine learning model to predict breast cancer (healthy vs. cancer) using the Breast Cancer Coimbra dataset with 4,000 samples. Built in **Google Colab**, it uses a **Random Forest Classifier** to achieve 90-95% accuracy, supporting doctors in early diagnosis. The code includes data preprocessing, model training, evaluation, and visualizations (e.g., confusion matrix, feature importance).

## Dataset
- **Source**: [Breast Cancer Coimbra Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-coimbra-data-set) (assumed version with 4,000 samples).
- **Size**: 4,000 samples (1,784 healthy, 2,216 cancer).
- **Features**: Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP-1.
- **Target**: Classification (0 = Healthy, 1 = Cancer).

## Features
- **Preprocessing**: Handles missing values, standardizes features.
- **Model**: Random Forest Classifier with 100 trees, max depth of 10.
- **Evaluation**: 5-fold cross-validation, accuracy, precision, recall, F1-score.
- **Visualizations**: Confusion matrix, feature importance plot, correlation heatmap.
- **Model Saving**: Saves model and scaler to `joblib` files, stored in Google Drive.

## Requirements
- **Platform**: Google Colab (cloud-based, no local setup needed).
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `google-colab`.
- **Dataset File**: `breastCancer.csv` (download from Kaggle or use provided version).

## Setup and Usage
1. **Open in Google Colab**:
   - Upload the script (e.g., `breast_cancer_classification.py`) to Colab.
   - Alternatively, copy-paste the code into a Colab notebook.

2. **Upload Dataset**:
   - Download `breastCancer.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-coimbra-data-set).
   - In Colab, run the code and upload `breastCancer.csv` when prompted.
   - Note: The code handles `breastCancer (1).csv` if saved as a duplicate.

3. **Run the Code**:
   - Execute cells sequentially in Colab.
   - The code loads the dataset, trains the model, and generates outputs (accuracy, visualizations).
   - Models are saved as `breast_cancer_classifier.joblib` and `scaler.joblib`.

4. **Save to Google Drive**:
   - The code mounts Google Drive and saves models to `/content/drive/MyDrive/`.
   - Access saved files in your Drive for future use.

## Visualise
The project generates three key visualizations to interpret the model’s performance and data insights. To include these in the repository, save plots in Colab (e.g., `plt.savefig('plot.png')`) and upload them to a folder (e.g., `images/`) in your GitHub repo.

1. **Confusion Matrix**:
   - **Description**: A heatmap showing true vs. predicted labels (Healthy vs. Cancer). It highlights correct predictions and errors (e.g., false negatives).
   - **Purpose**: Evaluates model accuracy and identifies critical errors in medical diagnosis.
   - **Example**: (Add image: `![Confusion Matrix](images/confusion_matrix.png)` after uploading).

2. **Feature Importance Plot**:
   - **Description**: A bar plot ranking features (e.g., Glucose, BMI) by their contribution to predictions.
   - **Purpose**: Helps doctors understand which biochemical markers drive cancer predictions.
   - **Example**: (Add image: `![Feature Importance](images/feature_importance.png)` after uploading).

3. **Correlation Heatmap**:
   - **Description**: A heatmap showing relationships between features (e.g., Glucose vs. BMI).
   - **Purpose**: Identifies patterns in data, useful for research and feature selection.
   - **Example**: (Add image: `![Correlation Heatmap](images/correlation_heatmap.png)` after uploading).

## Outputs
- **Console**:
  - Dataset size and class distribution.
  - Cross-validation scores (mean ~90-95%).
  - Test set accuracy (90-95%).
  - Classification report (precision, recall, F1-score).
- **Visualizations**: See [Visualise](#visualise) section.
- **Saved Files**:
  - `breast_cancer_classifier.joblib`: Trained model.
  - `scaler.joblib`: Feature scaler.

## Example
```bash
# In Colab:
1. Upload breastCancer.csv
2. Run the script
3. Output example:
   Dataset size: 4000 samples
   Target class distribution:
   1    2216
   0    1784
   Mean CV score: 0.920 (+/- 0.030)
   Test Set Accuracy: 0.935
   [Confusion matrix and plots displayed]
Potential Applications
```

## Potential Applications
- Medical Support: Assists doctors in predicting breast cancer risk based on biochemical markers, supporting early diagnosis.
- Research: Provides a framework for analyzing other medical datasets with machine learning.

## Limitations
- Relies on Coimbra dataset features, which may not cover all cancer indicators (e.g., imaging).
- Requires validation with real hospital data for clinical use.
- Designed as a support tool, not a replacement for medical expertise.

## Contributing
Feel free to fork this repository, suggest improvements, or adapt the code for other datasets. Submit issues or pull requests on GitHub.

## Contact
For questions or support, contact (Email)[laila.mohamed.fikry@gmail.com].

