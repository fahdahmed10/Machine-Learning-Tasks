# Heart Disease Prediction — UCI Dataset

A complete machine learning pipeline for binary classification of heart disease using the [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data). The project covers the full data science workflow: data cleaning, outlier handling, exploratory analysis, normality testing, feature engineering, standardization, and multi-model evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Outlier Handling](#2-outlier-handling)
  - [3. Descriptive Statistics](#3-descriptive-statistics)
  - [4. Feature Engineering & Encoding](#4-feature-engineering--encoding)
  - [5. Standardization](#5-standardization)
  - [6. Train/Test Split](#6-traintest-split)
  - [7. Feature Distribution & Normality Analysis](#7-feature-distribution--normality-analysis)
  - [8. Machine Learning Models](#8-machine-learning-models)
- [Feature Reference](#feature-reference)
- [Key Findings](#key-findings)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Project Overview

This notebook builds a heart disease binary classifier using a clinically-grounded data science approach. Rather than applying generic preprocessing, each decision — from imputation strategy to outlier retention — is justified using **medical domain knowledge**.

**Target variable:** `num` (binarized: 0 = No Disease, 1 = Disease Present)

**Models evaluated:**
- Gaussian Naive Bayes (implemented from scratch)
- Gaussian Naive Bayes (scikit-learn)
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression

---

## Dataset

| Property         | Value                                                  |
|------------------|--------------------------------------------------------|
| Source           | Kaggle — `redwankarimsony/heart-disease-data`          |
| File             | `heart_disease_uci.csv`                                |
| Patients         | Multi-site (Cleveland, Hungary, Switzerland, VA)       |
| Task             | Binary Classification (Heart Disease: Yes / No)        |

### Original Feature Set

| Feature     | Type         | Description                                       |
|-------------|--------------|---------------------------------------------------|
| `age`       | Quantitative | Age in years                                      |
| `sex`       | Categorical  | Male / Female                                     |
| `cp`        | Categorical  | Chest pain type (4 categories)                    |
| `trestbps`  | Quantitative | Resting blood pressure (mm Hg)                    |
| `chol`      | Quantitative | Serum cholesterol (mg/dl)                         |
| `fbs`       | Categorical  | Fasting blood sugar > 120 mg/dl (True/False)      |
| `restecg`   | Categorical  | Resting ECG results                               |
| `thalch`    | Quantitative | Maximum heart rate achieved                       |
| `exang`     | Categorical  | Exercise-induced angina (Yes/No)                  |
| `oldpeak`   | Quantitative | ST depression induced by exercise                 |
| `slope`     | Categorical  | Slope of peak exercise ST segment *(dropped)*     |
| `ca`        | Quantitative | Number of major vessels 0–3 *(dropped)*           |
| `thal`      | Categorical  | Thalassemia *(dropped)*                           |
| `num`       | Target       | Diagnosis of heart disease (0–4 → binarized)      |

---

## Project Structure

```
notebook.ipynb
│
├── Data Preparation
│   ├── Package imports
│   ├── Dataset loading & inspection
│   ├── Feature classification (quantitative / categorical)
│   ├── Missingness analysis (missingno matrix)
│   ├── Feature elimination (ca, slope, thal, id, dataset)
│   └── Remaining missing value imputation
│
├── Outlier Handling
│   ├── Box plot visualization (all features + individual)
│   ├── Clinical outlier assessment
│   └── Zero-value correction for trestbps and chol
│
├── Descriptive Statistics
│   ├── Quantitative stats table (mean, median, IQR, std, etc.)
│   └── Categorical frequency tables (count + percentage)
│
├── Feature Engineering & Encoding
│   ├── Binary encoding (sex, fbs, exang, num)
│   └── One-hot encoding (cp, restecg)
│
├── Standardization
│   └── Z-score normalization (quantitative features)
│
├── Train/Test Split
│   └── 80/20 stratified split
│
├── Feature Distribution Analysis (DatasetAnalyzer class)
│   ├── Histograms + KDE
│   ├── Q-Q plots
│   ├── Shapiro-Wilk normality test
│   ├── Anderson-Darling normality test
│   └── Conditional distributions by target class
│
└── Machine Learning Models
    ├── Naive Bayes (from scratch)
    ├── Gaussian Naive Bayes (sklearn)
    ├── Random Forest (300 estimators)
    ├── SVM (RBF kernel)
    └── Logistic Regression
```

---

## Pipeline Walkthrough

### 1. Data Preparation

**Libraries used:** `pandas`, `numpy`, `missingno`, `seaborn`, `matplotlib`

The dataset is loaded and immediately inspected using `df.info()` and a missingness matrix. Features are explicitly divided into:

- **Quantitative:** `age`, `trestbps`, `chol`, `thalch`, `oldpeak`
- **Categorical:** `sex`, `cp`, `fbs`, `restecg`, `exang`
- **Target:** `num`

**Feature elimination** removes `ca`, `slope`, `thal`, `id`, and `dataset` due to high missingness rates that would require unreliable imputation.

**Remaining missing values** are handled using:
- **Median** for quantitative features (robust against skewed biomedical distributions)
- **Mode** for categorical features (preserves the most representative category)

---

### 2. Outlier Handling

All quantitative features are visualized via box plots, then assessed using **clinical plausibility** rather than purely statistical rules.

| Feature    | Outlier Decision | Reason                                                            |
|------------|------------------|-------------------------------------------------------------------|
| `age`      | Retained         | Both young-onset and elderly CAD cases are clinically valid       |
| `trestbps` | Retained (except 0) | High BP values represent true hypertensive states              |
| `chol`     | Retained (except 0) | Extreme values can indicate familial hypercholesterolemia      |
| `thalch`   | Retained         | Range reflects normal cardiac variability                         |
| `oldpeak`  | Retained         | High values reflect myocardial ischemia; negatives are valid ECG artifacts |

> **Core principle:** Statistical extremity ≠ clinical invalidity. Outliers were removed only when physiologically impossible.

**Zero-value correction** is applied to `trestbps` and `chol` — zero is not a valid physiological measurement for either. Zeros are replaced with `NaN` and then imputed using the median.

---

### 3. Descriptive Statistics

**Quantitative features** are summarized with a full statistics table including: count, missing, mean, median, mode, standard deviation, variance, min, Q1, Q3, IQR, and max.

**Categorical features** are summarized using frequency tables showing counts and percentages for each category.

---

### 4. Feature Engineering & Encoding

| Feature   | Encoding Method       | Notes                                         |
|-----------|-----------------------|-----------------------------------------------|
| `sex`     | Binary (Male=1, Female=0) | Direct map                                |
| `fbs`     | Integer cast          | Already True/False                            |
| `exang`   | Integer cast          | Already True/False                            |
| `num`     | Binarized             | `>0 → 1` (any disease = positive class)       |
| `cp`      | One-hot (drop_first)  | 4 categories → 3 binary columns               |
| `restecg` | One-hot (drop_first)  | 3 categories → 2 binary columns               |

All resulting boolean columns are cast to `int` for compatibility with sklearn.

---

### 5. Standardization

Quantitative features are standardized using **z-score normalization**:

```
X_standardized = (X - mean) / std
```

Mean and standard deviation are computed from the **full dataset** before splitting. The standardized features have approximately zero mean and unit variance.

---

### 6. Train/Test Split

| Parameter        | Value              |
|------------------|--------------------|
| Split ratio      | 80% train / 20% test |
| Random state     | 40                 |
| Stratification   | Yes (`stratify=y`) |

Stratified sampling preserves the original class distribution in both training and test sets.

---

### 7. Feature Distribution & Normality Analysis

The `DatasetAnalyzer` class provides a systematic per-feature analysis with three visualizations and two statistical tests:

**Visualizations (per feature):**
1. Overall histogram with KDE overlay
2. Q-Q plot for normality assessment
3. Conditional distribution separated by target class

**Statistical Tests:**
- **Shapiro-Wilk:** Tests null hypothesis of normality (p > 0.05 → fail to reject)
- **Anderson-Darling:** Tests at multiple significance levels (15%, 10%, 5%, 2.5%, 1%)

#### Feature Analysis Summary

| Feature                  | Distribution     | Normality (Visual) | Predictive Power |
|--------------------------|------------------|--------------------|------------------|
| `age`                    | Approximately Gaussian | Yes (approx.) | Good — older age = higher risk |
| `sex`                    | Bernoulli (binary) | N/A | Strong — males show higher disease prevalence |
| `trestbps`               | Right-skewed     | No              | Weak — high class overlap |
| `chol`                   | Heavily right-skewed | No           | Weak — high class overlap |
| `fbs`                    | Bernoulli (sparse) | N/A           | Moderate — flags high-sugar minority |
| `thalch`                 | Approximately Gaussian | Yes (approx.) | **Strong** — lower max HR = higher disease risk |
| `exang`                  | Bernoulli (binary) | N/A          | **Very Strong** — clearest class separation |
| `oldpeak`                | Zero-inflated / exponential | No  | **Very Strong** — values > 0 strongly indicate disease |
| `cp_atypical angina`     | Bernoulli        | N/A              | Good — presence correlates with healthy outcome |
| `cp_non-anginal`         | Bernoulli        | N/A              | Good — presence rules out disease |
| `cp_typical angina`      | Bernoulli (sparse) | N/A           | Weak — counter-intuitive, very few positives |
| `restecg_normal`         | Bernoulli        | N/A              | Good — normal ECG → lower disease risk |
| `restecg_st-t abnormality` | Bernoulli      | N/A              | Good — abnormal ECG → higher disease risk |

---

### 8. Machine Learning Models

#### Naive Bayes (From Scratch)

A full implementation using NumPy:

- Computes per-class mean, variance, and prior probabilities
- Uses the Gaussian PDF for likelihood estimation
- Applies log-sum to avoid numerical underflow
- Adds a small epsilon (`1e-9`) to variance to prevent division by zero

**Bayes Theorem applied:**

$$P(y \mid \mathbf{X}) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)$$

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

**Prediction rule (log-space):**

$$\hat{y} = \arg\max_y \left( \log P(y) + \sum_{i=1}^{n} \log P(x_i \mid y) \right)$$

#### sklearn Models

| Model                  | Key Parameters                       |
|------------------------|--------------------------------------|
| GaussianNB             | Default                              |
| RandomForestClassifier | `n_estimators=300`, `random_state=42`|
| SVC                    | `kernel='rbf'`, `random_state=42`    |
| LogisticRegression     | `max_iter=1000`, `random_state=42`   |

#### Evaluation Metrics

The scratch Naive Bayes model is evaluated with:
- Accuracy (train and test)
- Confusion matrix
- Per-class Precision, Recall, F1-Score
- Macro and weighted average F1
- ROC-AUC score

---

## Results & Metrics

### Dataset Summary

| Property           | Value             |
|--------------------|-------------------|
| Total patients     | 920               |
| Training set size  | 736 (80%)         |
| Test set size      | 184 (20%)         |
| Features after encoding | 13          |

### Descriptive Statistics — Quantitative Features

| Feature    | Mean    | Median | Mode  | Std Dev | Min  | Q1     | Q3    | Max   |
|------------|---------|--------|-------|---------|------|--------|-------|-------|
| `age`      | 53.511  | 54.0   | 54.0  | 9.425   | 28.0 | 47.0   | 60.0  | 77.0  |
| `trestbps` | 132.137 | 130.0  | 130.0 | 17.930  | 80.0 | 120.0  | 140.0 | 200.0 |
| `chol`     | 244.030 | 236.0  | 236.0 | 52.011  | 85.0 | 217.75 | 267.0 | 603.0 |
| `thalch`   | 137.692 | 140.0  | 140.0 | 25.145  | 60.0 | 120.0  | 156.0 | 202.0 |
| `oldpeak`  | 0.853   | 0.5    | 0.0   | 1.058   | -2.6 | 0.0    | 1.5   | 6.2   |

### Frequency Tables — Categorical Features

**Sex**

| Category | Count | Percent |
|----------|-------|---------|
| Male     | 726   | 78.91%  |
| Female   | 194   | 21.09%  |

**Chest Pain Type (`cp`)**

| Category        | Count | Percent |
|-----------------|-------|---------|
| Asymptomatic    | 496   | 53.91%  |
| Non-anginal     | 204   | 22.17%  |
| Atypical Angina | 174   | 18.91%  |
| Typical Angina  | 46    | 5.00%   |

**Fasting Blood Sugar (`fbs`)**

| Category | Count | Percent |
|----------|-------|---------|
| False    | 782   | 85.00%  |
| True     | 138   | 15.00%  |

**Resting ECG (`restecg`)**

| Category          | Count | Percent |
|-------------------|-------|---------|
| Normal            | 553   | 60.11%  |
| LV Hypertrophy    | 188   | 20.43%  |
| ST-T Abnormality  | 179   | 19.46%  |

**Exercise-Induced Angina (`exang`)**

| Category | Count | Percent |
|----------|-------|---------|
| False    | 583   | 63.37%  |
| True     | 337   | 36.63%  |

---

### Normality Test Results (Training Set)

| Feature                    | Shapiro-Wilk Stat | Shapiro p-value | Anderson Stat | Normal? |
|----------------------------|-------------------|-----------------|---------------|---------|
| `age`                      | 0.992             | 0.0007          | 1.737         | No (but visually approx. Gaussian) |
| `sex`                      | 0.500             | 0.0000          | 192.092       | No (binary) |
| `trestbps`                 | 0.969             | 0.0000          | 7.157         | No (right-skewed) |
| `chol`                     | 0.877             | 0.0000          | 21.562        | No (heavily right-skewed) |
| `fbs`                      | 0.421             | 0.0000          | 219.819       | No (binary) |
| `thalch`                   | 0.990             | 0.0000          | 1.935         | No (but visually approx. Gaussian) |
| `exang`                    | 0.609             | 0.0000          | 145.141       | No (binary) |
| `oldpeak`                  | 0.855             | 0.0000          | 40.516        | No (zero-inflated) |
| `cp_atypical angina`       | 0.480             | 0.0000          | 199.506       | No (binary) |
| `cp_non-anginal`           | 0.512             | 0.0000          | 187.185       | No (binary) |
| `cp_typical angina`        | 0.213             | 0.0000          | 267.454       | No (binary, sparse) |
| `restecg_normal`           | 0.623             | 0.0000          | 138.641       | No (binary) |
| `restecg_st-t abnormality` | 0.485             | 0.0000          | 197.762       | No (binary) |

> **Note:** All Anderson-Darling tests reject normality at all significance levels (15%, 10%, 5%, 2.5%, 1%). Binary features are expected to fail normality tests by design.

---

### Model Performance — Test Set Accuracy

| Model                            | Test Accuracy | Train Accuracy |
|----------------------------------|---------------|----------------|
| **Naive Bayes (from scratch)**   | **78.80%**    | 80.57%         |
| Gaussian Naive Bayes (sklearn)   | 78.80%        | —              |
| Logistic Regression              | 80.98%        | —              |
| Random Forest (300 trees)        | 80.43%        | —              |
| SVM (RBF kernel)                 | **81.52%**    | —              |

> The scratch implementation matches sklearn's GaussianNB exactly, confirming correctness. SVM achieves the highest test accuracy at **81.52%**, followed closely by Logistic Regression at **80.98%**.

### Observations

- The small gap between train accuracy (80.57%) and test accuracy (78.80%) for Naive Bayes indicates minimal overfitting.
- All models perform within a narrow ~3% band (78.8% – 81.5%), suggesting the feature set has a natural accuracy ceiling under these conditions.
- SVM's RBF kernel likely benefits from capturing non-linear boundaries that the linear Logistic Regression and the independence-assuming Naive Bayes cannot model as well.
- Random Forest (80.43%) performs slightly below SVM, possibly because 300 trees is more than necessary for this dataset size (736 training samples).

---

## Feature Reference

| Symbol       | Full Name                          | Unit / Values          |
|--------------|------------------------------------|------------------------|
| `age`        | Patient age                        | Years                  |
| `sex`        | Biological sex                     | 0 = Female, 1 = Male   |
| `cp`         | Chest pain type                    | 4 categories (OHE)     |
| `trestbps`   | Resting blood pressure             | mm Hg                  |
| `chol`       | Serum cholesterol                  | mg/dL                  |
| `fbs`        | Fasting blood sugar > 120 mg/dL   | 0 = No, 1 = Yes        |
| `restecg`    | Resting ECG results                | 3 categories (OHE)     |
| `thalch`     | Maximum heart rate achieved        | bpm                    |
| `exang`      | Exercise-induced angina            | 0 = No, 1 = Yes        |
| `oldpeak`    | Exercise-induced ST depression     | mm (relative to rest)  |
| `num`        | Heart disease diagnosis (target)   | 0 = No, 1 = Yes        |

---

## Key Findings

- **Strongest predictors:** `thalch`, `exang`, and `oldpeak` show the clearest separation between healthy and diseased patients.
- **Weakest predictors:** `trestbps` and `chol` have high class overlap in isolation; they may interact with other features inside a model but do not separate classes on their own.
- **Binary features** (sex, exang, fbs) are not normally distributed by design but are valid and informative for Naive Bayes.
- **Clinically driven preprocessing** avoids discarding valid extreme values that represent genuinely high-risk patients — a critical consideration when building medical classifiers.

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
missingno
scipy
statsmodels
kagglehub
```

Install all dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn missingno scipy statsmodels kagglehub
```

---

## Usage

1. **Download the dataset** — the notebook uses `kagglehub` to pull `redwankarimsony/heart-disease-data` automatically. Make sure your Kaggle API credentials are configured, or replace the path with a local copy of `heart_disease_uci.csv`.

2. **Run cells in order** — the pipeline is sequential; each section depends on the cleaned dataframe produced by the previous one.

3. **Inspect individual features** — use the `DatasetAnalyzer` class directly:

```python
analyzer = DatasetAnalyzer(data_with_target, target="target")
analyzer.analyze_feature("thalch")   # single feature
analyzer.analyze_all()               # all features
```

4. **Train a custom model** — swap in any sklearn-compatible classifier after the standardization and split steps:

```python
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))
```

---

> **Note:** This project is intended for educational and research purposes. Model outputs should not be used for clinical diagnosis without appropriate validation by qualified medical professionals.
