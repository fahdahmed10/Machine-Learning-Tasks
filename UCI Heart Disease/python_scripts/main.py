# -*- coding: utf-8 -*-


# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import missingno as msn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             ConfusionMatrixDisplay, classification_report,
                             roc_auc_score)

# Import custom modules
from functions import one_hot_encoding, accuracy
from GausNB import NaiveBayes
from handel_outliers import handle_invalid_zeros
from statistical_tests import DatasetAnalyzer
from Ml_models import (train_random_forest, train_svm, 
                      train_logistic_regression, train_gaussian_nb)


# ============================================================================
# DATA PREPARATION
# ============================================================================
print("="*80)
print("LOADING DATASET")
print("="*80)

# Loading the dataset
df = pd.read_csv("heart_disease_uci.csv")
print("\nFirst 5 rows:")
print(df.head())

# Dataset info
print("\nDataset Info:")
df.info()


# ============================================================================
# FEATURE DEFINITION
# ============================================================================
quantitative_features = [
    "age",        # Age in years
    "trestbps",   # Resting blood pressure (mm Hg)
    "chol",       # Serum cholesterol (mg/dl)
    "thalach",    # Maximum heart rate achieved
    "oldpeak",    # ST depression induced by exercise
    "ca"          # Number of major vessels (0–3)
]

categorical_features = [
    "sex",        # Male / Female
    "cp",         # Chest pain type
    "fbs",        # Fasting blood sugar > 120 mg/dl
    "restecg",    # Resting ECG results
    "exang",      # Exercise-induced angina
    "slope",      # Slope of peak exercise ST segment
    "thal"        # Thalassemia
]

target_variable = "num"

print("\nQuantitative Features:", quantitative_features)
print("Categorical Features:", categorical_features)
print("Target Variable:", target_variable)


# ============================================================================
# DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("DATA CLEANING")
print("="*80)

# Check for null values
print("\nChecking for missing values:")
msn.matrix(df)

# Check for zero values
print("\nChecking for zero values:")
for feature in df.columns:
    print(f"{feature} : {df[feature].eq(0).any()}")

# Check for duplicates
print("\nChecking for duplicate rows:")
print(df.duplicated().any())


# ============================================================================
# FEATURE ELIMINATION
# ============================================================================
print("\n" + "="*80)
print("FEATURE ELIMINATION")
print("="*80)

# Drop Features with Excessive Missing Values
columns_to_drop = ["ca", "slope", "thal", "id", "dataset"]
df.drop(columns=columns_to_drop, inplace=True)

print("Remaining columns:")
print(df.columns)

# Update feature lists
quantitative_features = [
    "age",        # Age of the patient (years)
    "trestbps",   # Resting blood pressure (mm Hg)
    "chol",       # Serum cholesterol (mg/dl)
    "thalch",     # Maximum heart rate achieved
    "oldpeak"     # ST depression induced by exercise
]
categorical_features = [
    "sex",        # Male / Female
    "cp",         # Chest pain type
    "fbs",        # Fasting blood sugar > 120 mg/dl
    "restecg",    # Resting ECG results
    "exang",      # Exercise-induced angina
]

print("\nQuantitative Features:", quantitative_features)
print("Categorical Features:", categorical_features)


# ============================================================================
# HANDLING REMAINING MISSING VALUES
# ============================================================================
print("\n" + "="*80)
print("HANDLING MISSING VALUES")
print("="*80)

# Handle numerical features
for feature in quantitative_features:
    if feature in df.columns:
        if df[feature].isnull().sum() > 0:
            median_value = df[feature].median()
            df[feature] = df[feature].fillna(median_value)

# Handle categorical features
for feature in categorical_features:
    if feature in df.columns:
        if df[feature].isnull().sum() > 0:
            mode_value = df[feature].mode()[0]
            df[feature] = df[feature].fillna(mode_value)

print("\nDataset info after handling missing values:")
df.info()

msn.matrix(df)


# ============================================================================
# VISUALIZING OUTLIERS
# ============================================================================
print("\n" + "="*80)
print("VISUALIZING OUTLIERS")
print("="*80)

# Plotting box plots for all features
fig = plt.figure(figsize=(25, 10))
ax = fig.add_subplot(1, 1, 1)
ax.boxplot(df[quantitative_features])
ax.set_xticklabels(quantitative_features, rotation='vertical', fontsize="20")
ax.set_xlabel("Features", fontsize="20")
ax.set_ylabel("Values", fontsize="20")
ax.set_title("Box Plots for Quantitative Features",  fontsize="25")
plt.show()


# ============================================================================
# DEALING WITH OUTLIERS
# ============================================================================
print("\n" + "="*80)
print("DEALING WITH OUTLIERS")
print("="*80)

# Handle invalid zeros
invalid_zero_cols = ['trestbps', 'chol']
df = handle_invalid_zeros(df, invalid_zero_cols)

print("\nDataset info after handling outliers:")
df.info()

# Save cleaned data
df.to_csv("heart_disease_uci_cleand.csv")
print("\nCleaned data saved to 'heart_disease_uci_cleand.csv'")


# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

# Quantitative features statistics
q = df[quantitative_features]

desc_stats = pd.DataFrame({
    "Count (non-missing)": q.count(),
    "Missing": q.isna().sum(),
    "Mean": q.mean(),
    "Median": q.median(),
    "Mode": q.mode().iloc[0],
    "Std Dev": q.std(),
    "Variance": q.var(),
    "Min": q.min(),
    "Q1 (25%)": q.quantile(0.25),
    "Q3 (75%)": q.quantile(0.75),
    "IQR": q.quantile(0.75) - q.quantile(0.25),
    "Max": q.max(),
}).round(3)

print("\n=== Descriptive Statistics (Quantitative Features) ===")
print(desc_stats)

# Categorical features frequency tables
print("\n=== Frequency Tables (Categorical Features) ===")
for col in categorical_features:
    if col not in df.columns:
        print(f"\n--- {col} --- (SKIPPED: not found in df.columns)")
        continue

    counts = df[col].value_counts(dropna=False)
    perc = (counts / len(df) * 100).round(2)
    freq_table = pd.DataFrame({"Count": counts, "Percent (%)": perc})

    print(f"\n--- {col} ---")
    print(freq_table)


# ============================================================================
# ONE-HOT ENCODING
# ============================================================================
print("\n" + "="*80)
print("ONE-HOT ENCODING")
print("="*80)

df = one_hot_encoding(df)
print("\nFirst 5 rows after encoding:")
print(df.head(5))


# ============================================================================
# STANDARDIZATION
# ============================================================================
print("\n" + "="*80)
print("STANDARDIZATION")
print("="*80)

X = df[quantitative_features]

# Use descriptive statistics from the FULL dataset
global_mean = X.mean()
global_std  = X.std()

X_standardized = (X - global_mean) / global_std

# Replace in dataframe
df_standardized = df.copy()
df_standardized[quantitative_features] = X_standardized

print("\nStandardization verification (mean and std):")
print(X_standardized.describe().loc[['mean', 'std']])


# ============================================================================
# SPLITTING DATA
# ============================================================================
print("\n" + "="*80)
print("SPLITTING DATA")
print("="*80)

# Features (ALL columns except the target)
X = df_standardized.drop(columns=[target_variable])

# Target
y = df_standardized[target_variable]

# 80–20 split (random)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=40,
    stratify=y   # keeps class balance
)

print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])


# ============================================================================
# DATA INFERENCE - FEATURE DISTRIBUTION AND NORMALITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("DATA INFERENCE - FEATURE DISTRIBUTION AND NORMALITY ANALYSIS")
print("="*80)

# Prepare data for analysis
data_to_class = X_train.copy()
data_to_class["target"] = y_train

# Initialize analyzer
data_analyzer = DatasetAnalyzer(data_to_class)

# Analyze all features
print("\nAnalyzing all features...")
data_analyzer.analyze_all()


# ============================================================================
# NAIVE BAYES CLASSIFIER (FROM SCRATCH)
# ============================================================================
print("\n" + "="*80)
print("NAIVE BAYES CLASSIFIER (FROM SCRATCH)")
print("="*80)

NB_scratch = NaiveBayes()
NB_scratch.fit(X_train, y_train)

y_pred = NB_scratch.predict(X_test)
y_pred2 = NB_scratch.predict(X_train)

print(f"Naive Bayes accuracy on Test Set : {accuracy(y_test, y_pred):.4f}")
print(f"Naive Bayes accuracy on Train set: {accuracy(y_train, y_pred2):.4f}")


# ============================================================================
# DETAILED EVALUATION
# ============================================================================
print("\n" + "="*80)
print("DETAILED EVALUATION")
print("="*80)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy score for Naive Bayes (from scratch): {acc:.4f}")

# Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(
    y_test, y_pred), display_labels=['Class 0', 'Class 1'])

disp.plot(cmap=plt.cm.Blues)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================================
# OTHER MACHINE LEARNING MODELS
# ============================================================================
print("\n" + "="*80)
print("TRAINING OTHER MODELS")
print("="*80)

# Random Forest
print("\n--- Random Forest ---")
rf_classifier, rf_y_pred = train_random_forest(X_train, y_train, X_test, y_test)

# Support Vector Machine
print("\n--- Support Vector Machine ---")
svm_classifier, svm_y_pred = train_svm(X_train, y_train, X_test, y_test)

# Logistic Regression
print("\n--- Logistic Regression ---")
logreg, logreg_y_pred = train_logistic_regression(X_train, y_train, X_test, y_test)

# Gaussian Naive Bayes (sklearn)
print("\n--- Gaussian Naive Bayes (sklearn) ---")
gnb, gnb_y_pred = train_gaussian_nb(X_train, y_train, X_test, y_test)


# ============================================================================
# END
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)