# 🎮 Breast Cancer Classification Quest

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)  
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](https://github.com/yourusername/breast-cancer-classification/actions)  
[![Contributors](https://img.shields.io/badge/Contributors-Welcome-orange.svg)](https://github.com/yourusername/breast-cancer-classification/graphs/contributors)  
[![Last Updated](https://img.shields.io/badge/Last%20Updated-Jul%2030,%202025-lightgrey.svg)](https://github.com/yourusername/breast-cancer-classification)

Embark on a data-driven adventure with `eda.ipynb`! This Jupyter notebook transforms the Wisconsin Breast Cancer dataset into a battleground where machine learning models duel to classify tumors as malignant (M) or benign (B). 🛡️📊

## 🚀 Table of Contents

- [🎯 Problem Definition](#-problem-definition)
- [🛠️ Solution Approach](#-solution-approach)
- [📦 Installation](#-installation)
- [🎮 Usage](#-usage)
- [🏆 Results](#-results)
  - [📈 Model Performance](#-model-performance)
  - [🖼️ Confusion Matrices](#-confusion-matrices)
- [💾 Saved Artifacts](#-saved-artifacts)
- [🤝 Contributing](#-contributing)
- [🌟 Future Work](#-future-work)
- [📜 License](#-license)

## 🎯 Problem Definition

Breast cancer is a global health boss fight! Early detection is your ultimate power-up. The dataset, packed with features like radius, texture, perimeter, and concavity from biopsy images, challenges us to craft a model that distinguishes malignant from benign tumors—leveling up medical diagnosis! 🎮💉

## 🛠️ Solution Approach

Gear up for this quest with a strategic pipeline:

1. **Data Prep ⚙️**: Load the dataset with pandas and encode `diagnosis` (M = 1, B = 0).
2. **EDA Exploration 🗺️**: Uncover data secrets with stats and shapes.
3. **Model Showdown 🗡️**: Train Logistic Regression, SGD Classifier, SVC, KNeighbors Classifier, Decision Tree, XGBClassifier, and RandomForestClassifier. Save the RandomForest champ!
4. **Scoreboard 📋**: Compare train/test accuracies, saved to CSV.
5. **Visual Loot 🕹️**: Unlock confusion matrices in the `plots` folder.

## 📦 Installation

→ Clone the repo and gear up:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-classification.git
   cd breast-cancer-classification
   ```

→ Install your arsenal with `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

→ Place `breast_cancer.csv` in the `data` folder.

## 🎮 Usage

→ Launch the quest:
   ```bash
   jupyter notebook eda.ipynb
   ```

→ Follow the cells to battle data, train models, and claim your visuals! 🎯

## 🏆 Results

### 📈 Model Performance

Check the leaderboard! Accuracies show train/test prowess, with notes on overfitting or balance.

| Model                  | Train Accuracy | Test Accuracy | Notes                              |
|------------------------|-----------------|---------------|------------------------------------|
| SVC                    | 97.96%          | 98.84%        | 🥇 Top-tier generalization         |
| KNeighborsClassifier   | 100.00%         | 98.84%        | ⚠️ Overfit risk, but strong!       |
| SGDClassifier          | 96.21%          | 97.67%        | 🎯 Solid consistency               |
| DecisionTreeClassifier | 98.25%          | 97.67%        | ⚔️ Well-rounded fighter            |
| XGBClassifier          | 99.13%          | 97.67%        | 🔥 High train, slight test dip     |
| LogisticRegression     | 98.54%          | 96.51%        | 🛡️ Steady performer                |
| RandomForestClassifier | 98.54%          | 96.51%        | 💾 Saved champ for next level      |

→ **Insight**: SVC and KNeighborsClassifier reign with 98.84% test accuracy. KNeighbors might overfit (100% train), while RandomForest (96.51% both sets) is your saved hero for deployment.

### 🖼️ Confusion Matrices

Unlock these epic visuals in `plots`! They reveal true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)—your battle stats!

- [Comparison](plots/Comparison.png) → See the big fight overview!
- [DecisionTreeClassifier](plots/Confusion_metric_for_DecisionTreeClassifier.png) → 97.67% with balanced TP/TN.
- [KNN](plots/Confusion_metric_for_KNN.png) → 98.84% with rare missteps.
- [LogisticRegression](plots/Confusion_metric_for_LogisticRegression.png) → 96.51% with minor FP/FN.
- [RandomForestClassifier](plots/Confusion_metric_for_RandomForestClassifier.png) → 96.51% with saved glory.
- [SGDClassifier](plots/Confusion_metric_for_SGDClassifier.png) → 97.67% with moderate FP.
- [SVC](plots/Confusion_metric_for_SVC.png) → 98.84% with epic TP/TN.
- [XGBBoost](plots/Confusion_metric_for_XGBBoost.png) → 97.67% with FN focus.

→ These are your treasure maps—critical for spotting weaknesses, especially false negatives in this health quest!

## 💾 Saved Artifacts

- **Model**: `models/RandomForestClassifier.pkl` → Your saved hero!
- **Accuracy Table**: `models/accuracy_table.csv` → The full scoreboard.

## 🤝 Contributing

→ Join the guild! Fork, branch, and PR your improvements. Report bugs or ideas via issues. 🌐

## 🌟 Future Work

→ Level up the game:
- **Hyperparameter Tuning 🎛️**: Boost accuracy with GridSearchCV.
- **Feature Crafting 🛠️**: Add derived features or PCA.
- **Cross-Validation 🔄**: Strengthen with k-fold checks.
- **Web Deployment 🌐**: Build a Flask app for real-time battles.
- **Deep Learning 🚀**: Try CNNs on image data.

## 📜 License

This quest is under the [MIT License](LICENSE). Play on! 🎮