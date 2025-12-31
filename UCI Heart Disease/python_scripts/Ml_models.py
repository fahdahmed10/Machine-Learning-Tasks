# -*- coding: utf-8 -*-
"""
Machine Learning Models Training and Evaluation
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Random Forest Classifier
    """
    rf_classifier = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_classifier.fit(X_train, y_train)
    rf_y_pred = rf_classifier.predict(X_test)
    
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))
    return rf_classifier, rf_y_pred


def train_svm(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Support Vector Machine
    """
    svm_classifier = SVC(kernel='rbf', random_state=42)
    svm_classifier.fit(X_train, y_train)
    svm_y_pred = svm_classifier.predict(X_test)
    
    print("SVM Accuracy:", accuracy_score(y_test, svm_y_pred))
    return svm_classifier, svm_y_pred


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Logistic Regression
    """
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    logreg_y_pred = logreg.predict(X_test)
    
    print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_y_pred))
    return logreg, logreg_y_pred


def train_gaussian_nb(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Gaussian Naive Bayes (sklearn version)
    """
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_y_pred = gnb.predict(X_test)
    
    print("Gaussian Naive Bayes Accuracy:", accuracy_score(y_test, gnb_y_pred))
    return gnb, gnb_y_pred