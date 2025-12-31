"""
GausNB.py
Gaussian Naive Bayes classifier implementation from scratch.
"""

import numpy as np


class NaiveBayes:
    """
    Gaussian Naive Bayes classifier implemented from scratch.
    
    This classifier assumes that features follow a Gaussian (normal) distribution
    and are conditionally independent given the class label.
    
    Attributes
    ----------
    _classes : np.ndarray
        Unique class labels
    _mean : np.ndarray
        Mean of each feature for each class
    _var : np.ndarray
        Variance of each feature for each class
    _priors : np.ndarray
        Prior probability of each class
    """
    
    def __init__(self):
        self._classes = None
        self._mean = None
        self._var = None
        self._priors = None
    
    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier to training data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
        """
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize parameters
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Calculate mean, variance, and priors for each class
        for i, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[i] = X_c.mean(axis=0)
            self._var[i] = X_c.var(axis=0) + 1e-9  # Add small value to avoid zero variance
            self._priors[i] = X_c.shape[0] / n_samples
        
        print(f"Model trained on {n_samples} samples with {n_features} features")
        print(f"Classes: {self._classes}")
        print(f"Class priors: {self._priors}")

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        np.ndarray
            Predicted class labels
        """
        X = np.array(X)
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        """
        Predict class label for a single sample.
        
        Parameters
        ----------
        x : array-like, shape (n_features,)
            Single sample
            
        Returns
        -------
        class label
            Predicted class
        """
        posteriors = []

        for i, c in enumerate(self._classes):
            # log(P(class))
            log_prior = np.log(self._priors[i])

            # log(P(x|class)) = sum(log(pdf(x_i|class)))
            log_likelihood = np.sum(np.log(self._pdf(i, x)))

            # Posterior = log(P(class)) + log(P(x|class))
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)

        # Return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Calculate Gaussian probability density function for each feature.
        
        P(x_i|class) = (1/sqrt(2*pi*var)) * exp(-((x_i - mean)^2) / (2*var))
        
        Parameters
        ----------
        class_idx : int
            Index of the class
        x : array-like
            Feature values
            
        Returns
        -------
        np.ndarray
            PDF values for each feature
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        pdf = numerator / denominator

        # Clip to avoid log(0) in likelihood calculation
        return np.clip(pdf, 1e-9, None)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        X = np.array(X)
        probas = np.array([self._predict_proba_single(x) for x in X])
        return probas
    
    def _predict_proba_single(self, x):
        """
        Predict class probabilities for a single sample.
        
        Parameters
        ----------
        x : array-like, shape (n_features,)
            Single sample
            
        Returns
        -------
        np.ndarray
            Probabilities for each class
        """
        posteriors = []

        for i in range(len(self._classes)):
            log_prior = np.log(self._priors[i])
            log_likelihood = np.sum(np.log(self._pdf(i, x)))
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)

        # Convert log probabilities to probabilities
        posteriors = np.array(posteriors)
        posteriors = np.exp(posteriors - np.max(posteriors))  # Numerical stability
        posteriors = posteriors / np.sum(posteriors)  # Normalize
        
        return posteriors


def accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns
    -------
    float
        Accuracy score
    """
    return np.sum(y_true == y_pred) / len(y_true)