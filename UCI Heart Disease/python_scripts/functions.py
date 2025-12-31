# -*- coding: utf-8 -*-
"""
Helper functions for data preprocessing and analysis
"""

import pandas as pd
import numpy as np

def one_hot_encoding(df):
    """
    Apply one-hot encoding to categorical features
    """
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['fbs'] = df['fbs'].astype(int)
    df['exang'] = df['exang'].astype(int)
    df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    categorical_cols = ['cp', 'restecg']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def accuracy(y_true, y_pred):
    """
    Calculate accuracy score
    """
    return np.sum(y_true == y_pred) / len(y_true)