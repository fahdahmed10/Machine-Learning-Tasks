# -*- coding: utf-8 -*-
"""
Outlier handling functions
"""

import numpy as np


def handle_invalid_zeros(df, invalid_zero_cols):
    """
    Replace zeros with NaN for columns where zero is physiologically impossible
    and impute using median
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset
    invalid_zero_cols : list
        List of column names where zero is invalid
    
    Returns:
    --------
    df : pandas.DataFrame
        Dataset with zeros handled
    """
    # Replace zeros with NaN
    df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)

    # Median imputation (clinically safe and robust)
    df[invalid_zero_cols] = df[invalid_zero_cols].fillna(
        df[invalid_zero_cols].median()
    )
    
    return df