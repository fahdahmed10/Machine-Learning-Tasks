"""
statistical_tests.py
Statistical inference, normality testing, and distribution analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, anderson
import statsmodels.api as sm

sns.set(style="whitegrid")


class DatasetAnalyzer:
    """
    Perform distribution analysis and normality testing
    for all numerical features in a dataset.
    """

    def __init__(self, df: pd.DataFrame, target: str = "target"):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing features and target
        target : str
            Name of label/target column
        """
        self.df = df
        self.target = target

        # Select numerical features only
        self.features = df.select_dtypes(include=np.number).columns.tolist()
        
        # Remove target from the feature list if it exists
        if target in self.features:
            self.features.remove(target)

    def plot_histogram(self, feature: str):
        """
        Plot histogram with KDE overlay for a feature.
        
        Parameters
        ----------
        feature : str
            Feature name
        """
        plt.figure(figsize=(8, 5))
        data = self.df[feature].dropna()
        sns.histplot(data, kde=True, bins=30, color='teal')
        plt.title(f"Distribution of {feature}", fontsize=14)
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_all_histograms(self):
        """
        Plot histograms for all features.
        """
        for feature in self.features:
            self.plot_histogram(feature)

    def qq_plot(self, feature: str):
        """
        Generate Q-Q plot to assess normality visually.
        
        Parameters
        ----------
        feature : str
            Feature name
        """
        data = self.df[feature].dropna()
        plt.figure(figsize=(6, 6))
        sm.qqplot(data, line='s')
        plt.title(f"Q-Q Plot for {feature}", fontsize=14)
        plt.tight_layout()
        plt.show()

    def shapiro_test(self, feature: str) -> str:
        """
        Perform Shapiro-Wilk test for normality.
        
        H0: Feature follows a normal distribution
        H1: Feature does not follow a normal distribution
        
        Parameters
        ----------
        feature : str
            Feature name
            
        Returns
        -------
        str
            Test result string
        """
        x = self.df[feature].dropna()
        stat, p = shapiro(x)

        result = "Normal" if p > 0.05 else "Not Normal"
        return f"Shapiro-Wilk: Stat={stat:.4f}, p={p:.4f} ({result})"

    def anderson_test(self, feature: str):
        """
        Perform Anderson-Darling test for normality.
        
        H0: Feature is normally distributed
        H1: Feature is not normally distributed
        
        Parameters
        ----------
        feature : str
            Feature name
        """
        x = self.df[feature].dropna()
        res = anderson(x)

        print(f"  [Anderson-Darling] Stat: {res.statistic:.4f}")
        for sig, crit in zip(res.significance_level, res.critical_values):
            status = "Normal" if res.statistic < crit else "NOT Normal"
            print(f"    At {sig}%: {status}")

    def conditional_distribution(self, feature: str):
        """
        Plot conditional distribution of feature given target classes.
        
        Parameters
        ----------
        feature : str
            Feature name
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=self.df,
            x=feature,
            hue=self.target,
            kde=True,
            bins=30,
            element="step",
            stat="density",
            common_norm=False,
            palette='coolwarm'
        )
        plt.title(f"Conditional Distribution of {feature} by Target", fontsize=14)
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend(title=self.target)
        plt.tight_layout()
        plt.show()

    def analyze_feature(self, feature: str):
        """
        Complete analysis for a single feature:
        - Histogram
        - Q-Q plot
        - Normality tests
        - Conditional distribution
        
        Parameters
        ----------
        feature : str
            Feature name
        """
        print("="*80)
        print(f" ANALYSIS FOR FEATURE: {feature}")
        print("="*80)

        # Create 1x3 subplot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        data = self.df[feature].dropna()

        # Plot 1: Histogram
        sns.histplot(data, kde=True, bins=30, ax=axes[0], color='teal')
        axes[0].set_title(f"Overall Distribution\n({feature})")
        axes[0].set_xlabel(feature)

        # Plot 2: Q-Q Plot
        sm.qqplot(data, line='s', ax=axes[1])
        axes[1].set_title(f"Q-Q Plot\n(Normality Check)")

        # Plot 3: Conditional Distribution
        sns.histplot(
            data=self.df,
            x=feature,
            hue=self.target,
            kde=True,
            bins=30,
            element="step",
            stat="density",
            common_norm=False,
            palette='coolwarm',
            ax=axes[2]
        )
        axes[2].set_title(f"Conditional Distribution\n(Separation by Target)")

        plt.tight_layout()
        plt.show()

        # Print statistical test results
        print("\n--- Statistical Normality Tests ---")
        print(f"  {self.shapiro_test(feature)}")
        self.anderson_test(feature)
        print("\n")

    def analyze_all(self):
        """
        Analyze all numerical features in the dataset.
        """
        for feature in self.features:
            self.analyze_feature(feature)

    def generate_analysis_report(self) -> pd.DataFrame:
        """
        Generate a summary report of normality tests for all features.
        
        Returns
        -------
        pd.DataFrame
            Summary table with normality test results
        """
        report_data = []
        
        for feature in self.features:
            x = self.df[feature].dropna()
            
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = shapiro(x)
            shapiro_result = "Normal" if shapiro_p > 0.05 else "Not Normal"
            
            # Anderson-Darling test
            anderson_res = anderson(x)
            anderson_result = "Normal" if anderson_res.statistic < anderson_res.critical_values[2] else "Not Normal"
            
            # Skewness
            skewness = x.skew()
            
            report_data.append({
                'Feature': feature,
                'Shapiro-Wilk p-value': shapiro_p,
                'Shapiro Result': shapiro_result,
                'Anderson Statistic': anderson_res.statistic,
                'Anderson Result (5%)': anderson_result,
                'Skewness': skewness
            })
        
        report_df = pd.DataFrame(report_data)
        return report_df