from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SelectX(BaseEstimator, TransformerMixin):
    def __init__(self, optimal_features: List[str]) -> None:
        """sklearn pipeline compatible class to select features needed for modeling

        Args:
            optimal_features (List[str]): list of optimal features
        """
        self.optimal_features = optimal_features

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """method that does the column selection 

        Args:
            X (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: dataframe with the selected features only
        """
        return X[self.optimal_features]
