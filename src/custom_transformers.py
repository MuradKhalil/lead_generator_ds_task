from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SelectX(BaseEstimator, TransformerMixin):
    def __init__(self, optimal_features: List[str]) -> None:
        """_summary_

        Args:
            optimal_features (List[str]): _description_
        """
        self.optimal_features = optimal_features

    def fit(self, X, y=None):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """_summary_

        Args:
            X (pd.DataFrame): _description_
            y (_type_, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        return X[self.optimal_features]
