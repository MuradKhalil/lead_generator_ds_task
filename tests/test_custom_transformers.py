import pandas as pd
import pytest

from src.custom_transformers import SelectX


@pytest.mark.parametrize("columns, X, expected",
    [(
        ['col2', 'col5'],
        pd.DataFrame({
            'col1': ['a', 'b', 'c'], 
            'col2': [1, 2, 3],
            'col3': ['a', 'b', 'c'],
            'col4': [1, 2, 3],
            'col5': ['a', 'b', 'c'],
        }),
        pd.DataFrame({
            'col2': [1, 2, 3], 
            'col5': ['a', 'b', 'c'],
        })
    )]
)
def test_x_selector(columns, X, expected):
    x_selector = SelectX(columns)
    pd.testing.assert_frame_equal(x_selector.transform(X), expected)