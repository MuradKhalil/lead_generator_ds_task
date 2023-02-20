import argparse

import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from custom_transformers import SelectX

seed = 42
target = 'b_gekauft_gesamt'
features = ['b_specialisation_h', 'b_specialisation_j']

def parse_args():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    
    args=parser.parse_args()
    args = vars(args)
    return args

def main():
    """_summary_
    """
    args = parse_args()
    df = pd.read_csv(args['input_path'])
    y = df[target]
    feature_selector = SelectX(optimal_features=features)
    clf = XGBClassifier(scale_pos_weight=65, random_state=seed)
    model = make_pipeline(feature_selector, clf)
    model.fit(df, y)

    joblib.dump(model, args['output_path'])

if __name__ == "__main__":
    main()

