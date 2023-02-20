#!/bin/bash

echo "This is a shell script" 
python src/train.py --input_path data/CustomerData_LeadGenerator.csv --output_path models/model.joblib 
echo "Model is saved in the models folder"

