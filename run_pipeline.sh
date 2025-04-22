#!/bin/bash
set -e # Exit on error

echo "1. Prepare data..."
python3 src/prepare_data.py \
      --input "data/Titanic Dataset.csv" \
      --output "data/processed.pkl"
echo "1.1 Prepare data FINISHED"

echo "2. Training model..."
python3 src/train.py \
    --data "data/processed.pkl" \
    --model "titanic_model.pkl"
echo "2.1 Training model FINISHED"

echo "3. Evaluating..."
python3 src/evaluate.py \
    --data "data/processed.pkl" \
    --model "models/titanic_model.pkl"
echo "3.1 Evaluating FINISHED"

echo "Pipeline completed"
