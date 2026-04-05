# Fraud Detection with PyOD AutoEncoder

This project builds a fraud detection model using PyOD's AutoEncoder on the anonymized credit card transactions dataset from Kaggle.

## Dataset
Kaggle Fraud Detection Dataset:
https://www.kaggle.com/datasets/whenamancodes/fraud-detection

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
unzip data/creditcard.zip

## Run the program after all dependencies have been installed
python3 src/testAutoEncoder.py