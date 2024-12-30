# 🐶 Multi-Class Dog Breed Classification Project

## Problem Statement
The goal of this project is to develop a model capable of identifying the breed of a dog from a given image.

## Dataset
The dataset used for this project is sourced from Kaggle's Dog Breed Identification Competition. It includes:
- Training Dataset: Approximately 10,000 labeled images.
- Testing Dataset: Approximately 10,000 unlabeled images for prediction

## Features
- Utilizes TensorFlow and TensorFlow Hub for model training and inference.
- Supports classification for 120 different dog breeds.
- Handles both training and testing datasets effectively.

## Evaluation Metrics
- The evaluation metric for this project involves creating a submission file that contains the ID of each image along with the predicted probabilities for each breed.

## Goal
Train a model with an accuracy score of at least 93% (subject to evaluation of metric appropriateness).

## Project Structure
- `dog-visiob.ipynb`: Jupyter notebook containing the main analysis and model development
- `requirements.txt`: List of required Python packages
- `labels.csv` : The data file downloaded from Kaggle.
- `models` : A folder containing the different models I created and trained on the data.
- `app.py` : A Python Script implementing the model with a StreamLit frontend for users to engage with.

## Setup and Installation
1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook main.ipynb`

