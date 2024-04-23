# Parkinsons-disease-detection Model
## Overview
This project aims to develop a Parkinson's disease detection model using machine learning techniques. The model employs various algorithms including Logistic Regression, Support Vector Classifier (SVC), Decision Tree Classifier, Random Forest Classifier, and K-Nearest Neighbors Classifier. The performance of these algorithms is compared to determine the most effective approach for Parkinson's disease detection.

## Requirements
Python 3.x
Libraries: pandas, numpy, scikit-learn (sklearn), matplotlib, scipy
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/your_username/parkinsons-detection-model.git
Install the required Python libraries:
Copy code
pip install pandas numpy scikit-learn matplotlib scipy
## Usage
Navigate to the project directory:
bash
Copy code
cd parkinsons-detection-model
Run the main script to train and evaluate the model:
Copy code
python parkinsons_detection.py
Follow the prompts to load the dataset, perform exploratory data analysis, train the model, and evaluate its performance.
Dataset
The dataset used for this project contains observations from individuals with and without Parkinson's disease. Prior to model training, exploratory data analysis (EDA) is performed to identify and handle missing values, outliers, and other preprocessing tasks.
credit : https://www.kaggle.com/datasets/gargmanas/parkinsonsdataset

## Algorithms
The following algorithms are implemented and compared for Parkinson's disease detection:

Logistic Regression
Support Vector Classifier (SVC)
Decision Tree Classifier
Random Forest Classifier
K-Nearest Neighbors Classifier
Performance Comparison
The performance of each algorithm is evaluated based on metrics such as accuracy, precision, recall, and F1-score. Decision Tree Classifier and Random Forest Classifier emerge as equally effective models for Parkinson's disease detection.

## Contributing
Contributions to this project are welcome! If you have suggestions for improvements or would like to report issues, please submit a pull request or open an issue on GitHub.
