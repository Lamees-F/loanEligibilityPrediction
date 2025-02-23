# Loan Eligibility Prediction

This repository contains code and models for predicting loan eligibility using various machine learning algorithms.
## Repository Files
- `app.py`: Main application script.
- `model (2).joblib`: Pre-trained model file.
- `mymodel.ipynb`: Jupyter notebook containing the model training and evaluation code.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/loanEligibilityPrediction.git
    cd loanEligibilityPrediction
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```


## Model Training and Evaluation

The notebook [mymodel.ipynb](http://_vscodecontentref_/3) includes the following steps:

1. Data Preprocessing
2. Model Training
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Random Forest
3. Model Evaluation
    - Cross-Validation Scores
    - Mean Accuracy
4. Model Saving
    - The trained Logistic Regression model is saved as `model (2).joblib`.

## Results

### Logistic Regression Cross-Validation:
- Scores: [0.81300813, 0.7804878, 0.7804878, 0.85365854, 0.81147541]
- Mean Accuracy: 0.8078235372517659

### SVM Cross-Validation:
- Scores: [0.81300813, 0.7804878, 0.7804878, 0.85365854, 0.81967213]
- Mean Accuracy: 0.809462881514061

### Random Forest Cross-Validation:
- Scores: [0.7804878, 0.74796748, 0.77235772, 0.82926829, 0.77868852]
- Mean Accuracy: 0.7817539650806344
