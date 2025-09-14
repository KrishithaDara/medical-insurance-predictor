# Medical Expenses Prediction

## Overview

This project predicts individual medical expenses using machine learning models based on demographic and lifestyle factors.

## Dataset

**Features**:
- `age`: Age of the primary beneficiary
- `sex`: Gender (male/female) 
- `bmi`: Body Mass Index
- `children`: Number of dependents covered
- `smoker`: Smoking habit (yes/no)
- `region`: Residential region
- `expenses`: Medical costs (target variable)

## Analysis

### Data Exploration
- Smoking distribution analysis with pie chart
- Age distribution visualization 
- Age vs Expenses: "With increasing age, expense is expected to increase"
- BMI vs Expenses correlation analysis
- Smoking impact on expenses using boxplots

### Data Preprocessing
- Handled children outliers by capping values 4 & 5 to 3
- Applied categorical encoding
- Train-test split and feature scaling

### Models Implemented
- Linear Regression
- Lasso Regression  
- Ridge Regression
- Random Forest Regressor

## Key Findings

1. **Smoking Status**: Most significant predictor - smokers have dramatically higher medical costs
2. **Age**: Positive correlation with expenses
3. **BMI**: Higher BMI associated with increased costs
4. **Best Model**: Random Forest achieved superior performance by capturing non-linear relationships

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook medical_expenses_prediction.ipynb
```
