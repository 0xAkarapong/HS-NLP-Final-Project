
# Error Analysis Report - Baseline Model

## Overview
This report analyzes the errors made by the baseline TF-IDF + Logistic Regression model.

## Overall Performance
- **Accuracy**: 0.677
- **Macro F1-Score**: 0.655
- **Total Validation Samples**: 372
- **Correct Predictions**: 252
- **Incorrect Predictions**: 120

## Most Confident Predictions

### Most Confident Correct Prediction
- **True Department**: Finance
- **Predicted Department**: Finance
- **Confidence**: 0.624
- **Text Preview**: volunteer accountant summary cpa candidate with years of strong financial accounting and audit experience and knowledge of internal control, enterprise risk management and gl, pl, bs reconciliations, ...

### Most Confident Incorrect Prediction
- **True Department**: Engineering
- **Predicted Department**: Finance
- **Confidence**: 0.569
- **Text Preview**: project administrator engineering summary a consistent team leader with great analytic and interpersonal skills; highly focused in achieving and maintaining excellent customer relationships to assist ...


## Common Misclassification Patterns
1. **Marketing → HR**: 8 cases
2. **HR → Healthcare**: 7 cases
3. **Marketing → Sales**: 7 cases
4. **HR → Marketing**: 6 cases
5. **Healthcare → HR**: 6 cases


## Confidence Analysis
- **Average Confidence (Correct)**: 0.402
- **Average Confidence (Incorrect)**: 0.307
- **Confidence Difference**: 0.095

## Recommendations
1. Focus on improving Marketing vs HR distinction
2. Consider domain-specific features for better department separation
3. Investigate low-confidence predictions for model improvement
4. Use ensemble methods to boost performance on confused classes
