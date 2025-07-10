
# Performance Report - Baseline Model

## Model Architecture
- **Algorithm**: TF-IDF + Logistic Regression
- **Feature Engineering**: TF-IDF with 5000 features
- **N-grams**: (1, 2)
- **Regularization**: L2 (built-in to LogisticRegression)
- **Class Balancing**: Enabled

## Training Configuration
- **Training Samples**: 1,738
- **Validation Samples**: 372
- **Features**: 5,000
- **Solver**: lbfgs
- **Convergence**: Yes

## Performance Metrics
| Metric | Score |
|--------|-------|
| Accuracy | 0.677 |
| Precision (Macro) | 0.654 |
| Recall (Macro) | 0.665 |
| F1-Score (Macro) | 0.655 |

## Per-Department Performance
| Department | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Engineering | 0.812 | 0.776 | 0.794 | 67.0 |
| Finance | 0.787 | 0.923 | 0.850 | 52.0 |
| HR | 0.699 | 0.690 | 0.695 | 84.0 |
| Healthcare | 0.548 | 0.486 | 0.515 | 35.0 |
| IT | 0.568 | 0.583 | 0.575 | 36.0 |
| Marketing | 0.646 | 0.500 | 0.564 | 62.0 |
| Sales | 0.521 | 0.694 | 0.595 | 36.0 |


## Efficiency Metrics
- **Training Time**: 0.25 seconds
- **Inference Speed**: 68072.1 samples/second
- **Model Size**: ~0.3 MB

## Baseline Established
This model serves as the baseline for comparison with more advanced models:
- **Baseline F1-Score**: 0.655
- **Target for Deep Learning**: >0.705 (minimum 5% improvement)

## Key Strengths
- Fast training and inference
- Interpretable feature weights
- Good performance on balanced classes
- Robust to varying text lengths

## Areas for Improvement
- Better handling of semantic similarity
- Improved performance on minority classes
- Context-aware feature extraction
- Handling of domain-specific terminology
