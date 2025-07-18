# Multi-Department Resume Classification

A machine learning project that automatically classifies resumes into different departments using DistilBERT transformer model.

## Features

- **Multi-department Classification**: Classifies resumes into 7 departments (Engineering, Finance, HR, Healthcare, IT, Marketing, Sales)
- **DistilBERT-based Model**: Uses pre-trained DistilBERT for high-accuracy text classification
- **Confidence Scoring**: Provides confidence levels (HIGH, MEDIUM, LOW) for predictions
- **Batch Processing**: Efficient batch inference with performance metrics
- **Comprehensive Results**: Detailed analysis and reporting of classification results

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd HS-NLP-Final-Project
   ```

2. **Run Docker container**
   ```bash
   docker run -it --rm -v $(pwd):/workspace python:3.9
   ```

3. **Download the pre-trained model**
   ```bash
   wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=112AQwMc5uYggc5tBW0v6Fp-QcdK8tseu' -O models/distilBERT_final_model/model.safetensors
   ```

## Project Structure

```
HS-NLP-Final-Project/
├── src/                          # Source code
│   ├── pipeline.py               # Main classification pipeline
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocessor.py           # Text preprocessing
│   ├── postprocessor.py          # Results formatting and reporting
│   └── utils.py                  # Utility functions
├── models/                       # Pre-trained models
│   ├── distilBERT_final_model/   # DistilBERT model files
│   ├── distilBERT_tokenizer/     # Tokenizer files
│   └── label_encoder.pkl         # Label encoder
├── notebooks/                    # Jupyter notebooks
│   ├── Inference demo.ipynb      # Inference demonstration
│   └── Final Format Multi-Department Resume Classification.ipynb
├── project/dataset/              # Dataset files
├── results/                      # Output results
└── Info/                         # Project documentation
```

## Usage

### Inference Demo

The `notebooks/Inference demo.ipynb` demonstrates how to:
- Load the pre-trained model
- Classify sample resumes
- Generate confidence scores and performance metrics
- Export results in multiple formats (CSV, JSON, Parquet)

Key features of the demo:
- **Real-time Classification**: Classify resumes with confidence scores
- **Performance Metrics**: Processing speed, accuracy, and F1 scores
- **Department Analysis**: Breakdown of predictions by department
- **Export Options**: Save results in various formats

### Training (Google Colab)

The `notebooks/Final Format Multi-Department Resume Classification.ipynb` contains the complete training pipeline and **should be run in Google Colab** for optimal performance with GPU support.

## Model Performance

- **Architecture**: DistilBERT (66.9M parameters)
- **Departments**: 7 classes (Engineering, Finance, HR, Healthcare, IT, Marketing, Sales)
- **Confidence Levels**: HIGH (>0.8), MEDIUM (0.5-0.8), LOW (<0.5)
- **Processing Speed**: ~4.4 samples/second (CPU)

## Dependencies

- Python 3.9+
- PyTorch
- Transformers (Hugging Face)
- Polars (data processing)
- Scikit-learn
- Jupyter

## Results

The model generates:
- **Department Classifications**: Predicted department for each resume
- **Confidence Scores**: Probability scores for predictions
- **Performance Metrics**: Accuracy, F1 score, processing time
- **Department Summary**: Statistics by department

## License

This project is for educational purposes as part of an NLP final project.