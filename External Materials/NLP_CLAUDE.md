# Data Science & NLP Project Context

## Environment & Setup
- **Primary**: Google Colab (upload to Drive, mount with `drive.mount('/content/drive')`)
- **Local**: Jupyter Lab/Notebook for development
- **Python**: 3.8+ required
- **GPU**: Use Colab T4/A100 for transformer training
- **Memory**: Monitor RAM usage in Colab (crashes at ~12GB)

## Essential Commands
```bash
# Setup
pip install transformers datasets torch scikit-learn pandas seaborn wandb
# Save models
model.save_pretrained('./models/bert_classifier')
# Check GPU
torch.cuda.is_available()
```

## Data Science Workflow Rules
- **Always** start with `df.info()`, `df.describe()`, and missing value analysis
- **Never** skip train/val/test splits (70/15/15 typical)
- **Always** set random seeds: `np.random.seed(42)`, `torch.manual_seed(42)`
- **Check** class balance before modeling (use `value_counts()`)
- **Stratify** splits for classification tasks
- **Save** intermediate results to Drive: `/content/drive/MyDrive/project_name/`

## NLP & BERT Specifics
- **Tokenization**: Max length 512 for BERT, check with `tokenizer.tokenize()`
- **Models**: Use `bert-base-uncased` for English, `distilbert` for speed
- **Fine-tuning**: Learning rate 2e-5, batch size 16, epochs 3-5
- **Memory**: Use gradient_checkpointing=True for large models
- **Evaluation**: Always include confusion matrix, classification report
- **Baseline**: Start with simple models (LogisticRegression, Naive Bayes)

## Code Style
- Use descriptive variable names: `X_train_transformed`, `bert_classifier_model`
- Document preprocessing steps with comments
- Create functions for reusable code (tokenization, evaluation metrics)
- Use f-strings for printing: `print(f"Accuracy: {accuracy:.3f}")`
- Import style: `from sklearn.metrics import accuracy_score, classification_report`

## Data Handling
- **Text preprocessing**: lowercase, remove special chars, handle NaN
- **Encoding**: UTF-8 for text files, specify in `pd.read_csv(encoding='utf-8')`
- **Large files**: Use chunking with `pd.read_csv(chunksize=1000)`
- **Pandas**: Use `.copy()` to avoid SettingWithCopyWarning
- **Memory**: Use `del df` and `gc.collect()` for large datasets

## Model Training & Evaluation
- **Cross-validation**: Use StratifiedKFold for small datasets
- **Metrics**: Accuracy, Precision, Recall, F1 for classification
- **Imbalanced data**: Use class_weight='balanced' or SMOTE
- **Hyperparameters**: Log all params with comments
- **Checkpoints**: Save model state every epoch for long training
- **Early stopping**: Monitor validation loss

## Documentation Requirements
- **Notebook sections**: Data, EDA, Preprocessing, Modeling, Results, Conclusions
- **Method justification**: Explain why each preprocessing/model choice
- **Results interpretation**: Statistical significance, business impact
- **Reproducibility**: Include all hyperparameters, seeds, package versions

## Error Prevention
- **Text length**: Check max/min sequence lengths before tokenization
- **Memory errors**: Clear variables with `del`, use smaller batch sizes
- **Version conflicts**: Pin package versions in requirements
- **Colab timeouts**: Save work frequently, use session timeout extensions
- **Data leakage**: Never use test data for preprocessing parameters

## File Organization
```
project_name/
├── notebooks/           # Jupyter notebooks
├── data/               # Raw and processed data
├── models/             # Saved models and tokenizers  
├── results/            # Plots, metrics, outputs
├── requirements.txt    # Package dependencies
└── README.md          # Project overview
```

## Common Issues & Solutions
- **CUDA memory**: Use `torch.cuda.empty_cache()` between experiments
- **Tokenizer warnings**: Add `padding=True, truncation=True` to tokenizer
- **Colab disconnects**: Use `from google.colab import output; output.enable_custom_widget_manager()`
- **Large model loading**: Use `torch_dtype=torch.float16` for memory efficiency
- **Progress tracking**: Use `tqdm` for loops, `transformers` built-in progress bars