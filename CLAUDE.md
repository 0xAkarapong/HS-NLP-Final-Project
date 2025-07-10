# Multi-Department Resume Classification Project Context

## Project Overview
This project implements a multi-department resume classification system using both traditional ML and deep learning approaches. The system classifies resumes into 7 departments: Engineering, Finance, HR, Healthcare, IT, Marketing, and Sales.

## Environment & Setup
- **Primary**: Local development environment with GPU support (CUDA)
- **Alternative**: Google Colab with T4/A100 GPU for training
- **Python**: 3.8+ required
- **Key Dependencies**: transformers, torch, scikit-learn, pandas, seaborn
- **GPU**: Required for DistilBERT training (14GB+ memory recommended)

## Essential Commands
```bash
# Environment setup
pip install transformers datasets torch scikit-learn pandas seaborn matplotlib numpy

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run main notebook
jupyter notebook "project/Multi Department Resume Classification.ipynb"
```

## Project Structure Rules
- **Phase-based approach**: 7 distinct phases from data exploration to production
- **Always** follow the phase sequence: Data → Preprocessing → Baseline → BERT → Training → Evaluation → Production
- **Never** skip baseline model - establishes performance benchmark
- **Always** use stratified splits (70/15/15) to maintain class balance
- **Set** random seeds consistently: `np.random.seed(42)`, `torch.manual_seed(42)`

## Data Handling Specifics
- **Dataset**: 2,484 resumes with 24 original categories mapped to 7 departments
- **Text preprocessing**: Clean HTML, normalize text, handle special characters
- **Tokenization**: Max length 512 for BERT (89.6% truncation rate expected)
- **Class imbalance**: Use `class_weight='balanced'` and stratified sampling
- **Memory management**: Use `torch.cuda.empty_cache()` between experiments

## Model Architecture Guidelines
### Baseline Model (Phase 3)
- **Algorithm**: TF-IDF + Logistic Regression
- **Features**: 5000 TF-IDF features, ngram_range=(1,2)
- **Target**: Establish performance benchmark (~65.8% F1-Score)
- **Use**: `LogisticRegression(class_weight='balanced', max_iter=1000)`

### DistilBERT Model (Phase 4-5)
- **Model**: `distilbert-base-uncased` (66M parameters)
- **Sequence length**: 512 tokens (truncation required)
- **Batch size**: 16 (adjust based on GPU memory)
- **Learning rate**: 2e-5 with warmup
- **Epochs**: 3-4 (early stopping enabled)
- **Target**: >70.8% F1-Score (5% improvement over baseline)

## Training Configuration
```python
# DistilBERT Training Args
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
BATCH_SIZE = 16
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
MAX_LENGTH = 512
```

## Evaluation Standards
- **Primary metric**: Macro F1-Score (handles class imbalance)
- **Secondary metrics**: Accuracy, Precision, Recall
- **Always** include confusion matrix and per-class analysis
- **Test set**: Final evaluation only - never for hyperparameter tuning
- **Validation**: Monitor for overfitting (val/test F1 difference <0.02)

## Code Style & Documentation
- **Notebook structure**: Clear markdown headers for each phase
- **Function naming**: Descriptive names like `clean_resume_text()`, `analyze_tokenization_lengths()`
- **Progress tracking**: Use phase completion checkpoints
- **Results saving**: Save models, metrics, and summaries for each phase
- **Memory cleanup**: `del` large variables, `gc.collect()` when needed

## Department Mapping Strategy
```python
DEPARTMENT_MAPPING = {
    'INFORMATION-TECHNOLOGY': 'IT',
    'BPO': 'IT',
    'DIGITAL-MEDIA': 'IT',
    'ENGINEERING': 'Engineering',
    'CONSTRUCTION': 'Engineering',
    # ... (24 categories → 7 departments)
}
```

## Performance Benchmarks
- **Baseline**: 65.8% F1-Score (TF-IDF + LogReg)
- **Target**: >70.8% F1-Score (DistilBERT)
- **Inference speed**: Balance accuracy vs speed (expect ~100x slower than baseline)
- **Memory usage**: Monitor GPU memory (peak ~15GB for training)

## Error Prevention
- **Text truncation**: 89.6% of resumes exceed 512 tokens - verify truncation doesn't lose critical info
- **Class imbalance**: Always use stratified splits and balanced class weights
- **Memory errors**: Reduce batch size if OOM, use gradient checkpointing
- **Training instability**: Monitor validation loss, use early stopping
- **Reproducibility**: Set all random seeds, save preprocessing config

## File Organization
```
project/
├── Multi Department Resume Classification.ipynb  # Main notebook
├── dataset/Resume/Resume.csv                     # Raw data
├── External Materials/NLP_CLAUDE.md             # Reference guide
└── models/                                       # Saved models
    ├── baseline_tfidf_logistic.pkl
    ├── distilbert_final_model/
    └── label_encoder.pkl
```

## Phase Completion Checklist
- **Phase 1**: Data exploration and department mapping (7 departments confirmed)
- **Phase 2**: Text preprocessing and stratified splits (train/val/test ready)
- **Phase 3**: Baseline model training and evaluation (benchmark established)
- **Phase 4**: BERT tokenization and dataset preparation (truncation analyzed)
- **Phase 5**: DistilBERT training and validation (target performance achieved)
- **Phase 6**: Test evaluation and generalization analysis (final metrics)
- **Phase 7**: Production pipeline and documentation (deployment ready)

## Common Issues & Solutions
- **CUDA memory**: Reduce batch size to 8 or 4 if OOM errors
- **Long training time**: Use DistilBERT over BERT-base, enable FP16
- **Poor convergence**: Check learning rate, increase warmup steps
- **Overfitting**: Reduce epochs, increase weight decay, monitor val/test gap
- **Low F1-Score**: Verify class balancing, check text preprocessing quality

## Production Deployment Notes
- **Model size**: 268MB (DistilBERT) vs 20MB (baseline)
- **Inference time**: ~10ms per resume vs 0.008ms (baseline)
- **Memory requirements**: 2GB GPU memory for inference
- **Scalability**: Consider batch processing for high-volume scenarios
- **Monitoring**: Track F1-Score degradation over time with new data

## Research Extensions
- **Attention visualization**: Use model attention weights for interpretability
- **Ensemble methods**: Combine DistilBERT with baseline for robustness
- **Domain adaptation**: Fine-tune on industry-specific resume datasets
- **Multi-label classification**: Extend to predict multiple relevant departments
- **Active learning**: Implement uncertainty sampling for efficient labeling

## Success Metrics
- **Technical**: >70.8% F1-Score on test set
- **Practical**: <50ms inference time for production
- **Scientific**: Reproducible methodology with clear documentation
- **Business**: Automated resume routing with 95%+ accuracy for HR workflows