
# Resume Classification System - Deployment Guide

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download model files:
   - distilbert_final_model/
   - distilbert_tokenizer/
   - label_encoder.pkl

3. Initialize pipeline:
   ```python
   from resume_pipeline import ResumeClassificationPipeline

   pipeline = ResumeClassificationPipeline(
       model_path='./distilbert_final_model',
       tokenizer_path='./distilbert_tokenizer',
       label_encoder_path='./label_encoder.pkl'
   )

   result = pipeline.predict_single("Software engineer with Python experience...")
   print(f"Department: {result['department']}")
   print(f"Confidence: {result['confidence']:.3f}")
   ```

## Performance Metrics

- **Accuracy**: 89.1% F1-Score (+16.9% over baseline)
- **Speed**: ~20ms per resume
- **Throughput**: 50+ resumes/second
- **Interpretability**: Attention-based explanations available

## Production Features

Single and batch processing
Confidence scoring with human review triggers
Attention-based interpretability
Performance monitoring and logging
REST API interface
Health checks and error handling
Quality assurance alerts

## Deployment Options

1. **Local Development**: Run directly with Python
2. **Docker Container**: Containerized deployment
3. **Cloud Services**: AWS/GCP/Azure ML endpoints
4. **Enterprise Integration**: REST API for HR systems

## Model Information

- **Architecture**: DistilBERT (66M parameters)
- **Departments**: Engineering, Finance, HR, Healthcare, IT, Marketing, Sales
- **Input**: Resume text (max 512 tokens)
- **Output**: Department + confidence score + explanations
