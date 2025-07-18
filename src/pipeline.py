import time
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

class ResumeClassificationPipeline:
    def __init__(self, model_path: str, tokenizer_path: str, label_encoder_path: str, 
                 confidence_threshold: float = 0.7):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.label_encoder_path = label_encoder_path
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.metrics = {
            'total_predictions': 0,
            'total_time': 0.0,
            'successful_predictions': 0,
            'failed_predictions': 0
        }
    
    def initialize(self):
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.tokenizer_path)
        
        with open(self.label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
    
    def predict_single(self, text: str, include_probabilities: bool = False) -> Dict:
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(
                text, return_tensors='pt', max_length=512, 
                truncation=True, padding='max_length'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities.max().item()
            
            department = self.label_encoder.classes_[predicted_class]
            
            if confidence >= self.confidence_threshold:
                confidence_level = 'HIGH'
            elif confidence >= 0.5:
                confidence_level = 'MEDIUM'
            else:
                confidence_level = 'LOW'
            
            processing_time = time.time() - start_time
            
            self.metrics['total_predictions'] += 1
            self.metrics['total_time'] += processing_time
            self.metrics['successful_predictions'] += 1
            
            result = {
                'department': department,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'needs_human_review': confidence < self.confidence_threshold,
                'processing_time': processing_time
            }
            
            if include_probabilities:
                result['all_probabilities'] = {
                    dept: float(prob) for dept, prob in 
                    zip(self.label_encoder.classes_, probabilities[0].cpu().numpy())
                }
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics['total_predictions'] += 1
            self.metrics['total_time'] += processing_time
            self.metrics['failed_predictions'] += 1
            
            return {
                'department': 'Unknown',
                'confidence': 0.0,
                'confidence_level': 'LOW',
                'needs_human_review': True,
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def predict_batch(self, texts: List[str], batch_size: int = 8, 
                     include_probabilities: bool = False) -> List[Dict]:
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            for text in batch_texts:
                result = self.predict_single(text, include_probabilities)
                results.append(result)
        
        return results
    
    def get_performance_metrics(self) -> Dict:
        if self.metrics['total_predictions'] == 0:
            return {
                'total_predictions': 0,
                'success_rate': 0.0,
                'avg_processing_time': 0.0,
                'throughput_per_second': 0.0
            }
        
        success_rate = self.metrics['successful_predictions'] / self.metrics['total_predictions']
        avg_time = self.metrics['total_time'] / self.metrics['total_predictions']
        throughput = self.metrics['total_predictions'] / self.metrics['total_time']
        
        return {
            'total_predictions': self.metrics['total_predictions'],
            'success_rate': success_rate,
            'avg_processing_time': avg_time,
            'throughput_per_second': throughput
        }
    
    def get_info(self) -> Dict:
        return {
            'is_initialized': self.model is not None,
            'departments': list(self.label_encoder.classes_) if self.label_encoder else [],
            'num_departments': len(self.label_encoder.classes_) if self.label_encoder else 0,
            'confidence_threshold': self.confidence_threshold,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'device': str(self.device)
        }