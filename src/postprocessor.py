import json
import pandas as pd
import polars as pl
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

class ResultFormatter:
    def __init__(self):
        self.department_descriptions = {
            'Engineering': 'Technical roles involving design, development, and maintenance',
            'Finance': 'Financial analysis, accounting, budgeting, and financial planning',
            'HR': 'Human resources, talent acquisition, and organizational development',
            'Healthcare': 'Medical, nursing, healthcare administration, and patient care',
            'IT': 'Information technology, software development, and technical support',
            'Marketing': 'Marketing strategy, digital marketing, and brand management',
            'Sales': 'Sales management, business development, and customer relations'
        }
    
    def format_single_result(self, result: Dict, include_metadata: bool = True) -> Dict:
        formatted = {
            'department': result.get('department', 'Unknown'),
            'confidence': result.get('confidence', 0.0),
            'confidence_level': result.get('confidence_level', 'LOW'),
            'needs_human_review': result.get('needs_human_review', True),
            'recommendation': self._get_recommendation(result),
            'status': 'success' if 'error' not in result else 'error'
        }
        
        if include_metadata:
            formatted['metadata'] = {
                'processing_time': result.get('processing_time', 0.0),
                'timestamp': datetime.now().isoformat(),
                'model_version': '1.0.0'
            }
        
        if 'all_probabilities' in result:
            formatted['all_departments'] = [
                {
                    'department': dept,
                    'probability': prob,
                    'description': self.department_descriptions.get(dept, 'No description')
                }
                for dept, prob in sorted(result['all_probabilities'].items(), 
                                       key=lambda x: x[1], reverse=True)
            ]
        
        return formatted
    
    def format_batch_results(self, results: List[Dict]) -> Dict:
        formatted_results = [self.format_single_result(result) for result in results]
        
        confidences = [r['confidence'] for r in results if 'error' not in r]
        departments = [r['department'] for r in results if 'error' not in r]
        
        batch_statistics = {
            'total_processed': len(results),
            'successful_predictions': sum(1 for r in results if 'error' not in r),
            'failed_predictions': sum(1 for r in results if 'error' in r),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'human_review_needed': sum(1 for r in results if r.get('needs_human_review', True)),
            'confidence_distribution': {
                'HIGH': sum(1 for r in results if r.get('confidence_level') == 'HIGH'),
                'MEDIUM': sum(1 for r in results if r.get('confidence_level') == 'MEDIUM'),
                'LOW': sum(1 for r in results if r.get('confidence_level') == 'LOW')
            },
            'department_distribution': {dept: departments.count(dept) for dept in set(departments)}
        }
        
        return {
            'results': formatted_results,
            'batch_statistics': batch_statistics,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_recommendation(self, result: Dict) -> str:
        confidence = result.get('confidence', 0.0)
        department = result.get('department', 'Unknown')
        
        if confidence >= 0.8:
            return f"High confidence - route to {department}"
        elif confidence >= 0.6:
            return f"Moderate confidence - review for {department}"
        else:
            return "Low confidence - requires human review"
    
    def to_polars_dataframe(self, results: List[Dict]) -> pl.DataFrame:
        data = []
        for i, result in enumerate(results):
            data.append({
                'id': i + 1,
                'department': result.get('department', 'Unknown'),
                'confidence': result.get('confidence', 0.0),
                'confidence_level': result.get('confidence_level', 'LOW'),
                'needs_review': result.get('needs_human_review', True),
                'processing_time_ms': result.get('processing_time', 0.0) * 1000,
                'status': 'success' if 'error' not in result else 'error'
            })
        
        return pl.DataFrame(data)

class ReportGenerator:
    def __init__(self):
        self.category_mapping = {
            'HR': 'HR',
            'ENGINEERING': 'Engineering',
            'FINANCE': 'Finance',
            'HEALTHCARE': 'Healthcare',
            'INFORMATION-TECHNOLOGY': 'IT',
            'DESIGNER': 'Marketing',
            'SALES': 'Sales'
        }
    
    def generate_comprehensive_report(self, results: List[Dict], 
                                    original_categories: List[str] = None) -> Dict:
        if not results:
            return {'error': 'No results to analyze'}
        
        valid_results = [r for r in results if 'error' not in r]
        
        # Basic statistics
        confidences = [r['confidence'] for r in valid_results]
        departments = [r['department'] for r in valid_results]
        processing_times = [r['processing_time'] for r in valid_results]
        
        # Performance metrics
        if original_categories:
            true_labels = [self.category_mapping.get(cat, cat) for cat in original_categories]
            predictions = [r['department'] for r in results]
            
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='macro', zero_division=0
            )
            
            per_class_report = classification_report(
                true_labels, predictions, output_dict=True, zero_division=0
            )
        else:
            accuracy = precision = recall = f1 = 0.0
            per_class_report = {}
        
        return {
            'model_type': 'DistilBERT',
            'sample_size': len(results),
            'inference_time': sum(processing_times),
            'samples_per_second': len(results) / sum(processing_times) if processing_times else 0,
            'sample_accuracy': accuracy,
            'sample_precision': precision,
            'sample_recall': recall,
            'sample_f1': f1,
            'baseline_f1': 0.762,
            'confidence_statistics': {
                'mean_confidence': np.mean(confidences) if confidences else 0,
                'median_confidence': np.median(confidences) if confidences else 0,
                'std_confidence': np.std(confidences) if confidences else 0,
                'min_confidence': np.min(confidences) if confidences else 0,
                'max_confidence': np.max(confidences) if confidences else 0
            },
            'human_review_statistics': {
                'total_needing_review': sum(1 for r in results if r.get('needs_human_review', True)),
                'review_rate': sum(1 for r in results if r.get('needs_human_review', True)) / len(results),
                'high_confidence_predictions': sum(1 for r in results if r.get('confidence_level') == 'HIGH'),
                'medium_confidence_predictions': sum(1 for r in results if r.get('confidence_level') == 'MEDIUM'),
                'low_confidence_predictions': sum(1 for r in results if r.get('confidence_level') == 'LOW')
            },
            'department_distribution': {dept: departments.count(dept) for dept in set(departments)} if departments else {},
            'per_class_metrics': per_class_report,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_polars_summary(self, results: List[Dict]) -> pl.DataFrame:
        if not results:
            return pl.DataFrame()
        
        valid_results = [r for r in results if 'error' not in r]
        departments = [r['department'] for r in valid_results]
        
        summary_data = []
        for dept in set(departments):
            dept_results = [r for r in valid_results if r['department'] == dept]
            confidences = [r['confidence'] for r in dept_results]
            
            summary_data.append({
                'department': dept,
                'count': len(dept_results),
                'avg_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'high_confidence_count': sum(1 for r in dept_results if r['confidence_level'] == 'HIGH'),
                'needs_review_count': sum(1 for r in dept_results if r['needs_human_review'])
            })
        
        return pl.DataFrame(summary_data).sort('count', descending=True)

class DataExporter:
    @staticmethod
    def save_results(results: Dict, filepath: str):
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json_results = convert_numpy_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    @staticmethod
    def save_polars_csv(df: pl.DataFrame, filepath: str):
        df.write_csv(filepath)
    
    @staticmethod
    def save_polars_parquet(df: pl.DataFrame, filepath: str):
        df.write_parquet(filepath)