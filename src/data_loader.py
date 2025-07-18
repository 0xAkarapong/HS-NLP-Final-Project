import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

class DataLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.df = None
        self.category_mapping = {
            'HR': 'HR',
            'ENGINEERING': 'Engineering',
            'FINANCE': 'Finance',
            'HEALTHCARE': 'Healthcare',
            'INFORMATION-TECHNOLOGY': 'IT',
            'DESIGNER': 'Marketing',
            'SALES': 'Sales'
        }
    
    def load_dataset(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.dataset_path)
        return self.df
    
    def get_sample_data(self, sample_size: int = 5, 
                       target_departments: List[str] = None) -> Tuple[List[Dict], List[str]]:
        if self.df is None:
            self.load_dataset()
        
        if target_departments is None:
            target_departments = ['HR', 'ENGINEERING', 'FINANCE', 'HEALTHCARE', 
                                'INFORMATION-TECHNOLOGY', 'DESIGNER', 'SALES']
        
        np.random.seed(42)
        sample_resumes = []
        original_categories = []
        
        for dept in target_departments:
            if dept in self.df['Category'].values:
                dept_resumes = self.df[self.df['Category'] == dept].sample(
                    n=min(sample_size, len(self.df[self.df['Category'] == dept])), 
                    random_state=42
                )
                sample_resumes.extend(dept_resumes.to_dict('records'))
                original_categories.extend([dept] * len(dept_resumes))
        
        return sample_resumes, original_categories
    
    def get_resume_texts(self, sample_data: List[Dict]) -> List[str]:
        return [resume['Resume_str'] for resume in sample_data]
    
    def get_dataset_info(self) -> Dict:
        if self.df is None:
            self.load_dataset()
        
        return {
            'total_resumes': len(self.df),
            'total_categories': len(self.df['Category'].unique()),
            'categories': list(self.df['Category'].unique()),
            'category_counts': self.df['Category'].value_counts().to_dict(),
            'target_departments': list(self.category_mapping.keys()),
            'department_mapping': self.category_mapping
        }