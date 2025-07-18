import torch
import polars as pl
from pathlib import Path
from typing import Dict, Any

class ModelUtils:
    @staticmethod
    def get_device() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @staticmethod
    def clear_gpu_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        return {
            'cuda_available': torch.cuda.is_available(),
            'device': ModelUtils.get_device(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

class DisplayUtils:
    @staticmethod
    def format_polars_table(df: pl.DataFrame, title: str = None) -> None:
        if title:
            print(f"\n{title}")
            print("=" * len(title))
        
        print(df.to_pandas().to_string(index=False))
        print()
    
    @staticmethod
    def format_summary_stats(stats: Dict) -> None:
        print("\nSummary Statistics:")
        print("-" * 20)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        print()
    
    @staticmethod
    def format_confidence_distribution(results: list) -> pl.DataFrame:
        confidence_levels = [r.get('confidence_level', 'UNKNOWN') for r in results]
        
        data = []
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            count = confidence_levels.count(level)
            percentage = (count / len(results)) * 100 if results else 0
            data.append({
                'confidence_level': level,
                'count': count,
                'percentage': percentage
            })
        
        return pl.DataFrame(data)

class FileUtils:
    @staticmethod
    def create_output_dir(path: str) -> Path:
        output_dir = Path(path)
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    @staticmethod
    def check_model_files(model_path: str, tokenizer_path: str, 
                         label_encoder_path: str) -> Dict[str, bool]:
        return {
            'model': Path(model_path).exists(),
            'tokenizer': Path(tokenizer_path).exists(),
            'label_encoder': Path(label_encoder_path).exists()
        }