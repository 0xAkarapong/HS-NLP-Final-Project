import re
from typing import Dict, List, Optional, Union
from bs4 import BeautifulSoup
import torch
from transformers import DistilBertTokenizer


class TextPreprocessor:
    """
    Handles text preprocessing for resume classification.
    
    This class provides methods to clean resume text by removing HTML tags,
    normalizing text, and preparing it for tokenization.
    """
    
    def __init__(self):
        """Initialize the text preprocessor."""
        self.min_word_count = 50
        self.max_word_count = 10000
        
    def clean_resume_text(self, text: str) -> str:
        """
        Clean and normalize resume text.
        
        Args:
            text: Raw resume text string
            
        Returns:
            Cleaned text string
            
        Raises:
            ValueError: If text is empty or too short
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
            
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s\.,;:!?()-]', ' ', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[.,;:!?()-]{2,}', ' ', text)
        
        # Remove single digits and small numbers (artifacts)
        text = re.sub(r'\b\d{1,2}\b(?!\d)', ' ', text)
        
        # Replace common patterns with standardized tokens
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        text = re.sub(r'http[s]?://\S+', '[URL]', text)
        
        # Final whitespace cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def validate_text_length(self, text: str) -> bool:
        """
        Validate if text meets minimum requirements.
        
        Args:
            text: Cleaned text string
            
        Returns:
            True if text meets requirements, False otherwise
        """
        if not text:
            return False
            
        word_count = len(text.split())
        return self.min_word_count <= word_count <= self.max_word_count
    
    def preprocess_single(self, text: str) -> Dict[str, Union[str, bool, int]]:
        """
        Preprocess a single resume text.
        
        Args:
            text: Raw resume text
            
        Returns:
            Dictionary containing processed text and metadata
        """
        try:
            # Clean the text
            cleaned_text = self.clean_resume_text(text)
            
            # Validate text length
            is_valid = self.validate_text_length(cleaned_text)
            
            # Calculate statistics
            word_count = len(cleaned_text.split())
            char_count = len(cleaned_text)
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'is_valid': is_valid,
                'word_count': word_count,
                'char_count': char_count,
                'processing_successful': True
            }
            
        except Exception as e:
            return {
                'original_text': text,
                'cleaned_text': '',
                'is_valid': False,
                'word_count': 0,
                'char_count': 0,
                'processing_successful': False,
                'error': str(e)
            }
    
    def preprocess_batch(self, texts: List[str]) -> List[Dict[str, Union[str, bool, int]]]:
        """
        Preprocess a batch of resume texts.
        
        Args:
            texts: List of raw resume texts
            
        Returns:
            List of preprocessing results
        """
        results = []
        for text in texts:
            result = self.preprocess_single(text)
            results.append(result)
        return results


class ResumeTokenizer:
    """
    Handles tokenization using DistilBERT tokenizer.
    
    This class provides methods to tokenize preprocessed text
    for model input preparation.
    """
    
    def __init__(self, tokenizer_path: str, max_length: int = 512):
        """
        Initialize the tokenizer.
        
        Args:
            tokenizer_path: Path to the DistilBERT tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        
    def tokenize_single(self, text: str, return_tensors: str = 'pt') -> Dict[str, torch.Tensor]:
        """
        Tokenize a single text.
        
        Args:
            text: Preprocessed text string
            return_tensors: Format for return tensors ('pt' for PyTorch)
            
        Returns:
            Dictionary containing tokenized inputs
        """
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors=return_tensors
            )
            
            # Add metadata
            tokens = self.tokenizer.tokenize(text)
            is_truncated = len(tokens) > self.max_length - 2  # Account for [CLS] and [SEP]
            
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'tokens': tokens,
                'is_truncated': is_truncated,
                'original_length': len(tokens),
                'tokenization_successful': True
            }
            
        except Exception as e:
            return {
                'input_ids': torch.tensor([]),
                'attention_mask': torch.tensor([]),
                'tokens': [],
                'is_truncated': False,
                'original_length': 0,
                'tokenization_successful': False,
                'error': str(e)
            }
    
    def tokenize_batch(self, texts: List[str], return_tensors: str = 'pt') -> Dict[str, Union[torch.Tensor, List]]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of preprocessed text strings
            return_tensors: Format for return tensors ('pt' for PyTorch)
            
        Returns:
            Dictionary containing batch tokenized inputs
        """
        try:
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors=return_tensors
            )
            
            # Calculate metadata for batch
            batch_metadata = []
            for text in texts:
                tokens = self.tokenizer.tokenize(text)
                is_truncated = len(tokens) > self.max_length - 2
                batch_metadata.append({
                    'tokens': tokens,
                    'is_truncated': is_truncated,
                    'original_length': len(tokens)
                })
            
            return {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'batch_metadata': batch_metadata,
                'tokenization_successful': True
            }
            
        except Exception as e:
            return {
                'input_ids': torch.tensor([]),
                'attention_mask': torch.tensor([]),
                'batch_metadata': [],
                'tokenization_successful': False,
                'error': str(e)
            }
    
    def decode_tokens(self, input_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode tokenized input back to text.
        
        Args:
            input_ids: Tokenized input tensor
            skip_special_tokens: Whether to skip special tokens in decoding
            
        Returns:
            Decoded text string
        """
        try:
            return self.tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            return f"Decoding error: {str(e)}"
    
    def analyze_tokenization_stats(self, texts: List[str]) -> Dict[str, Union[float, int]]:
        """
        Analyze tokenization statistics for a batch of texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Dictionary containing tokenization statistics
        """
        token_lengths = []
        truncated_count = 0
        
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            token_length = len(tokens)
            token_lengths.append(token_length)
            
            if token_length > self.max_length - 2:
                truncated_count += 1
        
        return {
            'total_texts': len(texts),
            'avg_token_length': sum(token_lengths) / len(token_lengths),
            'max_token_length': max(token_lengths),
            'min_token_length': min(token_lengths),
            'truncated_count': truncated_count,
            'truncation_rate': truncated_count / len(texts),
            'vocab_size': self.tokenizer.vocab_size,
            'max_length': self.max_length
        }


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline combining text cleaning and tokenization.
    
    This class orchestrates the entire preprocessing workflow from raw text
    to model-ready inputs.
    """
    
    def __init__(self, tokenizer_path: str, max_length: int = 512):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            tokenizer_path: Path to the DistilBERT tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.text_processor = TextPreprocessor()
        self.tokenizer = ResumeTokenizer(tokenizer_path, max_length)
        
    def process_single(self, text: str) -> Dict[str, Union[str, bool, int, torch.Tensor]]:
        """
        Process a single resume through the complete pipeline.
        
        Args:
            text: Raw resume text
            
        Returns:
            Dictionary containing all processing results
        """
        # Text preprocessing
        preprocess_result = self.text_processor.preprocess_single(text)
        
        if not preprocess_result['processing_successful'] or not preprocess_result['is_valid']:
            return {
                **preprocess_result,
                'tokenization_result': {
                    'input_ids': torch.tensor([]),
                    'attention_mask': torch.tensor([]),
                    'tokenization_successful': False
                }
            }
        
        # Tokenization
        tokenization_result = self.tokenizer.tokenize_single(preprocess_result['cleaned_text'])
        
        return {
            **preprocess_result,
            'tokenization_result': tokenization_result
        }
    
    def process_batch(self, texts: List[str]) -> Dict[str, Union[List, torch.Tensor]]:
        """
        Process a batch of resumes through the complete pipeline.
        
        Args:
            texts: List of raw resume texts
            
        Returns:
            Dictionary containing batch processing results
        """
        # Text preprocessing
        preprocess_results = self.text_processor.preprocess_batch(texts)
        
        # Filter valid texts
        valid_texts = []
        valid_indices = []
        
        for i, result in enumerate(preprocess_results):
            if result['processing_successful'] and result['is_valid']:
                valid_texts.append(result['cleaned_text'])
                valid_indices.append(i)
        
        # Tokenization for valid texts
        if valid_texts:
            tokenization_result = self.tokenizer.tokenize_batch(valid_texts)
        else:
            tokenization_result = {
                'input_ids': torch.tensor([]),
                'attention_mask': torch.tensor([]),
                'batch_metadata': [],
                'tokenization_successful': False
            }
        
        return {
            'preprocess_results': preprocess_results,
            'tokenization_result': tokenization_result,
            'valid_indices': valid_indices,
            'valid_count': len(valid_texts),
            'total_count': len(texts)
        }
    
    def get_preprocessing_stats(self, texts: List[str]) -> Dict[str, Union[float, int]]:
        """
        Get comprehensive preprocessing statistics.
        
        Args:
            texts: List of raw resume texts
            
        Returns:
            Dictionary containing preprocessing statistics
        """
        preprocess_results = self.text_processor.preprocess_batch(texts)
        
        valid_texts = [result['cleaned_text'] for result in preprocess_results 
                      if result['processing_successful'] and result['is_valid']]
        
        # Text statistics
        text_stats = {
            'total_texts': len(texts),
            'valid_texts': len(valid_texts),
            'invalid_texts': len(texts) - len(valid_texts),
            'validity_rate': len(valid_texts) / len(texts) if texts else 0
        }
        
        # Tokenization statistics
        if valid_texts:
            tokenization_stats = self.tokenizer.analyze_tokenization_stats(valid_texts)
        else:
            tokenization_stats = {
                'avg_token_length': 0,
                'max_token_length': 0,
                'min_token_length': 0,
                'truncated_count': 0,
                'truncation_rate': 0
            }
        
        return {
            **text_stats,
            **tokenization_stats
        }