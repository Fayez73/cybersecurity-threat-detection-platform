import os
import json
import pickle
import logging
import boto3
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import joblib
from botocore.exceptions import ClientError


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_model(model: Any, filepath: str) -> None:
    """Save model to file using joblib"""
    try:
        joblib.dump(model, filepath)
        logging.info(f"Model saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        raise


def load_model(filepath: str) -> Any:
    """Load model from file using joblib"""
    try:
        model = joblib.load(filepath)
        logging.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise


def save_json(data: Dict, filepath: str) -> None:
    """Save dictionary to JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logging.info(f"Data saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving JSON: {str(e)}")
        raise


def load_json(filepath: str) -> Dict:
    """Load dictionary from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logging.info(f"Data loaded from {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON: {str(e)}")
        raise


def create_directory(directory: str) -> None:
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")


class S3Helper:
    """Helper class for S3 operations"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region)
        self.logger = logging.getLogger(__name__)
    
    def upload_file(self, local_filepath: str, s3_key: str) -> bool:
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(local_filepath, self.bucket_name, s3_key)
            self.logger.info(f"File {local_filepath} uploaded to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            self.logger.error(f"Error uploading file: {str(e)}")
            return False
    
    def download_file(self, s3_key: str, local_filepath: str) -> bool:
        """Download file from S3"""
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_filepath)
            self.logger.info(f"File s3://{self.bucket_name}/{s3_key} downloaded to {local_filepath}")
            return True
        except ClientError as e:
            self.logger.error(f"Error downloading file: {str(e)}")
            return False
    
    def list_objects(self, prefix: str = '') -> List[str]:
        """List objects in S3 bucket with given prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            objects = [obj['Key'] for obj in response.get('Contents', [])]
            return objects
        except ClientError as e:
            self.logger.error(f"Error listing objects: {str(e)}")
            return []
    
    def delete_object(self, s3_key: str) -> bool:
        """Delete object from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            self.logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            self.logger.error(f"Error deleting object: {str(e)}")
            return False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """Calculate common classification metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_prob is not None:
        metrics['auc_score'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print formatted metrics"""
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    for metric, value in metrics.items():
        print(f"{metric.upper():.<30} {value:.4f}")
    
    print("="*50 + "\n")


def feature_importance_analysis(model, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """Analyze and return top feature importances"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    else:
        logging.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()


def memory_usage_check() -> Dict[str, float]:
    """Check current memory usage"""
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'memory_mb': memory_info.rss / 1024 / 1024,
        'memory_percent': process.memory_percent()
    }


def validate_input_data(data: pd.DataFrame, required_columns: List[str] = None) -> Tuple[bool, List[str]]:
    """Validate input data format and completeness"""
    errors = []
    
    if data.empty:
        errors.append("Dataset is empty")
        return False, errors
    
    if required_columns:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {list(missing_columns)}")
    
    # Check for high percentage of missing values
    missing_percentage = data.isnull().sum() / len(data) * 100
    high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
    if high_missing_cols:
        errors.append(f"Columns with >50% missing values: {high_missing_cols}")
    
    # Check data types
    object_cols = data.select_dtypes(include=['object']).columns
    if len(object_cols) > 20:
        errors.append(f"High number of categorical columns ({len(object_cols)})")
    
    return len(errors) == 0, errors


def create_timestamp() -> str:
    """Create timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def chunks(lst: List, chunk_size: int):
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers, return default if division by zero"""
    try:
        return a / b if b != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Timer:
    """Context manager for timing code execution"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logging.info(f"Starting {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logging.info(f"Completed {self.description} in {duration:.2f} seconds")


# Global logger instance
logger = setup_logging()