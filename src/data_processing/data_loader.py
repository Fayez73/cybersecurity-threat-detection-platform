import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
import boto3
from io import StringIO
import os

from ..utils.helpers import S3Helper, Timer
from ..utils.config import Config


class DataLoader:
    """Data loader for cybersecurity threat detection datasets"""
    
    def __init__(self, config: Config):
        self.config = config
        self.s3_helper = S3Helper(config.aws.s3_bucket, config.aws.region)
        self.logger = logging.getLogger(__name__)
        
    def load_kdd_cup_data(self, filepath: str) -> pd.DataFrame:
        """Load KDD Cup 99 dataset with proper column names"""
        
        # KDD Cup 99 column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'attack_type'
        ]
        
        try:
            with Timer("Loading KDD Cup data"):
                data = pd.read_csv(filepath, names=columns, header=None)
                
                # Create binary target (normal vs attack)
                data['is_attack'] = (data['attack_type'] != 'normal').astype(int)
                
                self.logger.info(f"Loaded {len(data)} records from KDD Cup dataset")
                self.logger.info(f"Attack distribution: {data['is_attack'].value_counts().to_dict()}")
                
                return data
                
        except Exception as e:
            self.logger.error(f"Error loading KDD Cup data: {str(e)}")
            raise
    
    def load_nsl_kdd_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load NSL-KDD dataset (improved version of KDD Cup 99)"""
        
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
        ]
        
        try:
            with Timer("Loading NSL-KDD data"):
                # Load training data
                train_data = pd.read_csv(train_path, names=columns, header=None)
                train_data = train_data.drop('difficulty_level', axis=1)
                train_data['is_attack'] = (train_data['attack_type'] != 'normal').astype(int)
                
                # Load test data
                test_data = pd.read_csv(test_path, names=columns, header=None)
                test_data = test_data.drop('difficulty_level', axis=1)
                test_data['is_attack'] = (test_data['attack_type'] != 'normal').astype(int)
                
                self.logger.info(f"Loaded {len(train_data)} training records")
                self.logger.info(f"Loaded {len(test_data)} test records")
                
                return train_data, test_data
                
        except Exception as e:
            self.logger.error(f"Error loading NSL-KDD data: {str(e)}")
            raise
    
    def load_csv_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Generic CSV data loader with error handling"""
        try:
            with Timer(f"Loading CSV data from {filepath}"):
                data = pd.read_csv(filepath, **kwargs)
                
                self.logger.info(f"Loaded {len(data)} records with {len(data.columns)} columns")
                self.logger.info(f"Columns: {list(data.columns)}")
                
                return data
                
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {str(e)}")
            raise
    
    def load_from_s3(self, s3_key: str, **kwargs) -> pd.DataFrame:
        """Load CSV data directly from S3"""
        try:
            with Timer(f"Loading data from S3: {s3_key}"):
                # Get object from S3
                s3_client = boto3.client('s3', region_name=self.config.aws.region)
                obj = s3_client.get_object(Bucket=self.config.aws.s3_bucket, Key=s3_key)
                
                # Read CSV from S3 object
                data = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')), **kwargs)
                
                self.logger.info(f"Loaded {len(data)} records from S3")
                
                return data
                
        except Exception as e:
            self.logger.error(f"Error loading data from S3: {str(e)}")
            raise
    
    def create_sample_threat_data(self, n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
        """Create synthetic cybersecurity threat data for testing"""
        np.random.seed(seed)
        
        # Network connection features
        data = {
            'duration': np.random.exponential(10, n_samples),
            'src_bytes': np.random.lognormal(8, 2, n_samples).astype(int),
            'dst_bytes': np.random.lognormal(6, 3, n_samples).astype(int),
            'count': np.random.poisson(5, n_samples),
            'srv_count': np.random.poisson(3, n_samples),
            'serror_rate': np.random.beta(1, 10, n_samples),
            'srv_serror_rate': np.random.beta(1, 10, n_samples),
            'rerror_rate': np.random.beta(1, 15, n_samples),
            'srv_rerror_rate': np.random.beta(1, 15, n_samples),
            'same_srv_rate': np.random.beta(5, 2, n_samples),
            'diff_srv_rate': np.random.beta(2, 5, n_samples),
            'srv_diff_host_rate': np.random.beta(1, 10, n_samples),
            
            # Host-based features
            'dst_host_count': np.random.poisson(10, n_samples),
            'dst_host_srv_count': np.random.poisson(5, n_samples),
            'dst_host_same_srv_rate': np.random.beta(5, 2, n_samples),
            'dst_host_diff_srv_rate': np.random.beta(2, 5, n_samples),
            'dst_host_same_src_port_rate': np.random.beta(3, 3, n_samples),
            'dst_host_srv_diff_host_rate': np.random.beta(1, 10, n_samples),
            'dst_host_serror_rate': np.random.beta(1, 15, n_samples),
            'dst_host_srv_serror_rate': np.random.beta(1, 15, n_samples),
            'dst_host_rerror_rate': np.random.beta(1, 20, n_samples),
            'dst_host_srv_rerror_rate': np.random.beta(1, 20, n_samples),
            
            # Categorical features
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples, p=[0.7, 0.2, 0.1]),
            'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'telnet', 'other'], 
                                     n_samples, p=[0.4, 0.1, 0.1, 0.1, 0.05, 0.25]),
            'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR', 'SH', 'other'], 
                                   n_samples, p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05]),
            
            # Binary features
            'land': np.random.binomial(1, 0.01, n_samples),
            'wrong_fragment': np.random.binomial(1, 0.05, n_samples),
            'urgent': np.random.binomial(1, 0.01, n_samples),
            'hot': np.random.poisson(0.1, n_samples),
            'num_failed_logins': np.random.poisson(0.2, n_samples),
            'logged_in': np.random.binomial(1, 0.7, n_samples),
            'num_compromised': np.random.poisson(0.05, n_samples),
            'root_shell': np.random.binomial(1, 0.05, n_samples),
            'su_attempted': np.random.binomial(1, 0.02, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on suspicious patterns
        threat_score = (
            (df['serror_rate'] > 0.5).astype(int) * 0.3 +
            (df['rerror_rate'] > 0.3).astype(int) * 0.2 +
            (df['num_failed_logins'] > 2).astype(int) * 0.4 +
            (df['hot'] > 3).astype(int) * 0.3 +
            (df['num_compromised'] > 0).astype(int) * 0.5 +
            (df['root_shell'] == 1).astype(int) * 0.4 +
            (df['duration'] > 100).astype(int) * 0.1 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        # Binary classification: threat vs normal
        df['is_attack'] = (threat_score > 0.4).astype(int)
        
        # Add attack types for detailed analysis
        attack_types = np.where(
            df['is_attack'] == 1,
            np.random.choice(['dos', 'probe', 'r2l', 'u2r'], df['is_attack'].sum(),
                           p=[0.4, 0.3, 0.2, 0.1]),
            'normal'
        )
        df['attack_type'] = attack_types
        
        self.logger.info(f"Created synthetic dataset with {n_samples} samples")
        self.logger.info(f"Attack distribution: {df['is_attack'].value_counts().to_dict()}")
        
        return df
    
    def load_and_validate_data(self, filepath: str, dataset_type: str = 'auto') -> pd.DataFrame:
        """Load and validate data with automatic type detection"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Auto-detect dataset type
        if dataset_type == 'auto':
            if 'kdd' in filepath.lower():
                dataset_type = 'kdd'
            elif 'nsl' in filepath.lower():
                dataset_type = 'nsl'
            else:
                dataset_type = 'generic'
        
        # Load data based on type
        if dataset_type == 'kdd':
            data = self.load_kdd_cup_data(filepath)
        elif dataset_type == 'nsl':
            # For NSL-KDD, assume single file (train or test)
            data = self.load_kdd_cup_data(filepath)
        else:
            data = self.load_csv_data(filepath)
        
        # Basic validation
        if data.empty:
            raise ValueError("Loaded dataset is empty")
        
        if 'is_attack' not in data.columns:
            self.logger.warning("No 'is_attack' column found. Creating binary target if 'attack_type' exists.")
            if 'attack_type' in data.columns:
                data['is_attack'] = (data['attack_type'] != 'normal').astype(int)
            else:
                raise ValueError("No target column found ('is_attack' or 'attack_type')")
        
        self.logger.info(f"Data validation passed. Shape: {data.shape}")
        
        return data
    
    def save_processed_data(self, data: pd.DataFrame, filepath: str, 
                          upload_to_s3: bool = True) -> None:
        """Save processed data locally and optionally to S3"""
        try:
            # Save locally
            data.to_csv(filepath, index=False)
            self.logger.info(f"Data saved to {filepath}")
            
            # Upload to S3 if requested
            if upload_to_s3:
                s3_key = f"processed/{os.path.basename(filepath)}"
                if self.s3_helper.upload_file(filepath, s3_key):
                    self.logger.info(f"Data uploaded to S3: {s3_key}")
                
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict:
        """Get comprehensive data summary"""
        summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
        }
        
        # Target distribution if available
        if 'is_attack' in data.columns:
            summary['target_distribution'] = data['is_attack'].value_counts().to_dict()
        
        if 'attack_type' in data.columns:
            summary['attack_type_distribution'] = data['attack_type'].value_counts().to_dict()
        
        # Memory usage
        summary['memory_usage_mb'] = data.memory_usage(deep=True).sum() / 1024 / 1024
        
        return summary