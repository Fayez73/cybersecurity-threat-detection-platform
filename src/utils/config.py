import os
from typing import Dict, Any
from dataclasses import dataclass
import yaml


@dataclass
class ModelConfig:
    """XGBoost model configuration"""
    objective: str = 'binary:logistic'
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    eval_metric: str = 'auc'
    early_stopping_rounds: int = 10


@dataclass
class DataConfig:
    """Data processing configuration"""
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    normalize_features: bool = True
    remove_outliers: bool = True
    outlier_threshold: float = 3.0
    categorical_encoding: str = 'onehot'  # 'onehot', 'label', 'target'


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 1000
    max_epochs: int = 1000
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    hyperparameter_tuning: bool = True
    tuning_trials: int = 50


@dataclass
class InferenceConfig:
    """Inference configuration"""
    batch_size: int = 1000
    output_format: str = 'json'  # 'json', 'csv'
    include_probabilities: bool = True
    threshold: float = 0.5


@dataclass
class AWSConfig:
    """AWS-specific configuration"""
    region: str = os.getenv('AWS_REGION', 'us-east-1')
    s3_bucket: str = os.getenv('S3_BUCKET', 'cybersecurity-threat-detection')
    sagemaker_role: str = os.getenv('SAGEMAKER_ROLE')
    instance_type: str = 'ml.m5.large'
    training_instance_type: str = 'ml.m5.xlarge'
    endpoint_name: str = 'threat-detection-endpoint'


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = None):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.aws = AWSConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations from file
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(self.training, key):
                    setattr(self.training, key, value)
        
        if 'inference' in config_dict:
            for key, value in config_dict['inference'].items():
                if hasattr(self.inference, key):
                    setattr(self.inference, key, value)
        
        if 'aws' in config_dict:
            for key, value in config_dict['aws'].items():
                if hasattr(self.aws, key):
                    setattr(self.aws, key, value)
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'aws': self.aws.__dict__
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'aws': self.aws.__dict__
        }


# Global configuration instance
config = Config()

# Environment-specific configurations
ENVIRONMENTS = {
    'dev': {
        'aws': {
            's3_bucket': 'cybersecurity-threat-detection-dev',
            'instance_type': 'ml.t3.medium',
            'training_instance_type': 'ml.m5.large'
        }
    },
    'prod': {
        'aws': {
            's3_bucket': 'cybersecurity-threat-detection-prod',
            'instance_type': 'ml.m5.large',
            'training_instance_type': 'ml.m5.2xlarge'
        }
    }
}


def get_config(environment: str = None) -> Config:
    """Get configuration for specific environment"""
    cfg = Config()
    
    if environment and environment in ENVIRONMENTS:
        env_config = ENVIRONMENTS[environment]
        
        # Update AWS config for environment
        if 'aws' in env_config:
            for key, value in env_config['aws'].items():
                if hasattr(cfg.aws, key):
                    setattr(cfg.aws, key, value)
    
    return cfg