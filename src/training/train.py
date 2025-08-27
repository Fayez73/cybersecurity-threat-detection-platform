"""
Training script for cybersecurity threat detection model.
"""

import os
import sys
import argparse
import json
import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import joblib

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.preprocessing import DataPreprocessor
from src.models.xgboost_model import CyberThreatXGBoostModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Main training class for the cybersecurity threat detection model."""
    
    def __init__(self, config):
        """Initialize trainer with configuration."""
        self.config = config
        self.s3_client = boto3.client('s3')
        self.preprocessor = DataPreprocessor()
        self.model = None
        
    def download_data_from_s3(self):
        """Download training data from S3."""
        try:
            data_bucket = self.config.get('data_bucket')
            data_key = self.config.get('data_key', 'data/training/network_logs.csv')
            local_path = '/opt/ml/input/data/training/network_logs.csv'
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            logger.info(f"Downloading data from s3://{data_bucket}/{data_key}")
            self.s3_client.download_file(data_bucket, data_key, local_path)
            logger.info("Data downloaded successfully")
            
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading data from S3: {str(e)}")
            raise
    
    def prepare_data(self, data_path):
        """Prepare training and validation data."""
        logger.info("Starting data preparation")
        
        # Load and preprocess data
        processed_data = self.preprocessor.preprocess_pipeline(
            data_path, 
            target_column=self.config.get('target_column', 'label')
        )
        
        # Create validation split from training data
        X_train, X_val, y_train, y_val = train_test_split(
            processed_data['X_train'],
            processed_data['y_train'],
            test_size=0.2,
            random_state=42,
            stratify=processed_data['y_train']
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': processed_data['X_test'],
            'y_train': y_train,
            'y_val': y_val,
            'y_test': processed_data['y_test'],
            'feature_names': processed_data['feature_names']
        }
    
    def train_model(self, data):
        """Train the XGBoost model."""
        logger.info("Starting model training")
        
        # Model parameters from config
        model_params = self.config.get('model_params', {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        })
        
        # Initialize model
        self.model = CyberThreatXGBoostModel(**model_params)
        
        # Regular training
        self.model.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            early_stopping_rounds=10
        )
        
        logger.info("Model training completed")
    
    def evaluate_model(self, data):
        """Evaluate model performance."""
        logger.info("Starting model evaluation")
        
        # Evaluate on test set
        test_results = self.model.evaluate(data['X_test'], data['y_test'])
        
        # Evaluate on validation set
        val_results = self.model.evaluate(data['X_val'], data['y_val'])
        
        evaluation_metrics = {
            'test_metrics': test_results,
            'validation_metrics': val_results
        }
        
        # Log key metrics
        logger.info(f"Test AUC: {test_results['auc_score']:.4f}")
        logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Test Precision: {test_results['precision']:.4f}")
        logger.info(f"Test Recall: {test_results['recall']:.4f}")
        
        return evaluation_metrics
    
    def save_model_artifacts(self, data, evaluation_metrics):
        """Save model and related artifacts."""
        logger.info("Saving model artifacts")
        
        model_dir = self.config.get('model_dir', '/opt/ml/model')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'model.pkl')
        self.model.save_model(model_path)
        
        # Save preprocessor
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        joblib.dump(self.preprocessor, preprocessor_path)
        
        # Save feature names
        feature_names_path = os.path.join(model_dir, 'feature_names.json')
        with open(feature_names_path, 'w') as f:
            json.dump(data['feature_names'], f)
        
        # Save evaluation metrics
        metrics_path = os.path.join(model_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj
            
            # Convert nested dictionary
            converted_metrics = {}
            for key, value in evaluation_metrics.items():
                if isinstance(value, dict):
                    converted_metrics[key] = {k: convert_numpy(v) for k, v in value.items()}
                else:
                    converted_metrics[key] = convert_numpy(value)
            
            json.dump(converted_metrics, f, indent=2)
        
        # Save feature importance
        feature_importance = self.model.get_feature_importance(data['feature_names'])
        importance_path = os.path.join(model_dir, 'feature_importance.csv')
        feature_importance.to_csv(importance_path, index=False)
        
        logger.info("Model artifacts saved successfully")
    
    def upload_model_to_s3(self):
        """Upload trained model to S3."""
        try:
            model_bucket = self.config.get('model_bucket')
            model_prefix = self.config.get('model_prefix', 'models/cyberthreat-xgboost')
            model_dir = self.config.get('model_dir', '/opt/ml/model')
            
            # Create tar.gz file for SageMaker
            import tarfile
            
            tar_path = '/tmp/model.tar.gz'
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(model_dir, arcname='.')
            
            # Upload to S3
            s3_key = f"{model_prefix}/model.tar.gz"
            logger.info(f"Uploading model to s3://{model_bucket}/{s3_key}")
            
            self.s3_client.upload_file(tar_path, model_bucket, s3_key)
            logger.info("Model uploaded to S3 successfully")
            
            return f"s3://{model_bucket}/{s3_key}"
            
        except Exception as e:
            logger.error(f"Error uploading model to S3: {str(e)}")
            raise
    
    def run_training(self):
        """Execute the complete training pipeline."""
        try:
            # Download data
            data_path = self.download_data_from_s3()
            
            # Prepare data
            data = self.prepare_data(data_path)
            
            # Train model
            self.train_model(data)
            
            # Evaluate model
            evaluation_metrics = self.evaluate_model(data)
            
            # Save artifacts
            self.save_model_artifacts(data, evaluation_metrics)
            
            # Upload to S3
            model_s3_path = self.upload_model_to_s3()
            
            logger.info(f"Training completed successfully. Model saved to: {model_s3_path}")
            
            return {
                'model_path': model_s3_path,
                'metrics': evaluation_metrics
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train cybersecurity threat detection model')
    
    parser.add_argument('--data-bucket', type=str, required=True,
                        help='S3 bucket containing training data')
    parser.add_argument('--data-key', type=str, default='data/training/network_logs.csv',
                        help='S3 key for training data')
    parser.add_argument('--model-bucket', type=str, required=True,
                        help='S3 bucket for saving trained model')
    parser.add_argument('--model-prefix', type=str, default='models/cyberthreat-xgboost',
                        help='S3 prefix for model artifacts')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model',
                        help='Local directory for saving model')
    parser.add_argument('--target-column', type=str, default='label',
                        help='Target column name in the dataset')
    
    # Model parameters
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Configuration
    config = {
        'data_bucket': args.data_bucket,
        'data_key': args.data_key,
        'model_bucket': args.model_bucket,
        'model_prefix': args.model_prefix,
        'model_dir': args.model_dir,
        'target_column': args.target_column,
        'model_params': {
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'n_estimators': args.n_estimators,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'random_state': 42
        }
    }
    
    # Create trainer and run training
    trainer = ModelTrainer(config)
    result = trainer.run_training()
    
    print(f"Training completed successfully!")
    print(f"Model path: {result['model_path']}")
    print(f"Test AUC: {result['metrics']['test_metrics']['auc_score']:.4f}")


if __name__ == "__main__":
    main()