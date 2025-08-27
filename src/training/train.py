import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import logging
import os
import joblib
from typing import Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.config import Config
from ..utils.helpers import Timer, save_model, calculate_metrics, print_metrics, create_directory
from ..data_processing.data_loader import DataLoader
from ..data_processing.feature_engineering import FeatureEngineer
from ..data_processing.data_validation import DataValidator


class ThreatDetectionTrainer:
    """XGBoost trainer for cybersecurity threat detection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.feature_engineer = FeatureEngineer(config)
        self.data_validator = DataValidator(config)
        self.training_history = {}
        
    def prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and prepare data for training"""
        
        self.logger.info("Loading and preparing data...")
        
        # Load data
        data_loader = DataLoader(self.config)
        data = data_loader.load_and_validate_data(data_path)
        
        # Validate data
        validation_result = self.data_validator.comprehensive_validation(data)
        if not validation_result.is_valid:
            self.logger.error("Data validation failed!")
            self.data_validator.print_validation_report(validation_result)
            raise ValueError("Data validation failed")
        else:
            self.logger.info("Data validation passed")
        
        # Feature engineering
        data = self.feature_engineer.process_features(data, fit=True)
        
        # Separate features and target
        X = data.drop(['is_attack', 'attack_type'], axis=1, errors='ignore')
        y = data['is_attack']
        
        # Validate model input
        input_validation = self.data_validator.validate_model_input(X, y)
        if not input_validation.is_valid:
            self.logger.error("Model input validation failed!")
            raise ValueError("Model input validation failed")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
            stratify=y
        )
        
        self.logger.info(f"Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        self.logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
        self.logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def create_xgb_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier with configuration"""
        
        model_params = {
            'objective': self.config.model.objective,
            'max_depth': self.config.model.max_depth,
            'learning_rate': self.config.model.learning_rate,
            'n_estimators': self.config.model.n_estimators,
            'subsample': self.config.model.subsample,
            'colsample_bytree': self.config.model.colsample_bytree,
            'random_state': self.config.model.random_state,
            'eval_metric': self.config.model.eval_metric,
            'early_stopping_rounds': self.config.model.early_stopping_rounds,
            'verbosity': 1,
            'n_jobs': -1,
            'use_label_encoder': False
        }
        
        model = xgb.XGBClassifier(**model_params)
        
        self.logger.info(f"Created XGBoost model with parameters: {model_params}")
        
        return model
    
    def train_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   y_train: pd.Series, y_test: pd.Series) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        
        self.logger.info("Starting model training...")
        
        with Timer("Model training"):
            # Create model
            model = self.create_xgb_model()
            
            # Prepare validation set for early stopping
            if self.config.training.validation_split > 0:
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train,
                    test_size=self.config.training.validation_split,
                    random_state=self.config.model.random_state,
                    stratify=y_train
                )
                
                eval_set = [(X_val, y_val)]
                
                # Train model
                model.fit(
                    X_train_split, y_train_split,
                    eval_set=eval_set,
                    verbose=False
                )
            else:
                # Train without validation set
                model.fit(X_train, y_train, verbose=False)
            
            self.model = model
            
            # Store training history
            if hasattr(model, 'evals_result_'):
                self.training_history = model.evals_result_
            
            self.logger.info("Model training completed")
            
            return model
    
    def evaluate_model(self, model: xgb.XGBClassifier, 
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate trained model"""
        
        self.logger.info("Evaluating model...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        # Additional XGBoost-specific metrics
        if hasattr(model, 'best_score'):
            metrics['best_validation_score'] = model.best_score
        
        if hasattr(model, 'best_iteration'):
            metrics['best_iteration'] = model.best_iteration
        
        # Feature importance
        feature_importance = dict(zip(X_test.columns, model.feature_importances_))
        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20])
        
        evaluation_results = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'top_20_features': top_features,
            'predictions': y_pred,
            'probabilities': y_prob,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Print metrics
        print_metrics(metrics)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.logger.info("Model evaluation completed")
        
        return evaluation_results
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform cross-validation"""
        
        self.logger.info("Performing cross-validation...")
        
        model = self.create_xgb_model()
        
        # Stratified K-Fold cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.training.cross_validation_folds,
            shuffle=True,
            random_state=self.config.model.random_state
        )
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            model, X, y, cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        cv_results = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'cv_folds': self.config.training.cross_validation_folds
        }
        
        self.logger.info(f"Cross-validation completed - Mean AUC: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']*2:.4f})")
        
        return cv_results
    
    def plot_training_curves(self, save_path: str = None) -> None:
        """Plot training curves if available"""
        
        if not self.training_history:
            self.logger.warning("No training history available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot validation curves
        for eval_set, metrics in self.training_history.items():
            for metric_name, values in metrics.items():
                ax.plot(values, label=f"{eval_set}_{metric_name}")
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Metric Value')
        ax.set_title('XGBoost Training Curves')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model: xgb.XGBClassifier, 
                               feature_names: list, top_n: int = 20,
                               save_path: str = None) -> None:
        """Plot feature importance"""
        
        # Get feature importance
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        
        # Sort and get top N features
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Create plot
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance (XGBoost)')
        plt.gca().invert_yaxis()
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            save_path: str = None) -> None:
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def save_model_and_artifacts(self, model: xgb.XGBClassifier, 
                               evaluation_results: Dict[str, Any],
                               output_dir: str) -> None:
        """Save model and training artifacts"""
        
        create_directory(output_dir)
        
        # Save model
        model_path = os.path.join(output_dir, 'threat_detection_model.pkl')
        save_model(model, model_path)
        
        # Save feature engineer
        feature_engineer_path = os.path.join(output_dir, 'feature_engineer.pkl')
        save_model(self.feature_engineer, feature_engineer_path)
        
        # Save evaluation results
        import json
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = evaluation_results.copy()
        json_results['predictions'] = json_results['predictions'].tolist()
        json_results['probabilities'] = json_results['probabilities'].tolist()
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save training configuration
        config_path = os.path.join(output_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        self.logger.info(f"Model and artifacts saved to {output_dir}")
    
    def train_complete_pipeline(self, data_path: str, output_dir: str = 'models') -> Dict[str, Any]:
        """Complete training pipeline"""
        
        self.logger.info("Starting complete training pipeline...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(data_path)
        
        # Cross-validation
        cv_results = self.cross_validate_model(
            pd.concat([X_train, X_test]), 
            pd.concat([y_train, y_test])
        )
        
        # Train model
        model = self.train_model(X_train, X_test, y_train, y_test)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(model, X_test, y_test)
        evaluation_results['cross_validation'] = cv_results
        
        # Create output directory
        create_directory(output_dir)
        
        # Generate plots
        plots_dir = os.path.join(output_dir, 'plots')
        create_directory(plots_dir)
        
        # Training curves
        training_curves_path = os.path.join(plots_dir, 'training_curves.png')
        self.plot_training_curves(training_curves_path)
        
        # Feature importance
        feature_importance_path = os.path.join(plots_dir, 'feature_importance.png')
        self.plot_feature_importance(model, X_test.columns.tolist(), save_path=feature_importance_path)
        
        # Confusion matrix
        confusion_matrix_path = os.path.join(plots_dir, 'confusion_matrix.png')
        self.plot_confusion_matrix(y_test, evaluation_results['predictions'], confusion_matrix_path)
        
        # Save model and artifacts
        self.save_model_and_artifacts(model, evaluation_results, output_dir)
        
        self.logger.info("Complete training pipeline finished successfully!")
        
        return evaluation_results


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train XGBoost threat detection model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory for model and artifacts')
    parser.add_argument('--config_path', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = Config(args.config_path)
    
    # Create trainer
    trainer = ThreatDetectionTrainer(config)
    
    # Run training pipeline
    results = trainer.train_complete_pipeline(args.data_path, args.output_dir)
    
    print("\nTraining completed successfully!")
    print(f"Final model AUC: {results['metrics']['auc_score']:.4f}")
    print(f"Model and artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()