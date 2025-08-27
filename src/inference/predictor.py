import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Dict, List, Union, Optional, Any, Tuple
import joblib
from datetime import datetime

from ..utils.config import Config
from ..utils.helpers import Timer, load_model, create_directory, chunks
from ..data_processing.data_validation import DataValidator


class ThreatPredictor:
    """Real-time threat prediction using trained XGBoost model"""
    
    def __init__(self, model_path: str = None, feature_engineer_path: str = None, 
                 config: Config = None):
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.feature_engineer = None
        self.data_validator = DataValidator(self.config)
        self.model_metadata = {}
        
        if model_path and feature_engineer_path:
            self.load_model_artifacts(model_path, feature_engineer_path)
    
    def load_model_artifacts(self, model_path: str, feature_engineer_path: str) -> None:
        """Load trained model and feature engineering pipeline"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.path.exists(feature_engineer_path):
            raise FileNotFoundError(f"Feature engineer file not found: {feature_engineer_path}")
        
        # Load model and feature engineer
        self.model = load_model(model_path)
        self.feature_engineer = load_model(feature_engineer_path)
        
        # Load model metadata if available
        model_dir = os.path.dirname(model_path)
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
        
        self.logger.info("Model artifacts loaded successfully")
    
    def predict_single(self, network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict threat for a single network connection"""
        
        if self.model is None or self.feature_engineer is None:
            raise RuntimeError("Model artifacts not loaded. Call load_model_artifacts() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([network_data])
        
        # Validate input data
        validation_result = self.data_validator.validate_model_input(df)
        if not validation_result.is_valid:
            return {
                'prediction': None,
                'probability': None,
                'confidence': None,
                'error': f"Input validation failed: {validation_result.errors}",
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            with Timer("Single prediction"):
                # Apply feature engineering
                processed_df = self.feature_engineer.process_features(df, fit=False)
                
                # Remove target columns if they exist
                feature_cols = [col for col in processed_df.columns 
                              if col not in ['is_attack', 'attack_type']]
                X = processed_df[feature_cols]
                
                # Make prediction
                prediction = self.model.predict(X)[0]
                probability = self.model.predict_proba(X)[0, 1]
                
                # Calculate confidence
                confidence = self._calculate_confidence(probability)
                
                # Determine threat level
                threat_level = self._determine_threat_level(probability)
                
                result = {
                    'prediction': int(prediction),
                    'prediction_label': 'Attack' if prediction == 1 else 'Normal',
                    'probability': float(probability),
                    'confidence': confidence,
                    'threat_level': threat_level,
                    'timestamp': datetime.now().isoformat(),
                    'model_version': self.model_metadata.get('version', 'unknown')
                }
                
                return result
        
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            return {
                'prediction': None,
                'probability': None,
                'confidence': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict threats for batch of network connections"""
        
        if self.model is None or self.feature_engineer is None:
            raise RuntimeError("Model artifacts not loaded. Call load_model_artifacts() first.")
        
        self.logger.info(f"Starting batch prediction for {len(data)} samples")
        
        # Validate input data
        validation_result = self.data_validator.validate_model_input(data)
        if not validation_result.is_valid:
            raise ValueError(f"Input validation failed: {validation_result.errors}")
        
        try:
            with Timer("Batch prediction"):
                # Apply feature engineering
                processed_data = self.feature_engineer.process_features(data.copy(), fit=False)
                
                # Remove target columns if they exist
                feature_cols = [col for col in processed_data.columns 
                              if col not in ['is_attack', 'attack_type']]
                X = processed_data[feature_cols]
                
                # Make predictions
                predictions = self.model.predict(X)
                probabilities = self.model.predict_proba(X)[:, 1]
                
                # Calculate additional metrics
                confidences = [self._calculate_confidence(p) for p in probabilities]
                threat_levels = [self._determine_threat_level(p) for p in probabilities]
                
                # Create results DataFrame
                results = data.copy()
                results['prediction'] = predictions
                results['prediction_label'] = ['Attack' if p == 1 else 'Normal' for p in predictions]
                results['probability'] = probabilities
                results['confidence'] = confidences
                results['threat_level'] = threat_levels
                results['timestamp'] = datetime.now().isoformat()
                
                self.logger.info("Batch prediction completed successfully")
                
                return results
        
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def predict_streaming(self, data_stream, batch_size: int = 100):
        """Predict threats for streaming data"""
        
        batch = []
        batch_count = 0
        
        for record in data_stream:
            batch.append(record)
            
            if len(batch) >= batch_size:
                # Process batch
                df_batch = pd.DataFrame(batch)
                results = self.predict_batch(df_batch)
                
                batch_count += 1
                self.logger.info(f"Processed batch {batch_count} with {len(batch)} records")
                
                # Yield results
                for _, row in results.iterrows():
                    yield row.to_dict()
                
                # Clear batch
                batch = []
        
        # Process remaining records
        if batch:
            df_batch = pd.DataFrame(batch)
            results = self.predict_batch(df_batch)
            
            for _, row in results.iterrows():
                yield row.to_dict()
    
    def predict_with_explanation(self, network_data: Dict[str, Any], 
                               top_features: int = 10) -> Dict[str, Any]:
        """Predict with feature importance explanation"""
        
        # Get basic prediction
        result = self.predict_single(network_data)
        
        if result.get('error'):
            return result
        
        try:
            # Convert to DataFrame for feature processing
            df = pd.DataFrame([network_data])
            processed_df = self.feature_engineer.process_features(df, fit=False)
            
            feature_cols = [col for col in processed_df.columns 
                          if col not in ['is_attack', 'attack_type']]
            X = processed_df[feature_cols]
            
            # Get feature importance from model
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
                
                # Sort by importance and get top features
                top_important_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:top_features]
                
                # Get feature values for explanation
                feature_values = X.iloc[0].to_dict()
                
                explanation = {
                    'top_features': [
                        {
                            'feature': feature,
                            'importance': importance,
                            'value': feature_values.get(feature, 'N/A')
                        }
                        for feature, importance in top_important_features
                    ]
                }
                
                result['explanation'] = explanation
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error generating explanation: {str(e)}")
            result['explanation_error'] = str(e)
            return result
    
    def _calculate_confidence(self, probability: float) -> str:
        """Calculate confidence level based on probability"""
        
        # Distance from decision boundary (0.5)
        distance = abs(probability - 0.5)
        
        if distance >= 0.4:
            return "High"
        elif distance >= 0.2:
            return "Medium"
        else:
            return "Low"
    
    def _determine_threat_level(self, probability: float) -> str:
        """Determine threat level based on probability"""
        
        if probability >= 0.8:
            return "Critical"
        elif probability >= 0.6:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        elif probability >= 0.2:
            return "Low"
        else:
            return "Minimal"
    
    def get_prediction_stats(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for a batch of predictions"""
        
        if not predictions:
            return {}
        
        # Filter out predictions with errors
        valid_predictions = [p for p in predictions if p.get('prediction') is not None]
        
        if not valid_predictions:
            return {'error': 'No valid predictions found'}
        
        probabilities = [p['probability'] for p in valid_predictions]
        predictions_binary = [p['prediction'] for p in valid_predictions]
        confidences = [p['confidence'] for p in valid_predictions]
        threat_levels = [p['threat_level'] for p in valid_predictions]
        
        stats = {
            'total_predictions': len(predictions),
            'valid_predictions': len(valid_predictions),
            'attack_predictions': sum(predictions_binary),
            'normal_predictions': len(predictions_binary) - sum(predictions_binary),
            'attack_rate': sum(predictions_binary) / len(predictions_binary) * 100,
            'probability_stats': {
                'mean': np.mean(probabilities),
                'median': np.median(probabilities),
                'std': np.std(probabilities),
                'min': np.min(probabilities),
                'max': np.max(probabilities)
            },
            'confidence_distribution': {
                'High': confidences.count('High'),
                'Medium': confidences.count('Medium'),
                'Low': confidences.count('Low')
            },
            'threat_level_distribution': {
                level: threat_levels.count(level) 
                for level in ['Critical', 'High', 'Medium', 'Low', 'Minimal']
            }
        }
        
        return stats
    
    def save_predictions(self, predictions: Union[pd.DataFrame, List[Dict]], 
                        output_path: str, format: str = 'csv') -> None:
        """Save predictions to file"""
        
        create_directory(os.path.dirname(output_path))
        
        if isinstance(predictions, list):
            df = pd.DataFrame(predictions)
        else:
            df = predictions
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format.lower() == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Predictions saved to {output_path}")
    
    def create_alert(self, prediction: Dict[str, Any], 
                    alert_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """Create security alert for high-risk predictions"""
        
        if prediction.get('probability', 0) >= alert_threshold:
            alert = {
                'alert_id': f"THREAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'severity': prediction['threat_level'],
                'probability': prediction['probability'],
                'confidence': prediction['confidence'],
                'timestamp': prediction['timestamp'],
                'description': f"High-risk threat detected with {prediction['probability']:.1%} probability",
                'recommended_action': self._get_recommended_action(prediction['threat_level']),
                'model_version': prediction.get('model_version', 'unknown')
            }
            
            return alert
        
        return None
    
    def _get_recommended_action(self, threat_level: str) -> str:
        """Get recommended action based on threat level"""
        
        actions = {
            'Critical': 'Immediate isolation and investigation required',
            'High': 'Priority investigation and monitoring',
            'Medium': 'Enhanced monitoring and logging',
            'Low': 'Standard monitoring',
            'Minimal': 'Continue normal operations'
        }
        
        return actions.get(threat_level, 'Unknown threat level')


class BatchThreatPredictor:
    """Optimized batch processing for large datasets"""
    
    def __init__(self, predictor: ThreatPredictor):
        self.predictor = predictor
        self.logger = logging.getLogger(__name__)
    
    def predict_large_dataset(self, data_path: str, output_path: str, 
                            batch_size: int = 10000) -> Dict[str, Any]:
        """Process large datasets in chunks"""
        
        self.logger.info(f"Starting large dataset prediction from {data_path}")
        
        # Read data in chunks
        chunk_reader = pd.read_csv(data_path, chunksize=batch_size)
        
        total_processed = 0
        all_results = []
        
        with Timer("Large dataset prediction"):
            for chunk_idx, chunk in enumerate(chunk_reader):
                try:
                    # Process chunk
                    results = self.predictor.predict_batch(chunk)
                    all_results.append(results)
                    
                    total_processed += len(chunk)
                    self.logger.info(f"Processed chunk {chunk_idx + 1}: {len(chunk)} records "
                                   f"(Total: {total_processed})")
                
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
                    continue
        
        # Combine all results
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            
            # Save results
            self.predictor.save_predictions(final_results, output_path, 
                                          format=self.predictor.config.inference.output_format)
            
            # Calculate statistics
            predictions_list = final_results.to_dict('records')
            stats = self.predictor.get_prediction_stats(predictions_list)
            
            return {
                'total_processed': total_processed,
                'output_path': output_path,
                'statistics': stats
            }
        else:
            raise RuntimeError("No data was successfully processed")


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Threat detection inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--feature_engineer_path', type=str, required=True, help='Path to feature engineer')
    parser.add_argument('--input_data', type=str, required=True, help='Path to input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path for output predictions')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'large'], 
                       default='batch', help='Prediction mode')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create predictor
    config = Config()
    predictor = ThreatPredictor(args.model_path, args.feature_engineer_path, config)
    
    if args.mode == 'single':
        # Single prediction mode (expects JSON input)
        with open(args.input_data, 'r') as f:
            network_data = json.load(f)
        
        result = predictor.predict_single(network_data)
        
        with open(args.output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Prediction: {result['prediction_label']} ({result['probability']:.3f})")
    
    elif args.mode == 'batch':
        # Batch prediction mode
        data = pd.read_csv(args.input_data)
        results = predictor.predict_batch(data)
        predictor.save_predictions(results, args.output_path)
        
        # Print statistics
        predictions_list = results.to_dict('records')
        stats = predictor.get_prediction_stats(predictions_list)
        print(f"Processed {stats['total_predictions']} predictions")
        print(f"Attack rate: {stats['attack_rate']:.1f}%")
    
    else:  # large mode
        # Large dataset prediction mode
        batch_predictor = BatchThreatPredictor(predictor)
        results = batch_predictor.predict_large_dataset(
            args.input_data, 
            args.output_path, 
            args.batch_size
        )
        
        print(f"Processed {results['total_processed']} records")
        print(f"Results saved to: {results['output_path']}")


if __name__ == "__main__":
    main()