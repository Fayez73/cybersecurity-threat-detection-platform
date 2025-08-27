import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.calibration import calibration_curve
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
import joblib

from ..utils.config import Config
from ..utils.helpers import Timer, load_model, calculate_metrics, create_directory


class ModelEvaluator:
    """Comprehensive model evaluation for threat detection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_model_artifacts(self, model_dir: str) -> Tuple[Any, Any]:
        """Load trained model and feature engineer"""
        
        model_path = os.path.join(model_dir, 'threat_detection_model.pkl')
        feature_engineer_path = os.path.join(model_dir, 'feature_engineer.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if not os.path.exists(feature_engineer_path):
            raise FileNotFoundError(f"Feature engineer not found at {feature_engineer_path}")
        
        model = load_model(model_path)
        feature_engineer = load_model(feature_engineer_path)
        
        self.logger.info("Model and feature engineer loaded successfully")
        
        return model, feature_engineer
    
    def comprehensive_evaluation(self, model: Any, X_test: pd.DataFrame, 
                               y_test: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive model evaluation"""
        
        self.logger.info("Starting comprehensive model evaluation...")
        
        with Timer("Model evaluation"):
            # Get predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Basic metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'auc_score': roc_auc_score(y_test, y_prob),
                'average_precision': average_precision_score(y_test, y_prob)
            }
            
            # Class-specific metrics
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
            
            # Threshold analysis
            threshold_analysis = self.analyze_thresholds(y_test, y_prob)
            
            # Feature importance analysis
            feature_importance = self.analyze_feature_importance(model, X_test.columns.tolist())
            
            # Prediction confidence analysis
            confidence_analysis = self.analyze_prediction_confidence(y_prob, y_test)
            
            results = {
                'basic_metrics': metrics,
                'classification_report': class_report,
                'confusion_matrix': cm.tolist(),
                'roc_curve': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist(),
                    'auc': roc_auc
                },
                'precision_recall_curve': {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist()
                },
                'threshold_analysis': threshold_analysis,
                'feature_importance': feature_importance,
                'confidence_analysis': confidence_analysis,
                'predictions': {
                    'y_pred': y_pred.tolist(),
                    'y_prob': y_prob.tolist(),
                    'y_true': y_test.tolist()
                }
            }
            
            self.logger.info("Comprehensive evaluation completed")
            
            return results
    
    def analyze_thresholds(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Analyze performance across different probability thresholds"""
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            
            # Calculate metrics for this threshold
            acc = accuracy_score(y_true, y_pred_thresh)
            prec = precision_score(y_true, y_pred_thresh, zero_division=0)
            rec = recall_score(y_true, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            
            threshold_metrics.append({
                'threshold': threshold,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1
            })
        
        # Find optimal threshold based on F1 score
        optimal_f1_idx = max(range(len(threshold_metrics)), 
                           key=lambda i: threshold_metrics[i]['f1_score'])
        optimal_threshold = threshold_metrics[optimal_f1_idx]
        
        return {
            'threshold_metrics': threshold_metrics,
            'optimal_threshold': optimal_threshold
        }
    
    def analyze_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance from the model"""
        
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), 
                                     key=lambda x: x[1], reverse=True)
            
            # Get top 20 and bottom 10
            top_20 = dict(sorted_importance[:20])
            bottom_10 = dict(sorted_importance[-10:])
            
            # Calculate importance statistics
            importance_values = list(importance_dict.values())
            
            return {
                'all_features': importance_dict,
                'top_20_features': top_20,
                'bottom_10_features': bottom_10,
                'importance_stats': {
                    'mean': np.mean(importance_values),
                    'std': np.std(importance_values),
                    'max': np.max(importance_values),
                    'min': np.min(importance_values)
                }
            }
        else:
            return {'error': 'Model does not have feature_importances_ attribute'}
    
    def analyze_prediction_confidence(self, y_prob: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence and calibration"""
        
        # Confidence bins
        confidence_bins = np.arange(0, 1.1, 0.1)
        bin_analysis = []
        
        for i in range(len(confidence_bins) - 1):
            bin_start, bin_end = confidence_bins[i], confidence_bins[i + 1]
            
            # Find predictions in this confidence bin
            in_bin = (y_prob >= bin_start) & (y_prob < bin_end)
            
            if in_bin.sum() > 0:
                bin_accuracy = y_true[in_bin].mean()
                bin_count = in_bin.sum()
                avg_confidence = y_prob[in_bin].mean()
                
                bin_analysis.append({
                    'bin_range': f'{bin_start:.1f}-{bin_end:.1f}',
                    'count': int(bin_count),
                    'avg_confidence': avg_confidence,
                    'accuracy': bin_accuracy,
                    'calibration_error': abs(avg_confidence - bin_accuracy)
                })
        
        # Overall calibration
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            calibration_error = np.mean(np.abs(prob_pred - prob_true))
        except:
            prob_true, prob_pred = [], []
            calibration_error = 0
        
        return {
            'bin_analysis': bin_analysis,
            'calibration_curve': {
                'prob_true': prob_true.tolist() if len(prob_true) > 0 else [],
                'prob_pred': prob_pred.tolist() if len(prob_pred) > 0 else []
            },
            'expected_calibration_error': calibration_error
        }
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, 
                      roc_auc: float, save_path: str = None) -> None:
        """Plot ROC curve"""
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, precision: np.ndarray, recall: np.ndarray,
                                   avg_precision: float, save_path: str = None) -> None:
        """Plot Precision-Recall curve"""
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"PR curve saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None) -> None:
        """Plot confusion matrix heatmap"""
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_threshold_analysis(self, threshold_data: List[Dict], save_path: str = None) -> None:
        """Plot threshold analysis"""
        
        thresholds = [d['threshold'] for d in threshold_data]
        accuracy = [d['accuracy'] for d in threshold_data]
        precision = [d['precision'] for d in threshold_data]
        recall = [d['recall'] for d in threshold_data]
        f1_score = [d['f1_score'] for d in threshold_data]
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(thresholds, accuracy, 'b-', label='Accuracy', linewidth=2)
        plt.plot(thresholds, precision, 'r-', label='Precision', linewidth=2)
        plt.plot(thresholds, recall, 'g-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1_score, 'm-', label='F1-Score', linewidth=2)
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Performance Metrics vs. Classification Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0.1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Threshold analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, importance_dict: Dict[str, float], 
                              top_n: int = 20, save_path: str = None) -> None:
        """Plot feature importance"""
        
        # Get top N features
        sorted_features = sorted(importance_dict.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        
        # Color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_calibration_curve(self, prob_true: np.ndarray, prob_pred: np.ndarray,
                             save_path: str = None) -> None:
        """Plot calibration curve"""
        
        if len(prob_true) == 0 or len(prob_pred) == 0:
            self.logger.warning("No calibration data available for plotting")
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, 's-', label='Model')
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot (Reliability Diagram)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Calibration curve saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any],
                                 output_dir: str) -> None:
        """Generate comprehensive evaluation report"""
        
        create_directory(output_dir)
        plots_dir = os.path.join(output_dir, 'evaluation_plots')
        create_directory(plots_dir)
        
        # Generate all plots
        
        # ROC Curve
        roc_data = evaluation_results['roc_curve']
        self.plot_roc_curve(
            np.array(roc_data['fpr']),
            np.array(roc_data['tpr']),
            roc_data['auc'],
            os.path.join(plots_dir, 'roc_curve.png')
        )
        
        # Precision-Recall Curve
        pr_data = evaluation_results['precision_recall_curve']
        self.plot_precision_recall_curve(
            np.array(pr_data['precision']),
            np.array(pr_data['recall']),
            evaluation_results['basic_metrics']['average_precision'],
            os.path.join(plots_dir, 'precision_recall_curve.png')
        )
        
        # Confusion Matrix
        cm = np.array(evaluation_results['confusion_matrix'])
        self.plot_confusion_matrix(cm, os.path.join(plots_dir, 'confusion_matrix.png'))
        
        # Threshold Analysis
        threshold_data = evaluation_results['threshold_analysis']['threshold_metrics']
        self.plot_threshold_analysis(threshold_data, os.path.join(plots_dir, 'threshold_analysis.png'))
        
        # Feature Importance
        if 'all_features' in evaluation_results['feature_importance']:
            importance_dict = evaluation_results['feature_importance']['all_features']
            self.plot_feature_importance(importance_dict, 20, 
                                       os.path.join(plots_dir, 'feature_importance.png'))
        
        # Calibration Curve
        calib_data = evaluation_results['confidence_analysis']['calibration_curve']
        if calib_data['prob_true'] and calib_data['prob_pred']:
            self.plot_calibration_curve(
                np.array(calib_data['prob_true']),
                np.array(calib_data['prob_pred']),
                os.path.join(plots_dir, 'calibration_curve.png')
            )
        
        # Save evaluation results
        import json
        results_path = os.path.join(output_dir, 'detailed_evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Generate text report
        self.generate_text_report(evaluation_results, 
                                os.path.join(output_dir, 'evaluation_report.txt'))
        
        self.logger.info(f"Comprehensive evaluation report generated in {output_dir}")
    
    def generate_text_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Generate text-based evaluation report"""
        
        with open(output_path, 'w') as f:
            f.write("CYBERSECURITY THREAT DETECTION MODEL - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic Metrics
            f.write("BASIC PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            metrics = results['basic_metrics']
            for metric, value in metrics.items():
                f.write(f"{metric.upper():.<25} {value:.4f}\n")
            f.write("\n")
            
            # Optimal Threshold
            f.write("OPTIMAL THRESHOLD ANALYSIS\n")
            f.write("-" * 30 + "\n")
            optimal = results['threshold_analysis']['optimal_threshold']
            f.write(f"Optimal Threshold: {optimal['threshold']:.3f}\n")
            f.write(f"  - Accuracy: {optimal['accuracy']:.4f}\n")
            f.write(f"  - Precision: {optimal['precision']:.4f}\n")
            f.write(f"  - Recall: {optimal['recall']:.4f}\n")
            f.write(f"  - F1-Score: {optimal['f1_score']:.4f}\n\n")
            
            # Top Features
            f.write("TOP 10 MOST IMPORTANT FEATURES\n")
            f.write("-" * 35 + "\n")
            if 'top_20_features' in results['feature_importance']:
                top_features = results['feature_importance']['top_20_features']
                for i, (feature, importance) in enumerate(list(top_features.items())[:10], 1):
                    f.write(f"{i:2d}. {feature:.<30} {importance:.4f}\n")
            f.write("\n")
            
            # Class-specific Performance
            f.write("CLASS-SPECIFIC PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            class_report = results['classification_report']
            for class_label in ['0', '1']:  # Normal and Attack
                if class_label in class_report:
                    class_name = 'Normal' if class_label == '0' else 'Attack'
                    class_data = class_report[class_label]
                    f.write(f"{class_name} Class:\n")
                    f.write(f"  Precision: {class_data['precision']:.4f}\n")
                    f.write(f"  Recall: {class_data['recall']:.4f}\n")
                    f.write(f"  F1-Score: {class_data['f1-score']:.4f}\n")
                    f.write(f"  Support: {class_data['support']}\n\n")
            
            # Model Calibration
            f.write("MODEL CALIBRATION\n")
            f.write("-" * 20 + "\n")
            calib_error = results['confidence_analysis']['expected_calibration_error']
            f.write(f"Expected Calibration Error: {calib_error:.4f}\n")
            f.write("(Lower is better, <0.1 is well-calibrated)\n\n")
            
            # Confusion Matrix
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 20 + "\n")
            cm = results['confusion_matrix']
            f.write("           Predicted\n")
            f.write("         Normal Attack\n")
            f.write(f"Normal   {cm[0][0]:6d} {cm[0][1]:6d}\n")
            f.write(f"Attack   {cm[1][0]:6d} {cm[1][1]:6d}\n\n")
            
            f.write("Report generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        
        self.logger.info(f"Text report saved to {output_path}")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate threat detection model')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained model')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = Config()
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Load model
    model, feature_engineer = evaluator.load_model_artifacts(args.model_dir)
    
    # Load and prepare test data
    from ..data_processing.data_loader import DataLoader
    data_loader = DataLoader(config)
    test_data = data_loader.load_and_validate_data(args.test_data)
    
    # Apply feature engineering
    test_data = feature_engineer.process_features(test_data, fit=False)
    X_test = test_data.drop(['is_attack', 'attack_type'], axis=1, errors='ignore')
    y_test = test_data['is_attack']
    
    # Evaluate model
    results = evaluator.comprehensive_evaluation(model, X_test, y_test)
    
    # Generate report
    evaluator.generate_evaluation_report(results, args.output_dir)
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")
    print(f"Model AUC: {results['basic_metrics']['auc_score']:.4f}")


if __name__ == "__main__":
    main()