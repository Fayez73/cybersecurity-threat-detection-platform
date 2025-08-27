import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime

from ..utils.config import Config


@dataclass
class ValidationResult:
    """Data validation result container"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class DataValidator:
    """Comprehensive data validation for cybersecurity datasets"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def validate_schema(self, data: pd.DataFrame, expected_columns: List[str] = None) -> ValidationResult:
        """Validate data schema and structure"""
        errors = []
        warnings = []
        metrics = {}
        
        # Basic shape validation
        if data.empty:
            errors.append("Dataset is empty")
            return ValidationResult(False, errors, warnings, metrics)
        
        metrics['num_rows'] = len(data)
        metrics['num_columns'] = len(data.columns)
        
        # Column validation
        if expected_columns:
            missing_columns = set(expected_columns) - set(data.columns)
            extra_columns = set(data.columns) - set(expected_columns)
            
            if missing_columns:
                errors.append(f"Missing required columns: {list(missing_columns)}")
            
            if extra_columns:
                warnings.append(f"Extra columns found: {list(extra_columns)}")
        
        # Data types validation
        metrics['column_types'] = data.dtypes.value_counts().to_dict()
        
        # Check for completely empty columns
        empty_columns = data.columns[data.isnull().all()].tolist()
        if empty_columns:
            errors.append(f"Completely empty columns: {empty_columns}")
        
        # Check for constant columns
        constant_columns = []
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].nunique() == 1:
                constant_columns.append(col)
        
        if constant_columns:
            warnings.append(f"Constant columns (no variation): {constant_columns}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, metrics)
    
    def validate_data_quality(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data quality metrics"""
        errors = []
        warnings = []
        metrics = {}
        
        # Missing value analysis
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data) * 100).round(2)
        
        metrics['missing_value_summary'] = {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': (missing_counts > 0).sum(),
            'max_missing_percentage': missing_percentages.max()
        }
        
        # High missing value columns (>50%)
        high_missing_cols = missing_percentages[missing_percentages > 50].index.tolist()
        if high_missing_cols:
            warnings.append(f"Columns with >50% missing values: {high_missing_cols}")
        
        # Extremely high missing value columns (>90%)
        very_high_missing_cols = missing_percentages[missing_percentages > 90].index.tolist()
        if very_high_missing_cols:
            errors.append(f"Columns with >90% missing values: {very_high_missing_cols}")
        
        # Duplicate rows
        duplicate_count = data.duplicated().sum()
        metrics['duplicate_rows'] = duplicate_count
        
        if duplicate_count > len(data) * 0.1:  # More than 10% duplicates
            warnings.append(f"High number of duplicate rows: {duplicate_count} ({duplicate_count/len(data)*100:.1f}%)")
        
        # Data type consistency
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].dtype == 'object':
                errors.append(f"Column {col} should be numeric but contains non-numeric values")
        
        # Check for infinite values
        inf_columns = []
        for col in numeric_columns:
            if np.isinf(data[col]).any():
                inf_columns.append(col)
        
        if inf_columns:
            errors.append(f"Columns with infinite values: {inf_columns}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, metrics)
    
    def validate_target_variable(self, data: pd.DataFrame, target_col: str = 'is_attack') -> ValidationResult:
        """Validate target variable properties"""
        errors = []
        warnings = []
        metrics = {}
        
        if target_col not in data.columns:
            errors.append(f"Target column '{target_col}' not found")
            return ValidationResult(False, errors, warnings, metrics)
        
        target = data[target_col]
        
        # Basic target statistics
        metrics['target_stats'] = {
            'unique_values': target.nunique(),
            'missing_values': target.isnull().sum(),
            'value_counts': target.value_counts().to_dict()
        }
        
        # Check for missing target values
        if target.isnull().sum() > 0:
            errors.append(f"Target variable has {target.isnull().sum()} missing values")
        
        # Binary classification validation
        unique_values = target.dropna().unique()
        if len(unique_values) != 2:
            warnings.append(f"Target variable has {len(unique_values)} unique values, expected 2 for binary classification")
        
        # Check class imbalance
        if len(unique_values) == 2:
            value_counts = target.value_counts()
            minority_class_ratio = value_counts.min() / value_counts.sum()
            metrics['class_balance'] = {
                'minority_class_ratio': minority_class_ratio,
                'imbalance_ratio': value_counts.max() / value_counts.min()
            }
            
            if minority_class_ratio < 0.05:  # Less than 5% minority class
                warnings.append(f"Severe class imbalance: minority class represents only {minority_class_ratio:.1%}")
            elif minority_class_ratio < 0.1:  # Less than 10% minority class
                warnings.append(f"Class imbalance detected: minority class represents {minority_class_ratio:.1%}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, metrics)
    
    def validate_feature_distributions(self, data: pd.DataFrame) -> ValidationResult:
        """Validate feature distributions and detect anomalies"""
        errors = []
        warnings = []
        metrics = {}
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['is_attack']]
        
        distribution_stats = {}
        
        for col in numeric_columns:
            if data[col].dtype in [np.number]:
                col_data = data[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                stats = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'q25': col_data.quantile(0.25),
                    'q50': col_data.quantile(0.50),
                    'q75': col_data.quantile(0.75)
                }
                
                distribution_stats[col] = stats
                
                # Check for extreme skewness
                if abs(stats['skewness']) > 3:
                    warnings.append(f"Column {col} has extreme skewness: {stats['skewness']:.2f}")
                
                # Check for extreme kurtosis
                if abs(stats['kurtosis']) > 10:
                    warnings.append(f"Column {col} has extreme kurtosis: {stats['kurtosis']:.2f}")
                
                # Check for negative values where they shouldn't be
                if col in ['duration', 'src_bytes', 'dst_bytes', 'count'] and stats['min'] < 0:
                    errors.append(f"Column {col} has negative values which may be invalid")
        
        metrics['distribution_stats'] = distribution_stats
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, metrics)
    
    def validate_categorical_features(self, data: pd.DataFrame) -> ValidationResult:
        """Validate categorical features"""
        errors = []
        warnings = []
        metrics = {}
        
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        categorical_stats = {}
        
        for col in categorical_columns:
            col_data = data[col].dropna()
            
            stats = {
                'unique_count': col_data.nunique(),
                'most_frequent': col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None,
                'most_frequent_count': col_data.value_counts().iloc[0] if len(col_data) > 0 else 0,
                'value_counts': col_data.value_counts().head(10).to_dict()
            }
            
            categorical_stats[col] = stats
            
            # Check for high cardinality
            if stats['unique_count'] > len(col_data) * 0.8:
                warnings.append(f"Column {col} has very high cardinality: {stats['unique_count']} unique values")
            
            # Check for rare categories
            value_counts = col_data.value_counts()
            rare_categories = value_counts[value_counts == 1]
            
            if len(rare_categories) > stats['unique_count'] * 0.5:
                warnings.append(f"Column {col} has many rare categories: {len(rare_categories)} categories appear only once")
        
        metrics['categorical_stats'] = categorical_stats
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, metrics)
    
    def validate_cybersecurity_specific(self, data: pd.DataFrame) -> ValidationResult:
        """Validate cybersecurity-specific data patterns"""
        errors = []
        warnings = []
        metrics = {}
        
        # Expected cybersecurity features
        expected_features = [
            'duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
            'serror_rate', 'rerror_rate', 'same_srv_rate', 'diff_srv_rate'
        ]
        
        missing_features = [f for f in expected_features if f not in data.columns]
        if missing_features:
            warnings.append(f"Missing common cybersecurity features: {missing_features}")
        
        # Validate rate features (should be between 0 and 1)
        rate_features = [col for col in data.columns if 'rate' in col.lower()]
        
        for col in rate_features:
            if col in data.columns:
                col_data = data[col].dropna()
                
                if col_data.min() < 0 or col_data.max() > 1:
                    errors.append(f"Rate feature {col} has values outside [0,1] range: [{col_data.min():.3f}, {col_data.max():.3f}]")
        
        # Validate protocol types
        if 'protocol_type' in data.columns:
            protocols = data['protocol_type'].value_counts()
            expected_protocols = ['tcp', 'udp', 'icmp']
            unexpected_protocols = set(protocols.index) - set(expected_protocols)
            
            if unexpected_protocols:
                warnings.append(f"Unexpected protocol types found: {list(unexpected_protocols)}")
            
            metrics['protocol_distribution'] = protocols.to_dict()
        
        # Validate service types
        if 'service' in data.columns:
            services = data['service'].value_counts()
            metrics['service_distribution'] = services.head(10).to_dict()
            
            if services.nunique() > 100:
                warnings.append(f"Very high number of service types: {services.nunique()}")
        
        # Check for logical inconsistencies
        if all(col in data.columns for col in ['src_bytes', 'dst_bytes', 'duration']):
            # Check for zero duration with non-zero bytes
            zero_duration_with_bytes = data[
                (data['duration'] == 0) & 
                ((data['src_bytes'] > 0) | (data['dst_bytes'] > 0))
            ]
            
            if len(zero_duration_with_bytes) > 0:
                warnings.append(f"Found {len(zero_duration_with_bytes)} records with zero duration but non-zero bytes")
        
        # Check attack type consistency
        if all(col in data.columns for col in ['is_attack', 'attack_type']):
            inconsistent_attacks = data[
                ((data['is_attack'] == 1) & (data['attack_type'] == 'normal')) |
                ((data['is_attack'] == 0) & (data['attack_type'] != 'normal'))
            ]
            
            if len(inconsistent_attacks) > 0:
                errors.append(f"Found {len(inconsistent_attacks)} records with inconsistent attack labels")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, metrics)
    
    def comprehensive_validation(self, data: pd.DataFrame, 
                               expected_columns: List[str] = None,
                               target_col: str = 'is_attack') -> ValidationResult:
        """Run comprehensive validation pipeline"""
        
        self.logger.info("Starting comprehensive data validation")
        
        all_errors = []
        all_warnings = []
        all_metrics = {}
        
        # Schema validation
        schema_result = self.validate_schema(data, expected_columns)
        all_errors.extend(schema_result.errors)
        all_warnings.extend(schema_result.warnings)
        all_metrics['schema'] = schema_result.metrics
        
        # Data quality validation
        quality_result = self.validate_data_quality(data)
        all_errors.extend(quality_result.errors)
        all_warnings.extend(quality_result.warnings)
        all_metrics['quality'] = quality_result.metrics
        
        # Target variable validation
        target_result = self.validate_target_variable(data, target_col)
        all_errors.extend(target_result.errors)
        all_warnings.extend(target_result.warnings)
        all_metrics['target'] = target_result.metrics
        
        # Feature distribution validation
        distribution_result = self.validate_feature_distributions(data)
        all_errors.extend(distribution_result.errors)
        all_warnings.extend(distribution_result.warnings)
        all_metrics['distributions'] = distribution_result.metrics
        
        # Categorical feature validation
        categorical_result = self.validate_categorical_features(data)
        all_errors.extend(categorical_result.errors)
        all_warnings.extend(categorical_result.warnings)
        all_metrics['categorical'] = categorical_result.metrics
        
        # Cybersecurity-specific validation
        cyber_result = self.validate_cybersecurity_specific(data)
        all_errors.extend(cyber_result.errors)
        all_warnings.extend(cyber_result.warnings)
        all_metrics['cybersecurity'] = cyber_result.metrics
        
        # Overall validation summary
        is_valid = len(all_errors) == 0
        all_metrics['summary'] = {
            'total_errors': len(all_errors),
            'total_warnings': len(all_warnings),
            'validation_passed': is_valid,
            'data_shape': data.shape,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        result = ValidationResult(is_valid, all_errors, all_warnings, all_metrics)
        
        # Log validation results
        if is_valid:
            self.logger.info(f"Data validation PASSED with {len(all_warnings)} warnings")
        else:
            self.logger.error(f"Data validation FAILED with {len(all_errors)} errors and {len(all_warnings)} warnings")
        
        return result
    
    def print_validation_report(self, validation_result: ValidationResult) -> None:
        """Print formatted validation report"""
        
        print("\n" + "="*80)
        print("DATA VALIDATION REPORT")
        print("="*80)
        
        print(f"Validation Status: {'PASSED' if validation_result.is_valid else 'FAILED'}")
        print(f"Timestamp: {validation_result.timestamp}")
        
        if validation_result.metrics.get('summary'):
            summary = validation_result.metrics['summary']
            print(f"Data Shape: {summary['data_shape']}")
            print(f"Total Errors: {summary['total_errors']}")
            print(f"Total Warnings: {summary['total_warnings']}")
        
        # Print errors
        if validation_result.errors:
            print("\n" + "-"*40)
            print("ERRORS:")
            print("-"*40)
            for i, error in enumerate(validation_result.errors, 1):
                print(f"{i}. {error}")
        
        # Print warnings
        if validation_result.warnings:
            print("\n" + "-"*40)
            print("WARNINGS:")
            print("-"*40)
            for i, warning in enumerate(validation_result.warnings, 1):
                print(f"{i}. {warning}")
        
        # Print key metrics
        if validation_result.metrics:
            print("\n" + "-"*40)
            print("KEY METRICS:")
            print("-"*40)
            
            if 'quality' in validation_result.metrics:
                quality = validation_result.metrics['quality']
                if 'missing_value_summary' in quality:
                    mvs = quality['missing_value_summary']
                    print(f"Missing Values: {mvs['total_missing']} total, {mvs['max_missing_percentage']:.1f}% max per column")
                
                if 'duplicate_rows' in quality:
                    print(f"Duplicate Rows: {quality['duplicate_rows']}")
            
            if 'target' in validation_result.metrics:
                target = validation_result.metrics['target']
                if 'class_balance' in target:
                    cb = target['class_balance']
                    print(f"Class Balance: {cb['minority_class_ratio']:.1%} minority class")
        
        print("="*80 + "\n")
    
    def save_validation_report(self, validation_result: ValidationResult, 
                             filepath: str) -> None:
        """Save validation report to JSON file"""
        import json
        
        report_data = {
            'is_valid': validation_result.is_valid,
            'errors': validation_result.errors,
            'warnings': validation_result.warnings,
            'metrics': validation_result.metrics,
            'timestamp': validation_result.timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Validation report saved to {filepath}")
    
    def validate_model_input(self, X: pd.DataFrame, y: pd.Series = None) -> ValidationResult:
        """Validate data for model training/inference"""
        errors = []
        warnings = []
        metrics = {}
        
        # Basic shape validation
        if X.empty:
            errors.append("Feature matrix X is empty")
            return ValidationResult(False, errors, warnings, metrics)
        
        if y is not None and len(X) != len(y):
            errors.append(f"Feature matrix X ({len(X)}) and target y ({len(y)}) have different lengths")
        
        # Check for non-numeric features
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            errors.append(f"Non-numeric columns found in feature matrix: {non_numeric_cols}")
        
        # Check for infinite or NaN values
        if X.isnull().any().any():
            null_cols = X.columns[X.isnull().any()].tolist()
            errors.append(f"NaN values found in columns: {null_cols}")
        
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            inf_cols = X.columns[np.isinf(X.select_dtypes(include=[np.number])).any()].tolist()
            errors.append(f"Infinite values found in columns: {inf_cols}")
        
        # Feature statistics
        metrics['feature_stats'] = {
            'num_features': X.shape[1],
            'num_samples': X.shape[0],
            'feature_names': X.columns.tolist()
        }
        
        if y is not None:
            metrics['target_stats'] = {
                'unique_values': y.nunique(),
                'value_counts': y.value_counts().to_dict()
            }
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, metrics)