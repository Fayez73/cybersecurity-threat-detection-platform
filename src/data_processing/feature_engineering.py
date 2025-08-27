import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging

from ..utils.config import Config
from ..utils.helpers import Timer


class FeatureEngineer:
    """Feature engineering for cybersecurity threat detection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.feature_stats = {}
        
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        data = data.copy()
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if data[col].isnull().sum() > 0:
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
                self.logger.info(f"Filled {col} missing values with median: {median_val}")
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                mode_val = data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'unknown'
                data[col].fillna(mode_val, inplace=True)
                self.logger.info(f"Filled {col} missing values with mode: {mode_val}")
        
        return data
    
    def remove_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from numeric features"""
        if not self.config.data.remove_outliers:
            return data
        
        data = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['is_attack']]
        
        initial_shape = data.shape
        
        for col in numeric_columns:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores < self.config.data.outlier_threshold]
        
        final_shape = data.shape
        removed_rows = initial_shape[0] - final_shape[0]
        
        self.logger.info(f"Removed {removed_rows} outlier rows ({removed_rows/initial_shape[0]*100:.2f}%)")
        
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame, 
                                  fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        data = data.copy()
        categorical_columns = data.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col not in ['attack_type']]
        
        if self.config.data.categorical_encoding == 'onehot':
            for col in categorical_columns:
                if fit:
                    encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
                    encoded_features = encoder.fit_transform(data[[col]])
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                    self.encoders[col] = encoder
                else:
                    if col in self.encoders:
                        encoder = self.encoders[col]
                        encoded_features = encoder.transform(data[[col]])
                        feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                    else:
                        continue
                
                # Add encoded features
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=data.index)
                data = pd.concat([data, encoded_df], axis=1)
                data.drop(col, axis=1, inplace=True)
        
        elif self.config.data.categorical_encoding == 'label':
            for col in categorical_columns:
                if fit:
                    encoder = LabelEncoder()
                    data[col] = encoder.fit_transform(data[col])
                    self.encoders[col] = encoder
                else:
                    if col in self.encoders:
                        encoder = self.encoders[col]
                        # Handle unknown categories
                        data[col] = data[col].map(lambda x: encoder.transform([x])[0] 
                                                if x in encoder.classes_ else -1)
        
        return data
    
    def create_network_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced network-based features"""
        data = data.copy()
        
        # Byte ratios and rates
        data['bytes_ratio'] = data['src_bytes'] / (data['dst_bytes'] + 1)
        data['bytes_per_second'] = (data['src_bytes'] + data['dst_bytes']) / (data['duration'] + 1)
        
        # Connection patterns
        if 'count' in data.columns and 'srv_count' in data.columns:
            data['srv_connection_ratio'] = data['srv_count'] / (data['count'] + 1)
        
        # Error rate combinations
        if all(col in data.columns for col in ['serror_rate', 'rerror_rate']):
            data['total_error_rate'] = data['serror_rate'] + data['rerror_rate']
            data['error_rate_diff'] = abs(data['serror_rate'] - data['rerror_rate'])
        
        # Host-based patterns
        if 'dst_host_count' in data.columns:
            data['host_connection_density'] = data['dst_host_srv_count'] / (data['dst_host_count'] + 1)
        
        # Service diversity
        if all(col in data.columns for col in ['same_srv_rate', 'diff_srv_rate']):
            data['srv_diversity'] = 1 - data['same_srv_rate']
            data['srv_anomaly'] = data['diff_srv_rate'] * (1 - data['same_srv_rate'])
        
        # Time-based features
        if 'duration' in data.columns:
            data['duration_log'] = np.log1p(data['duration'])
            data['is_short_connection'] = (data['duration'] < 1).astype(int)
            data['is_long_connection'] = (data['duration'] > 100).astype(int)
        
        self.logger.info("Created network-based engineered features")
        
        return data
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features for anomaly detection"""
        data = data.copy()
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['is_attack']]
        
        # Moving statistics (window-based features)
        window_size = min(100, len(data) // 10)
        
        for col in ['src_bytes', 'dst_bytes', 'duration']:
            if col in data.columns:
                # Rolling statistics
                data[f'{col}_rolling_mean'] = data[col].rolling(window=window_size, min_periods=1).mean()
                data[f'{col}_rolling_std'] = data[col].rolling(window=window_size, min_periods=1).std()
                data[f'{col}_zscore'] = (data[col] - data[f'{col}_rolling_mean']) / (data[f'{col}_rolling_std'] + 1e-8)
        
        # Percentile features
        for col in numeric_columns[:5]:  # Limit to first 5 numeric columns for performance
            if col in data.columns:
                percentiles = [25, 50, 75, 90]
                col_percentiles = data[col].quantile([p/100 for p in percentiles])
                
                for p in percentiles:
                    data[f'{col}_above_p{p}'] = (data[col] > col_percentiles[p/100]).astype(int)
        
        self.logger.info("Created statistical engineered features")
        
        return data
    
    def create_interaction_features(self, data: pd.DataFrame, max_interactions: int = 10) -> pd.DataFrame:
        """Create interaction features between important variables"""
        data = data.copy()
        
        # Key interaction pairs based on domain knowledge
        interaction_pairs = [
            ('serror_rate', 'srv_serror_rate'),
            ('rerror_rate', 'srv_rerror_rate'),
            ('same_srv_rate', 'diff_srv_rate'),
            ('dst_host_same_srv_rate', 'dst_host_diff_srv_rate'),
            ('count', 'srv_count'),
            ('src_bytes', 'dst_bytes')
        ]
        
        created_features = 0
        for col1, col2 in interaction_pairs:
            if created_features >= max_interactions:
                break
            
            if col1 in data.columns and col2 in data.columns:
                # Multiplicative interaction
                data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                
                # Ratio interaction
                data[f'{col1}_div_{col2}'] = data[col1] / (data[col2] + 1e-8)
                
                created_features += 2
        
        self.logger.info(f"Created {created_features} interaction features")
        
        return data
    
    def scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features"""
        if not self.config.data.normalize_features:
            return data
        
        data = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['is_attack']]
        
        if fit:
            scaler = StandardScaler()
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            self.scalers['standard'] = scaler
        else:
            if 'standard' in self.scalers:
                scaler = self.scalers['standard']
                data[numeric_columns] = scaler.transform(data[numeric_columns])
        
        self.logger.info(f"Scaled {len(numeric_columns)} numeric features")
        
        return data
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'mutual_info', k: int = 50,
                       fit: bool = True) -> pd.DataFrame:
        """Select top k features using specified method"""
        
        if method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            score_func = f_classif
        
        if fit:
            selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            self.feature_selectors[method] = selector
            self.feature_stats['selected_features'] = selected_features
            self.feature_stats['feature_scores'] = dict(zip(X.columns, selector.scores_))
        else:
            if method in self.feature_selectors:
                selector = self.feature_selectors[method]
                X_selected = selector.transform(X)
                selected_features = self.feature_stats.get('selected_features', X.columns.tolist())
            else:
                X_selected = X.values
                selected_features = X.columns.tolist()
        
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        self.logger.info(f"Selected {X_selected_df.shape[1]} features using {method}")
        
        return X_selected_df
    
    def process_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        
        with Timer("Feature engineering pipeline"):
            # Handle missing values
            data = self.handle_missing_values(data)
            
            # Remove outliers
            data = self.remove_outliers(data)
            
            # Create engineered features
            data = self.create_network_features(data)
            data = self.create_statistical_features(data)
            data = self.create_interaction_features(data)
            
            # Encode categorical features
            data = self.encode_categorical_features(data, fit=fit)
            
            # Scale features
            data = self.scale_features(data, fit=fit)
            
            self.logger.info(f"Feature engineering completed. Final shape: {data.shape}")
            
            return data
    
    def get_feature_importance_summary(self) -> Dict:
        """Get summary of feature importance and selection"""
        summary = {}
        
        if 'selected_features' in self.feature_stats:
            summary['selected_features'] = self.feature_stats['selected_features']
            summary['num_selected'] = len(self.feature_stats['selected_features'])
        
        if 'feature_scores' in self.feature_stats:
            # Top 20 features by score
            sorted_features = sorted(
                self.feature_stats['feature_scores'].items(),
                key=lambda x: x[1], reverse=True
            )
            summary['top_20_features'] = dict(sorted_features[:20])
        
        return summary