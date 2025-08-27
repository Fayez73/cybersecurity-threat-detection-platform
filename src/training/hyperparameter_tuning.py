import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
import optuna
import logging
from typing import Dict, Any, Optional, Tuple
import json
import os

from ..utils.config import Config
from ..utils.helpers import Timer, create_directory


class HyperparameterTuner:
    """Hyperparameter tuning for XGBoost threat detection model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.best_params = {}
        self.tuning_results = {}
        
    def get_param_space(self) -> Dict[str, Any]:
        """Define hyperparameter search space for XGBoost"""
        
        param_space = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [100, 200, 300, 500, 700, 1000],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5, 1, 2],
            'min_child_weight': [1, 2, 3, 5, 7],
            'reg_alpha': [0, 0.1, 0.5, 1, 2],
            'reg_lambda': [0, 0.1, 0.5, 1, 2]
        }
        
        return param_space
    
    def get_optuna_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define Optuna hyperparameter search space"""
        
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.config.model.random_state,
            'verbosity': 0
        }
        
        return params
    
    def random_search_tuning(self, X: pd.DataFrame, y: pd.Series, 
                           n_iter: int = 50) -> Dict[str, Any]:
        """Perform random search hyperparameter tuning"""
        
        self.logger.info(f"Starting random search tuning with {n_iter} iterations...")
        
        with Timer("Random search tuning"):
            # Create base model
            base_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=self.config.model.random_state,
                verbosity=0,
                n_jobs=-1
            )
            
            # Define parameter space
            param_space = self.get_param_space()
            
            # Setup cross-validation
            cv = StratifiedKFold(
                n_splits=self.config.training.cross_validation_folds,
                shuffle=True,
                random_state=self.config.model.random_state
            )
            
            # Setup scorer
            scorer = make_scorer(roc_auc_score, needs_proba=True)
            
            # Perform random search
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_space,
                n_iter=n_iter,
                cv=cv,
                scoring=scorer,
                n_jobs=-1,
                verbose=1,
                random_state=self.config.model.random_state
            )
            
            random_search.fit(X, y)
            
            results = {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'cv_results': {
                    'mean_test_scores': random_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_scores': random_search.cv_results_['std_test_score'].tolist(),
                    'params': random_search.cv_results_['params']
                }
            }
            
            self.best_params = random_search.best_params_
            
            self.logger.info(f"Random search completed. Best AUC: {random_search.best_score_:.4f}")
            
            return results
    
    def grid_search_tuning(self, X: pd.DataFrame, y: pd.Series,
                          param_grid: Dict[str, list] = None) -> Dict[str, Any]:
        """Perform grid search hyperparameter tuning"""
        
        if param_grid is None:
            # Define a smaller grid for computational efficiency
            param_grid = {
                'max_depth': [4, 5, 6],
                'learning_rate': [0.05, 0.1, 0.15],
                'n_estimators': [200, 300, 500],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }
        
        self.logger.info("Starting grid search tuning...")
        
        with Timer("Grid search tuning"):
            # Create base model
            base_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=self.config.model.random_state,
                verbosity=0,
                n_jobs=-1
            )
            
            # Setup cross-validation
            cv = StratifiedKFold(
                n_splits=self.config.training.cross_validation_folds,
                shuffle=True,
                random_state=self.config.model.random_state
            )
            
            # Setup scorer
            scorer = make_scorer(roc_auc_score, needs_proba=True)
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring=scorer,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': {
                    'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                }
            }
            
            self.best_params = grid_search.best_params_
            
            self.logger.info(f"Grid search completed. Best AUC: {grid_search.best_score_:.4f}")
            
            return results
    
    def optuna_tuning(self, X: pd.DataFrame, y: pd.Series, 
                     n_trials: int = 100) -> Dict[str, Any]:
        """Perform Optuna-based hyperparameter tuning"""
        
        self.logger.info(f"Starting Optuna tuning with {n_trials} trials...")
        
        def objective(trial):
            # Get parameters from Optuna
            params = self.get_optuna_param_space(trial)
            
            # Create model
            model = xgb.XGBClassifier(**params)
            
            # Setup cross-validation
            cv = StratifiedKFold(
                n_splits=self.config.training.cross_validation_folds,
                shuffle=True,
                random_state=self.config.model.random_state
            )
            
            # Perform cross-validation
            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold, verbose=False)
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                
                auc_score = roc_auc_score(y_val_fold, y_pred_proba)
                scores.append(auc_score)
            
            return np.mean(scores)
        
        with Timer("Optuna tuning"):
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.config.model.random_state)
            )
            
            # Optimize
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            results = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'optimization_history': [trial.value for trial in study.trials if trial.value is not None],
                'param_importance': optuna.importance.get_param_importances(study)
            }
            
            self.best_params = study.best_params
            
            self.logger.info(f"Optuna tuning completed. Best AUC: {study.best_value:.4f}")
            
            return results
    
    def bayesian_optimization(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Bayesian optimization using scikit-optimize"""
        
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            from skopt.utils import use_named_args
        except ImportError:
            self.logger.error("scikit-optimize not installed. Using Optuna instead.")
            return self.optuna_tuning(X, y)
        
        self.logger.info("Starting Bayesian optimization tuning...")
        
        # Define search space
        dimensions = [
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
            Integer(100, 1000, name='n_estimators'),
            Real(0.5, 1.0, name='subsample'),
            Real(0.5, 1.0, name='colsample_bytree'),
            Real(0, 5, name='gamma'),
            Integer(1, 10, name='min_child_weight'),
            Real(0, 5, name='reg_alpha'),
            Real(0, 5, name='reg_lambda')
        ]
        
        @use_named_args(dimensions)
        def objective(**params):
            # Create model
            model = xgb.XGBClassifier(
                **params,
                objective='binary:logistic',
                eval_metric='auc',
                random_state=self.config.model.random_state,
                verbosity=0,
                n_jobs=-1
            )
            
            # Setup cross-validation
            cv = StratifiedKFold(
                n_splits=self.config.training.cross_validation_folds,
                shuffle=True,
                random_state=self.config.model.random_state
            )
            
            # Perform cross-validation
            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold, verbose=False)
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                
                auc_score = roc_auc_score(y_val_fold, y_pred_proba)
                scores.append(auc_score)
            
            # Return negative score for minimization
            return -np.mean(scores)
        
        with Timer("Bayesian optimization"):
            # Perform optimization
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=50,
                random_state=self.config.model.random_state
            )
            
            # Extract best parameters
            best_params = {
                'max_depth': result.x[0],
                'learning_rate': result.x[1],
                'n_estimators': result.x[2],
                'subsample': result.x[3],
                'colsample_bytree': result.x[4],
                'gamma': result.x[5],
                'min_child_weight': result.x[6],
                'reg_alpha': result.x[7],
                'reg_lambda': result.x[8]
            }
            
            results = {
                'best_params': best_params,
                'best_score': -result.fun,
                'n_calls': len(result.func_vals),
                'optimization_history': [-val for val in result.func_vals]
            }
            
            self.best_params = best_params
            
            self.logger.info(f"Bayesian optimization completed. Best AUC: {-result.fun:.4f}")
            
            return results
    
    def comprehensive_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive hyperparameter tuning using multiple methods"""
        
        self.logger.info("Starting comprehensive hyperparameter tuning...")
        
        all_results = {}
        
        # Random Search (fast exploration)
        self.logger.info("Phase 1: Random Search")
        random_results = self.random_search_tuning(X, y, n_iter=30)
        all_results['random_search'] = random_results
        
        # Optuna (sophisticated optimization)
        self.logger.info("Phase 2: Optuna Optimization")
        optuna_results = self.optuna_tuning(X, y, n_trials=50)
        all_results['optuna'] = optuna_results
        
        # Compare results and select best
        random_score = random_results['best_score']
        optuna_score = optuna_results['best_score']
        
        if optuna_score > random_score:
            self.best_params = optuna_results['best_params']
            best_method = 'optuna'
            best_score = optuna_score
        else:
            self.best_params = random_results['best_params']
            best_method = 'random_search'
            best_score = random_score
        
        # Fine-tuning with grid search around best parameters
        self.logger.info("Phase 3: Fine-tuning with Grid Search")
        fine_tune_grid = self._create_fine_tune_grid(self.best_params)
        if fine_tune_grid:
            grid_results = self.grid_search_tuning(X, y, fine_tune_grid)
            all_results['grid_search_fine_tune'] = grid_results
            
            if grid_results['best_score'] > best_score:
                self.best_params = grid_results['best_params']
                best_method = 'grid_search_fine_tune'
                best_score = grid_results['best_score']
        
        # Summary
        summary = {
            'best_method': best_method,
            'best_params': self.best_params,
            'best_score': best_score,
            'all_results': all_results
        }
        
        self.tuning_results = summary
        
        self.logger.info(f"Comprehensive tuning completed. Best method: {best_method}, Best AUC: {best_score:.4f}")
        
        return summary
    
    def _create_fine_tune_grid(self, best_params: Dict[str, Any]) -> Dict[str, list]:
        """Create a fine-tuning grid around the best parameters"""
        
        fine_tune_grid = {}
        
        # Fine-tune key parameters with small variations
        if 'max_depth' in best_params:
            max_depth = best_params['max_depth']
            fine_tune_grid['max_depth'] = [max_depth - 1, max_depth, max_depth + 1]
            fine_tune_grid['max_depth'] = [max(3, min(10, d)) for d in fine_tune_grid['max_depth']]
        
        if 'learning_rate' in best_params:
            lr = best_params['learning_rate']
            fine_tune_grid['learning_rate'] = [lr * 0.8, lr, lr * 1.2]
            fine_tune_grid['learning_rate'] = [max(0.01, min(0.3, l)) for l in fine_tune_grid['learning_rate']]
        
        if 'n_estimators' in best_params:
            n_est = best_params['n_estimators']
            fine_tune_grid['n_estimators'] = [max(100, n_est - 100), n_est, n_est + 100]
        
        return fine_tune_grid
    
    def create_tuned_model(self) -> xgb.XGBClassifier:
        """Create XGBoost model with tuned hyperparameters"""
        
        if not self.best_params:
            self.logger.warning("No tuned parameters available. Using default configuration.")
            params = {
                'objective': self.config.model.objective,
                'max_depth': self.config.model.max_depth,
                'learning_rate': self.config.model.learning_rate,
                'n_estimators': self.config.model.n_estimators,
                'random_state': self.config.model.random_state
            }
        else:
            params = self.best_params.copy()
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': self.config.model.random_state,
                'verbosity': 0,
                'n_jobs': -1
            })
        
        model = xgb.XGBClassifier(**params)
        
        self.logger.info("Created tuned XGBoost model")
        
        return model
    
    def save_tuning_results(self, output_dir: str) -> None:
        """Save hyperparameter tuning results"""
        
        create_directory(output_dir)
        
        # Save tuning results
        results_path = os.path.join(output_dir, 'hyperparameter_tuning_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.tuning_results, f, indent=2, default=str)
        
        # Save best parameters
        best_params_path = os.path.join(output_dir, 'best_hyperparameters.json')
        with open(best_params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        self.logger.info(f"Tuning results saved to {output_dir}")
    
    def plot_optimization_history(self, results: Dict[str, Any], save_path: str = None) -> None:
        """Plot optimization history"""
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, len(results['all_results']), figsize=(15, 5))
        if len(results['all_results']) == 1:
            axes = [axes]
        
        for idx, (method, method_results) in enumerate(results['all_results'].items()):
            if 'optimization_history' in method_results:
                history = method_results['optimization_history']
                axes[idx].plot(history, 'b-', linewidth=2)
                axes[idx].set_title(f'{method.replace("_", " ").title()}')
                axes[idx].set_xlabel('Iteration')
                axes[idx].set_ylabel('AUC Score')
                axes[idx].grid(True, alpha=0.3)
                
                # Mark best score
                best_idx = np.argmax(history)
                axes[idx].plot(best_idx, history[best_idx], 'ro', markersize=8)
                axes[idx].annotate(f'Best: {history[best_idx]:.4f}', 
                                 xy=(best_idx, history[best_idx]),
                                 xytext=(10, 10), textcoords='offset points')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Optimization history plot saved to {save_path}")
        
        plt.show()
    
    def plot_parameter_importance(self, results: Dict[str, Any], save_path: str = None) -> None:
        """Plot parameter importance from Optuna results"""
        
        if 'optuna' not in results['all_results']:
            self.logger.warning("No Optuna results available for parameter importance plot")
            return
        
        optuna_results = results['all_results']['optuna']
        
        if 'param_importance' not in optuna_results:
            self.logger.warning("No parameter importance data available")
            return
        
        import matplotlib.pyplot as plt
        
        importance = optuna_results['param_importance']
        params = list(importance.keys())
        values = list(importance.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(params, values)
        plt.xlabel('Importance')
        plt.title('Hyperparameter Importance (Optuna)')
        plt.tight_layout()
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Parameter importance plot saved to {save_path}")
        
        plt.show()


def main():
    """Main hyperparameter tuning function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for threat detection model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--method', type=str, choices=['random', 'grid', 'optuna', 'bayesian', 'comprehensive'], 
                       default='comprehensive', help='Tuning method')
    parser.add_argument('--output_dir', type=str, default='tuning_results', help='Output directory')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for optimization')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = Config()
    
    # Load and prepare data
    from ..data_processing.data_loader import DataLoader
    from ..data_processing.feature_engineering import FeatureEngineer
    
    data_loader = DataLoader(config)
    data = data_loader.load_and_validate_data(args.data_path)
    
    feature_engineer = FeatureEngineer(config)
    data = feature_engineer.process_features(data, fit=True)
    
    X = data.drop(['is_attack', 'attack_type'], axis=1, errors='ignore')
    y = data['is_attack']
    
    # Create tuner
    tuner = HyperparameterTuner(config)
    
    # Perform tuning based on method
    if args.method == 'random':
        results = tuner.random_search_tuning(X, y, n_iter=args.n_trials)
    elif args.method == 'grid':
        results = tuner.grid_search_tuning(X, y)
    elif args.method == 'optuna':
        results = tuner.optuna_tuning(X, y, n_trials=args.n_trials)
    elif args.method == 'bayesian':
        results = tuner.bayesian_optimization(X, y)
    else:  # comprehensive
        results = tuner.comprehensive_tuning(X, y)
    
    # Save results
    tuner.save_tuning_results(args.output_dir)
    
    # Generate plots
    plots_dir = os.path.join(args.output_dir, 'plots')
    create_directory(plots_dir)
    
    if args.method == 'comprehensive':
        tuner.plot_optimization_history(results, os.path.join(plots_dir, 'optimization_history.png'))
        tuner.plot_parameter_importance(results, os.path.join(plots_dir, 'parameter_importance.png'))
    
    print(f"\nHyperparameter tuning completed!")
    print(f"Best parameters: {tuner.best_params}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()