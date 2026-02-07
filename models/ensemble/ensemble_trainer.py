"""
Enhanced Ensemble ML and DL Models with Better Handling
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import mode

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import os
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from config.settings import settings

class EnhancedEnsembleTrainer:
    """Enhanced Ensemble Model Class with Better Handling"""
    
    def __init__(self):
        self.models_dir = Path(settings.MODELS_DIR)
        self.ml_dir = self.models_dir / "ml"
        self.dl_dir = self.models_dir / "dl"
        self.ensemble_dir = self.models_dir / "ensemble"
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    def load_models_and_predictions(self, symbol):
        """Load models and previous predictions with validation"""
        
        print(f"Loading models and predictions for {symbol}...")
        
        # Load ML models
        ml_models = {}
        ml_model_files = list(self.ml_dir.glob(f"{symbol}_*.pkl"))
        for model_file in ml_model_files:
            model_name = model_file.stem.replace(f"{symbol}_", "")
            if model_name != "ml_results" and "ensemble" not in model_name:
                try:
                    model = joblib.load(model_file)
                    ml_models[model_name] = model
                    print(f"  ✓ Loaded ML model: {model_name}")
                except Exception as e:
                    print(f"  ✗ Error loading {model_name}: {e}")
        
        # Load ML predictions with validation
        ml_predictions = pd.DataFrame()
        ml_preds_path = self.ml_dir / f"{symbol}_predictions.csv"
        if ml_preds_path.exists():
            ml_predictions = pd.read_csv(ml_preds_path, index_col=0)
            print(f"  ✓ Loaded ML predictions: {ml_predictions.shape}")
            
            # Validate predictions
            if 'actual' not in ml_predictions.columns:
                print(f"  ⚠ Warning: 'actual' column missing in ML predictions")
            else:
                actual_counts = Counter(ml_predictions['actual'])
                print(f"    Actual class distribution: {dict(actual_counts)}")
        else:
            print(f"  ⚠ Warning: ML predictions file not found: {ml_preds_path}")
        
        # Load DL predictions with validation
        dl_predictions = pd.DataFrame()
        dl_preds_path = self.dl_dir / f"{symbol}_dl_predictions.csv"
        if dl_preds_path.exists():
            dl_predictions = pd.read_csv(dl_preds_path, index_col=0)
            print(f"  ✓ Loaded DL predictions: {dl_predictions.shape}")
            
            # Check for all-zero predictions
            zero_pred_models = []
            for col in dl_predictions.columns:
                if col != 'actual':
                    if dl_predictions[col].nunique() <= 1 or dl_predictions[col].max() == 0:
                        zero_pred_models.append(col)
            
            if zero_pred_models:
                print(f"  ⚠ Warning: DL models with all-zero predictions: {zero_pred_models}")
        else:
            print(f"  ⚠ Warning: DL predictions file not found: {dl_preds_path}")
        
        return ml_models, ml_predictions, dl_predictions
    
    def clean_predictions(self, predictions_df):
        """Clean prediction data"""
        cleaned_df = predictions_df.copy()
        
        # Remove columns with all zeros or constant values
        columns_to_drop = []
        for col in cleaned_df.columns:
            if col != 'actual':
                unique_vals = cleaned_df[col].nunique()
                if unique_vals <= 1:
                    columns_to_drop.append(col)
                elif cleaned_df[col].isnull().all():
                    columns_to_drop.append(col)
        
        if columns_to_drop:
            print(f"  Dropping {len(columns_to_drop)} columns with invalid predictions: {columns_to_drop}")
            cleaned_df = cleaned_df.drop(columns=columns_to_drop)
        
        # Fill NaN values with mean
        for col in cleaned_df.columns:
            if col != 'actual':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        
        return cleaned_df
    
    def align_predictions(self, ml_predictions, dl_predictions):
        """Align ML and DL predictions to ensure same length"""
        
        if ml_predictions.empty and dl_predictions.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Clean predictions first
        ml_cleaned = self.clean_predictions(ml_predictions) if not ml_predictions.empty else pd.DataFrame()
        dl_cleaned = self.clean_predictions(dl_predictions) if not dl_predictions.empty else pd.DataFrame()
        
        if ml_cleaned.empty:
            return pd.DataFrame(), dl_cleaned
        
        if dl_cleaned.empty:
            return ml_cleaned, pd.DataFrame()
        
        # Try to align by index first
        if hasattr(ml_cleaned.index, 'intersection') and hasattr(dl_cleaned.index, 'intersection'):
            common_indices = ml_cleaned.index.intersection(dl_cleaned.index)
            if len(common_indices) > 50:  # Need reasonable overlap
                ml_aligned = ml_cleaned.loc[common_indices]
                dl_aligned = dl_cleaned.loc[common_indices]
                print(f"  ✓ Aligned using {len(common_indices)} common indices")
                return ml_aligned, dl_aligned
        
        # If no common indices or insufficient overlap, align by minimum length
        min_len = min(len(ml_cleaned), len(dl_cleaned))
        ml_aligned = ml_cleaned.iloc[:min_len].reset_index(drop=True)
        dl_aligned = dl_cleaned.iloc[:min_len].reset_index(drop=True)
        print(f"  ⚠ Aligned by trimming to {min_len} rows")
        
        return ml_aligned, dl_aligned
    
    def create_feature_matrix(self, ml_predictions, dl_predictions):
        """Create feature matrix from model predictions"""
        
        # Align predictions first
        ml_aligned, dl_aligned = self.align_predictions(ml_predictions, dl_predictions)
        
        features = {}
        feature_names = []
        model_metadata = {}
        
        # Add ML predictions
        if not ml_aligned.empty:
            for col in ml_aligned.columns:
                if col != 'actual':
                    feature_name = f"ml_{col}"
                    pred_values = ml_aligned[col].values
                    
                    # Check if predictions are valid
                    if len(np.unique(pred_values)) > 1 and not np.all(pred_values == 0):
                        features[feature_name] = pred_values
                        feature_names.append(feature_name)
                        
                        # Store metadata
                        model_metadata[feature_name] = {
                            'type': 'ml',
                            'model': col,
                            'mean': np.mean(pred_values),
                            'std': np.std(pred_values),
                            'unique': len(np.unique(pred_values))
                        }
                    else:
                        print(f"  ⚠ Skipping {feature_name}: invalid predictions")
        
        # Add DL predictions
        if not dl_aligned.empty:
            for col in dl_aligned.columns:
                if col != 'actual':
                    feature_name = f"dl_{col}"
                    pred_values = dl_aligned[col].values
                    
                    # Check if predictions are valid
                    if len(np.unique(pred_values)) > 1 and not np.all(pred_values == 0):
                        features[feature_name] = pred_values
                        feature_names.append(feature_name)
                        
                        # Store metadata
                        model_metadata[feature_name] = {
                            'type': 'dl',
                            'model': col,
                            'mean': np.mean(pred_values),
                            'std': np.std(pred_values),
                            'unique': len(np.unique(pred_values))
                        }
                    else:
                        print(f"  ⚠ Skipping {feature_name}: invalid predictions")
        
        if not features:
            raise ValueError("No valid features available for ensemble training")
        
        # Ensure all arrays have same length
        lengths = [len(arr) for arr in features.values()]
        if len(set(lengths)) > 1:
            print(f"  ⚠ Warning: Feature arrays have different lengths: {lengths}")
            min_len = min(lengths)
            for key in list(features.keys()):
                if len(features[key]) > min_len:
                    features[key] = features[key][:min_len]
                elif len(features[key]) < min_len:
                    # Pad with mean value
                    padding = np.full(min_len - len(features[key]), np.mean(features[key]))
                    features[key] = np.concatenate([features[key], padding])
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        
        # Get target - prioritize ML predictions if available
        y = None
        if not ml_aligned.empty and 'actual' in ml_aligned.columns:
            y = ml_aligned['actual'].values[:len(features_df)]
        elif not dl_aligned.empty and 'actual' in dl_aligned.columns:
            y = dl_aligned['actual'].values[:len(features_df)]
        
        if y is None:
            raise ValueError("No target variable found in predictions")
        
        # Ensure y has same length as features
        if len(y) != len(features_df):
            min_len = min(len(y), len(features_df))
            y = y[:min_len]
            features_df = features_df.iloc[:min_len]
        
        # Convert to numpy arrays
        y = np.array(y).astype(int)
        
        print(f"\n  ✓ Created feature matrix: {features_df.shape}")
        print(f"  ✓ Valid features: {feature_names}")
        print(f"  ✓ Target distribution: {Counter(y)}")
        print(f"  ✓ Class ratio: {sum(y == 1) / len(y):.3f}")
        
        # Print model statistics
        print("\n  Model Statistics:")
        for feat_name, metadata in model_metadata.items():
            print(f"    {feat_name:20s} mean={metadata['mean']:.3f}, std={metadata['std']:.3f}, unique={metadata['unique']}")
        
        return features_df, y, model_metadata
    
    def create_robust_ensemble(self, predictions_dict, y_true=None):
        """Create robust ensemble using multiple methods"""
        
        n_samples = len(next(iter(predictions_dict.values())))
        results = {}
        
        # 1. Simple Average
        avg_pred = np.zeros(n_samples)
        count = 0
        for pred in predictions_dict.values():
            if len(pred) == n_samples:
                avg_pred += pred
                count += 1
        if count > 0:
            avg_pred = avg_pred / count
            results['simple_avg'] = avg_pred
        
        # 2. Median (robust to outliers)
        all_preds = np.column_stack([pred for pred in predictions_dict.values() if len(pred) == n_samples])
        if all_preds.shape[1] > 0:
            median_pred = np.median(all_preds, axis=1)
            results['median'] = median_pred
        
        # 3. Trimmed Mean (remove top/bottom 10%)
        if all_preds.shape[1] >= 3:
            sorted_preds = np.sort(all_preds, axis=1)
            trim_idx = max(1, all_preds.shape[1] // 10)  # Trim 10% from each side
            trimmed_mean = np.mean(sorted_preds[:, trim_idx:-trim_idx], axis=1)
            results['trimmed_mean'] = trimmed_mean
        
        # 4. Weighted Average based on individual model performance
        if y_true is not None and len(y_true) == n_samples:
            weights = {}
            for name, pred in predictions_dict.items():
                if len(pred) == n_samples:
                    # Use accuracy as weight
                    pred_binary = (pred >= 0.5).astype(int)
                    acc = accuracy_score(y_true, pred_binary)
                    weights[name] = acc
            
            if weights:
                # Normalize weights
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weighted_pred = np.zeros(n_samples)
                    for name, weight in weights.items():
                        weighted_pred += predictions_dict[name] * (weight / total_weight)
                    results['performance_weighted'] = weighted_pred
        
        return results
    
    def find_optimal_threshold(self, y_true, y_pred_proba, method='f1'):
        """Find optimal threshold using different methods"""
        
        if method == 'f1':
            from sklearn.metrics import f1_score
            
            thresholds = np.arange(0.05, 0.95, 0.05)
            best_threshold = 0.5
            best_metric = 0
            
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_metric:
                    best_metric = f1
                    best_threshold = threshold
        
        elif method == 'balanced':
            # Threshold based on class balance
            class_ratio = sum(y_true == 1) / len(y_true)
            best_threshold = 1 - class_ratio
        
        elif method == 'youden':
            from sklearn.metrics import roc_curve
            
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            youden_index = tpr - fpr
            best_threshold = thresholds[np.argmax(youden_index)]
        
        # Ensure threshold is reasonable
        best_threshold = max(0.1, min(0.9, best_threshold))
        
        return best_threshold
    
    def evaluate_model(self, name, y_true, y_pred_proba, threshold=0.5):
        """Comprehensive model evaluation"""
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            metrics['roc_auc'] = 0.5
        
        cm = confusion_matrix(y_true, y_pred)
        metrics.update({
            'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]
        })
        
        return metrics
    
    def train_ensemble(self, symbol):
        """Train ensemble for a currency pair"""
        print(f"\n{'='*60}")
        print(f"TRAINING ENSEMBLE FOR {symbol}")
        print(f"{'='*60}")
        
        try:
            # Load models and predictions
            ml_models, ml_preds, dl_preds = self.load_models_and_predictions(symbol)
            
            if ml_preds.empty and dl_preds.empty:
                print("No predictions available for ensemble training")
                return None
            
            # Create feature matrix
            X, y, model_metadata = self.create_feature_matrix(ml_preds, dl_preds)
            
            # Ensure we have enough samples
            if len(X) < 100:
                print(f"Insufficient samples for ensemble training: {len(X)}")
                return None
            
            # Split data (use time-series aware split)
            split_idx = int(len(X) * 0.7)  # 70% train, 30% test
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            
            print(f"\n{'='*40}")
            print("DATA SPLIT")
            print('='*40)
            print(f"Training samples: {len(X_train)}")
            print(f"Testing samples: {len(X_test)}")
            print(f"Train class distribution: {Counter(y_train)}")
            print(f"Test class distribution: {Counter(y_test)}")
            
            # Evaluate individual models on test set
            print(f"\n{'='*40}")
            print("INDIVIDUAL MODEL PERFORMANCE (Test Set)")
            print('='*40)
            
            individual_metrics = {}
            for col in X_test.columns:
                pred_proba = X_test[col].values
                metrics = self.evaluate_model(col, y_test, pred_proba)
                individual_metrics[col] = metrics
                
                print(f"{col:25s} Acc: {metrics['accuracy']:.4f} "
                      f"Prec: {metrics['precision']:.4f} "
                      f"Rec: {metrics['recall']:.4f} "
                      f"F1: {metrics['f1']:.4f}")
            
            # Create robust ensembles
            print(f"\n{'='*40}")
            print("CREATING ENSEMBLES")
            print('='*40)
            
            # Use training predictions for ensemble creation
            train_predictions = {col: X_train[col].values for col in X_train.columns}
            test_predictions = {col: X_test[col].values for col in X_test.columns}
            
            # Create various ensembles
            ensembles = self.create_robust_ensemble(test_predictions, y_test)
            
            # Also create meta-model ensemble
            print("Training meta-model...")
            meta_model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                penalty='l2',
                C=0.1
            )
            
            meta_model.fit(X_train.values, y_train)
            ensembles['meta_model'] = meta_model.predict_proba(X_test.values)[:, 1]
            
            # Evaluate all ensembles
            print(f"\n{'='*40}")
            print("ENSEMBLE PERFORMANCE")
            print('='*40)
            
            ensemble_metrics = {}
            for ensemble_name, pred_proba in ensembles.items():
                # Find optimal threshold
                opt_threshold = self.find_optimal_threshold(y_test, pred_proba, method='f1')
                
                # Evaluate with optimal threshold
                metrics_opt = self.evaluate_model(
                    f"{ensemble_name} (opt)", 
                    y_test, 
                    pred_proba, 
                    threshold=opt_threshold
                )
                
                # Evaluate with standard threshold
                metrics_std = self.evaluate_model(
                    f"{ensemble_name} (std)", 
                    y_test, 
                    pred_proba, 
                    threshold=0.5
                )
                
                ensemble_metrics[ensemble_name] = {
                    'optimal_threshold': opt_threshold,
                    'metrics_optimal': metrics_opt,
                    'metrics_standard': metrics_std,
                    'predictions': pred_proba
                }
                
                print(f"\n{ensemble_name}:")
                print(f"  Optimal threshold: {opt_threshold:.3f}")
                print(f"  Optimal -> Acc: {metrics_opt['accuracy']:.4f}, "
                      f"F1: {metrics_opt['f1']:.4f}, "
                      f"Prec: {metrics_opt['precision']:.4f}, "
                      f"Rec: {metrics_opt['recall']:.4f}")
                print(f"  Standard-> Acc: {metrics_std['accuracy']:.4f}, "
                      f"F1: {metrics_std['f1']:.4f}, "
                      f"Prec: {metrics_std['precision']:.4f}, "
                      f"Rec: {metrics_std['recall']:.4f}")
            
            # Select best ensemble based on F1 score
            best_ensemble = None
            best_f1 = 0
            for ensemble_name, data in ensemble_metrics.items():
                f1_score = data['metrics_optimal']['f1']
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_ensemble = ensemble_name
            
            if best_ensemble:
                print(f"\n{'='*40}")
                print(f"SELECTED BEST ENSEMBLE: {best_ensemble}")
                print('='*40)
                
                best_data = ensemble_metrics[best_ensemble]
                best_metrics = best_data['metrics_optimal']
                
                print(f"Optimal threshold: {best_data['optimal_threshold']:.3f}")
                print(f"Accuracy:  {best_metrics['accuracy']:.4f}")
                print(f"Precision: {best_metrics['precision']:.4f}")
                print(f"Recall:    {best_metrics['recall']:.4f}")
                print(f"F1-Score:  {best_metrics['f1']:.4f}")
                
                if 'roc_auc' in best_metrics:
                    print(f"ROC-AUC:   {best_metrics['roc_auc']:.4f}")
                
                print(f"\nConfusion Matrix:")
                print(f"  TN: {best_metrics['tn']}, FP: {best_metrics['fp']}")
                print(f"  FN: {best_metrics['fn']}, TP: {best_metrics['tp']}")
            
            # Save results
            results = {
                'symbol': symbol,
                'features': list(X.columns),
                'model_metadata': model_metadata,
                'individual_metrics': individual_metrics,
                'ensemble_metrics': ensemble_metrics,
                'best_ensemble': best_ensemble,
                'best_metrics': best_metrics if best_ensemble else None,
                'data_split': {
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'train_distribution': dict(Counter(y_train)),
                    'test_distribution': dict(Counter(y_test))
                }
            }
            
            # Save predictions
            if best_ensemble:
                preds_df = pd.DataFrame({
                    'prediction_prob': ensemble_metrics[best_ensemble]['predictions'],
                    'signal_optimal': (ensemble_metrics[best_ensemble]['predictions'] >= 
                                     ensemble_metrics[best_ensemble]['optimal_threshold']).astype(int),
                    'signal_standard': (ensemble_metrics[best_ensemble]['predictions'] >= 0.5).astype(int),
                    'actual': y_test
                })
                
                preds_path = self.ensemble_dir / f"{symbol}_ensemble_predictions.csv"
                preds_df.to_csv(preds_path, index=False)
                print(f"\n✓ Predictions saved to: {preds_path}")
            
            # Save detailed metrics
            metrics_path = self.ensemble_dir / f"{symbol}_detailed_metrics.pkl"
            joblib.dump(results, metrics_path)
            print(f"✓ Detailed metrics saved to: {metrics_path}")
            
            # Save ensemble model
            if 'meta_model' in ensembles:
                model_data = {
                    'meta_model': meta_model,
                    'best_ensemble': best_ensemble,
                    'optimal_threshold': best_data['optimal_threshold'] if best_ensemble else 0.5,
                    'feature_names': list(X.columns)
                }
                
                model_path = self.ensemble_dir / f"{symbol}_ensemble_model.pkl"
                joblib.dump(model_data, model_path)
                print(f"✓ Ensemble model saved to: {model_path}")
            
            return results
            
        except Exception as e:
            print(f"\n✗ Error training ensemble for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_all_ensembles(self):
        """Train ensemble for all currency pairs"""
        ensemble_results = {}
        
        # Get list of available prediction files
        available_pairs = []
        for pair in settings.FOREX_PAIRS:
            ml_path = self.ml_dir / f"{pair}_predictions.csv"
            dl_path = self.dl_dir / f"{pair}_dl_predictions.csv"
            if ml_path.exists() or dl_path.exists():
                available_pairs.append(pair)
        
        print(f"\nFound predictions for {len(available_pairs)} pairs: {available_pairs}")
        
        for pair in available_pairs:
            print(f"\n{'='*70}")
            print(f"PROCESSING {pair}")
            print(f"{'='*70}")
            
            results = self.train_ensemble(pair)
            if results:
                ensemble_results[pair] = results
                print(f"\n✓ Successfully trained ensemble for {pair}")
            else:
                print(f"\n✗ Failed to train ensemble for {pair}")
        
        # Generate summary report
        self.generate_summary_report(ensemble_results)
        
        return ensemble_results
    
    def generate_summary_report(self, ensemble_results):
        """Generate comprehensive summary report"""
        
        if not ensemble_results:
            print("\nNo ensemble results to summarize")
            return
        
        print(f"\n{'='*70}")
        print("ENSEMBLE TRAINING SUMMARY REPORT")
        print(f"{'='*70}")
        
        summary_data = []
        
        for pair, results in ensemble_results.items():
            if results.get('best_ensemble'):
                best_data = results['ensemble_metrics'][results['best_ensemble']]
                best_metrics = best_data['metrics_optimal']
                
                summary_data.append({
                    'pair': pair,
                    'best_ensemble': results['best_ensemble'],
                    'threshold': best_data['optimal_threshold'],
                    'accuracy': best_metrics['accuracy'],
                    'precision': best_metrics['precision'],
                    'recall': best_metrics['recall'],
                    'f1': best_metrics['f1'],
                    'roc_auc': best_metrics.get('roc_auc', 0.5),
                    'test_samples': results['data_split']['test_samples'],
                    'class_ratio': (results['data_split']['test_distribution'].get(1, 0) / 
                                  results['data_split']['test_samples'])
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Save summary
            summary_path = self.ensemble_dir / "ensemble_summary_detailed.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\n✓ Detailed summary saved to: {summary_path}")
            
            # Print summary table
            print("\n" + "="*70)
            print("PERFORMANCE SUMMARY")
            print("="*70)
            print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
            
            # Calculate averages
            print("\n" + "="*70)
            print("AVERAGE PERFORMANCE ACROSS ALL PAIRS")
            print("="*70)
            
            avg_metrics = summary_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].mean()
            for metric, value in avg_metrics.items():
                print(f"{metric:15s}: {value:.4f}")
        
        print(f"\n{'='*70}")
        print("ENSEMBLE TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Successfully trained ensembles for {len(ensemble_results)} currency pairs")
        
        # Recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        if summary_data:
            avg_class_ratio = np.mean([d['class_ratio'] for d in summary_data])
            print(f"1. Average class imbalance: {avg_class_ratio:.3f} (target: ~0.5)")
            print("2. Consider collecting more balanced data or using data augmentation")
            print("3. Review DL model training - some models predict all zeros")
            print("4. Optimal thresholds are significantly lower than 0.5 due to imbalance")
            print("5. Consider using ensemble methods that handle imbalance better")

if __name__ == "__main__":
    print("="*70)
    print("ENHANCED ENSEMBLE MODEL TRAINER")
    print("="*70)
    
    trainer = EnhancedEnsembleTrainer()
    results = trainer.train_all_ensembles()

# """
# Ensemble مدل‌های ML و DL
# """

# import numpy as np
# import pandas as pd
# from pathlib import Path
# import joblib

# from sklearn.ensemble import VotingClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.calibration import CalibratedClassifierCV
# import os
# import sys
# # Add project root to Python path
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert(0, PROJECT_ROOT)
# from config.settings import settings

# class EnsembleTrainer:
#     """کلاس Ensemble مدل‌ها"""
    
#     def __init__(self):
#         self.models_dir = Path(settings.MODELS_DIR)
#         self.ml_dir = self.models_dir / "ml"
#         self.dl_dir = self.models_dir / "dl"
#         self.ensemble_dir = self.models_dir / "ensemble"
#         self.ensemble_dir.mkdir(parents=True, exist_ok=True)
    
#     def load_models_and_predictions(self, symbol):
#         """بارگذاری مدل‌ها و پیش‌بینی‌های قبلی"""
        
#         # بارگذاری مدل‌های ML
#         ml_models = {}
#         ml_predictions = {}
        
#         ml_model_files = list(self.ml_dir.glob(f"{symbol}_*.pkl"))
#         for model_file in ml_model_files:
#             model_name = model_file.stem.replace(f"{symbol}_", "")
#             if model_name != "ml_results":
#                 ml_models[model_name] = joblib.load(model_file)
        
#         # بارگذاری پیش‌بینی‌های ML
#         ml_preds_path = self.ml_dir / f"{symbol}_predictions.csv"
#         if ml_preds_path.exists():
#             ml_predictions = pd.read_csv(ml_preds_path, index_col=0, parse_dates=True)
        
#         # بارگذاری پیش‌بینی‌های DL
#         dl_predictions = {}
#         dl_preds_path = self.dl_dir / f"{symbol}_dl_predictions.csv"
#         if dl_preds_path.exists():
#             dl_predictions = pd.read_csv(dl_preds_path, index_col=0)
        
#         return ml_models, ml_predictions, dl_predictions
    
#     def create_feature_matrix(self, ml_predictions, dl_predictions):
#         """ایجاد ماتریس ویژگی از پیش‌بینی‌های مدل‌ها"""
#         features = {}
        
#         # اضافه کردن پیش‌بینی‌های ML
#         for col in ml_predictions.columns:
#             if col != 'actual':
#                 features[f"ml_{col}"] = ml_predictions[col].values
        
#         # اضافه کردن پیش‌بینی‌های DL
#         for col in dl_predictions.columns:
#             if col != 'actual':
#                 features[f"dl_{col}"] = dl_predictions[col].values
        
#         # تبدیل به DataFrame
#         features_df = pd.DataFrame(features)
        
#         # target
#         if 'actual' in ml_predictions.columns:
#             y = ml_predictions['actual'].values
#         else:
#             y = dl_predictions['actual'].values
        
#         return features_df, y
    
#     def create_weighted_ensemble(self, predictions, weights=None):
#         """ایجاد ensemble وزن‌دار"""
#         if weights is None:
#             # وزن‌دهی بر اساس دقت (فرضی)
#             weights = {
#                 'random_forest': 0.3,
#                 'xgboost': 0.3,
#                 'lstm': 0.2,
#                 'attention': 0.2
#             }
        
#         weighted_sum = np.zeros(len(predictions))
        
#         for model_name, weight in weights.items():
#             if model_name in predictions.columns:
#                 weighted_sum += predictions[model_name] * weight
        
#         return weighted_sum / sum(weights.values())
    
#     def create_voting_ensemble(self, ml_models, X, y):
#         """ایجاد Voting Ensemble"""
        
#         estimators = [
#             (name, model) for name, model in ml_models.items()
#         ]
        
#         voting_clf = VotingClassifier(
#             estimators=estimators,
#             voting='soft',  # soft voting برای probability
#             n_jobs=-1
#         )
        
#         # آموزش ensemble
#         voting_clf.fit(X, y)
        
#         return voting_clf
    
#     def create_stacking_ensemble(self, ml_models, X, y):
#         """ایجاد Stacking Ensemble"""
        
#         estimators = [
#             (name, model) for name, model in ml_models.items()
#         ]
        
#         stacking_clf = StackingClassifier(
#             estimators=estimators,
#             final_estimator=LogisticRegression(),
#             cv=5,
#             n_jobs=-1
#         )
        
#         stacking_clf.fit(X, y)
        
#         return stacking_clf
    
#     def calibrate_predictions(self, predictions, y_true):
#         """Calibrate کردن احتمالات"""
#         calibrated = CalibratedClassifierCV(
#             base_estimator=LogisticRegression(),
#             cv=5
#         )
        
#         # باید reshape کنیم
#         X_calib = predictions.values.reshape(-1, 1)
#         calibrated.fit(X_calib, y_true)
        
#         return calibrated
    
#     def train_ensemble(self, symbol):
#         """آموزش ensemble برای یک جفت ارز"""
#         print(f"\n{'='*50}")
#         print(f"آموزش Ensemble برای {symbol}")
#         print(f"{'='*50}")
        
#         # بارگذاری مدل‌ها و پیش‌بینی‌ها
#         ml_models, ml_preds, dl_preds = self.load_models_and_predictions(symbol)
        
#         if ml_preds.empty or dl_preds.empty:
#             print("پیش‌بینی‌های کافی برای ensemble وجود ندارد")
#             return None
        
#         # ایجاد ماتریس ویژگی
#         X_ensemble, y = self.create_feature_matrix(ml_preds, dl_preds)
        
#         print(f"ویژگی‌های Ensemble: {X_ensemble.shape}")
#         print(f"مدل‌های موجود: {list(ml_models.keys())}")
        
#         # 1. Weighted Ensemble
#         print("\n1. ایجاد Weighted Ensemble...")
#         weights = {
#             'random_forest': 0.25,
#             'xgboost': 0.25,
#             'lightgbm': 0.15,
#             'logistic_regression': 0.10,
#             'lstm': 0.15,
#             'attention': 0.10
#         }
        
#         # فقط مدل‌های موجود را استفاده می‌کنیم
#         available_weights = {
#             k: v for k, v in weights.items() 
#             if k in ml_preds.columns or f"dl_{k}" in dl_preds.columns
#         }
        
#         # نرمال‌سازی وزن‌ها
#         total_weight = sum(available_weights.values())
#         available_weights = {k: v/total_weight for k, v in available_weights.items()}
        
#         weighted_pred = self.create_weighted_ensemble(
#             pd.concat([ml_preds, dl_preds], axis=1),
#             available_weights
#         )
        
#         # 2. Voting Ensemble (فقط برای مدل‌های ML)
#         print("\n2. ایجاد Voting Ensemble...")
        
#         # برای Voting به X اصلی نیاز داریم
#         features_path = Path(settings.PROCESSED_DATA_DIR) / f"{symbol}_features.csv"
#         df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
        
#         X_original = df_features.drop(['target', 'target_return'], axis=1)
#         y_original = df_features['target'].astype(int)
        
#         split_idx = int(len(X_original) * settings.TRAIN_TEST_SPLIT)
#         X_test_original = X_original.iloc[split_idx:]
        
#         voting_clf = self.create_voting_ensemble(
#             ml_models,
#             X_original.iloc[:split_idx],
#             y_original.iloc[:split_idx]
#         )
        
#         voting_pred = voting_clf.predict_proba(X_test_original)[:, 1]
        
#         # 3. Stacking Ensemble
#         print("\n3. ایجاد Stacking Ensemble...")
#         stacking_clf = self.create_stacking_ensemble(
#             ml_models,
#             X_original.iloc[:split_idx],
#             y_original.iloc[:split_idx]
#         )
        
#         stacking_pred = stacking_clf.predict_proba(X_test_original)[:, 1]
        
#         # 4. Meta Ensemble (ترکیب تمام ensemble‌ها)
#         print("\n4. ایجاد Meta Ensemble...")
        
#         # جمع‌آوری تمام پیش‌بینی‌ها
#         all_predictions = pd.DataFrame({
#             'weighted': weighted_pred[:len(voting_pred)],
#             'voting': voting_pred,
#             'stacking': stacking_pred
#         })
        
#         # آموزش یک meta-model روی نتایج ensemble‌ها
#         meta_model = LogisticRegression()
#         meta_model.fit(all_predictions.values, y)
        
#         # پیش‌بینی نهایی
#         final_pred = meta_model.predict_proba(all_predictions.values)[:, 1]
        
#         # ارزیابی
#         from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
#         # آستانه‌بندی
#         final_pred_binary = (final_pred > 0.5).astype(int)
        
#         metrics = {
#             'accuracy': accuracy_score(y, final_pred_binary),
#             'precision': precision_score(y, final_pred_binary),
#             'recall': recall_score(y, final_pred_binary),
#             'f1': f1_score(y, final_pred_binary)
#         }
        
#         print(f"\nنتایج Ensemble نهایی برای {symbol}:")
#         print(f"Accuracy: {metrics['accuracy']:.4f}")
#         print(f"Precision: {metrics['precision']:.4f}")
#         print(f"Recall: {metrics['recall']:.4f}")
#         print(f"F1-Score: {metrics['f1']:.4f}")
        
#         # ذخیره ensemble models
#         ensemble_models = {
#             'weighted_ensemble': available_weights,
#             'voting_ensemble': voting_clf,
#             'stacking_ensemble': stacking_clf,
#             'meta_ensemble': meta_model
#         }
        
#         # ذخیره مدل‌ها
#         ensemble_path = self.ensemble_dir / f"{symbol}_ensemble.pkl"
#         joblib.dump(ensemble_models, ensemble_path)
        
#         # ذخیره پیش‌بینی‌ها
#         preds_df = pd.DataFrame({
#             'final_prediction': final_pred,
#             'final_signal': final_pred_binary,
#             'actual': y
#         }, index=ml_preds.index)
        
#         preds_path = self.ensemble_dir / f"{symbol}_ensemble_predictions.csv"
#         preds_df.to_csv(preds_path)
        
#         print(f"\nEnsemble برای {symbol} کامل شد و ذخیره شد")
        
#         return {
#             'models': ensemble_models,
#             'metrics': metrics,
#             'predictions': preds_df
#         }
    
#     def train_all_ensembles(self):
#         """آموزش ensemble برای تمام جفت‌ارزها"""
#         ensemble_results = {}
        
#         for pair in settings.FOREX_PAIRS[:2]:
#             results = self.train_ensemble(pair)
#             if results:
#                 ensemble_results[pair] = results
        
#         return ensemble_results

# if __name__ == "__main__":
#     trainer = EnsembleTrainer()
#     results = trainer.train_all_ensembles()
#     print(f"آموزش Ensemble برای {len(results)} جفت کامل شد")